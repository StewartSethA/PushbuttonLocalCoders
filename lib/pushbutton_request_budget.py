"""Request admission; estimates are never represented as tokenizer counts."""
from __future__ import annotations

import http.client
import json
from dataclasses import dataclass
from urllib.parse import urlsplit


class BudgetError(ValueError):
    pass


SLA_WARNING = "decode SLA UNKNOWN: min_tps is a target, not a guarantee"


def integer(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise BudgetError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def direct_admission_limit(capacity):
    limit = capacity.get("admission_limit")
    limit = 1 if limit is None else integer(limit, "admission_limit")
    slots = integer(capacity.get("slots", 1), "slots")
    explicit = capacity.get("admission_explicit") is True
    if "admission_explicit" not in capacity:
        # Resolved plans default to C1; a larger supplied limit is explicit.
        explicit = limit > 1
    if explicit or capacity.get("admission_proven") is True:
        return min(limit, slots)
    return 1


def output_tokens(body, capacity):
    values = [integer(body[k], k) for k in ("max_tokens", "max_completion_tokens", "max_output_tokens") if k in body]
    if len(set(values)) > 1:
        raise BudgetError("conflicting output token limits")
    output = values[0] if values else integer(capacity.get("output_tokens", 4096), "output_tokens")
    if output > integer(capacity.get("output_tokens", output), "output_tokens"):
        raise BudgetError("requested output exceeds this route's output budget (reasoning tokens included); lower max_tokens")
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        reasoning = integer(thinking.get("budget_tokens"), "thinking.budget_tokens")
        if reasoning >= output:
            raise BudgetError("thinking budget must fit inside the total output budget, with room for the answer")
    return output


def estimate_tokens(body):
    """UTF-8 byte-sized admission estimate, including all request fields.

    This deliberately overestimates ordinary text, but cannot bound image/audio
    expansion or unknown chat templates. A backend hard guard is still required.
    """
    return len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 32


def backend_json(endpoint, path, body=None, headers=None):
    u = urlsplit(endpoint)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = cls(u.hostname, u.port, timeout=5)
    base = u.path.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    raw = None if body is None else json.dumps(body).encode()
    try:
        safe_headers = {k: v for k, v in (headers or {}).items()
                        if k.lower() not in {"content-length", "host", "transfer-encoding", "connection", "accept-encoding"}}
        conn.request("GET" if body is None else "POST", base + path, raw,
                     {"Content-Type": "application/json", **safe_headers})
        response = conn.getresponse()
        if response.status in (404, 405, 501):
            response.read()
            return None
        if response.status != 200:
            response.read()
            raise BudgetError(f"backend token/context validation failed (HTTP {response.status})")
        return json.loads(response.read())
    except BudgetError:
        raise
    except Exception:
        raise BudgetError("backend token/context validation unavailable; retry after checking backend health") from None
    finally:
        conn.close()


class TokenCount(int):
    def __new__(cls, value, context_limit=None):
        count = super().__new__(cls, value)
        count.context_limit = context_limit
        return count


def is_vllm(backend):
    return isinstance(backend, str) and backend.lower().startswith("vllm")


def has_multimodal(body):
    return bool(body.get("attachments") or body.get("images") or body.get("audio")) or any(
        isinstance(message.get("content"), list) and any(
            isinstance(block, dict) and block.get("type") not in {"text", "input_text"}
            for block in message["content"])
        for message in body.get("messages", []) if isinstance(message, dict))


def vllm_tokens(endpoint, body, headers):
    if "messages" in body and not isinstance(body["messages"], list):
        raise BudgetError("vLLM native tokenization requires a messages array")
    if has_multimodal(body):
        return None
    if "messages" in body:
        request = {k: body[k] for k in (
            "model", "messages", "tools", "tool_choice", "chat_template",
            "chat_template_kwargs") if k in body}
        request.update(add_generation_prompt=True, continue_final_message=False, add_special_tokens=True)
    else:
        if not isinstance(body.get("prompt"), str):
            return None
        request = {"model": body.get("model"), "prompt": body["prompt"], "add_special_tokens": True}
    result = backend_json(endpoint, "/tokenize", request, headers)
    if result is None:
        return None
    if not isinstance(result, dict):
        raise BudgetError("vLLM native /tokenize did not return a count object")
    count = integer(result.get("count"), "vLLM full-request count", zero=True)
    if "tokens" in result and (not isinstance(result["tokens"], list) or len(result["tokens"]) != count):
        raise BudgetError("vLLM native /tokenize count and tokens disagree; cannot trust partial counts")
    context = result.get("max_model_len")
    if context is not None:
        context = integer(context, "vLLM max_model_len")
    return TokenCount(count, context)


def authoritative_tokens(endpoint, body, protocol="openai", headers=None, backend=None):
    if protocol == "anthropic":
        count_body = {k: v for k, v in body.items() if k not in {
            "max_tokens", "stream", "temperature", "top_p", "top_k", "chat_template_kwargs"
        }}
        result = backend_json(endpoint, "/v1/messages/count_tokens", count_body, headers)
        if result is None:
            return None
        return integer(result.get("input_tokens"), "backend input_tokens", zero=True)
    if is_vllm(backend):
        return vllm_tokens(endpoint, body, headers)
    if "messages" in body:
        rendered = backend_json(endpoint, "/apply-template", {**body, "add_generation_prompt": True}, headers)
        if rendered is None:
            return None
        prompt = rendered.get("prompt")
        if not isinstance(prompt, str):
            raise BudgetError("backend apply-template did not return a rendered prompt")
        # Tokenizing a text-only rendering cannot prove multimodal expansion.
        if has_multimodal(body):
            return None
    else:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            return None
    # A possibly duplicated BOS is conservative; omitting a required BOS is not.
    result = backend_json(endpoint, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True}, headers)
    if result is None:
        return None
    tokens = result.get("tokens")
    if not isinstance(tokens, list):
        raise BudgetError("backend tokenize did not return tokens")
    return len(tokens)


def per_slot_context(props):
    for key in ("n_ctx_slot", "n_ctx_per_slot"):
        if props.get(key) is not None:
            return props[key]
    return (props.get("default_generation_settings") or {}).get("n_ctx")


def verify_backend_capacity(endpoint, capacity):
    """Startup proof from per-slot properties, never aggregate KV context."""
    if not isinstance(capacity, dict):
        raise BudgetError("capacity must be a JSON object")
    requested_context = integer(capacity.get("context"), "requested context")
    requested_slots = integer(capacity.get("slots", 1), "requested slots")
    props = backend_json(endpoint, "/props")
    if not isinstance(props, dict):
        raise BudgetError("backend /props cannot prove per-slot capacity")
    actual = per_slot_context(props)
    actual = integer(actual, "backend per-slot context")
    slots = props.get("total_slots")
    if slots is None:
        slots = props.get("n_parallel")
    slots = integer(slots, "backend slots")
    if actual < requested_context or slots < requested_slots:
        raise BudgetError(f"backend proves context={actual}, slots={slots}; requested context={requested_context}, slots={requested_slots}")
    return {"context": actual, "slots": slots, "source": "backend-props"}


@dataclass(frozen=True)
class Admission:
    input_tokens: int
    output_tokens: int
    safety_tokens: int
    context: int
    source: str

    @property
    def total(self):
        return self.input_tokens + self.output_tokens + self.safety_tokens


def enforce(body, capacity, endpoint, protocol="openai", headers=None, *, count_only=False, context_cache=None, backend=None):
    context = integer(capacity.get("context"), "route context")
    safety = integer(capacity.get("safety_tokens", 0), "safety_tokens", zero=True)
    output = 0 if count_only else output_tokens(body, capacity)
    if output + safety >= context:
        raise BudgetError("output plus safety exceeds route context; lower output budget")
    if context_cache is not None and endpoint in context_cache:
        props = context_cache[endpoint]
    else:
        props = backend_json(endpoint, "/props", headers=headers)
        if context_cache is not None:
            context_cache[endpoint] = props
    if isinstance(props, dict):
        actual = per_slot_context(props)
        if actual is not None:
            context = min(context, integer(actual, "backend per-slot context"))
    else:
        actual = None
    context_proven = actual is not None
    if output + safety > context:
        raise BudgetError("output plus safety exceeds backend per-slot context; lower output budget")
    count = authoritative_tokens(endpoint, body, protocol, headers, backend=backend)
    source = "backend-tokenizer"
    if isinstance(count, TokenCount):
        source += "/vllm-native"
        if count.context_limit is not None:
            context = min(context, count.context_limit)
            context_proven = True
    if is_vllm(backend) and (count is None or not context_proven):
        raise BudgetError("vLLM native /tokenize must expose full-request count and max_model_len; upgrade backend tokenization support or select a guarded llama.cpp route")
    if backend is not None and str(backend).lower() not in {"resident", "llama.cpp", "llamampere", "volta-llama"} and not is_vllm(backend):
        if count is None or not context_proven:
            raise BudgetError(f"{backend} lacks validated full-request token counting or physical-context proof; enable supported native tokenization and ceiling reporting or select a guarded llama.cpp route")
    if not context_proven and capacity.get("no_context_shift") is not True:
        raise BudgetError("backend /props cannot prove per-slot context; require verified --no-context-shift before serving this route")
    if output + safety > context:
        raise BudgetError("output plus safety exceeds backend per-slot context; lower output budget")
    if count is None:
        if capacity.get("no_context_shift") is not True:
            raise BudgetError("backend cannot count the full request; require a verified --no-context-shift backend or supported token endpoints")
        count = estimate_tokens(body)
        source = "conservative-estimate; uncertain tokenization; backend --no-context-shift is final hard guard"
    elif not context_proven:
        source += "; backend context unverified; --no-context-shift is final hard guard"
    limit = min(integer(capacity.get("input_tokens", context - output - safety), "input_tokens"),
                context - output - safety)
    if count > limit:
        raise BudgetError(
            f"request input {count} ({source}) exceeds route input budget {limit}; "
            "compact/summarize history and oversized tool results or reduce attachments; history was not truncated")
    return Admission(count, output, safety, context, source)


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Verify a launched backend's per-slot capacity")
    parser.add_argument("--check-backend", required=True)
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--slots", default=1, type=int)
    args = parser.parse_args()
    try:
        proof = verify_backend_capacity(args.check_backend, {"context": args.context, "slots": args.slots})
    except BudgetError as exc:
        print(f"Capacity readiness failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(proof))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
