"""Request admission; estimates are never represented as tokenizer counts."""
from __future__ import annotations

import http.client
import json
from dataclasses import dataclass
from urllib.parse import urlsplit


class BudgetError(ValueError):
    pass


def integer(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise BudgetError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


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
                        if k.lower() not in {"content-length", "host", "transfer-encoding", "connection"}}
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


def authoritative_tokens(endpoint, body, protocol="openai", headers=None):
    if protocol == "anthropic":
        count_body = {k: v for k, v in body.items() if k not in {
            "max_tokens", "stream", "temperature", "top_p", "top_k", "chat_template_kwargs"
        }}
        result = backend_json(endpoint, "/v1/messages/count_tokens", count_body, headers)
        if result is None:
            return None
        return integer(result.get("input_tokens"), "backend input_tokens", zero=True)
    if "messages" in body:
        rendered = backend_json(endpoint, "/apply-template", body, headers)
        if rendered is None:
            return None
        prompt = rendered.get("prompt")
        if not isinstance(prompt, str):
            raise BudgetError("backend apply-template did not return a rendered prompt")
        # Tokenizing a text-only rendering cannot prove multimodal expansion.
        if any(isinstance(m.get("content"), list) and any(
                isinstance(b, dict) and b.get("type") not in {"text", "input_text"}
                for b in m["content"]) for m in body.get("messages", []) if isinstance(m, dict)):
            return None
    else:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            return None
    result = backend_json(endpoint, "/tokenize", {"content": prompt, "add_special": True}, headers)
    if result is None:
        return None
    tokens = result.get("tokens")
    if not isinstance(tokens, list):
        raise BudgetError("backend tokenize did not return tokens")
    return len(tokens)


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


def enforce(body, capacity, endpoint, protocol="openai", headers=None, *, count_only=False):
    context = integer(capacity.get("context"), "route context")
    safety = integer(capacity.get("safety_tokens", 0), "safety_tokens", zero=True)
    output = 0 if count_only else output_tokens(body, capacity)
    if output + safety >= context:
        raise BudgetError("output plus safety exceeds route context; lower output budget")
    props = backend_json(endpoint, "/props", headers=headers)
    if isinstance(props, dict):
        defaults = props.get("default_generation_settings") or {}
        actual = defaults.get("n_ctx") or props.get("n_ctx")
        if actual is not None:
            context = min(context, integer(actual, "backend per-slot context"))
    count = authoritative_tokens(endpoint, body, protocol, headers)
    source = "backend-tokenizer"
    if count is None:
        if not capacity.get("no_context_shift"):
            raise BudgetError("backend cannot count the full request; require a verified --no-context-shift backend or supported token endpoints")
        count = estimate_tokens(body)
        source = "conservative-estimate; uncertain tokenization; backend --no-context-shift is final hard guard"
    limit = min(integer(capacity.get("input_tokens", context - output - safety), "input_tokens"),
                context - output - safety)
    if count > limit:
        raise BudgetError(
            f"request input {count} ({source}) exceeds route input budget {limit}; "
            "compact/summarize history and oversized tool results or reduce attachments; history was not truncated")
    return Admission(count, output, safety, context, source)
