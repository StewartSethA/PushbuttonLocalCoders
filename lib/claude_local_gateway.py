#!/usr/bin/env python3
"""Anthropic-wire router for claude-local.

Routes Claude family/model aliases to local llama-server instances and applies a
low-latency Qwen agent policy. Qwen3.6/3.8 use their native XML tool dialect;
llama.cpp parses that into Anthropic `tool_use` blocks.

For tool-bearing requests this gateway buffers the backend response until tool
arguments have been validated against the exact JSON schemas Claude Code sent.
Malformed/raw tool markup is retried before any bytes are committed downstream,
and is never handed to Claude Code as a bogus invocation.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}
RAW_TOOL_MARKERS = (
    "<tool_call", "</tool_call", "<function=", "</function>",
    "<parameter=", "</parameter>", "</prompt>", "</tool>", "</tool_calls>",
)


def family_for(model: str) -> str | None:
    m = (model or "").lower()
    for family in ("haiku", "sonnet", "opus", "fable"):
        if family in m:
            return family
    return None


def thinking_enabled() -> bool:
    return os.environ.get("CLAUDE_LOCAL_ENABLE_THINKING", "0").lower() in {"1", "true", "yes", "on"}


def apply_local_policy(body: dict) -> None:
    """Keep agent turns fast and use Qwen's native tool format."""
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}

    # Do not force a JSON tool dialect. Current Qwen3.6/3.8 models are trained to
    # emit tagged XML tools and llama.cpp's native Qwen parser is built for that.
    kwargs.pop("tool_call_format", None)

    if not thinking_enabled():
        # Cover the names used by both Qwen templates and recent llama.cpp builds.
        kwargs["enable_thinking"] = False
        kwargs["preserve_thinking"] = False
        kwargs["preserve_reasoning"] = False

    body["chat_template_kwargs"] = kwargs

    # Tool serialization is not a creativity task. Lower temperature materially
    # reduces malformed Qwen3.6 XML/argument output in current llama.cpp.
    if body.get("tools"):
        body["temperature"] = 0.0


class Router:
    def __init__(self, config: dict):
        self.roles: dict[str, dict] = config["roles"]
        self.alias_map: dict[str, dict] = {}
        for role, route in self.roles.items():
            self.alias_map[str(route["model_id"]).lower()] = route
            self.alias_map[role] = route

    def resolve(self, model: str) -> dict:
        key = (model or "").lower()
        if key in self.alias_map:
            return self.alias_map[key]
        fam = family_for(key)
        if fam and fam in self.roles:
            return self.roles[fam]
        return self.roles["sonnet"]

    def unique_routes(self) -> list[dict]:
        out: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for route in self.roles.values():
            key = (str(route.get("url", "")), str(route.get("backend_alias", "")))
            if key not in seen:
                seen.add(key)
                out.append(route)
        return out


def backend_connection(route: dict) -> http.client.HTTPConnection:
    parsed = urlsplit(route["url"])
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    return cls(parsed.hostname, parsed.port, timeout=route.get("timeout", 3600))


def tool_schemas(tools: object) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not isinstance(tools, list):
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        # Anthropic shape.
        if isinstance(t.get("name"), str):
            schema = t.get("input_schema")
            if isinstance(schema, dict):
                out[t["name"]] = schema
            continue
        # OpenAI shape, useful for direct/probe compatibility.
        fn = t.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            schema = fn.get("parameters")
            if isinstance(schema, dict):
                out[fn["name"]] = schema
    return out


def validate_schema(value, schema: dict, path: str = "$") -> list[str]:
    """Small JSON-Schema validator for the subset used by Claude Code tools."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors

    if "enum" in schema and value not in schema.get("enum", []):
        errors.append(f"{path}: value {value!r} not in enum {schema.get('enum')!r}")
        return errors

    typ = schema.get("type")
    if isinstance(typ, list):
        candidates = [dict(schema, type=t) for t in typ]
        if not any(not validate_schema(value, s, path) for s in candidates):
            errors.append(f"{path}: value does not match any allowed type {typ!r}")
        return errors

    if typ == "object" or (typ is None and ("properties" in schema or "required" in schema)):
        if not isinstance(value, dict):
            return [f"{path}: expected object, got {type(value).__name__}"]
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for k in required:
            if k not in value:
                errors.append(f"{path}: missing required property {k!r}")
        for k, v in value.items():
            if k in props and isinstance(props[k], dict):
                errors.extend(validate_schema(v, props[k], f"{path}.{k}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {k!r}")
        return errors

    if typ == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array, got {type(value).__name__}"]
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                errors.extend(validate_schema(item, items, f"{path}[{i}]"))
        return errors
    if typ == "string" and not isinstance(value, str):
        errors.append(f"{path}: expected string, got {type(value).__name__}")
    elif typ == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        errors.append(f"{path}: expected integer, got {type(value).__name__}")
    elif typ == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        errors.append(f"{path}: expected number, got {type(value).__name__}")
    elif typ == "boolean" and not isinstance(value, bool):
        errors.append(f"{path}: expected boolean, got {type(value).__name__}")
    elif typ == "null" and value is not None:
        errors.append(f"{path}: expected null")
    return errors


def validate_tool_uses(uses: list[dict], tools: object) -> list[str]:
    schemas = tool_schemas(tools)
    errors: list[str] = []
    for call in uses:
        name = call.get("name")
        inp = call.get("input")
        if not isinstance(name, str):
            errors.append("tool call has no string name")
            continue
        schema = schemas.get(name)
        if schema is None:
            errors.append(f"unknown tool {name!r}")
            continue
        errors.extend(f"{name}: {e}" for e in validate_schema(inp, schema))
    return errors


def parse_nonstream_tool_uses(payload: bytes) -> tuple[list[dict], list[str]]:
    try:
        obj = json.loads(payload)
    except Exception as exc:
        return [], [f"non-stream response is not JSON: {exc}"]
    serialized = json.dumps(obj, ensure_ascii=False)
    leaks = [m for m in RAW_TOOL_MARKERS if m in serialized]
    blocks = obj.get("content") if isinstance(obj, dict) else None
    uses = [b for b in blocks or [] if isinstance(b, dict) and b.get("type") == "tool_use"] if isinstance(blocks, list) else []
    return uses, ([f"raw tool markup leaked: {leaks}"] if leaks else [])


def parse_sse_tool_uses(payload: bytes) -> tuple[list[dict], list[str]]:
    starts: dict[int, dict] = {}
    partial: dict[int, str] = {}
    errors: list[str] = []
    text = payload.decode("utf-8", "replace")
    leaks = [m for m in RAW_TOOL_MARKERS if m in text]
    if leaks:
        errors.append(f"raw tool markup leaked: {leaks}")
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except Exception:
            continue
        idx = event.get("index")
        if not isinstance(idx, int):
            continue
        if event.get("type") == "content_block_start":
            block = event.get("content_block")
            if isinstance(block, dict) and block.get("type") == "tool_use":
                starts[idx] = {"type": "tool_use", "name": block.get("name"), "input": block.get("input") or {}}
                partial[idx] = ""
        elif event.get("type") == "content_block_delta" and idx in starts:
            delta = event.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
                partial[idx] += str(delta.get("partial_json") or "")
    uses: list[dict] = []
    for idx, block in starts.items():
        raw = partial.get(idx, "")
        if raw:
            try:
                block["input"] = json.loads(raw)
            except Exception as exc:
                errors.append(f"{block.get('name')}: malformed streamed input JSON: {exc}: {raw[:500]!r}")
        uses.append(block)
    return uses, errors


def validate_backend_payload(payload: bytes, content_type: str, tools: object) -> list[str]:
    if "text/event-stream" in content_type:
        uses, errors = parse_sse_tool_uses(payload)
    else:
        uses, errors = parse_nonstream_tool_uses(payload)
    errors.extend(validate_tool_uses(uses, tools))
    return errors


def probe_tool_route(route: dict) -> None:
    """Use a realistic Claude-style multi-field tool schema as a startup gate."""
    expected = {
        "pattern": "needle",
        "path": ".",
        "output_mode": "files_with_matches",
        "head_limit": 7,
    }
    body = {
        "model": route["backend_alias"],
        "max_tokens": 384,
        "temperature": 0,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": "Call Grep exactly once with pattern=needle, path=., output_mode=files_with_matches, head_limit=7. Do not answer in prose.",
        }],
        "tools": [{
            "name": "Grep",
            "description": "Search files for a pattern and return matching file paths.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                    "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
                    "head_limit": {"type": "integer"},
                },
                "required": ["pattern", "path", "output_mode", "head_limit"],
                "additionalProperties": False,
            },
        }],
        "tool_choice": {"type": "tool", "name": "Grep"},
    }
    apply_local_policy(body)
    raw = json.dumps(body, separators=(",", ":")).encode()
    conn = backend_connection(route)
    try:
        conn.request("POST", "/v1/messages", body=raw, headers={"content-type": "application/json", "content-length": str(len(raw))})
        resp = conn.getresponse()
        payload = resp.read()
        content_type = (resp.getheader("content-type") or "").lower()
    finally:
        conn.close()
    text = payload.decode("utf-8", "replace")
    if not (200 <= resp.status < 300):
        raise RuntimeError(f"tool probe HTTP {resp.status} for {route['backend_alias']}: {text[:2500]}")
    uses, parse_errors = parse_nonstream_tool_uses(payload)
    errors = parse_errors + validate_tool_uses(uses, body["tools"])
    good = any(u.get("name") == "Grep" and u.get("input") == expected for u in uses)
    if not good:
        errors.append(f"expected exact Grep input {expected!r}; got {uses!r}")
    if errors:
        raise RuntimeError(f"Claude-shaped tool probe failed for {route['backend_alias']}: {'; '.join(errors)}; response={text[:3000]}")


def probe_all_routes(router: Router) -> None:
    for route in router.unique_routes():
        print(f"claude-local gateway: validating Claude-shaped tool arguments on {route['backend_alias']}...", flush=True)
        probe_tool_route(route)
        print(f"claude-local gateway: tool schema probe PASS on {route['backend_alias']}", flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "claude-local/0.4"

    @property
    def router(self) -> Router:
        return self.server.router  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        if self.server.verbose:  # type: ignore[attr-defined]
            sys.stderr.write("gateway: " + (fmt % args) + "\n")
            sys.stderr.flush()

    def _json(self, status: int, obj: dict):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/health", "/health/liveliness"):
            return self._json(200, {"status": "ok", "local": True, "tool_probe": "claude-shaped-pass", "thinking": thinking_enabled()})
        if path in ("/v1/models", "/models"):
            seen = set(); data = []
            for route in self.router.roles.values():
                mid = route["model_id"]
                if mid not in seen:
                    data.append({"id": mid, "object": "model", "owned_by": "claude-local"}); seen.add(mid)
            return self._json(200, {"object": "list", "data": data})
        return self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": "local gateway route not found"}})

    def _forward_headers(self, resp: http.client.HTTPResponse, content_length: str | None):
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            kl = k.lower()
            if kl in HOP_HEADERS or kl == "content-length":
                continue
            self.send_header(k, v)
        if content_length:
            self.send_header("content-length", content_length)
        else:
            self.send_header("connection", "close"); self.close_connection = True
        self.end_headers()

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/v1/messages", "/v1/messages/count_tokens"):
            return self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": "local gateway route not found"}})
        try:
            n = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            return self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": f"bad JSON: {exc}"}})

        route = self.router.resolve(str(body.get("model", "")))
        body["model"] = route["backend_alias"]
        apply_local_policy(body)
        raw = json.dumps(body, separators=(",", ":")).encode()
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        headers["content-type"] = "application/json"; headers["content-length"] = str(len(raw))

        attempts = int(route.get("proxy_attempts", 2))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            conn = backend_connection(route)
            committed = False
            try:
                conn.request("POST", path, body=raw, headers=headers)
                resp = conn.getresponse()
                content_type = (resp.getheader("content-type") or "").lower()
                content_length = resp.getheader("content-length")

                if resp.status in (500, 502, 503, 504) and attempt < attempts:
                    try: resp.read()
                    except Exception: pass
                    last_exc = RuntimeError(f"backend HTTP {resp.status}")
                    continue

                # Tool-bearing turns are buffered so malformed arguments can be
                # rejected/retried before Claude Code executes them.
                if body.get("tools") and 200 <= resp.status < 300:
                    payload = resp.read()
                    errors = validate_backend_payload(payload, content_type, body.get("tools"))
                    if errors:
                        last_exc = RuntimeError("invalid backend tool arguments: " + "; ".join(errors[:12]))
                        self.log_message("%s; retrying=%s", last_exc, attempt < attempts)
                        if attempt < attempts:
                            continue
                        break
                    self.send_response(resp.status, resp.reason)
                    for k, v in resp.getheaders():
                        kl = k.lower()
                        if kl in HOP_HEADERS or kl == "content-length":
                            continue
                        self.send_header(k, v)
                    self.send_header("content-length", str(len(payload)))
                    self.end_headers(); committed = True
                    self.wfile.write(payload); self.wfile.flush(); return

                first = b""
                if "text/event-stream" in content_type and 200 <= resp.status < 300:
                    first = resp.read1(65536)
                    if not first:
                        raise ConnectionError("backend closed SSE stream before first event")
                self._forward_headers(resp, content_length); committed = True
                if first:
                    self.wfile.write(first); self.wfile.flush()
                while True:
                    chunk = resp.read1(65536)
                    if not chunk: break
                    self.wfile.write(chunk); self.wfile.flush()
                return
            except (BrokenPipeError, ConnectionResetError) as exc:
                last_exc = exc
                if committed:
                    self.close_connection = True; return
            except Exception as exc:
                last_exc = exc
                if committed:
                    self.close_connection = True; return
            finally:
                conn.close()
            if attempt < attempts:
                continue
            break

        return self._json(502, {"type": "error", "error": {"type": "api_error", "message": f"local tool/backend validation failure before response: {last_exc}"}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    router = Router(config)
    probe_all_routes(router)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.router = router  # type: ignore[attr-defined]
    server.verbose = args.verbose  # type: ignore[attr-defined]
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"claude-local gateway listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
