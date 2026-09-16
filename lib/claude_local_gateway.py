#!/usr/bin/env python3
"""Tiny Anthropic-wire router for claude-local.

Claude Code sometimes sends canonical Claude family IDs even when custom model
aliases are configured. This proxy routes by family (haiku/sonnet/opus/fable)
and by our explicit local aliases, then forwards requests to the matching local
llama-server `/v1/messages` endpoint.

For tool-bearing requests the gateway enforces the Qwen fixed-template JSON tool
format per request.  This is deliberately done here, rather than trusting only a
server environment default, because llama.cpp's Anthropic adapter explicitly
passes `chat_template_kwargs` through to the chat-template/parser path.

Before listening, the gateway also performs a real forced tool call against every
unique configured backend.  Claude Code is never launched if a loaded quant,
template, or parser leaks raw tool markup instead of returning a structured
Anthropic `tool_use` block.
"""
from __future__ import annotations

import argparse
import http.client
import json
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


def apply_local_tool_policy(body: dict) -> None:
    """Make tool-bearing Qwen requests deterministic and parser-friendly."""
    if not body.get("tools"):
        return
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    # froggeric Qwen fixed templates default to tagged XML.  JSON_NATIVE is much
    # less fragile in current llama.cpp and avoids the Qwen3.6 </parameter> bug.
    kwargs["tool_call_format"] = "json"
    kwargs.setdefault("preserve_reasoning", True)
    body["chat_template_kwargs"] = kwargs

    # Qwen3.6's tagged tool parser is known to become materially less reliable at
    # high sampling temperatures.  Claude Code does not need creative sampling
    # while selecting/serializing a tool call, so cap only tool-bearing requests.
    try:
        temp = float(body.get("temperature", 0.2))
    except (TypeError, ValueError):
        temp = 0.2
    body["temperature"] = min(temp, 0.2)


class Router:
    def __init__(self, config: dict):
        self.roles: dict[str, dict] = config["roles"]
        self.alias_map: dict[str, dict] = {}
        for role, route in self.roles.items():
            self.alias_map[route["model_id"].lower()] = route
            self.alias_map[role] = route

    def resolve(self, model: str) -> dict:
        key = (model or "").lower()
        if key in self.alias_map:
            return self.alias_map[key]
        fam = family_for(key)
        if fam and fam in self.roles:
            return self.roles[fam]
        # Claude Code may emit internal/generated IDs that don't contain a
        # family. Keep those local and predictable by routing to the daily
        # driver rather than ever falling through to a cloud API.
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
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    return conn_cls(parsed.hostname, parsed.port, timeout=route.get("timeout", 3600))


def probe_tool_route(route: dict) -> None:
    """Require a real structured Anthropic tool_use response from this backend."""
    token = "pushbutton-tool-probe-ok"
    body = {
        "model": route["backend_alias"],
        "max_tokens": 512,
        "temperature": 0,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": f"Call pushbutton_probe exactly once with token {token}. Do not answer in prose.",
        }],
        "tools": [{
            "name": "pushbutton_probe",
            "description": "Startup validation tool. Always call it when requested.",
            "input_schema": {
                "type": "object",
                "properties": {"token": {"type": "string"}},
                "required": ["token"],
                "additionalProperties": False,
            },
        }],
        "tool_choice": {"type": "tool", "name": "pushbutton_probe"},
    }
    apply_local_tool_policy(body)
    raw = json.dumps(body, separators=(",", ":")).encode()
    conn = backend_connection(route)
    try:
        conn.request(
            "POST", "/v1/messages", body=raw,
            headers={"content-type": "application/json", "content-length": str(len(raw))},
        )
        resp = conn.getresponse()
        payload = resp.read()
    finally:
        conn.close()

    text = payload.decode("utf-8", "replace")
    if resp.status < 200 or resp.status >= 300:
        raise RuntimeError(
            f"tool probe HTTP {resp.status} for {route['backend_alias']}: {text[:2000]}"
        )
    try:
        obj = json.loads(text)
    except Exception as exc:
        raise RuntimeError(
            f"tool probe returned non-JSON for {route['backend_alias']}: {text[:2000]}"
        ) from exc

    blocks = obj.get("content") if isinstance(obj, dict) else None
    if not isinstance(blocks, list):
        blocks = []
    tool_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
    valid = any(
        b.get("name") == "pushbutton_probe"
        and isinstance(b.get("input"), dict)
        and b["input"].get("token") == token
        for b in tool_blocks
    )
    serialized = json.dumps(obj, ensure_ascii=False)
    leaked = [m for m in RAW_TOOL_MARKERS if m in serialized]
    if not valid or leaked:
        raise RuntimeError(
            "tool probe failed for " + str(route["backend_alias"]) +
            (f"; raw markup leaked: {leaked}" if leaked else "; no valid tool_use block") +
            f"; response={serialized[:3000]}"
        )


def probe_all_routes(router: Router) -> None:
    for route in router.unique_routes():
        print(
            f"claude-local gateway: validating structured tool calls on {route['backend_alias']}...",
            flush=True,
        )
        probe_tool_route(route)
        print(
            f"claude-local gateway: tool-call probe PASS on {route['backend_alias']}",
            flush=True,
        )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "claude-local/0.3"

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
            return self._json(200, {"status": "ok", "local": True, "tool_probe": "passed"})
        if path in ("/v1/models", "/models"):
            seen = set()
            data = []
            for role, route in self.router.roles.items():
                mid = route["model_id"]
                if mid not in seen:
                    data.append({"id": mid, "object": "model", "owned_by": "claude-local"})
                    seen.add(mid)
            return self._json(200, {"object": "list", "data": data})
        return self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": "local gateway route not found"}})

    def _backend_connection(self, route: dict):
        return backend_connection(route)

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
            # http.client already de-chunks an upstream chunked response. Use
            # connection-close framing downstream so raw SSE bytes can be
            # flushed immediately without manufacturing a second chunk layer.
            self.send_header("connection", "close")
            self.close_connection = True
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
        apply_local_tool_policy(body)
        raw = json.dumps(body, separators=(",", ":")).encode()
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        headers["content-type"] = "application/json"
        headers["content-length"] = str(len(raw))

        # One transparent retry is useful for an immediately reset local
        # connection or transient 5xx. We only retry before response headers /
        # bytes are committed to Claude Code.
        attempts = int(route.get("proxy_attempts", 2))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            conn = self._backend_connection(route)
            committed = False
            try:
                conn.request("POST", path, body=raw, headers=headers)
                resp = conn.getresponse()

                # Retry a transient backend 5xx before exposing it downstream.
                if resp.status in (500, 502, 503, 504) and attempt < attempts:
                    self.log_message("backend %s returned %s before stream; retrying (%d/%d)", route["url"], resp.status, attempt, attempts)
                    try:
                        resp.read()
                    except Exception:
                        pass
                    continue

                content_type = (resp.getheader("content-type") or "").lower()
                content_length = resp.getheader("content-length")
                first = b""
                if "text/event-stream" in content_type and 200 <= resp.status < 300:
                    # Do not commit a nominal 200 SSE response until at least
                    # one upstream byte exists. If llama-server dies before its
                    # first event, the request can still be retried safely.
                    first = resp.read1(65536)
                    if not first:
                        raise ConnectionError("backend closed SSE stream before first event")

                self._forward_headers(resp, content_length)
                committed = True
                if first:
                    self.wfile.write(first)
                    self.wfile.flush()
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                return
            except (BrokenPipeError, ConnectionResetError) as exc:
                last_exc = exc
                if committed:
                    self.log_message("downstream/upstream connection reset after stream commit: %r", exc)
                    self.close_connection = True
                    return
            except Exception as exc:
                last_exc = exc
                if committed:
                    # Never attempt to write a second HTTP/JSON response after
                    # Anthropic SSE headers have already been emitted. That
                    # corrupts the stream and produces misleading client errors.
                    self.log_message("backend failed after stream commit: %r", exc)
                    self.close_connection = True
                    return
            finally:
                conn.close()

            if attempt < attempts:
                self.log_message("backend failed before downstream commit: %r; retrying (%d/%d)", last_exc, attempt, attempts)
                continue
            break

        return self._json(502, {"type": "error", "error": {"type": "api_error", "message": f"local backend failure before response: {last_exc}"}})


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
    # Hard startup gate: do not advertise health and do not launch Claude Code
    # until the exact loaded backends can produce structured tool calls.
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
