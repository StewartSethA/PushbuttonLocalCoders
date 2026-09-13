#!/usr/bin/env python3
"""Tiny Anthropic-wire router for claude-local.

Claude Code sometimes sends canonical Claude family IDs even when custom model
aliases are configured. This proxy routes by family (haiku/sonnet/opus/fable)
and by our explicit local aliases, then forwards the request unchanged except
for the model field to the matching llama-server `/v1/messages` endpoint.
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


def family_for(model: str) -> str | None:
    m = (model or "").lower()
    for family in ("haiku", "sonnet", "opus", "fable"):
        if family in m:
            return family
    return None


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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "claude-local/0.1"

    @property
    def router(self) -> Router:
        return self.server.router  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        if self.server.verbose:  # type: ignore[attr-defined]
            sys.stderr.write("gateway: " + (fmt % args) + "\n")

    def _json(self, status: int, obj: dict):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/health", "/health/liveliness"):
            return self._json(200, {"status": "ok", "local": True})
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
        raw = json.dumps(body, separators=(",", ":")).encode()
        parsed = urlsplit(route["url"])
        conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(parsed.hostname, parsed.port, timeout=route.get("timeout", 3600))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        headers["content-type"] = "application/json"
        headers["content-length"] = str(len(raw))
        try:
            conn.request("POST", path, body=raw, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status, resp.reason)
            content_length = resp.getheader("content-length")
            for k, v in resp.getheaders():
                kl = k.lower()
                if kl in HOP_HEADERS or kl == "content-length":
                    continue
                self.send_header(k, v)
            if content_length:
                self.send_header("content-length", content_length)
            else:
                # End-of-stream is signalled by connection close. This lets us
                # forward SSE incrementally without re-chunking it ourselves.
                self.send_header("connection", "close")
                self.close_connection = True
            self.end_headers()
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            if not self.wfile.closed:
                try:
                    self._json(502, {"type": "error", "error": {"type": "api_error", "message": f"local backend failure: {exc}"}})
                except Exception:
                    pass
        finally:
            conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.router = Router(config)  # type: ignore[attr-defined]
    server.verbose = args.verbose  # type: ignore[attr-defined]
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"claude-local gateway listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
