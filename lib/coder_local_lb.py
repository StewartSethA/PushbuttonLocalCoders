#!/usr/bin/env python3
"""Tiny round-robin OpenAI-compatible router for replica-aware local coders."""
from __future__ import annotations

import argparse
import http.client
import json
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


class Pool:
    def __init__(self, backends: list[dict]):
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = backends
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> dict:
        with self._lock:
            b = self.backends[self._i % len(self.backends)]
            self._i += 1
            return b


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "coder-local-lb/0.1"

    @property
    def pool(self) -> Pool:
        return self.server.pool  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        if self.server.verbose:  # type: ignore[attr-defined]
            sys.stderr.write("coder-local-lb: " + (fmt % args) + "\n")

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
            return self._json(200, {"status": "ok", "replicas": len(self.pool.backends)})
        if path in ("/v1/models", "/models"):
            models = []
            seen = set()
            for b in self.pool.backends:
                mid = b.get("model", "local-coder")
                if mid not in seen:
                    models.append({"id": mid, "object": "model", "owned_by": "coder-local"})
                    seen.add(mid)
            return self._json(200, {"object": "list", "data": models})
        return self._json(404, {"error": {"message": "route not found"}})

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/v1/chat/completions", "/v1/responses", "/v1/embeddings"):
            return self._json(404, {"error": {"message": "route not found"}})
        try:
            n = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(n)
            body = json.loads(raw or b"{}")
        except Exception as exc:
            return self._json(400, {"error": {"message": f"bad JSON: {exc}"}})

        backend = self.pool.next()
        if backend.get("alias"):
            body["model"] = backend["alias"]
        raw = json.dumps(body, separators=(",", ":")).encode()
        parsed = urlsplit(backend["url"])
        conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(parsed.hostname, parsed.port, timeout=backend.get("timeout", 3600))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        headers["content-type"] = "application/json"
        headers["content-length"] = str(len(raw))
        try:
            conn.request("POST", path, body=raw, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status, resp.reason)
            content_length = resp.getheader("content-length")
            for k, v in resp.getheaders():
                if k.lower() in HOP_HEADERS or k.lower() == "content-length":
                    continue
                self.send_header(k, v)
            if content_length:
                self.send_header("content-length", content_length)
            else:
                self.send_header("connection", "close")
                self.close_connection = True
            self.send_header("x-coder-local-backend", str(backend.get("name", backend["url"])))
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
                    self._json(502, {"error": {"message": f"local replica failure: {exc}"}})
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
    server.pool = Pool(config["backends"])  # type: ignore[attr-defined]
    server.verbose = args.verbose  # type: ignore[attr-defined]
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"coder-local replica router listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
