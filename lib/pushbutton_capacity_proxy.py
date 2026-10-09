#!/usr/bin/env python3
"""Small, dependency-free OpenAI request guard for a single local backend."""
import argparse
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pushbutton_request_budget as budget

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
       "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def do_GET(self):
        if urlsplit(self.path).path in {"/health", "/health/liveliness"}:
            return self.reply(200, {"status": "ok"})
        if urlsplit(self.path).path in {"/v1/models", "/models"}:
            return self.reply(200, {"object": "list", "data": [{"id": self.server.alias, "object": "model"}]})
        return self.reply(404, {"error": {"message": "unknown guard endpoint"}})

    def do_POST(self):
        if urlsplit(self.path).path not in {"/v1/chat/completions", "/v1/completions"}:
            return self.reply(404, {"error": {"message": "unknown guard endpoint"}})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if not isinstance(body, dict):
                raise budget.BudgetError("request body must be an object")
            body["model"] = self.server.alias
            budget.output_tokens(body, self.server.capacity)
        except budget.BudgetError as exc:
            return self.reply(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        except (ValueError, TypeError):
            return self.reply(400, {"error": {"message": "invalid JSON or output budget; output must be a positive integer including reasoning"}})
        if not self.server.slots.acquire(timeout=60):
            return self.reply(429, {"error": {"message": "route admission limit reached; retry later"}})
        try:
            try:
                admission = budget.enforce(body, self.server.capacity, self.server.upstream, headers=dict(self.headers))
            except budget.BudgetError as exc:
                return self.reply(400, {"error": {"message": str(exc), "type": "context_budget_exceeded"}})
            if not any(k in body for k in ("max_tokens", "max_completion_tokens", "max_output_tokens")):
                body["max_tokens"] = admission.output_tokens
            self.forward(body, admission.source)
        finally:
            self.server.slots.release()

    def forward(self, body, source):
        u = urlsplit(self.server.upstream)
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn = cls(u.hostname, u.port, timeout=3600)
        base = u.path.rstrip("/")
        path = self.path if base and self.path.startswith(base + "/") else base + self.path
        committed = False
        try:
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            conn.request("POST", path, json.dumps(body).encode(), headers)
            resp = conn.getresponse()
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("X-Pushbutton-Token-Source", source)
            self.send_header("Connection", "close")
            self.end_headers()
            committed = True
            while True:
                data = resp.read1(65536)
                if not data:
                    break
                self.wfile.write(data)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            if not committed:
                self.reply(502, {"error": {"message": "local backend unavailable"}})
        finally:
            self.close_connection = True
            conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--capacity", required=True, help="capacity JSON")
    ap.add_argument("--alias", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    capacity = json.loads(args.capacity)
    budget.integer(capacity.get("context"), "context")
    limit = budget.integer(capacity.get("admission_limit", 1), "admission_limit")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.capacity = capacity
    server.upstream = args.upstream
    server.alias = args.alias
    server.slots = threading.BoundedSemaphore(limit)
    server.serve_forever()


if __name__ == "__main__":
    main()
