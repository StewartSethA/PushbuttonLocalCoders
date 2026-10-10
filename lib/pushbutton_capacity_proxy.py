#!/usr/bin/env python3
"""Dependency-free multi-route guard, also embeddable on a single backend server."""
import argparse
import http.client
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pushbutton_request_budget as budget
verify_backend_capacity = budget.verify_backend_capacity

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
            return self.reply(200, {"status": "ok", "decode_sla": "UNKNOWN"})
        if urlsplit(self.path).path in {"/v1/models", "/models"}:
            routes = getattr(self.server, "routes", None)
            names = list(routes) if routes is not None else [self.server.alias]
            return self.reply(200, {"object": "list", "data": [{"id": name, "object": "model"} for name in names]})
        return self.reply(404, {"error": {"message": "unknown guard endpoint"}})

    def do_POST(self):
        if urlsplit(self.path).path not in {"/v1/chat/completions", "/v1/completions"}:
            return self.reply(404, {"error": {"message": "unknown guard endpoint"}})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if not isinstance(body, dict):
                raise budget.BudgetError("request body must be an object")
            routes = getattr(self.server, "routes", None)
            if routes is None:
                route = {"url": self.server.upstream, "backend_alias": self.server.alias,
                         "capacity": self.server.capacity, "gate": self.server.slots}
            else:
                if body.get("model") is None:
                    if len(routes) != 1:
                        raise budget.BudgetError("model is required for a multi-route guard; select an advertised model")
                    route = next(iter(routes.values()))
                else:
                    route = routes.get(body.get("model"))
                if route is None:
                    return self.reply(404, {"error": {"message": "unknown model; select an advertised local route"}})
            body["model"] = route["backend_alias"]
            budget.output_tokens(body, route["capacity"])
        except budget.BudgetError as exc:
            return self.reply(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        except (ValueError, TypeError):
            return self.reply(400, {"error": {"message": "invalid JSON or output budget; output must be a positive integer including reasoning"}})
        route["gate"].acquire()
        try:
            try:
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
                admission = budget.enforce(body, route["capacity"], route["url"], headers=headers,
                                           backend=route.get("backend", "llama.cpp"))
            except budget.BudgetError as exc:
                return self.reply(400, {"error": {"message": str(exc), "type": "context_budget_exceeded"}})
            if not any(k in body for k in ("max_tokens", "max_completion_tokens", "max_output_tokens")):
                body["max_tokens"] = admission.output_tokens
            self.forward(body, admission.source, route["url"])
        finally:
            route["gate"].release()

    def forward(self, body, source, upstream):
        u = urlsplit(upstream)
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
                    self.send_header(k.replace("\r", "").replace("\n", ""), v.replace("\r", "").replace("\n", ""))
            self.send_header("X-Pushbutton-Token-Source", source)
            self.send_header("X-Pushbutton-SLA-Warning", budget.SLA_WARNING)
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
    ap.add_argument("--upstream")
    ap.add_argument("--port", type=int)
    ap.add_argument("--capacity", "--capacity-json", dest="capacity", help="capacity JSON")
    ap.add_argument("--alias")
    ap.add_argument("--config", help="JSON file containing model-keyed routes")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--check-backend", help="verify backend capacity and exit")
    args = ap.parse_args()
    if args.check_backend:
        if not args.capacity:
            ap.error("--check-backend requires --capacity-json")
        try:
            proof = verify_backend_capacity(args.check_backend, json.loads(args.capacity))
        except (budget.BudgetError, ValueError, TypeError) as exc:
            print(f"Capacity readiness failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(proof))
        return 0
    if args.port is None:
        ap.error("--port is required for serving")
    if args.config:
        if any((args.upstream, args.capacity, args.alias)):
            ap.error("--config cannot be combined with single-backend options")
        with open(args.config, encoding="utf-8") as f:
            routes = json.load(f)["routes"]
    else:
        if not all((args.upstream, args.capacity, args.alias)):
            ap.error("provide --config or --upstream, --capacity, and --alias")
        routes = {args.alias: {"url": args.upstream, "backend_alias": args.alias,
                               "capacity": json.loads(args.capacity)}}
    if not isinstance(routes, dict) or not routes:
        ap.error("routes must be a nonempty model-keyed object")
    limits = {}
    for route in routes.values():
        capacity = route["capacity"]
        budget.integer(capacity.get("context"), "context")
        limit = budget.direct_admission_limit(capacity)
        key = (route["url"], route["backend_alias"])
        limits[key] = min(limit, limits.get(key, limit))
    gates = {key: threading.BoundedSemaphore(limit) for key, limit in limits.items()}
    for route in routes.values():
        route["gate"] = gates[(route["url"], route["backend_alias"])]
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.routes = routes
    print(f"[pushbutton-capacity] {budget.SLA_WARNING}; unproven admission defaults to C1", file=sys.stderr, flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
