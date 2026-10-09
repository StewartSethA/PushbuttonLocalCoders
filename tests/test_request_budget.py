import pathlib
import sys
import unittest
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
import pushbutton_request_budget as budget
import pushbutton_capacity_proxy as proxy


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.capacity = {"context": 128, "output_tokens": 32, "input_tokens": 80, "safety_tokens": 16}
        self.body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32}

    def test_full_request_template_and_authoritative_count(self):
        self.body["tools"] = [{"type": "function", "function": {"name": "run", "parameters": {"type": "object"}}}]
        seen = []
        def backend(endpoint, path, body=None, headers=None):
            seen.append((path, body))
            return {"/props": None, "/apply-template": {"prompt": "rendered-with-tools"},
                    "/tokenize": {"tokens": [1] * 79}}[path]
        with patch.object(budget, "backend_json", side_effect=backend):
            result = budget.enforce(self.body, self.capacity, "http://local/v1")
        self.assertEqual(result.input_tokens, 79)
        self.assertEqual(result.source, "backend-tokenizer")
        self.assertIn("tools", seen[1][1])
        self.assertEqual(seen[2][1]["content"], "rendered-with-tools")

    def test_overflow_and_tool_compaction_error(self):
        with patch.object(budget, "backend_json", return_value=None), patch.object(budget, "authoritative_tokens", return_value=81):
            with self.assertRaisesRegex(budget.BudgetError, "compact/summarize.*tool results"):
                budget.enforce(self.body, self.capacity, "http://local")

    def test_output_is_integral_and_not_clamped(self):
        for output in (0, -1, True, "12", 2.5, 33):
            with self.subTest(output=output), self.assertRaises(budget.BudgetError):
                budget.output_tokens({"max_tokens": output}, self.capacity)
        with self.assertRaisesRegex(budget.BudgetError, "thinking budget"):
            budget.output_tokens({"max_tokens": 32, "thinking": {"type": "enabled", "budget_tokens": 32}}, self.capacity)

    def test_input_output_safety_total(self):
        capacity = dict(self.capacity, context=100, input_tokens=90)
        with patch.object(budget, "backend_json", return_value=None), patch.object(budget, "authoritative_tokens", return_value=53):
            with self.assertRaises(budget.BudgetError):
                budget.enforce(self.body, capacity, "http://local")

    def test_backend_props_smaller_per_slot_context(self):
        with patch.object(budget, "backend_json", return_value={"default_generation_settings": {"n_ctx": 64}}), patch.object(budget, "authoritative_tokens", return_value=20):
            with self.assertRaises(budget.BudgetError):
                budget.enforce(self.body, self.capacity, "http://local")

    def test_unsupported_requires_explicit_hard_guard(self):
        with patch.object(budget, "backend_json", return_value=None), patch.object(budget, "authoritative_tokens", return_value=None):
            with self.assertRaisesRegex(budget.BudgetError, "no-context-shift"):
                budget.enforce(self.body, self.capacity, "http://local")
            large = dict(self.capacity, context=4096, input_tokens=4000, no_context_shift=True)
            result = budget.enforce(self.body, large, "http://local")
            self.assertIn("uncertain", result.source)

    def test_estimate_includes_system_tools_and_attachments(self):
        small = budget.estimate_tokens(self.body)
        self.body.update(system="rules" * 40, tools=[{"description": "tool" * 50}],
                         attachments=[{"data": "bytes" * 30}])
        self.assertGreater(budget.estimate_tokens(self.body), small + 400)

    def test_anthropic_count_only_still_enforces_route_input(self):
        def backend(endpoint, path, body=None, headers=None):
            return None if path == "/props" else {"input_tokens": 81}
        with patch.object(budget, "backend_json", side_effect=backend):
            with self.assertRaises(budget.BudgetError):
                budget.enforce(self.body, self.capacity, "http://local", "anthropic", count_only=True)

    def test_smaller_route_switch_rechecks_capacity(self):
        with patch.object(budget, "backend_json", return_value=None), patch.object(budget, "authoritative_tokens", return_value=70):
            budget.enforce(self.body, self.capacity, "http://large")
            with self.assertRaises(budget.BudgetError):
                budget.enforce(self.body, dict(self.capacity, context=96, input_tokens=48), "http://small")


class FakeBackend(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_error(404)

    def do_POST(self):
        obj = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            self.server.rendered_request = obj
            payload = {"prompt": "rendered full request"}
        elif self.path == "/tokenize":
            payload = {"tokens": [1] * self.server.token_count}
        else:
            self.server.calls += 1
            raw = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
            if obj.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)
                self.close_connection = True
                return
            payload = {"error": {"message": "upstream rejected request"}}
            self.send_response(422)
            raw = json.dumps(payload).encode()
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.backend = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackend)
        self.backend.token_count = 12
        self.backend.calls = 0
        self.guard = ThreadingHTTPServer(("127.0.0.1", 0), proxy.Handler)
        self.guard.upstream = f"http://127.0.0.1:{self.backend.server_port}/v1"
        self.guard.alias = "local"
        self.guard.capacity = {"context": 128, "output_tokens": 32, "input_tokens": 80, "safety_tokens": 16}
        self.guard.slots = threading.BoundedSemaphore(1)
        for server in (self.backend, self.guard):
            threading.Thread(target=server.serve_forever, daemon=True).start()

    def tearDown(self):
        for server in (self.guard, self.backend):
            server.shutdown()
            server.server_close()

    def request(self, **kwargs):
        conn = http.client.HTTPConnection("127.0.0.1", self.guard.server_port, timeout=5)
        body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32, **kwargs}
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        result = (response.status, response.read(), dict(response.getheaders()))
        conn.close()
        return result

    def test_stream_is_transparent_and_tools_are_rendered(self):
        status, raw, headers = self.request(stream=True, tools=[{"function": {"name": "run"}}])
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"data: [DONE]\n\n"))
        self.assertIn("tools", self.backend.rendered_request)
        self.assertEqual(headers["X-Pushbutton-Token-Source"], "backend-tokenizer")

    def test_backend_error_is_transparent(self):
        status, raw, _ = self.request()
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(raw)["error"]["message"], "upstream rejected request")

    def test_overflow_never_reaches_generation(self):
        self.backend.token_count = 81
        status, raw, _ = self.request()
        self.assertEqual(status, 400)
        self.assertIn(b"compact/summarize", raw)
        self.assertEqual(self.backend.calls, 0)


if __name__ == "__main__":
    unittest.main()
