import pathlib
import sys
import unittest
import http.client
import json
import threading
import subprocess
import tempfile
import time
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
            return {"/props": {"n_ctx_per_slot": 128}, "/apply-template": {"prompt": "rendered-with-tools"},
                    "/tokenize": {"tokens": [1] * 79}}[path]
        with patch.object(budget, "backend_json", side_effect=backend):
            result = budget.enforce(self.body, self.capacity, "http://local/v1")
        self.assertEqual(result.input_tokens, 79)
        self.assertEqual(result.source, "backend-tokenizer")
        self.assertIn("tools", seen[1][1])
        self.assertTrue(seen[1][1]["add_generation_prompt"])
        self.assertEqual(seen[2][1]["content"], "rendered-with-tools")
        self.assertTrue(seen[2][1]["add_special"])
        self.assertTrue(seen[2][1]["parse_special"])

    def test_overflow_and_tool_compaction_error(self):
        with patch.object(budget, "backend_json", return_value={"n_ctx_per_slot": 128}), patch.object(budget, "authoritative_tokens", return_value=81):
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
        with patch.object(budget, "backend_json", return_value={"n_ctx_per_slot": 128}), patch.object(budget, "authoritative_tokens", return_value=53):
            with self.assertRaises(budget.BudgetError):
                budget.enforce(self.body, capacity, "http://local")

    def test_backend_props_smaller_per_slot_context(self):
        with patch.object(budget, "backend_json", return_value={"default_generation_settings": {"n_ctx": 64}}), patch.object(budget, "authoritative_tokens", return_value=20):
            with self.assertRaises(budget.BudgetError):
                budget.enforce(self.body, self.capacity, "http://local")
        with patch.object(budget, "backend_json", return_value={
                "n_ctx_slot": 64, "default_generation_settings": {"n_ctx": 128}}), patch.object(budget, "authoritative_tokens", return_value=20):
            with self.assertRaises(budget.BudgetError):
                budget.enforce(self.body, self.capacity, "http://local")
        with patch.object(budget, "backend_json", return_value={"default_generation_settings": {"n_ctx": 0}}):
            with self.assertRaisesRegex(budget.BudgetError, "backend per-slot context"):
                budget.enforce(self.body, self.capacity, "http://local")

    def test_unsupported_requires_explicit_hard_guard(self):
        with patch.object(budget, "backend_json", return_value=None), patch.object(budget, "authoritative_tokens", return_value=None):
            with self.assertRaisesRegex(budget.BudgetError, "no-context-shift"):
                budget.enforce(self.body, self.capacity, "http://local")
            large = dict(self.capacity, context=4096, input_tokens=4000, no_context_shift=True)
            result = budget.enforce(self.body, large, "http://local")
            self.assertIn("uncertain", result.source)

    def test_legacy_token_count_does_not_bypass_unknown_physical_context(self):
        with patch.object(budget, "backend_json", return_value=None), patch.object(budget, "authoritative_tokens", return_value=10):
            with self.assertRaisesRegex(budget.BudgetError, "cannot prove per-slot context"):
                budget.enforce(self.body, self.capacity, "http://local")
            result = budget.enforce(self.body, dict(self.capacity, no_context_shift=True), "http://local")
            self.assertTrue(result.source.startswith("backend-tokenizer"))
            self.assertIn("context unverified", result.source)

    def test_estimate_includes_system_tools_and_attachments(self):
        small = budget.estimate_tokens(self.body)
        self.body.update(system="rules" * 40, tools=[{"description": "tool" * 50}],
                         attachments=[{"data": "bytes" * 30}])
        self.assertGreater(budget.estimate_tokens(self.body), small + 400)

    def test_anthropic_count_only_still_enforces_route_input(self):
        def backend(endpoint, path, body=None, headers=None):
            return {"n_ctx_per_slot": 128} if path == "/props" else {"input_tokens": 81}
        with patch.object(budget, "backend_json", side_effect=backend):
            with self.assertRaises(budget.BudgetError):
                budget.enforce(self.body, self.capacity, "http://local", "anthropic", count_only=True)

    def test_smaller_route_switch_rechecks_capacity(self):
        with patch.object(budget, "backend_json", return_value={"n_ctx_per_slot": 128}), patch.object(budget, "authoritative_tokens", return_value=70):
            budget.enforce(self.body, self.capacity, "http://large")
            with self.assertRaises(budget.BudgetError):
                budget.enforce(self.body, dict(self.capacity, context=96, input_tokens=48), "http://small")

    def test_startup_proves_per_slot_context_and_slots(self):
        self.assertIs(proxy.verify_backend_capacity, budget.verify_backend_capacity)
        with patch.object(budget, "backend_json", return_value={
                "default_generation_settings": {"n_ctx": 128}, "total_slots": 2}):
            proof = budget.verify_backend_capacity("http://local", {"context": 128, "slots": 2})
        self.assertEqual(proof, {"context": 128, "slots": 2, "source": "backend-props"})
        for props in (None, {"n_ctx": 256, "total_slots": 2},
                      {"n_ctx_slot": 64, "default_generation_settings": {"n_ctx": 128}, "total_slots": 2},
                      {"n_ctx_per_slot": 64, "total_slots": 2},
                      {"n_ctx_per_slot": 128, "total_slots": 1},
                      {"n_ctx_per_slot": 128}):
            with self.subTest(props=props), patch.object(budget, "backend_json", return_value=props):
                with self.assertRaises(budget.BudgetError):
                    budget.verify_backend_capacity("http://local", {"context": 128, "slots": 2})

    def test_direct_slots_are_not_admission_or_decode_proof(self):
        capacity = {"slots": 4, "admission_limit": 1, "min_tps": 50}
        self.assertEqual(budget.direct_admission_limit(capacity), 1)
        self.assertEqual(budget.direct_admission_limit({"slots": 4}), 1)
        self.assertEqual(budget.direct_admission_limit(dict(capacity, admission_limit=3)), 3)
        self.assertEqual(budget.direct_admission_limit(dict(capacity, admission_limit=4, admission_explicit=True)), 4)
        self.assertEqual(budget.direct_admission_limit(dict(capacity, admission_limit=4, admission_proven=True)), 4)
        with self.assertRaises(budget.BudgetError):
            budget.direct_admission_limit({"slots": 4, "admission_limit": 0})

    def test_multimodal_estimate_is_uncertain_and_oversized_payload_fails_closed(self):
        body = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://local/image"}}]}], "max_tokens": 32}
        capacity = dict(self.capacity, context=4096, input_tokens=4000, no_context_shift=True)
        seen = []
        def backend(endpoint, path, body=None, headers=None):
            seen.append(path)
            return {"n_ctx_per_slot": 4096} if path == "/props" else {"prompt": "image placeholder"}
        with patch.object(budget, "backend_json", side_effect=backend):
            result = budget.enforce(body, capacity, "http://local")
            self.assertIn("conservative-estimate", result.source)
            self.assertIn("uncertain", result.source)
            self.assertNotIn("/tokenize", seen)
            with self.assertRaisesRegex(budget.BudgetError, "no-context-shift"):
                budget.enforce(body, dict(capacity, no_context_shift=False), "http://local")
            body["attachments"] = [{"data": "x" * 10000}]
            with self.assertRaisesRegex(budget.BudgetError, "reduce attachments"):
                budget.enforce(body, capacity, "http://local")

    def test_known_vllm_native_chat_counts_full_request_and_proves_context(self):
        body = dict(self.body, model="native", tools=[{"function": {"name": "run", "parameters": {"type": "object"}}}])
        seen = []
        def backend(endpoint, path, body=None, headers=None):
            seen.append((path, body))
            return None if path == "/props" else {"count": 70, "tokens": [1] * 70, "max_model_len": 128}
        with patch.object(budget, "backend_json", side_effect=backend):
            result = budget.enforce(body, self.capacity, "http://native/v1", backend="vllm-qwen38-3090")
        self.assertEqual(result.input_tokens, 70)
        self.assertEqual(result.context, 128)
        self.assertEqual(result.source, "backend-tokenizer/vllm-native")
        self.assertEqual([x[0] for x in seen], ["/props", "/tokenize"])
        request = seen[1][1]
        self.assertEqual(request["messages"], body["messages"])
        self.assertEqual(request["tools"], body["tools"])
        self.assertEqual(request["model"], "native")
        self.assertTrue(request["add_generation_prompt"])
        self.assertTrue(request["add_special_tokens"])
        self.assertNotIn("content", request)

    def test_vllm_rejects_unsupported_partial_counts_and_small_physical_context(self):
        invalid = (None, {"tokens": [1]}, {"count": True, "max_model_len": 128},
                   {"count": 70, "tokens": [1], "max_model_len": 128},
                   {"count": 70}, {"count": 70, "max_model_len": 100})
        for response in invalid:
            def backend(endpoint, path, body=None, headers=None):
                return None if path == "/props" else response
            with self.subTest(response=response), patch.object(budget, "backend_json", side_effect=backend):
                with self.assertRaises(budget.BudgetError):
                    budget.enforce(self.body, self.capacity, "http://native", backend="vllm")
        with patch.object(budget, "backend_json", return_value=None) as backend:
            self.assertIsNone(budget.authoritative_tokens("http://unknown", self.body))
            self.assertEqual(backend.call_args.args[1], "/apply-template")
        with patch.object(budget, "backend_json", return_value=None):
            with self.assertRaisesRegex(budget.BudgetError, "sglang.*native tokenization"):
                budget.enforce(self.body, dict(self.capacity, no_context_shift=True), "http://unknown", backend="sglang-v100")


class FakeBackend(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if getattr(self.server, "native_vllm", False):
            self.send_error(404)
            return
        if self.path == "/props" and hasattr(self.server, "props"):
            raw = json.dumps(self.server.props).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self):
        obj = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            if getattr(self.server, "native_vllm", False) or not getattr(self.server, "tokenization_supported", True):
                self.send_error(404)
                return
            self.server.rendered_request = obj
            payload = {"prompt": "rendered full request"}
        elif self.path == "/tokenize":
            if getattr(self.server, "native_vllm", False):
                if "messages" not in obj and "prompt" not in obj:
                    self.send_error(400)
                    return
                self.server.rendered_request = obj
                payload = {"count": self.server.token_count, "tokens": [1] * self.server.token_count,
                           "max_model_len": getattr(self.server, "max_model_len", 128)}
            else:
                payload = {"tokens": [1] * self.server.token_count}
        else:
            self.server.calls += 1
            with self.server.lock:
                self.server.active += 1
                self.server.max_active = max(self.server.max_active, self.server.active)
            time.sleep(getattr(self.server, "delay", 0))
            with self.server.lock:
                self.server.active -= 1
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
        self.backend.props = {"default_generation_settings": {"n_ctx": 128}, "total_slots": 1}
        self.backend.token_count = 12
        self.backend.calls = 0
        self.backend.lock = threading.Lock()
        self.backend.active = self.backend.max_active = 0
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
        self.assertIn("UNKNOWN", headers["X-Pushbutton-SLA-Warning"])

    def test_backend_error_is_transparent(self):
        status, raw, _ = self.request()
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(raw)["error"]["message"], "upstream rejected request")

    def test_capacity_verifier_cli(self):
        self.backend.props = {"default_generation_settings": {"n_ctx": 128}, "total_slots": 2}
        command = [sys.executable, str(ROOT / "lib/pushbutton_request_budget.py"),
                   "--check-backend", self.guard.upstream, "--slots", "2", "--context"]
        good = subprocess.run([*command, "128"], capture_output=True, text=True, timeout=5)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(json.loads(good.stdout)["slots"], 2)
        bad = subprocess.run([*command, "129"], capture_output=True, text=True, timeout=5)
        self.assertEqual(bad.returncode, 1)
        self.assertIn("requested context=129", bad.stderr)
        proxy_check = [sys.executable, str(ROOT / "lib/pushbutton_capacity_proxy.py"),
                       "--check-backend", self.guard.upstream, "--capacity-json"]
        good = subprocess.run([*proxy_check, json.dumps({"context": 128, "slots": 2})],
                              capture_output=True, text=True, timeout=5)
        self.assertEqual(good.returncode, 0, good.stderr)
        bad = subprocess.run([*proxy_check, json.dumps({"context": 128, "slots": 3})],
                             capture_output=True, text=True, timeout=5)
        self.assertEqual(bad.returncode, 1)

    def test_overflow_never_reaches_generation(self):
        self.backend.token_count = 81
        status, raw, _ = self.request()
        self.assertEqual(status, 400)
        self.assertIn(b"compact/summarize", raw)
        self.assertEqual(self.backend.calls, 0)

    def test_default_c1_and_explicit_route_admission(self):
        self.backend.delay = .15
        self.backend.props["total_slots"] = 4
        for explicit, expected in ((False, 1), (True, 2)):
            capacity = dict(self.guard.capacity, slots=4, admission_limit=4, admission_explicit=explicit)
            self.guard.routes = {"local": {
                "url": self.guard.upstream, "backend_alias": "local", "capacity": capacity,
                "gate": threading.BoundedSemaphore(budget.direct_admission_limit(capacity))}}
            self.backend.max_active = 0
            results = []
            errors = []
            start = threading.Barrier(3)
            def run():
                start.wait()
                try:
                    results.append(self.request(stream=True)[0])
                except Exception as exc:
                    errors.append(exc)
            threads = [threading.Thread(target=run) for _ in range(2)]
            for thread in threads:
                thread.start()
            start.wait()
            for thread in threads:
                thread.join(5)
            self.assertEqual(errors, [])
            self.assertEqual(results, [200, 200])
            self.assertEqual(self.backend.max_active, expected)

    def test_direct_unsupported_tokenization_is_protected_and_estimate_labeled(self):
        self.backend.tokenization_supported = False
        self.backend.props = {"n_ctx_slot": 512, "total_slots": 1}
        self.guard.capacity = dict(self.guard.capacity, context=512, input_tokens=450)
        self.assertEqual(self.request(stream=True)[0], 400)
        self.assertEqual(self.backend.calls, 0)
        self.guard.capacity["no_context_shift"] = True
        status, _, headers = self.request(stream=True)
        self.assertEqual(status, 200)
        self.assertIn("conservative-estimate", headers["X-Pushbutton-Token-Source"])
        self.assertIn("uncertain", headers["X-Pushbutton-Token-Source"])
        self.assertEqual(self.request(stream=True, attachments=[{"data": "x" * 10000}])[0], 400)
        self.assertEqual(self.backend.calls, 1)

    def test_native_vllm_route_without_llama_props_or_shift_flag(self):
        self.backend.native_vllm = True
        self.guard.routes = {"native": {
            "url": self.guard.upstream, "backend_alias": "real-native", "backend": "vllm",
            "capacity": self.guard.capacity, "gate": threading.BoundedSemaphore(1)}}
        status, _, headers = self.request(stream=True, tools=[{"function": {"name": "run"}}])
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Pushbutton-Token-Source"], "backend-tokenizer/vllm-native")
        self.assertEqual(self.backend.rendered_request["model"], "real-native")
        self.assertIn("tools", self.backend.rendered_request)
        self.backend.max_model_len = 64
        self.backend.token_count = 20
        self.assertEqual(self.request(stream=True)[0], 400)
        self.assertEqual(self.backend.calls, 1)

    def test_missing_model_uses_only_route_not_an_arbitrary_route(self):
        self.guard.routes = {"one": {
            "url": self.guard.upstream, "backend_alias": "real-one",
            "capacity": self.guard.capacity, "gate": threading.BoundedSemaphore(1)}}
        self.assertEqual(self.request(stream=True)[0], 200)
        self.assertEqual(self.backend.rendered_request["model"], "real-one")
        self.guard.routes["two"] = dict(self.guard.routes["one"], backend_alias="real-two")
        self.assertEqual(self.request(stream=True)[0], 400)
        self.assertEqual(self.backend.calls, 1)

    def test_config_cli_routes_models_with_independent_budgets(self):
        self.backend.token_count = 40
        url = f"http://127.0.0.1:{self.backend.server_port}"
        config = {"routes": {
            "large": {"url": url, "backend_alias": "real-large", "capacity": self.guard.capacity},
            "small": {"url": url, "backend_alias": "real-small", "capacity": {
                "context": 64, "output_tokens": 16, "input_tokens": 32, "safety_tokens": 16}},
        }}
        import socket
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory) / "routes.json"
            path.write_text(json.dumps(config))
            proc = subprocess.Popen([sys.executable, str(ROOT / "lib/pushbutton_capacity_proxy.py"),
                                     "--config", str(path), "--port", str(port)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for _ in range(50):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                    try:
                        conn.request("GET", "/v1/models")
                        response = conn.getresponse()
                        models = json.loads(response.read())
                        conn.close()
                        break
                    except OSError:
                        conn.close()
                        time.sleep(.05)
                else:
                    self.fail("multi-route capacity proxy failed to start")
                self.assertEqual({x["id"] for x in models["data"]}, {"large", "small"})
                for model, output, expected in (("large", 32, 200), ("small", 16, 400), ("unknown", 16, 404), (None, 16, 400)):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    conn.request("POST", "/v1/chat/completions", json.dumps({
                        "model": model, "messages": [], "max_tokens": output, "stream": True}),
                        {"Content-Type": "application/json"})
                    response = conn.getresponse()
                    self.assertEqual(response.status, expected)
                    response.read()
                    conn.close()
                    if model == "large":
                        self.assertEqual(self.backend.rendered_request["model"], "real-large")
                self.assertEqual(self.backend.calls, 1)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    unittest.main()
