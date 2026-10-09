#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


planmod = load("claude_local_plan", ROOT / "lib" / "claude_local_plan.py")
gwmod = load("claude_local_gateway", ROOT / "lib" / "claude_local_gateway.py")


class PlannerTests(unittest.TestCase):
    def test_role_mapping(self):
        self.assertEqual(planmod.role_map(["q38"])["fable"], "qwen3.8:27b")
        r = planmod.role_map(["nemotron", "q36", "q38", "glm53"])
        self.assertEqual(
            [r[x] for x in planmod.ROLES],
            ["nemotron-3.5-lightning", "qwen3.6:35b", "qwen3.8:27b", "glm-5.3-flash"],
        )

    def test_heterogeneous_joint_placement(self):
        gpus = [
            planmod.GPU(0, "Tesla V100-SXM2-32GB", 32768, 32400, "7.0", "", 3, 16),
            planmod.GPU(1, "RTX 4060 Ti", 16380, 16000, "8.9", "", 2, 8),
        ]
        p = planmod.plan(["q38", "q36"], gpus, 262144)
        by_model = {s["model"]: s for s in p["servers"]}
        self.assertEqual(by_model["qwen3.6:35b"]["cuda_visible_devices"], "0")
        self.assertEqual(by_model["qwen3.6:35b"]["profile"]["quant"], "UD-Q5_K_M")
        self.assertEqual(by_model["qwen3.8:27b"]["cuda_visible_devices"], "1")
        self.assertEqual(by_model["qwen3.8:27b"]["profile"]["quant"], "IQ3_XXS")

    def test_full_context_4060ti_profiles(self):
        gpus = [planmod.GPU(0, "RTX 4060 Ti", 16380, 16100, "8.9", "", 4, 8)]
        self.assertEqual(planmod.plan(["q38"], gpus, 262144)["servers"][0]["profile"]["quant"], "IQ3_XXS")
        self.assertEqual(planmod.plan(["q36"], gpus, 262144)["servers"][0]["profile"]["quant"], "UD-IQ3_XXS")


class GatewayTests(unittest.TestCase):
    def setUp(self):
        cfg = {
            "roles": {
                r: {"model_id": "local-" + r, "backend_alias": "local-" + r, "url": "http://127.0.0.1:1"}
                for r in ("haiku", "sonnet", "opus", "fable")
            }
        }
        self.router = gwmod.Router(cfg)

    def test_canonical_claude_ids_route_by_family(self):
        self.assertEqual(self.router.resolve("claude-fable-5")["model_id"], "local-fable")
        self.assertEqual(self.router.resolve("claude-opus-5")["model_id"], "local-opus")
        self.assertEqual(self.router.resolve("claude-sonnet-5")["model_id"], "local-sonnet")
        self.assertEqual(self.router.resolve("claude-haiku-4-5")["model_id"], "local-haiku")

    def test_unknown_internal_model_stays_local(self):
        self.assertEqual(self.router.resolve("unexpected-internal-id")["model_id"], "local-sonnet")

    def test_count_tokens_cannot_bypass_smaller_route_capacity(self):
        import json
        import threading
        import urllib.request
        import urllib.error
        from http.server import ThreadingHTTPServer
        cfg = {"roles": {
            "sonnet": {"model_id": "large", "backend_alias": "large", "url": "http://local",
                       "capacity": {"context": 128, "output_tokens": 32, "input_tokens": 80}},
            "haiku": {"model_id": "small", "backend_alias": "small", "url": "http://local",
                      "capacity": {"context": 64, "output_tokens": 16, "input_tokens": 32}},
        }}
        server = ThreadingHTTPServer(("127.0.0.1", 0), gwmod.Handler)
        server.router = gwmod.Router(cfg)
        server.verbose = False
        threading.Thread(target=server.serve_forever, daemon=True).start()
        def backend(endpoint, path, body=None, headers=None):
            return {"n_ctx_per_slot": 128} if path == "/props" else {"input_tokens": 40}
        try:
            with patch.object(gwmod.request_budget, "backend_json", side_effect=backend):
                for model, expected in (("large", 200), ("small", 400)):
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/v1/messages/count_tokens",
                        json.dumps({"model": model, "messages": [], "tools": [{"name": "run"}]}).encode(),
                        {"Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(request, timeout=5) as response:
                            self.assertEqual(response.status, expected)
                            self.assertEqual(json.load(response)["input_tokens"], 40)
                    except urllib.error.HTTPError as exc:
                        self.assertEqual(exc.code, expected)
                        self.assertIn("compact/summarize", exc.read().decode())
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
