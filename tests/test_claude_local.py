#!/usr/bin/env python3
import importlib.util
import http.client
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

    def test_classifier_replica_reserves_gpu_even_for_same_model(self):
        gpus = [
            planmod.GPU(i, "RTX 3090", 24576, 24000) for i in (0, 1)
        ]
        p = planmod.plan(["q38"], gpus, 131072,
                         classifier_model="q38", classifier_gpu=1)
        main, classifier = p["servers"]
        self.assertEqual(main["cuda_visible_devices"], "0")
        self.assertEqual(classifier["cuda_visible_devices"], "1")
        self.assertEqual(classifier["context"], 32768)
        self.assertEqual(classifier["slots"], 1)
        self.assertEqual(p["classifier_id"], "local-classifier")
        self.assertNotEqual(p["role_ids"]["sonnet"], p["classifier_id"])
        self.assertEqual(p["unused_gpus"], [])

    def test_classifier_configuration_fails_closed(self):
        gpus = [planmod.GPU(0, "RTX 3090", 24576, 24000)]
        for kwargs in (
            {"classifier_model": "q38"},
            {"classifier_gpu": 0},
            {"classifier_model": "q38", "classifier_gpu": 1},
            {"classifier_model": "q38", "classifier_gpu": 0},
            {"slots": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                planmod.plan(["q38"], gpus, 262144, **kwargs)

    def test_classifier_can_use_a_different_registered_model(self):
        gpus = [planmod.GPU(i, "RTX 3090", 24576, 24000) for i in (0, 1)]
        p = planmod.plan(["q38"], gpus, 131072,
                         classifier_model="q36", classifier_gpu=1)
        self.assertEqual(p["servers"][-1]["model"], "qwen3.6:35b")
        self.assertEqual(p["servers"][-1]["cuda_visible_devices"], "1")

    def test_multiple_slots_budget_aggregate_context(self):
        gpus = [planmod.GPU(0, "RTX 3090", 24576, 24000)]
        p = planmod.plan(["q38"], gpus, 262144, slots=2)
        s = p["servers"][0]
        self.assertEqual(s["slots"], 2)
        self.assertEqual(s["context"], 262144)
        profile = next(x for x in planmod.PROFILES[s["model"]]
                       if x.quant == s["profile"]["quant"])
        self.assertGreater(s["required_mib"], profile.required_mib)


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

    def test_classifier_model_is_independent_of_main_model(self):
        classifier = {"model_id": "local-classifier", "backend_alias": "local-classifier",
                      "url": "http://127.0.0.1:2"}
        router = gwmod.Router({"roles": self.router.roles,
                               "models": {"local-classifier": classifier}})
        self.assertEqual(router.resolve("local-classifier"), classifier)
        self.assertEqual(router.resolve("local-sonnet")["url"], "http://127.0.0.1:1")
        self.assertEqual(router.resolve("unknown")["model_id"], "local-sonnet")

    def test_explicit_classifier_request_id_overrides_family_only_for_that_id(self):
        route = {"model_id": "claude-sonnet-5", "backend_alias": "local-classifier",
                 "url": "http://127.0.0.1:2"}
        router = gwmod.Router({"roles": self.router.roles,
                               "models": {"claude-sonnet-5": route}})
        self.assertEqual(router.resolve("claude-sonnet-5"), route)
        self.assertEqual(router.resolve("claude-sonnet-other")["model_id"], "local-sonnet")
        self.assertEqual(router.resolve("local-sonnet")["model_id"], "local-sonnet")

    def test_classifier_request_id_cannot_override_session_route(self):
        with self.assertRaises(ValueError):
            gwmod.Router({"roles": self.router.roles,
                          "models": {"LOCAL-SONNET": {}}})

    def test_classifier_failure_does_not_fall_back_to_main(self):
        requests = []

        class Backend(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append((self.path, json.loads(
                    self.rfile.read(int(self.headers["content-length"])))))
                raw = b'{"error":"classifier unavailable"}'
                self.send_response(503)
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_):
                pass

        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        gateway = ThreadingHTTPServer(("127.0.0.1", 0), gwmod.Handler)
        gateway.router = gwmod.Router({
            "roles": self.router.roles,
            "models": {"local-classifier": {
                "model_id": "local-classifier", "backend_alias": "local-classifier",
                "url": f"http://127.0.0.1:{backend.server_port}",
            }},
        })
        gateway.verbose = False
        threads = [threading.Thread(target=s.serve_forever) for s in (backend, gateway)]
        for t in threads:
            t.start()
        conn = http.client.HTTPConnection("127.0.0.1", gateway.server_port, timeout=5)
        try:
            conn.request("GET", "/v1/models")
            self.assertIn("local-classifier", [m["id"] for m in
                          json.loads(conn.getresponse().read())["data"]])
            for path in ("/v1/messages", "/v1/messages/count_tokens"):
                conn.request("POST", path, json.dumps({"model": "local-classifier"}),
                             {"content-type": "application/json"})
                resp = conn.getresponse()
                self.assertEqual(resp.status, 503)
                self.assertEqual(json.loads(resp.read())["error"], "classifier unavailable")
            self.assertEqual(len(requests), 4)
            self.assertTrue(all(body["model"] == "local-classifier" for _, body in requests))
        finally:
            conn.close()
            for s in (gateway, backend):
                s.shutdown()
                s.server_close()
            for t in threads:
                t.join()


class HarnessTests(unittest.TestCase):
    def test_default_concurrency_counts_main_slots_not_classifier(self):
        gpus = [planmod.GPU(i, "RTX 3090", 24576, 24000) for i in (0, 1)]
        p = planmod.plan(["q38"], gpus, 131072, slots=2,
                         classifier_model="q38", classifier_gpu=1)
        source = (ROOT / "claude-local").read_text()
        source = source[source.index("server_rows() {"):source.index("doctor() {")]
        with tempfile.TemporaryDirectory() as tmp:
            planfile = pathlib.Path(tmp) / "plan.json"
            planfile.write_text(json.dumps(p))
            for override, expected in (("", "2"), ("7", "7")):
                result = subprocess.run(
                    ["bash", "-c", source + '\nPLAN_FILE="$1"; '
                     'CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY="$2"; '
                     'GATEWAY_PORT=19000; CLIENT_CTX=100000; ENABLE_TEAMS=0; '
                     'CLAUDE_ARGS=(); CLASSIFIER_MODEL=""; say() { :; }; '
                     'claude() { echo "$CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY"; '
                     'echo "$ANTHROPIC_MODEL"; }; run_claude',
                     "claude-local", str(planfile), override],
                    text=True, capture_output=True, check=True,
                )
                self.assertEqual(result.stdout.splitlines(),
                                 [expected, p["role_ids"]["sonnet"]])

    def test_server_rows_and_gateway_config_include_classifier(self):
        gpus = [planmod.GPU(i, "RTX 3090", 24576, 24000) for i in (0, 1)]
        p = planmod.plan(["q38"], gpus, 131072, slots=2,
                         classifier_model="q38", classifier_gpu=1)
        with tempfile.TemporaryDirectory() as tmp:
            state = pathlib.Path(tmp)
            planfile = state / "plan.json"
            planfile.write_text(json.dumps(p))
            backends = state / "backends"
            backends.write_text("\n".join(
                f"{s['id']}\t{19000+i}\tmodel" for i, s in enumerate(p["servers"])))
            source = (ROOT / "claude-local").read_text()
            source = source[source.index("server_rows() {"):source.index("contains_claude_flag()")]
            result = subprocess.run(
                ["bash", "-c", source + '\nPLAN_FILE="$1"; STATE_DIR="$2"; '
                 'BACKENDS_TSV="$3"; CLASSIFIER_REQUEST_MODELS=(claude-sonnet-5); '
                 'server_rows; write_gateway_config; '
                 'cat "$GATEWAY_CONFIG"; '
                 'CACHE_DIR="$STATE_DIR"; PORT_BASE=19000; ALLOW_OFFLOAD=0; '
                 'LLAMA_SERVER=unused; QWEN_TEMPLATE=unused; PIDS=(); '
                 'say() { :; }; free_port() { echo "$1"; }; curl() { return 0; }; '
                 'start_log_follower() { echo 0; }; stop_log_follower() { :; }; '
                 'env() { printf "%s\\n" "$@" > "$STATE_DIR/args.${1#CUDA_VISIBLE_DEVICES=}"; }; '
                 'start_backends; wait', "claude-local",
                 str(planfile), str(state), str(backends)],
                text=True, capture_output=True, check=True,
            )
            rows = result.stdout.splitlines()
            main, classifier = [row.split("\x1f") for row in rows[:2]]
            self.assertEqual(main[-2:], ["131072", "2"])
            self.assertEqual(classifier[-2:], ["32768", "1"])
            cfg = json.loads("\n".join(rows[2:]))
            self.assertEqual(cfg["models"]["local-classifier"]["url"],
                             "http://127.0.0.1:19001")
            self.assertEqual(cfg["roles"]["sonnet"]["url"],
                             "http://127.0.0.1:19000")
            self.assertEqual(cfg["models"]["claude-sonnet-5"]["backend_alias"],
                             "local-classifier")
            for gpu, context, slots in ((0, "262144", "2"), (1, "32768", "1")):
                args = (state / f"args.{gpu}").read_text().splitlines()
                self.assertEqual(args[args.index("-c")+1], context)
                self.assertEqual(args[args.index("-np")+1], slots)

    def test_classifier_requires_explicit_distinct_request_id(self):
        for args in (
            ["--local-classifier-model", "q38", "--local-classifier-gpu", "1"],
            ["--local-classifier-request-model", "claude-sonnet-5"],
            ["--local-classifier-model", "q38", "--local-classifier-gpu", "1",
             "--local-classifier-request-model", "claude-sonnet-5", "--model", "sonnet"],
            ["--local-slots", "0"],
            ["--local-classifier-request-model"],
        ):
            with self.subTest(args=args):
                result = subprocess.run(["bash", str(ROOT / "claude-local"), *args],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Installing", result.stdout)


if __name__ == "__main__":
    unittest.main()
