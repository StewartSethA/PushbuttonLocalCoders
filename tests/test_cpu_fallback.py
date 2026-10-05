import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
import claude_local_plan as base
import claude_local_resources as resources
import claude_local_validate as validate
import claude_local_gateway as gateway
import http.client
import time


def calibration(model="q38", mode="gpu", ngl=None, **overrides):
    profile = base.PROFILES[base.canonical_model(model)][0]
    entry = {
        "hf_spec": profile.hf_spec, "source": "test fixture; not production measurements",
        "mode": mode, "layer_count": 64,
        "ngl": ngl if ngl is not None else {"gpu": 65, "hybrid": 32, "cpu": 0}[mode],
        "weights_host_mib": 2000 if mode == "gpu" else 4000,
        "weights_gpu_mib": 0 if mode == "cpu" else 4000,
        "cache_host_mib_per_token": 0 if mode == "gpu" else .001,
        "cache_gpu_mib_per_token": 0 if mode == "cpu" else .002,
        "buffer_host_mib": 300, "buffer_gpu_mib": 0 if mode == "cpu" else 200,
        "max_context": 262144, "max_slots": 2, "max_threads": 32,
        "batch": 512, "ubatch": 256, "kv_k": "f16", "kv_v": "f16", "flash_attn": "off",
    }
    entry.update(overrides)
    return entry


class StartupPlannerTests(unittest.TestCase):
    def setUp(self):
        self.host = resources.Host(64000, None, 64000, tuple(range(8)), 8, ())
        self.gpus = [base.GPU(i, "fixture GPU", 12000, 11000) for i in range(3)]

    def plan(self, models=None, entries=None, **kwargs):
        return resources.startup_plan(
            base, models or ["q38"], kwargs.pop("gpus", self.gpus),
            kwargs.pop("context", 131072), host=kwargs.pop("host", self.host),
            metadata={"version": 1, "placements": entries or [calibration()]}, **kwargs)

    def test_default_is_gpu_only(self):
        p = self.plan(entries=[calibration(mode="cpu"), calibration(mode="gpu")])
        self.assertEqual(p["startup_policy"], "gpu-only")
        self.assertTrue(all(s["mode"] == "gpu" for s in p["servers"]))

    def test_gpu_only_without_gpu_rejects(self):
        with self.assertRaisesRegex(ValueError, "no safe gpu-only"):
            self.plan(gpus=[], entries=[calibration(mode="cpu")])

    def test_no_uncalibrated_cpu_guess(self):
        for policy in ("allow-cpu-only",):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                resources.startup_plan(base, ["q38"], [], 131072, host=self.host,
                                       startup_policy=policy)

    def test_cpu_only_preserves_slots_context_model(self):
        p = self.plan(gpus=[], entries=[calibration(mode="cpu")], startup_policy="allow-cpu-only")
        s = p["servers"][0]
        self.assertEqual((s["ngl"], s["context"], s["slots"]), (0, 131072, 2))
        self.assertEqual((s["model"], s["profile"]["quant"]), ("qwen3.8:27b", "Q8_0"))
        self.assertEqual(s["gpus"], [])
        self.assertTrue(s["cache_on_cpu"])
        self.assertEqual(s["ram_required_mib"], 4563)
        self.assertEqual(s["required_mib"], 0)

    def test_cpu_replica_is_provisioned_with_gpu(self):
        p = self.plan(entries=[calibration(), calibration(mode="cpu")],
                      startup_policy="allow-cpu-only")
        main, replica = p["servers"]
        self.assertEqual(main["mode"], "gpu")
        self.assertEqual(replica["mode"], "cpu")
        self.assertEqual(main["fallback_id"], replica["id"])
        self.assertEqual(main["profile"]["hf_spec"], replica["profile"]["hf_spec"])
        self.assertEqual((replica["context"], replica["slots"]), (131072, 2))
        self.assertEqual(p["ram_required_mib"], sum(s["ram_required_mib"] for s in p["servers"]))

    def test_cpu_replica_requires_matching_weights(self):
        with self.assertRaisesRegex(ValueError, "replica"):
            self.plan(entries=[calibration()], startup_policy="allow-cpu-only", min_quality=100)

    def test_cpu_calibration_can_pair_with_conservative_gpu_envelope(self):
        p = self.plan(entries=[calibration(mode="cpu")], gpus=[base.GPU(0, "GPU", 64000, 64000)],
                      startup_policy="allow-cpu-only", min_quality=100)
        main, replica = p["servers"]
        self.assertEqual(main["mode"], "gpu")
        self.assertEqual(replica["mode"], "cpu")
        self.assertIn("legacy_envelope_mib", main["memory_estimate"])

    def test_joint_replica_ram_budget_selects_cpu_not_unbudgeted_overflow(self):
        host = resources.Host(6000, 6000, 6000, tuple(range(8)), 8, ())
        p = self.plan(entries=[calibration(), calibration(mode="cpu")],
                      startup_policy="allow-cpu-only", host=host, min_quality=100)
        self.assertEqual([s["mode"] for s in p["servers"]], ["cpu"])
        self.assertNotIn("fallback_id", p["servers"][0])

    def test_classifier_replica_remains_distinct_with_reserved_gpu(self):
        p = self.plan(entries=[calibration(), calibration(mode="cpu")],
                      startup_policy="allow-cpu-only", classifier_model="q38", classifier_gpu=1)
        main, main_cpu, classifier, classifier_cpu = p["servers"]
        self.assertEqual(classifier["cuda_visible_devices"], "1")
        self.assertEqual(classifier["fallback_id"], "local-classifier-cpu")
        self.assertEqual(classifier_cpu["context"], 32768)
        self.assertTrue(classifier_cpu["classifier"])
        self.assertNotEqual(main_cpu["id"], classifier_cpu["id"])
        self.assertNotEqual(main["cuda_visible_devices"], "1")

    def test_hybrid_policy_and_metadata_rejected(self):
        with self.assertRaises(ValueError):
            self.plan(startup_policy="allow-hybrid")
        with self.assertRaises(ValueError):
            self.plan(entries=[calibration(mode="hybrid")])

    def test_main_stays_gpu_secondary_moves_to_cpu(self):
        entries = [calibration("q36", "gpu"), calibration("q36", "cpu"),
                   calibration("q38", "gpu"), calibration("q38", "cpu")]
        p = self.plan(["q36", "q38"], entries, gpus=self.gpus[:1], startup_policy="allow-cpu-only")
        by_model = {s["model"]: s for s in p["servers"] if not s.get("fallback")}
        self.assertEqual(by_model["qwen3.8:27b"]["mode"], "gpu")
        self.assertEqual(by_model["qwen3.6:35b"]["mode"], "cpu")
        self.assertLessEqual(sum(s["threads"] for s in p["servers"]), 8)

    def test_four_distinct_roles_two_gpus_and_classifier(self):
        models = ["q3-4b", "q38", "q36", "q38next"]
        entries = [calibration(model, mode) for model in models for mode in ("gpu", "cpu")]
        p = self.plan(models, entries, context=262144, gpus=self.gpus[:2],
                      classifier_model="q3-4b", classifier_gpu=1,
                      startup_policy="allow-cpu-only")
        primaries = {s["model"]: s for s in p["servers"]
                     if not s["classifier"] and not s.get("fallback")}
        self.assertEqual(len(primaries), 4)
        self.assertEqual(len(set(p["role_ids"].values())), 4)
        self.assertEqual(primaries[p["roles"]["sonnet"]]["cuda_visible_devices"], "0")
        for role in ("haiku", "opus", "fable"):
            self.assertEqual(primaries[p["roles"][role]]["mode"], "cpu")
        classifier = next(s for s in p["servers"] if s["id"] == p["classifier_id"])
        self.assertEqual(classifier["cuda_visible_devices"], "1")
        self.assertNotEqual(classifier["id"], p["role_ids"]["haiku"])
        self.assertEqual(classifier["model"], p["roles"]["haiku"])
        self.assertEqual(classifier["context"], 32768)
        self.assertEqual(len(p["servers"]), 7)
        by_id = {s["id"]: s for s in p["servers"]}
        for s in p["servers"]:
            self.assertEqual(s["slots"], 2)
            self.assertIn("test fixture; not production", s["estimate_source"])
            if not s["classifier"]:
                self.assertEqual(s["context"], 262144)
            if s["mode"] == "gpu":
                replica = by_id[s["fallback_id"]]
                self.assertEqual(replica["mode"], "cpu")
                for field in ("model", "context", "slots", "classifier"):
                    self.assertEqual(replica[field], s[field])
                self.assertEqual(replica["profile"]["hf_spec"], s["profile"]["hf_spec"])
        self.assertEqual(p["ram_required_mib"], sum(s["ram_required_mib"] for s in p["servers"]))
        self.assertLessEqual(sum(s["threads"] for s in p["servers"]), 8)

    def test_four_roles_without_classifier_have_six_servers(self):
        models = ["q3-4b", "q38", "q36", "q38next"]
        entries = [calibration(model, mode) for model in models for mode in ("gpu", "cpu")]
        p = self.plan(models, entries, gpus=self.gpus[:2], startup_policy="allow-cpu-only")
        self.assertEqual(len(p["servers"]), 6)
        self.assertIsNone(p["classifier_id"])
        self.assertEqual(sum(s["mode"] == "gpu" for s in p["servers"]), 2)
        self.assertEqual(sum(bool(s.get("fallback")) for s in p["servers"]), 2)
        self.assertEqual(len(set(p["role_ids"].values())), 4)

    def test_small_model_requires_calibration_even_on_large_gpu(self):
        for policy in ("gpu-only", "allow-cpu-only"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "calibrated"):
                resources.startup_plan(base, ["q3-4b"], [base.GPU(0, "fixture", 10**9, 10**9)],
                                       262144, host=self.host, startup_policy=policy)

    def test_small_model_both_quants_and_context_limit(self):
        for profile in base.PROFILES[base.canonical_model("q3-4b")]:
            entries = [calibration("q3-4b", mode, hf_spec=profile.hf_spec)
                       for mode in ("gpu", "cpu")]
            with self.subTest(quant=profile.quant):
                p = self.plan(["q3-4b"], entries, context=262144,
                              startup_policy="allow-cpu-only")
                self.assertEqual([s["profile"]["hf_spec"] for s in p["servers"]],
                                 [profile.hf_spec, profile.hf_spec])
                self.assertEqual(p["servers"][0]["fallback_id"], p["servers"][1]["id"])
                cpu = self.plan(["q3-4b"], entries, context=262144, gpus=[],
                                startup_policy="allow-cpu-only")
                self.assertEqual(len(cpu["servers"]), 1)
                self.assertEqual(cpu["servers"][0]["mode"], "cpu")
                self.assertEqual(cpu["servers"][0]["profile"]["hf_spec"], profile.hf_spec)
                with self.assertRaisesRegex(ValueError, "no safe"):
                    self.plan(["q3-4b"], entries, context=262145,
                              startup_policy="allow-cpu-only")

    def test_flash_next_catalogue_does_not_fit_two_24g_gpus(self):
        with self.assertRaisesRegex(ValueError, "no safe"):
            resources.startup_plan(base, ["q38next"],
                                   [base.GPU(i, "fixture 24G", 24576, 24576) for i in range(2)],
                                   262144, host=self.host)

    def test_joint_ram_budget_rejects_individually_fitting_models(self):
        host = resources.Host(7000, 7000, 7000, tuple(range(8)), 8, (), 1000)
        with self.assertRaisesRegex(ValueError, "joint RAM"):
            self.plan(["q36", "q38"], [calibration("q36", "cpu"), calibration("q38", "cpu")],
                      gpus=[], host=host, startup_policy="allow-cpu-only")

    def test_gpu_reserve_must_fit_not_just_weights(self):
        with self.assertRaises(ValueError):
            self.plan(gpus=[base.GPU(0, "GPU", 5000, 5000)], min_quality=100)

    def test_classifier_is_distinct_and_jointly_reserved(self):
        p = self.plan(classifier_model="q38", classifier_gpu=1)
        main, classifier = p["servers"]
        self.assertNotEqual(main["id"], classifier["id"])
        self.assertEqual(classifier["cuda_visible_devices"], "1")
        self.assertNotEqual(main["cuda_visible_devices"], "1")
        self.assertEqual(classifier["context"], 32768)
        self.assertEqual(p["ram_required_mib"], main["ram_required_mib"] + classifier["ram_required_mib"])

    def test_classifier_ram_not_ignored(self):
        host = resources.Host(5000, None, 5000, tuple(range(8)), 8, ())
        with self.assertRaises(ValueError):
            self.plan(host=host, classifier_model="q38", classifier_gpu=1)

    def test_classifier_cpu_only_explicit_policy(self):
        p = self.plan(entries=[calibration(mode="cpu")], gpus=[],
                      classifier_model="q38", startup_policy="allow-cpu-only")
        self.assertEqual(p["classifier_id"], "local-classifier")
        self.assertTrue(all(s["mode"] == "cpu" for s in p["servers"]))

    def test_reserved_classifier_never_migrates_to_cpu(self):
        with self.assertRaises(ValueError):
            self.plan(entries=[calibration(mode="cpu")], classifier_model="q38",
                      classifier_gpu=1, startup_policy="allow-cpu-only", min_quality=100)

    def test_joint_thread_quota(self):
        host = resources.Host(64000, None, 64000, (0, 1, 2), 1.5, ())
        with self.assertRaisesRegex(ValueError, "CPU quota"):
            self.plan(["q36", "q38"], [calibration("q36"), calibration("q38")], host=host)

    def test_calibration_thread_limit(self):
        p = self.plan(entries=[calibration(max_threads=1)], min_quality=100)
        self.assertEqual(p["servers"][0]["threads"], 1)

    def test_context_and_slots_are_not_silently_reduced(self):
        for kwargs in ({"context": 262145}, {"slots": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan(entries=[calibration(mode="cpu")], gpus=[],
                          startup_policy="allow-cpu-only", **kwargs)

    def test_bounded_complete_alternatives_and_quality(self):
        p = self.plan(entries=[calibration(), calibration(mode="cpu")],
                      startup_policy="allow-cpu-only", max_layouts=3, min_quality=100,
                      gpus=self.gpus[:1])
        self.assertEqual(len(p["alternatives"]), 1)
        self.assertEqual([layout["servers"][0]["mode"] for layout in [p, *p["alternatives"]]],
                         ["gpu", "cpu"])
        for layout in [p, *p["alternatives"]]:
            self.assertEqual(layout["role_ids"], p["role_ids"])
            self.assertTrue(all(s["profile"]["quality"] == 100 for s in layout["servers"]))

    def test_invalid_metadata_rejected(self):
        for field, value in (("weights_host_mib", -1), ("cache_gpu_mib_per_token", float("nan")),
                             ("ngl", True), ("source", ""), ("layer_count", 0),
                             ("mode", "cache-only"), ("max_slots", 0)):
            entry = calibration()
            entry[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                resources.validate_metadata({"version": 1, "placements": [entry]})

    def test_cpu_cache_quantization_and_cache_only_deferred(self):
        for entry in (calibration(mode="cpu", kv_k="q4_0"),
                      calibration(cache_host_mib_per_token=.01),
                      calibration(mode="cpu", weights_gpu_mib=1),
                      calibration(mode="gpu", ngl=5), calibration(mode="hybrid", ngl=65),
                      calibration(mode="gpu", weights_gpu_mib=0),
                      calibration(mode="hybrid", weights_host_mib=0)):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                resources.validate_metadata({"version": 1, "placements": [entry]})

    def test_legacy_envelope_is_not_split_85_15(self):
        p = resources.startup_plan(base, ["q38"], [base.GPU(0, "GPU", 64000, 64000)],
                                   131072, slots=1, host=self.host)
        s = p["servers"][0]
        profile = base.PROFILES["qwen3.8:27b"][0]
        self.assertEqual(s["required_mib"], profile.required_mib)
        self.assertEqual(s["ram_required_mib"], profile.required_mib)
        self.assertIn("legacy_envelope_mib", s["memory_estimate"])

    def test_aggregate_slot_bound_precedes_launch(self):
        with self.assertRaisesRegex(ValueError, "64 aggregate"):
            self.plan(slots=65)

    def test_nvidia_timeout_does_not_prevent_cpu_fallback(self):
        with mock.patch.object(base.shutil, "which", return_value="/not/nvidia-smi"), \
             mock.patch.object(base.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 10)):
            self.assertEqual(base.inventory(), [])


class HostInventoryTests(unittest.TestCase):
    def inventory(self, version=2, ancestor_limit="8388608000", leaf_limit="max"):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = pathlib.Path(directory)
            proc, sysroot = root / "proc", root / "sys"
            (proc / "self").mkdir(parents=True)
            (proc / "meminfo").write_text("MemAvailable: 12288000 kB\nSwapFree: 999999999 kB\n")
            (proc / "self/cgroup").write_text(
                "0::/parent/child\n" if version == 2 else "5:memory:/parent/child\n6:cpu:/parent/child\n")
            cg = sysroot / ("fs/cgroup" if version == 2 else "fs/cgroup/memory")
            for path, limit, used in ((cg, "max" if version == 2 else str(2**60), 0),
                                       (cg / "parent", ancestor_limit, 1048576000),
                                       (cg / "parent/child", leaf_limit, 0)):
                path.mkdir(parents=True, exist_ok=True)
                (path / ("memory.max" if version == 2 else "memory.limit_in_bytes")).write_text(limit)
                (path / ("memory.current" if version == 2 else "memory.usage_in_bytes")).write_text(str(used))
            if version == 2:
                (cg / "parent/cpu.max").write_text("150000 100000")
            else:
                cpu = sysroot / "fs/cgroup/cpu/parent/child"
                cpu.mkdir(parents=True)
                (cpu.parent / "cpu.cfs_quota_us").write_text("150000")
                (cpu.parent / "cpu.cfs_period_us").write_text("100000")
            numa = sysroot / "devices/system/node/node0"
            numa.mkdir(parents=True)
            (numa / "cpulist").write_text("0-3")
            with mock.patch.object(os, "sched_getaffinity", return_value={1, 3}):
                return resources.host_inventory(proc, sysroot)

    def test_v2_ancestors_affinity_quota_numa_no_swap(self):
        h = self.inventory()
        self.assertEqual(h.available_mib, 12000)
        self.assertEqual(h.cgroup_remaining_mib, 7000)
        self.assertEqual(h.usable_mib, 7000)
        self.assertEqual(h.cpu_ids, (1, 3))
        self.assertEqual(h.cpu_capacity, 1.5)
        self.assertEqual(h.numa_nodes, ("node0: 0-3",))

    def test_v1_remaining_quota(self):
        self.assertEqual(self.inventory(version=1, leaf_limit=str(2**60)).usable_mib, 7000)

    def test_exhausted_cgroup_is_not_negative(self):
        self.assertEqual(self.inventory(ancestor_limit="1024").usable_mib, 0)

    def test_leaf_stricter_than_ancestor(self):
        self.assertEqual(self.inventory(leaf_limit="2097152000").usable_mib, 2000)


class OverflowGatewayTests(unittest.TestCase):
    def setUp(self):
        self.release_stream = threading.Event()
        self.requests = []
        self.behavior = {"gpu": "json", "cpu": "json", "classifier-gpu": "json", "classifier-cpu": "json"}
        owner = self

        def handler(kind):
            class Backend(BaseHTTPRequestHandler):
                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    owner.requests.append((kind, body["model"]))
                    behavior = owner.behavior[kind]
                    if behavior == "error":
                        self.send_response(503)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    if behavior == "empty":
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    if behavior in ("stream", "broken"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Content-Length", "40")
                        self.end_headers()
                        self.wfile.write(b"data: first\n\n")
                        self.wfile.flush()
                        owner.release_stream.wait(5)
                        if behavior == "stream":
                            try:
                                self.wfile.write(b"x" * 27)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                pass
                        self.close_connection = True
                        return
                    raw = json.dumps({"model": body["model"]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)

                def log_message(self, *_):
                    pass
            return Backend

        self.cpu = ThreadingHTTPServer(("127.0.0.1", 0), handler("cpu"))
        self.gpu = ThreadingHTTPServer(("127.0.0.1", 0), handler("gpu"))
        self.primary = self.route(self.gpu, "main")
        self.fallback = self.route(self.cpu, "main-cpu", proxy_attempts=1)
        self.primary["cpu_fallback"] = self.fallback
        classifier_gpu = ThreadingHTTPServer(("127.0.0.1", 0), handler("classifier-gpu"))
        classifier_cpu = ThreadingHTTPServer(("127.0.0.1", 0), handler("classifier-cpu"))
        classifier = self.route(classifier_gpu, "local-classifier")
        classifier["cpu_fallback"] = self.route(classifier_cpu, "local-classifier-cpu", proxy_attempts=1)
        self.router = gateway.Router({
            "roles": {r: self.primary for r in ("haiku", "sonnet", "opus", "fable")},
            "models": {"classifier-request": classifier},
        })
        self.gateway = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
        self.gateway.router = self.router
        self.gateway.verbose = False
        self.servers = [self.gateway, self.gpu, self.cpu, classifier_gpu, classifier_cpu]
        self.threads = [threading.Thread(target=s.serve_forever, kwargs={"poll_interval": .05})
                        for s in self.servers]
        for thread in self.threads:
            thread.start()
        self.connections = []

    @staticmethod
    def route(server, alias, **kwargs):
        return {"url": f"http://127.0.0.1:{server.server_port}", "backend_alias": alias,
                "model_id": alias, "slots": 2, "timeout": 2, **kwargs}

    def tearDown(self):
        self.release_stream.set()
        for connection in self.connections:
            connection.close()
        for server in self.servers:
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join()

    def request(self, model="sonnet"):
        connection = http.client.HTTPConnection("127.0.0.1", self.gateway.server_port, timeout=5)
        self.connections.append(connection)
        connection.request("POST", "/v1/messages", json.dumps({"model": model}),
                           {"Content-Type": "application/json"})
        return connection, connection.getresponse()

    def assert_idle(self):
        deadline = time.monotonic() + 3
        while any(self.router.active.values()) and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(all(n == 0 for n in self.router.active.values()), self.router.active)

    def test_gpu_available_keeps_request_on_gpu(self):
        _, response = self.request()
        self.assertEqual(json.loads(response.read())["model"], "main")
        self.assertEqual(self.requests, [("gpu", "main")])
        self.assert_idle()

    def test_model_aliases_share_backend_capacity(self):
        alternate = {**self.primary, "backend_alias": "another-alias", "model_id": "another-alias"}
        router = gateway.Router({"roles": self.router.roles, "models": {"another": alternate}})
        self.assertTrue(router.acquire(self.primary))
        self.assertTrue(router.acquire(alternate))
        self.assertFalse(router.acquire(self.primary))
        router.release(self.primary)
        router.release(alternate)

    def test_two_busy_gpu_streams_overflow_to_cpu_then_exhaust(self):
        self.behavior.update(gpu="stream", cpu="stream")
        responses = []
        for model in ("sonnet", "haiku"):
            _, response = self.request(model)
            self.assertEqual(response.read(13), b"data: first\n\n")
            responses.append(response)
        self.assertEqual(self.router.active[self.router.key(self.primary)], 2)
        for model in ("unknown-internal-id", "fable"):
            _, response = self.request(model)
            self.assertEqual(response.read(13), b"data: first\n\n")
            responses.append(response)
        self.assertEqual(self.router.active[self.router.key(self.fallback)], 2)
        _, response = self.request("opus")
        self.assertEqual(response.status, 503)
        response.read()
        self.assertEqual(self.requests, [("gpu", "main")] * 2 + [("cpu", "main-cpu")] * 2)
        self.release_stream.set()
        for response in responses:
            self.assertEqual(response.read(), b"x" * 27)
        self.assert_idle()

    def test_gpu_precommit_http_failure_can_use_cpu(self):
        self.behavior["gpu"] = "error"
        _, response = self.request()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read())["model"], "main-cpu")
        self.assertEqual([kind for kind, _ in self.requests], ["gpu", "gpu", "cpu"])
        self.assert_idle()

    def test_empty_gpu_stream_can_use_cpu(self):
        self.behavior["gpu"] = "empty"
        _, response = self.request()
        self.assertEqual(json.loads(response.read())["model"], "main-cpu")
        self.assert_idle()

    def test_cpu_http_failure_surfaces_without_other_model(self):
        self.behavior.update(gpu="error", cpu="error")
        _, response = self.request()
        self.assertEqual(response.status, 503)
        response.read()
        self.assertEqual(self.requests[-1], ("cpu", "main-cpu"))
        self.assertEqual(sum(kind == "cpu" for kind, _ in self.requests), 1)
        self.assert_idle()

    def test_cpu_empty_stream_surfaces_gateway_error(self):
        self.behavior.update(gpu="error", cpu="empty")
        _, response = self.request()
        self.assertEqual(response.status, 502)
        response.read()
        self.assert_idle()

    def test_committed_gpu_failure_releases_slots_without_cpu_replay(self):
        self.behavior["gpu"] = "broken"
        _, response = self.request()
        self.assertEqual(response.read(13), b"data: first\n\n")
        self.assertEqual(self.router.active[self.router.key(self.primary)], 1)
        self.release_stream.set()
        with self.assertRaises(http.client.IncompleteRead):
            response.read()
        self.assertEqual(self.requests, [("gpu", "main")])
        self.assert_idle()

    def test_downstream_disconnect_releases_stream_slot(self):
        self.behavior["gpu"] = "stream"
        connection, response = self.request()
        response.read(13)
        response.close()
        connection.close()
        self.release_stream.set()
        self.assert_idle()

    def test_cpu_stream_holds_slot_and_failure_does_not_replay(self):
        self.behavior.update(gpu="error", cpu="broken")
        _, response = self.request()
        response.read(13)
        self.assertEqual(self.router.active[self.router.key(self.fallback)], 1)
        self.release_stream.set()
        with self.assertRaises(http.client.IncompleteRead):
            response.read()
        self.assertEqual(sum(kind == "cpu" for kind, _ in self.requests), 1)
        self.assert_idle()

    def test_classifier_overflow_is_independent_of_main(self):
        classifier = self.router.resolve("classifier-request")
        for _ in range(2):
            self.assertTrue(self.router.acquire(classifier))
        try:
            _, response = self.request("classifier-request")
            self.assertEqual(json.loads(response.read())["model"], "local-classifier-cpu")
            _, response = self.request("sonnet")
            self.assertEqual(json.loads(response.read())["model"], "main")
        finally:
            for _ in range(2):
                self.router.release(classifier)
        self.assert_idle()

    def test_classifier_cpu_failure_does_not_use_main_or_synthesize_verdict(self):
        self.behavior.update({"classifier-gpu": "error", "classifier-cpu": "error"})
        _, response = self.request("classifier-request")
        self.assertEqual(response.status, 503)
        response.read()
        self.assertEqual(self.requests, [("classifier-gpu", "local-classifier")] * 2 +
                         [("classifier-cpu", "local-classifier-cpu")])
        self.assert_idle()

    def test_gpu_only_exhaustion_does_not_create_implicit_cpu_route(self):
        self.primary.pop("cpu_fallback")
        for _ in range(2):
            self.router.acquire(self.primary)
        try:
            _, response = self.request()
            self.assertEqual(response.status, 503)
            response.read()
            self.assertEqual(self.requests, [])
        finally:
            for _ in range(2):
                self.router.release(self.primary)
        self.assert_idle()


class LauncherFallbackTests(unittest.TestCase):
    def source(self, start, end):
        source = (ROOT / "claude-local").read_text()
        return source[source.index(start):source.index(end)]

    def test_four_role_gateway_config_keeps_classifier_explicit(self):
        models = ["q3-4b", "q38", "q36", "q38next"]
        p = resources.startup_plan(
            base, models, [base.GPU(i, "fixture GPU", 12000, 11000) for i in range(2)],
            262144, startup_policy="allow-cpu-only", classifier_model="q3-4b",
            classifier_gpu=1, host=resources.Host(64000, None, 64000, tuple(range(8)), 8, ()),
            metadata={"version": 1, "placements": [
                calibration(model, mode) for model in models for mode in ("gpu", "cpu")]})
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            state = pathlib.Path(directory)
            (state / "plan.json").write_text(json.dumps(p))
            (state / "backends.tsv").write_text("".join(
                f"{s['id']}\t{19000+i}\n" for i, s in enumerate(p["servers"])))
            script = self.source("write_gateway_config() {", "start_gateway() {")
            result = subprocess.run(["bash", "-c", script + '''
STATE_DIR="$1"; PLAN_FILE="$1/plan.json"; BACKENDS_TSV="$1/backends.tsv";
CLASSIFIER_REQUEST_MODELS=(observed-safety-id);
write_gateway_config
cat "$GATEWAY_CONFIG"
''', "test", directory], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            router = gateway.Router(json.loads(result.stdout))
            for role, mid in p["role_ids"].items():
                self.assertEqual(router.resolve(role)["backend_alias"], mid)
                self.assertEqual(router.resolve("claude-" + role + "-5")["backend_alias"], mid)
            classifier = router.resolve("observed-safety-id")
            self.assertEqual(classifier["backend_alias"], "local-classifier")
            self.assertEqual(classifier["cpu_fallback"]["backend_alias"], "local-classifier-cpu")
            self.assertNotEqual(classifier["url"], router.resolve("haiku")["url"])
            self.assertNotEqual(classifier["url"], router.resolve("sonnet")["url"])

    def test_cancel_exits_without_retry_and_holds_lock_through_teardown(self):
        import fcntl

        script = (self.source("acquire_startup_lock() {", "inventory_arches() {") +
                  self.source("cleanup() {", "server_rows() {") +
                  self.source("start_layouts() {", "write_gateway_config()"))
        for sig, status in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            for phase in ("readiness", "retry-teardown"):
                with self.subTest(signal=sig, phase=phase), \
                     tempfile.TemporaryDirectory(dir=ROOT) as directory:
                    state = pathlib.Path(directory)
                    (state / "plan.json").write_text(json.dumps({"alternatives": [{}]}))
                    (state / "server.py").write_text('''
import pathlib, signal, sys, time
state = pathlib.Path(sys.argv[1])
def stop(signum, frame):
    with (state / "stops").open("a") as log:
        log.write("stop\\n")
    (state / "stopping").touch()
    while not (state / "release").exists():
        time.sleep(.02)
    (state / "stopped").touch()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
(state / "ready").touch()
while True:
    time.sleep(.02)
''')
                    launcher = subprocess.Popen(["bash", "-c", script + '''
set -euo pipefail
STATE_DIR="$1"; PLAN_FILE="$1/plan.json"; LOCK_FILE="$1/startup.lock";
KEEP_SERVERS=1; STARTUP_COMPLETE=0; STARTUP_LOCK_FD=""; PIDS=(); STARTUP_TIMEOUT=30;
say() { :; }; warn() { :; }; print_plan() { :; }; ensure_llamacpp() { :; };
verify_llama_options() { :; }; admit_layout() { :; }; validate_layout() { return 1; };
die() { exit 1; };
start_backends() {
    echo launch >> "$STATE_DIR/launches"
    python3 "$STATE_DIR/server.py" "$STATE_DIR" & PIDS+=("$!")
    echo "$!" >> "$STATE_DIR/pids"
    while [[ ! -e "$STATE_DIR/ready" ]]; do sleep .02; done
    [[ "$PHASE" == retry-teardown ]] && return 1
    while kill -0 "${PIDS[0]}" 2>/dev/null; do sleep .02; done
    return 1
}
acquire_startup_lock
start_layouts
echo gateway >> "$STATE_DIR/launches"
''',
                        "test", directory], env={**os.environ, "PHASE": phase},
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    try:
                        marker = state / ("ready" if phase == "readiness" else "stopping")
                        deadline = time.monotonic() + 5
                        while not marker.exists() and time.monotonic() < deadline:
                            time.sleep(.02)
                        self.assertTrue(marker.exists(), "launcher did not reach test phase")
                        launcher.send_signal(sig)
                        deadline = time.monotonic() + 5
                        while not (state / "stopping").exists() and time.monotonic() < deadline:
                            time.sleep(.02)
                        self.assertTrue((state / "stopping").exists(), "server was not terminated")
                        if phase == "readiness":
                            launcher.send_signal(signal.SIGTERM if sig == signal.SIGINT else signal.SIGINT)
                        with (state / "startup.lock").open("a") as lock:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            self.assertFalse((state / "stopped").exists())
                            (state / "release").touch()
                            stdout, stderr = launcher.communicate(timeout=5)
                            self.assertEqual(launcher.returncode, status, stdout + stderr)
                            self.assertTrue((state / "stopped").exists())
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        self.assertEqual((state / "launches").read_text().splitlines(), ["launch"])
                        # A signal arriving during cleanup must not re-enter teardown.
                        if phase == "readiness":
                            self.assertEqual((state / "stops").read_text().splitlines(), ["stop"])
                    finally:
                        (state / "release").touch()
                        if launcher.poll() is None:
                            launcher.kill()
                        launcher.communicate(timeout=5)
                        if (state / "pids").exists() and not (state / "stopped").exists():
                            for pid in (state / "pids").read_text().splitlines():
                                try:
                                    os.kill(int(pid), signal.SIGKILL)
                                except ProcessLookupError:
                                    pass

    def test_cleanup_keep_servers_only_after_success_and_preserves_exit_status(self):
        script = self.source("cleanup() {", "server_rows() {")
        for complete in (0, 1):
            for keep in (0, 1):
                for termination, status in (("exit 7", 7), ("kill -INT $$", 130),
                                            ("kill -TERM $$", 143)):
                    with self.subTest(complete=complete, keep=keep, termination=termination):
                        result = subprocess.run(["bash", "-c", script + '''
KEEP_SERVERS="$1"; STARTUP_COMPLETE="$2"; PIDS=();
stop_layout() { echo stop; PIDS=(); };
release_startup_lock() { echo unlock; };
''' + termination + "\necho resumed", "test", str(keep), str(complete)],
                            capture_output=True, text=True, timeout=5)
                        self.assertEqual(result.returncode, status, result.stderr)
                        expected = ["unlock"] if keep and complete else ["stop", "unlock"]
                        self.assertEqual(result.stdout.splitlines(), expected)

    def test_cpu_launch_exact_plan_flags(self):
        p = resources.startup_plan(base, ["q38"], [], 131072, startup_policy="allow-cpu-only",
                                   host=resources.Host(64000, None, 64000, (0, 1), 2, ()),
                                   metadata={"version": 1, "placements": [calibration(mode="cpu")]})
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            state = pathlib.Path(directory)
            plan = state / "plan.json"
            plan.write_text(json.dumps(p))
            script = self.source("server_rows() {", "contains_claude_flag()")
            result = subprocess.run(["bash", "-c", script + '''
PLAN_FILE="$1"; STATE_DIR="$2"; CACHE_DIR="$2"; PORT_BASE=19000; ALLOW_OFFLOAD=0;
LLAMA_SERVER=unused; QWEN_TEMPLATE=unused; PIDS=();
say() { :; }; free_port() { echo "$1"; }; curl() { return 0; };
start_log_follower() { echo 0; }; stop_log_follower() { :; };
env() { printf '%s\\n' "$@" > "$STATE_DIR/args"; };
start_backends; wait
''', "test", str(plan), str(state)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = (state / "args").read_text().splitlines()
            for flag, value in (("-ngl", "0"), ("--device", "none"), ("--fit", "off"),
                                ("-t", "2"), ("-tb", "2"), ("-c", "262144"), ("-np", "2")):
                self.assertEqual(args[args.index(flag) + 1], value)
            self.assertIn("--no-kv-offload", args)
            self.assertIn("CUDA_VISIBLE_DEVICES=", args)
            self.assertEqual(args[args.index("--cache-type-k") + 1], "f16")

    def test_clean_complete_layout_retry_before_gateway(self):
        p = resources.startup_plan(base, ["q38"], [], 131072, startup_policy="allow-cpu-only",
                                   host=resources.Host(64000, None, 64000, (0, 1), 2, ()),
                                   metadata={"version": 1, "placements": [calibration(mode="cpu"),
                                                                         calibration(mode="cpu", buffer_host_mib=400)]})
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            state = pathlib.Path(directory)
            plan = state / "plan.json"
            plan.write_text(json.dumps(p))
            script = self.source("start_layouts() {", "write_gateway_config()")
            result = subprocess.run(["bash", "-c", script + '''
PLAN_FILE="$1"; STATE_DIR="$2"; STARTUP_TIMEOUT=30; counter=0;
say() { :; }; warn() { :; }; print_plan() { :; }; ensure_llamacpp() { :; };
verify_llama_options() { return 0; };
admit_layout() { return 0; };
start_backends() { echo launch; counter=$((counter+1)); };
validate_layout() { (( counter > 1 )); };
stop_layout() { echo stop-all; };
die() { echo failure; exit 1; };
start_layouts; echo gateway
''', "test", str(plan), str(state)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["launch", "stop-all", "launch", "gateway"])
            self.assertNotIn("alternatives", json.loads(plan.read_text()))
            self.assertFalse(list(state.glob("layouts.*")))

    def test_cpu_build_does_not_call_cuda_functions(self):
        script = self.source("ensure_llamacpp() {", "fetch_qwen_template()")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            result = subprocess.run(["bash", "-c", script + '''
STATE_DIR="$1"; build_kind() { echo upstream; }; inventory_arches() { :; };
plan_uses_gpu() { return 1; }; say() { :; };
prepare_host_compiler() { echo forbidden; exit 9; }; activate_cuda() { echo forbidden; exit 9; };
git() { mkdir -p "$STATE_DIR/llama.cpp"; };
cmake() {
 printf '%s\\n' "$@" >> "$STATE_DIR/cmake-args";
 mkdir -p "$STATE_DIR/llama.cpp/build-cpu/bin";
 touch "$STATE_DIR/llama.cpp/build-cpu/bin/llama-server";
 chmod +x "$STATE_DIR/llama.cpp/build-cpu/bin/llama-server";
};
die() { exit 1; }; ensure_llamacpp
''', "test", directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = (pathlib.Path(directory) / "cmake-args").read_text()
            self.assertIn("-DGGML_CUDA=OFF", args)
            self.assertNotIn("CMAKE_CUDA_COMPILER", args)
            self.assertNotIn("forbidden", result.stdout)

    def test_generated_gateway_routes_have_explicit_independent_cpu_capacity(self):
        p = resources.startup_plan(
            base, ["q38"], [base.GPU(i, "GPU", 12000, 11000) for i in (0, 1)], 131072,
            startup_policy="allow-cpu-only", classifier_model="q38", classifier_gpu=1,
            host=resources.Host(64000, None, 64000, tuple(range(8)), 8, ()),
            metadata={"version": 1, "placements": [calibration(), calibration(mode="cpu")]})
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            state = pathlib.Path(directory)
            plan, backends = state / "plan.json", state / "backends"
            plan.write_text(json.dumps(p))
            backends.write_text("".join(f"{s['id']}\t{19000+i}\tmodel\n"
                                       for i, s in enumerate(p["servers"])))
            source = self.source("write_gateway_config() {", "start_gateway()")
            result = subprocess.run(["bash", "-c", source + '''
PLAN_FILE="$1"; STATE_DIR="$2"; BACKENDS_TSV="$3";
CLASSIFIER_REQUEST_MODELS=(classifier-request);
write_gateway_config; cat "$GATEWAY_CONFIG"
''', "test", str(plan), str(state), str(backends)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(result.stdout)
            main = config["roles"]["sonnet"]
            classifier = config["models"]["classifier-request"]
            self.assertEqual(main["cpu_fallback"]["backend_alias"], p["role_ids"]["sonnet"] + "-cpu")
            self.assertEqual(classifier["cpu_fallback"]["backend_alias"], "local-classifier-cpu")
            self.assertEqual(main["slots"], 2)
            self.assertEqual(classifier["cpu_fallback"]["slots"], 2)
            self.assertEqual(classifier["cpu_fallback"]["proxy_attempts"], 1)
            router = gateway.Router(config)
            self.assertEqual(len(router.active), 4)

    def test_readiness_failure_is_bounded(self):
        p = base.plan(["q38"], [base.GPU(0, "GPU", 64000, 64000)], 131072)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            plan = pathlib.Path(directory) / "plan.json"
            plan.write_text(json.dumps(p))
            script = self.source("server_rows() {", "plan_uses_gpu()")
            result = subprocess.run(["bash", "-c", script + '''
PLAN_FILE="$1"; STATE_DIR="$2"; CACHE_DIR="$2"; PORT_BASE=19000; ALLOW_OFFLOAD=0;
LLAMA_SERVER=unused; QWEN_TEMPLATE=unused; PIDS=(); LAYOUT_DEADLINE=0;
say() { :; }; warn() { :; }; free_port() { echo "$1"; }; curl() { return 1; };
start_log_follower() { echo 0; }; stop_log_follower() { :; }; env() { :; };
start_backends; rc=$?; wait; exit "$rc"
''', "test", str(plan), directory], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 1)

    def test_actual_cpu_dry_run_without_nvidia_tools(self):
        import shutil
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            state = pathlib.Path(directory)
            tools = state / "bin"
            tools.mkdir()
            for command in ("bash", "python3", "readlink", "dirname", "mkdir"):
                (tools / command).symlink_to(shutil.which(command))
            metadata = state / "memory.json"
            metadata.write_text(json.dumps({"version": 1, "placements": [calibration(mode="cpu")]}))
            env = {**os.environ, "PATH": str(tools), "CLAUDE_LOCAL_STATE": str(state / "state"),
                   "CLAUDE_LOCAL_CACHE": str(state / "cache"), "CLAUDE_LOCAL_STARTUP_POLICY": "gpu-only"}
            result = subprocess.run(
                [str(ROOT / "claude-local"), "q38", "--local-startup-policy", "allow-cpu-only",
                 "--local-memory-metadata", str(metadata), "--local-min-quality", "100", "--local-dry-run"],
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for detail in ("MemAvailable=", "cgroup remaining=", "placement=cpu ngl=0",
                           "capacity=", "no swap", "latency is unmeasured"):
                self.assertIn(detail, result.stdout)

    def test_warmup_process_deadline_is_bounded(self):
        source = self.source("validate_layout() {", "start_layouts()")
        result = subprocess.run(["bash", "-c", source + '''
LIB=unused; PLAN_FILE=unused; BACKENDS_TSV=unused; WARMUP_TIMEOUT=1;
LAYOUT_DEADLINE=$((SECONDS+10)); PIDS=();
python3() { exec sleep 10; }; warn() { :; };
validate_layout; rc=$?;
for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
exit "$rc"
'''], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1)


class RuntimeValidationTests(unittest.TestCase):
    def test_admission_rechecks_joint_ram_gpu_cpu_and_refreshes_baseline(self):
        p = resources.startup_plan(base, ["q38"], [base.GPU(0, "GPU", 12000, 11000)], 131072,
                                   host=resources.Host(64000, None, 64000, (0, 1), 2, ()),
                                   metadata={"version": 1, "placements": [calibration()]},
                                   min_quality=100)
        for host, gpus in (
            (resources.Host(1000, None, 1000, (0, 1), 2, ()), [base.GPU(0, "GPU", 12000, 11000)]),
            (resources.Host(64000, None, 64000, (0,), 1, ()), [base.GPU(0, "GPU", 12000, 11000)]),
            (resources.Host(64000, None, 64000, (0, 1), 2, ()), []),
            (resources.Host(64000, None, 64000, (0, 1), 2, ()), [base.GPU(0, "GPU", 12000, 4000)]),
        ):
            with self.subTest(host=host, gpus=gpus), self.assertRaises(ValueError):
                validate.admit_layout(p, host=host, gpus=gpus)
        validate.admit_layout(p, host=resources.Host(64000, None, 64000, (0, 1), 2, ()),
                              gpus=[base.GPU(0, "GPU", 12000, 10000)])
        self.assertEqual(p["servers"][0]["gpus"][0]["free_mib"], 10000)

    def test_all_servers_and_slots_receive_real_warmups(self):
        observed = []

        class Backend(BaseHTTPRequestHandler):
            def do_POST(self):
                observed.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = b'{"content":"ready"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        thread = threading.Thread(target=backend.serve_forever)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                root = pathlib.Path(directory)
                plan, backends = root / "plan.json", root / "backends"
                plan.write_text(json.dumps({"servers": [{"id": "main", "slots": 2, "mode": "cpu"},
                                                       {"id": "local-classifier", "slots": 2, "mode": "cpu"}]}))
                backends.write_text(f"main\t{backend.server_port}\tmodel\t{os.getpid()}\n"
                                    f"local-classifier\t{backend.server_port}\tmodel\t{os.getpid()}\n")
                with mock.patch.object(sys, "argv", ["validator", str(plan), str(backends), "--timeout", "5"]), \
                     mock.patch.object(validate, "check_memory") as check:
                    self.assertEqual(validate.main(), 0)
                self.assertEqual(len(observed), 4)
                self.assertTrue(all(r["n_predict"] == 1 and not r["cache_prompt"] for r in observed))
                check.assert_called_once()
        finally:
            backend.shutdown()
            backend.server_close()
            thread.join()

    def test_real_warmup_error_fails_closed(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        for body in ({"error": "classifier unavailable"}, {"content": 42}, {}):
            response.read.return_value = json.dumps(body).encode()
            with mock.patch.object(validate.urllib.request, "urlopen", return_value=response), \
                 self.assertRaises(ValueError):
                validate.warmup("http://127.0.0.1:1", "local-classifier", 0, 1)

    def test_offload_log_enforces_gpu_cpu_placement(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = pathlib.Path(directory)
            log = root / "main.log"
            for mode, ngl, report, accepted in (
                ("gpu", 999, "", False),
                ("gpu", 999, "offloaded 0/65 layers to GPU", False),
                ("gpu", 999, "offloaded 32/65 layers to GPU", False),
                ("gpu", 999, "offloaded 65/65 layers to GPU", True),
                ("hybrid", 32, "offloaded 31/65 layers to GPU", False),
                ("hybrid", 32, "offloaded 32/65 layers to GPU", False),
                ("hybrid", 65, "offloaded 65/65 layers to GPU", False),
                ("cpu", 0, "offloaded 1/65 layers to GPU", False),
                ("cpu", 0, "", True),
            ):
                log.write_text(report)
                p = {"servers": [{"id": "main", "mode": mode, "ngl": ngl}]}
                with self.subTest(mode=mode, report=report):
                    if accepted:
                        validate.check_offload(p, root)
                    else:
                        with self.assertRaises(ValueError):
                            validate.check_offload(p, root)

    def test_ram_reserve_fails_closed(self):
        host = resources.Host(500, 500, 500, (0,), 1, ())
        with self.assertRaisesRegex(ValueError, "RAM headroom"):
            validate.check_memory({"host": {"reserve_mib": 1024}, "servers": []}, {}, host=host)

    def test_gpu_usage_envelope_and_missing_gpu(self):
        host = resources.Host(64000, None, 64000, (0,), 1, ())
        server = {"id": "test", "ram_required_mib": 4000, "required_mib": 1000,
                  "gpus": [{"index": 0, "free_mib": 10000}]}
        p = {"host": {"reserve_mib": 1024}, "servers": [server]}
        for gpus in ([], [base.GPU(0, "GPU", 12000, 8000)], [base.GPU(0, "GPU", 12000, 100)]):
            with mock.patch.object(pathlib.Path, "read_text", return_value="VmRSS: 1024 kB\n"), \
                 self.assertRaises(ValueError):
                validate.check_memory(p, {"test": (1, 2)}, host=host, gpus=gpus)

    def test_resident_ram_envelope(self):
        host = resources.Host(64000, None, 64000, (0,), 1, ())
        server = {"id": "cpu", "ram_required_mib": 1000, "required_mib": 0, "gpus": []}
        with mock.patch.object(pathlib.Path, "read_text", return_value="VmRSS: 4096000 kB\n"), \
             self.assertRaisesRegex(ValueError, "resident RAM"):
            validate.check_memory({"host": {"reserve_mib": 1024}, "servers": [server]},
                                  {"cpu": (1, 2)}, host=host, gpus=[])


if __name__ == "__main__":
    unittest.main()
