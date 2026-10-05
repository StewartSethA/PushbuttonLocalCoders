import json
import os
import pathlib
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

    def test_no_uncalibrated_cpu_or_hybrid_guess(self):
        for policy in ("allow-hybrid", "allow-cpu-only"):
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

    def test_exact_nonuniform_calibrated_hybrid(self):
        p = self.plan(gpus=self.gpus[:1], entries=[calibration(mode="hybrid", ngl=7)],
                      startup_policy="allow-hybrid")
        s = p["servers"][0]
        self.assertEqual(s["ngl"], 7)
        self.assertEqual(s["required_mib"], 4725)
        self.assertEqual(s["ram_required_mib"], 4563)

    def test_hybrid_requires_some_gpu(self):
        with self.assertRaises(ValueError):
            self.plan(gpus=[], entries=[calibration(mode="cpu")], startup_policy="allow-hybrid")

    def test_main_stays_gpu_secondary_moves_to_cpu(self):
        entries = [calibration("q36", "gpu"), calibration("q36", "cpu"),
                   calibration("q38", "gpu"), calibration("q38", "cpu")]
        p = self.plan(["q36", "q38"], entries, gpus=self.gpus[:1], startup_policy="allow-hybrid")
        by_model = {s["model"]: s for s in p["servers"]}
        self.assertEqual(by_model["qwen3.8:27b"]["mode"], "gpu")
        self.assertEqual(by_model["qwen3.6:35b"]["mode"], "cpu")
        self.assertEqual(sum(s["threads"] for s in p["servers"]), 8)

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
        p = self.plan(entries=[calibration(), calibration(mode="hybrid"), calibration(mode="cpu")],
                      startup_policy="allow-cpu-only", max_layouts=3, min_quality=100)
        self.assertEqual(len(p["alternatives"]), 2)
        self.assertEqual([layout["servers"][0]["mode"] for layout in [p, *p["alternatives"]]],
                         ["gpu", "hybrid", "cpu"])
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


class LauncherFallbackTests(unittest.TestCase):
    def source(self, start, end):
        source = (ROOT / "claude-local").read_text()
        return source[source.index(start):source.index(end)]

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

    def test_offload_log_enforces_gpu_hybrid_cpu_placement(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = pathlib.Path(directory)
            log = root / "main.log"
            for mode, ngl, report, accepted in (
                ("gpu", 999, "", False),
                ("gpu", 999, "offloaded 0/65 layers to GPU", False),
                ("gpu", 999, "offloaded 32/65 layers to GPU", False),
                ("gpu", 999, "offloaded 65/65 layers to GPU", True),
                ("hybrid", 32, "offloaded 31/65 layers to GPU", False),
                ("hybrid", 32, "offloaded 32/65 layers to GPU", True),
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
