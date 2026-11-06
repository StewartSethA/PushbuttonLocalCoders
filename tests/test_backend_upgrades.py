import importlib.machinery
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, filename):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


backend = load("backend_upgrades", "pushbutton-backend")
bench = load("bench_upgrades", "pushbutton-bench")


class BackendUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name)
        for name, value in (("STATE", self.path / "state"), ("CACHE", self.path / "cache"), ("DRY", True)):
            patch = mock.patch.object(backend, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def command(self, model, profile, context=4096, runtime="ninfer-3090"):
        with mock.patch.object(backend, "exec_cmd") as execute:
            backend.serve(runtime, model, [0], 20880, context, profile)
        return execute.call_args.args[0]

    def test_prepare_downloads_exact_model_from_pinned_source(self):
        for model, expected in backend.NINFER_MODELS.items():
            with self.subTest(model=model), mock.patch.object(backend, "ensure_repo", return_value=self.path) as repo, mock.patch.object(backend, "run") as run:
                backend.prepare("ninfer-3090", model, [0])
                self.assertEqual(repo.call_args.args[2], backend.registry()["backends"]["ninfer-3090"]["source_ref"])
                self.assertEqual(run.call_args.args[0], ["bash", "scripts/download-model.sh", expected[0]])
                self.assertEqual(run.call_args.kwargs["env"]["NINFER_GRAFTS"], "off")

    def test_selected_model_not_largest_cached_artifact(self):
        cache = backend.CACHE / "ninfer-3090"
        cache.mkdir(parents=True)
        (cache / backend.NINFER_MODELS["qwen3.8:27b"][1]).write_bytes(b"x")
        (cache / backend.NINFER_MODELS["qwen3.6:35b"][1]).write_bytes(b"x" * 100)
        with mock.patch.object(backend, "DRY", False), mock.patch.object(backend, "docker_prefix", return_value=["docker"]):
            cmd = self.command("qwen3.8:27b", "mtp")
        self.assertIn("models/qwen3_8_27b.ninfer", cmd)
        self.assertNotIn("models/qwen3_6_35b_a3b.ninfer", cmd)

    def test_missing_artifact_fails_instead_of_serving_another_model(self):
        with mock.patch.object(backend, "DRY", False), mock.patch.object(backend, "docker_prefix", return_value=["docker"]), self.assertRaisesRegex(SystemExit, "missing model artifact"):
            self.command("qwen3.6:35b", "mtp")

    def test_kv_and_speculation_profiles(self):
        for profile, kv in (("mtp", "int8"), ("mtp-k8v4", "k8v4"), ("mtp-rk8v4", "rk8v4")):
            cmd = self.command("qwen3.6:35b", profile)
            self.assertEqual(cmd[cmd.index("--kv-dtype") + 1], kv)
            self.assertEqual(cmd[cmd.index("--spec") + 1], "mtp")
            self.assertEqual(cmd[cmd.index("--max-context") + 1], "4096")
            self.assertEqual(cmd[cmd.index("--kv-capacity") + 1], "4096")
            self.assertIn("127.0.0.1:20880:8080", cmd)
            self.assertNotIn("--no-cuda-graph", cmd)
            self.assertNotIn("--no-prefix-reuse", cmd)
        self.assertNotIn("--spec", self.command("qwen3.6:35b", "standard"))
        cmd = self.command("qwen3.8:27b", "prompt-lookup")
        self.assertEqual(cmd[cmd.index("--ngram-draft-tokens") + 1], "15")
        self.assertEqual(cmd[cmd.index("--spec") + 1], "dflash2")

    def test_invalid_profiles_and_context_fail_before_prepare(self):
        for options in (["--profile", "unknown"], ["--profile", "prompt-lookup"], ["--context", "0"], ["--context", "131073"]):
            with self.subTest(options=options), mock.patch.object(sys, "argv", ["pushbutton-backend", "--dry-run", "serve", "ninfer-3090", "qwen36", *options]), mock.patch.object(backend, "prepare") as prepare, self.assertRaises(SystemExit):
                backend.main()
            prepare.assert_not_called()
        with self.assertRaisesRegex(SystemExit, "no selectable profile"):
            backend.resolve_profile("llama.cpp", "qwen3.8:27b", "mtp", 4096)
        self.assertEqual(backend.resolve_profile("ninfer-3090", "qwen3.6:35b", "mtp-k8v4", None)[1], 4096)
        self.assertEqual(backend.resolve_profile("ninfer-3090", "qwen3.8:27b", "mtp", 262144)[1], 262144)

    def test_hyperqwen_context_is_not_ignored(self):
        cmd = self.command("qwen3.8:27b", "mtp-long", 150000, "vllm-qwen38-3090")
        for expected in ("MAX_LEN=150000", "SPEC=mtp", "CTX=long", "PTH_FA=1", "INT8_ACT=int8", "INT8_LAYERS=mlp", "PREFIX_CACHE=1", "ghcr.io/syv-ai/hyperqwen:latest"):
            self.assertIn(expected, cmd)
        self.assertIn("127.0.0.1:20880:18020", cmd)
        with self.assertRaisesRegex(SystemExit, "invalid context"):
            backend.resolve_profile("vllm-qwen38-3090", "qwen3.8:27b", "dflash2", 150000)

    def test_gpu_unlock_status_never_modifies_host(self):
        with mock.patch.object(backend.subprocess, "run") as run, mock.patch.object(backend, "exec_cmd") as execute:
            status = backend.gpu_unlock_status()
        self.assertFalse(status["throughput_verified"])
        self.assertIsInstance(status["boot_targets"], list)
        run.assert_not_called()
        execute.assert_not_called()

    def test_benchmark_propagates_profile_and_records_operator_state(self):
        output = self.path / "results"
        eligibility = json.dumps({"context": 4096, "profile": "mtp-k8v4"})
        argv = ["pushbutton-bench", "qwen36", "--backend", "ninfer-3090", "--profile", "mtp-k8v4", "--gpu-unlock-state", "stock", "--suite", "quick", "--output-dir", str(output)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(bench, "hardware", return_value={}), mock.patch.object(bench, "backend_check", return_value=(True, eligibility)), mock.patch.object(bench.subprocess, "Popen") as popen, mock.patch.object(bench, "wait_ready"), mock.patch.object(bench, "discover_model", return_value="qwen36"), mock.patch.object(bench, "run_case", return_value={"name": "1k-c1"}), mock.patch.object(bench, "fingerprint", return_value={"id": "ninfer-3090", "profile": "mtp-k8v4"}) as fingerprint, mock.patch.object(bench.os, "killpg"), mock.patch.dict(bench.os.environ, {"PUSHBUTTON_BACKEND_STATE": str(self.path)}):
            self.assertEqual(bench.main(), 0)
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--profile") + 1], "mtp-k8v4")
        self.assertEqual(cmd[cmd.index("--context") + 1], "4096")
        fingerprint.assert_called_once_with("ninfer-3090", "qwen3.6:35b", None, "mtp-k8v4", 4096)
        report = json.loads(next(output.glob("*.json")).read_text())
        self.assertEqual(report["context"], 4096)
        self.assertEqual(report["gpu_unlock_state"], "stock")
        self.assertEqual(report["gpu_unlock_state_origin"], "operator-reported")


if __name__ == "__main__":
    unittest.main()
