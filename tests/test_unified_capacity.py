import importlib.machinery
import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
loader = importlib.machinery.SourceFileLoader("unified_capacity", str(ROOT / "pushbutton"))
spec = importlib.util.spec_from_loader(loader.name, loader)
runtime = importlib.util.module_from_spec(spec)
loader.exec_module(runtime)


class UnifiedCapacityTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(model="qwen3.8:27b", context=65536, frontend="none",
                      telemetry_on=False, telemetry_off=False, telemetry_url=None,
                      slots=None, output=None, client_context=None, compact=None,
                      quant=None, kv_k=None, kv_v=None, min_tps=None, admission=None,
                      plan_only=True)
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_model_suffix_overrides_global_fallback(self):
        with patch.object(sys.stdin, "isatty", return_value=False):
            cfg = runtime.configure({}, self.args(model="qwen3.8:27b@slots=3,context=32768", slots="2"))
        req = runtime.worker_plan.parse_model_spec(cfg["model_spec"], cfg["models"])
        resolved = runtime.capacity.resolve_options(req.capacity, cfg["context"])
        self.assertEqual(resolved["slots"], 3)
        self.assertEqual(resolved["context"], 32768)

    def test_preview_does_not_write_configuration(self):
        with tempfile.TemporaryDirectory() as td, patch.object(runtime, "CONFIG", pathlib.Path(td) / "runtime.json"):
            with patch.object(sys.stdin, "isatty", return_value=False):
                runtime.configure({}, self.args())
            self.assertFalse(runtime.CONFIG.exists())

    def test_invalid_slot_count_is_rejected(self):
        with patch.object(sys.stdin, "isatty", return_value=False):
            with self.assertRaises(ValueError):
                runtime.configure({}, self.args(slots="0"))

    def test_pinned_quant_and_cache_reach_launch_command(self):
        req = runtime.worker_plan.parse_model_spec("qwen3.8:27b@slots=2,context=32768,kv_k=q8_0")
        gpus = runtime.worker_plan.synthetic_3090(1)
        with patch.object(runtime.plan, "inventory", return_value=gpus):
            proposed, row = runtime.capacity_candidate(req, 32768)
        self.assertEqual(row["capacity"]["slots"], 2)
        self.assertEqual(row["kv_k"], "q8_0")
        self.assertEqual(row["req"], proposed["workers"][0]["required_mib"])

    def test_existing_instance_requires_matching_budget(self):
        expected = runtime.capacity.resolve_options({"slots": 2}, 32768)
        instance = dict(id="i", model="qwen3.8:27b", aliases=[], healthy=True,
                        endpoint="http://127.0.0.1:1234/v1", max_context=32768,
                        framework_max_concurrency=2, capacity=expected)
        with patch.object(runtime, "registry", return_value={"instances": [instance]}), patch.object(runtime, "endpoint_ok", return_value=True):
            self.assertIs(runtime.healthy_resident("qwen3.8:27b", 32768, expected), instance)
            different = {**expected, "output_tokens": expected["output_tokens"] + 1}
            self.assertIsNone(runtime.healthy_resident("qwen3.8:27b", 32768, different))

    def test_instance_disables_context_shift_and_preserves_slot_context(self):
        instance_loader = importlib.machinery.SourceFileLoader("instance_capacity", str(ROOT / "pushbutton-instance"))
        instance_spec = importlib.util.spec_from_loader(instance_loader.name, instance_loader)
        instance = importlib.util.module_from_spec(instance_spec)
        instance_loader.exec_module(instance)
        argv = ["pushbutton-instance", "--hf-spec", "repo/model:Q4", "--alias", "model",
                "--port", "1234", "--context", "4096", "--slots", "3"]
        with patch.object(sys, "argv", argv), patch.object(instance, "ensure_server", return_value=pathlib.Path("/tmp/llama-server")), patch.object(instance.os, "execvpe") as execute:
            instance.main()
        command = execute.call_args.args[1]
        self.assertIn("--no-context-shift", command)
        self.assertEqual(command[command.index("-c") + 1], "12288")
        self.assertEqual(command[command.index("--parallel") + 1], "3")


if __name__ == "__main__":
    unittest.main()
