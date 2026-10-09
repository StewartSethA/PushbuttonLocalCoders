import importlib.machinery
import importlib.util
import contextlib
import pathlib
import json
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

    def test_suffix_options_persist_across_model_switches(self):
        with patch.object(sys.stdin, "isatty", return_value=False):
            cfg = runtime.configure({}, self.args(model="qwen3.8:27b@slots=3,context=32768", context=None))
            cfg = runtime.configure(cfg, self.args(model="qwen3.6:35b", context=None))
            cfg = runtime.configure(cfg, self.args(context=None))
        req = runtime.worker_plan.parse_model_spec(cfg["model_spec"], cfg["models"])
        self.assertEqual(req.capacity["slots"], 3)
        self.assertEqual(req.capacity["context"], 32768)

    def test_cli_can_update_saved_suffix_setting(self):
        with patch.object(sys.stdin, "isatty", return_value=False):
            cfg = runtime.configure({}, self.args(model="qwen3.8:27b@slots=3", context=None))
            cfg = runtime.configure(cfg, self.args(model=None, context=None, slots="2"))
        req = runtime.worker_plan.parse_model_spec(cfg["model_spec"], cfg["models"])
        self.assertEqual(req.capacity["slots"], 2)

    def test_preview_does_not_write_configuration(self):
        with tempfile.TemporaryDirectory() as td, patch.object(runtime, "CONFIG", pathlib.Path(td) / "runtime.json"):
            with patch.object(sys.stdin, "isatty", return_value=False):
                runtime.configure({}, self.args())
            self.assertFalse(runtime.CONFIG.exists())

    def test_invalid_slot_count_is_rejected(self):
        with patch.object(sys.stdin, "isatty", return_value=False):
            with self.assertRaises(ValueError):
                runtime.configure({}, self.args(slots="0"))

    def test_options_after_model_are_not_passed_to_frontend(self):
        argv = ["pushbutton", "qwen3.8:27b", "--slots", "2", "--context", "32768",
                "--plan-only", "--", "--resume"]
        req = runtime.worker_plan.parse_model_spec("qwen3.8:27b@slots=2,context=32768")
        with patch.object(sys, "argv", argv), patch.object(sys.stdin, "isatty", return_value=False), patch.object(runtime, "read_json", return_value={}), patch.object(runtime, "capacity_candidate", return_value=({"workers": []}, {})) as candidate, patch("builtins.print"):
            runtime.main()
        self.assertEqual(candidate.call_args.args[0].capacity, req.capacity)
        self.assertEqual(candidate.call_args.args[1], 32768)

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

    def test_actual_vram_lease_rejects_overallocation(self):
        with patch.object(runtime.subprocess, "check_output", side_effect=["0, GPU-A\n1, GPU-B\n", "123, GPU-A, 17000\n123, GPU-B, 15000\n"]):
            self.assertFalse(runtime.check_vram_lease(123, "0,1", 16384))

    def test_default_startup_preserves_measured_backend_selection(self):
        cfg = {"model": "qwen3.8:27b", "model_spec": "qwen3.8:27b", "models": {},
               "context": 65536, "frontend": "none"}
        row = {"backend": "vllm-qwen38-3090", "artifact": "native", "gpu": 0,
               "req": 18000, "tg": 70, "quality": 95, "evidence": "LOCAL MEASURED"}
        with patch.object(sys, "argv", ["pushbutton"]), patch.object(runtime, "configure", return_value=cfg), patch.object(runtime, "healthy_resident", return_value=None), patch.object(runtime, "selector_views", return_value=[row]), patch.object(runtime, "commission", return_value={}) as commission, patch.object(runtime, "ensure_broker", return_value=19780), patch.object(runtime.registry_store, "named_lock", return_value=contextlib.nullcontext()), patch.object(runtime, "capacity_candidate") as planned, patch("builtins.print"):
            runtime.main()
        self.assertEqual(commission.call_args.args[2]["backend"], "vllm-qwen38-3090")
        planned.assert_not_called()

    def test_default_preview_uses_same_measured_backend_selection(self):
        cfg = {"model": "qwen3.8:27b", "model_spec": "qwen3.8:27b", "models": {},
               "context": 65536, "frontend": "none"}
        row = {"backend": "vllm-qwen38-3090", "artifact": "native", "gpu": 0,
               "req": 18000, "tg": 70, "quality": 95, "evidence": "LOCAL MEASURED"}
        with patch.object(sys, "argv", ["pushbutton", "--plan-only"]), patch.object(runtime, "configure", return_value=cfg), patch.object(runtime, "selector_views", return_value=[row]), patch.object(runtime, "commission") as commission, patch("builtins.print") as printed:
            runtime.main()
        preview = json.loads(printed.call_args.args[0])
        self.assertEqual(preview["selected"]["backend"], "vllm-qwen38-3090")
        commission.assert_not_called()

    def test_qwen_receives_supported_output_and_compaction_settings(self):
        with tempfile.TemporaryDirectory() as td, patch.object(runtime, "STATE", pathlib.Path(td)), patch.object(runtime, "ensure_qwen", return_value="/tmp/qwen"), patch.object(runtime.os, "execvpe"), patch.object(runtime, "write_json") as write:
            runtime.launch_qwen(19780, 32768, [], "model", 4096, 22000)
        settings = write.call_args.args[1]
        config = settings["modelProviders"]["openai"][0]["generationConfig"]
        self.assertEqual(config["samplingParams"]["max_tokens"], 4096)
        self.assertAlmostEqual(settings["context"]["autoCompactThreshold"], 22000 / 32768)

    def test_opencode_receives_input_limit_and_compaction_reserve(self):
        with tempfile.TemporaryDirectory() as td, patch.object(runtime, "STATE", pathlib.Path(td)), patch.object(runtime, "ensure_opencode", return_value="/tmp/opencode"), patch.object(runtime.os, "execvpe"), patch.object(runtime, "write_json") as write:
            runtime.launch_opencode(19780, 32768, [], "model", 4096, 27648, 22000)
        settings = write.call_args.args[1]
        self.assertEqual(settings["provider"]["pushbutton"]["models"]["model"]["limit"]["input"], 27648)
        self.assertEqual(settings["compaction"]["reserved"], 5648)

    def test_opencode_wrapper_preserves_distinct_model_budgets(self):
        source = (ROOT / "opencode-local").read_text()
        python = source.split('python3 - "$QWEN_HOME/settings.json" "$config" <<\'PY\'\n', 1)[1].split("\nPY", 1)[0]
        models = [{"id": name, "name": name, "baseUrl": "http://127.0.0.1:1234/v1",
                   "generationConfig": {"contextWindowSize": context, "samplingParams": {"max_tokens": output}}}
                  for name, context, output in [("small", 32768, 4096), ("large", 200000, 8192)]]
        routes = {"small": {"capacity": {"input_tokens": 27648, "compact_trigger": 22000}},
                  "large": {"capacity": {"input_tokens": 170000, "compact_trigger": 130000}}}
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td)
            (path / "qwen.json").write_text(json.dumps({"modelProviders": {"openai": models}}))
            (path / "capacity.json").write_text(json.dumps({"routes": routes}))
            with patch.object(sys, "argv", ["wrapper", str(path / "qwen.json"), str(path / "opencode.json")]), patch.dict(runtime.os.environ, {"PUSHBUTTON_CAPACITY_CONFIG": str(path / "capacity.json")}):
                exec(compile(python, str(ROOT / "opencode-local"), "exec"), {})
            config = json.loads((path / "opencode.json").read_text())
        reserve = config["compaction"]["reserved"]
        self.assertEqual(reserve, 5648)
        for i, name in enumerate(["small", "large"], 1):
            limits = config["provider"][f"pushbutton{i}"]["models"][name]["limit"]
            self.assertEqual(limits["input"] - reserve, routes[name]["capacity"]["compact_trigger"])
            self.assertLessEqual(limits["input"], routes[name]["capacity"]["input_tokens"])
        self.assertEqual(config["provider"]["pushbutton1"]["models"]["small"]["limit"]["output"], 4096)

    def test_commission_rejects_unproved_requested_capacity(self):
        row = {"backend": "llama.cpp", "artifact": "IQ3_XXS", "gpu": "0",
               "capacity": runtime.capacity.resolve_options({"slots": 2}, 32768),
               "kv_k": "q4_0", "kv_v": "q4_0"}
        with tempfile.TemporaryDirectory() as td, patch.object(runtime, "STATE", pathlib.Path(td)), patch.object(runtime, "free_port", return_value=20880), patch.object(runtime, "hf_spec", return_value="repo/model:IQ3_XXS"), patch.object(runtime, "start_detached") as start, patch.object(runtime, "wait_endpoint", return_value=True), patch.object(runtime, "served_model", return_value="model"), patch.object(runtime.request_budget, "verify_backend_capacity", side_effect=runtime.request_budget.BudgetError("smaller per-slot context")), patch.object(runtime.registry_store, "append_instance") as append:
            with self.assertRaises(SystemExit):
                runtime.commission("qwen3.8:27b", 32768, row)
            start.return_value.terminate.assert_called_once()
            append.assert_not_called()
        self.assertEqual(row["capacity"]["slots"], 2)


if __name__ == "__main__":
    unittest.main()
