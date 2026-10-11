import importlib.machinery
import importlib.util
import contextlib
import pathlib
import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
loader = importlib.machinery.SourceFileLoader("unified_capacity", str(ROOT / "pushbutton"))
spec = importlib.util.spec_from_loader(loader.name, loader)
runtime = importlib.util.module_from_spec(spec)
loader.exec_module(runtime)


class UnifiedCapacityTests(unittest.TestCase):
    def test_short_native_custom_runtime_defaults_and_explicit_context_validation(self):
        source = "TestOrg/ShortNative-GGUF:Q4_K_M"
        with self.custom_source(source), patch.dict(runtime.plan.PROFILES, {source: (
                replace(runtime.plan.PROFILES[source][0], native_context=32768),)}), patch.object(sys.stdin, "isatty", return_value=False):
            cfg = runtime.configure({}, self.args(model=source, context=None))
            self.assertEqual(cfg["context"], 32768)
            for options in ({"context": 65536}, {"client_context": "65536"}):
                with self.assertRaises((ValueError, SystemExit)):
                    runtime.configure({}, self.args(model=source, **options))

    def test_small_model_alias_registers_before_runtime_configuration(self):
        preset = "qwen3.5:9b"
        with patch.dict(runtime.plan.ALIASES, {"small-test": preset, preset: preset}), patch.object(runtime.plan, "canonical_model", return_value=preset) as canonical:
            self.assertEqual(runtime.canon("SMALL-TEST"), preset)
        canonical.assert_called_once_with(preset)

    def test_canon_registers_custom_profile_from_metadata_without_lowercasing(self):
        import gguf_source
        source = "ExampleOrg/CaseSensitive-GGUF:Q4_K_M"
        metadata = {"repo": "ExampleOrg/CaseSensitive-GGUF", "quant": "Q4_K_M",
                    "context": 131072, "download_spec": source, "weight_bytes": 2**30,
                    "attn_layers": 32, "kv_dim": 1024}
        with patch.dict(runtime.plan.PROFILES), patch.object(gguf_source, "resolve_source", return_value=metadata) as resolve:
            self.assertEqual(runtime.canon(source), source)
            profile = runtime.plan.PROFILES[source][0]
            self.assertEqual(profile.hf_spec, source)
            self.assertEqual(profile.native_context, 131072)
            resolve.assert_called_once_with(source)

    @contextlib.contextmanager
    def custom_source(self, source):
        profile = replace(runtime.plan.PROFILES["qwen3.8:27b"][3], model=source, source_spec=source, required_mib=4096)
        original = runtime.plan.canonical_model
        def canonical(text):
            return text if text == source else original(text)
        module = types.SimpleNamespace(is_source=lambda text: text == source)
        with patch.dict(sys.modules, {"gguf_source": module}), patch.dict(runtime.plan.PROFILES, {source: (profile,)}), patch.object(runtime.plan, "canonical_model", side_effect=canonical):
            yield profile

    def test_custom_source_preserves_case_and_capacity_settings(self):
        source = "hf:ExampleOrg/CaseSensitive-GGUF/Model-Q4_K_M.gguf"
        with self.custom_source(source), patch.object(sys.stdin, "isatty", return_value=False):
            cfg = runtime.configure({}, self.args(model=source + "@slots=3,context=32768", slots="2"))
            req = runtime.worker_plan.parse_model_spec(cfg["model_spec"], cfg["models"])
            self.assertEqual(cfg["model"], source)
            self.assertEqual(req.model, source)
            self.assertEqual(req.capacity["slots"], 3)
            self.assertEqual(req.capacity["context"], 32768)
            self.assertEqual(runtime.hf_spec(source, runtime.plan.PROFILES[source][0].quant), source)
            with self.assertRaises(ValueError):
                runtime.configure({}, self.args(model=source + "@unknown=3"))

    def test_custom_commission_uses_safe_alias_and_never_native_backend(self):
        source = "https://example.com/CaseSensitive/Model-Q4_K_M.gguf"
        proc = types.SimpleNamespace(pid=123)
        with self.custom_source(source) as profile, tempfile.TemporaryDirectory(dir=ROOT) as td, patch.object(runtime, "STATE", pathlib.Path(td)), patch.object(runtime, "free_port", return_value=20880), patch.object(runtime, "start_detached", return_value=proc) as start, patch.object(runtime, "wait_endpoint", return_value=True), patch.object(runtime, "served_model", side_effect=lambda endpoint, alias: alias), patch.object(runtime.registry_store, "append_instance"):
            row = {"backend": "llama.cpp", "artifact": profile.quant, "gpu": None}
            instance = runtime.commission(source, 32768, row)
            cmd, log, _ = start.call_args.args
            alias = cmd[cmd.index("--alias") + 1]
            self.assertEqual(cmd[cmd.index("--hf-spec") + 1], source)
            self.assertRegex(alias, r"^gguf-[0-9a-f]{16}$")
            self.assertEqual(log.parent, pathlib.Path(td) / "logs")
            self.assertEqual(instance["model"], source)
            self.assertEqual(instance["served_model"], alias)
            self.assertIn(source, instance["aliases"])
            self.assertNotIn("/", instance["id"])
            with self.assertRaises(ValueError):
                runtime.commission(source, 32768, {**row, "backend": "vllm-qwen38-3090"})

    def test_standalone_bits_reaches_planner_before_or_after_model(self):
        for args in (["bits=3", "q38"], ["q38", "bits=3"], ["q38", "--bits", "3"]):
            with self.subTest(args=args), patch.object(sys, "argv", ["pushbutton", *args, "--plan-only"]), patch.object(sys.stdin, "isatty", return_value=False), patch.object(runtime, "read_json", return_value={}), patch.object(runtime, "capacity_candidate", return_value=({"workers": []}, {})) as candidate, patch("builtins.print"):
                runtime.main()
                self.assertEqual(candidate.call_args.args[0].capacity["bits"], 3)

    def test_resident_reuse_checks_actual_artifact_bitness(self):
        expected = runtime.capacity.resolve_options({"bits": 3}, 32768)
        instance = dict(model="qwen3.8:27b", healthy=True, max_context=32768,
                        endpoint="http://127.0.0.1:1234/v1", artifact="Q4_K_M",
                        capacity=expected)
        with patch.object(runtime, "registry", return_value={"instances": [instance]}), patch.object(runtime, "endpoint_ok", return_value=True):
            self.assertIsNone(runtime.healthy_resident("qwen3.8:27b", 32768, expected))
            instance["artifact"] = "IQ3_XXS"
            instance["capacity"] = {k: v for k, v in expected.items() if k != "bits"}
            self.assertIs(runtime.healthy_resident("qwen3.8:27b", 32768, expected), instance)

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

    def test_catalog_only_models_keep_context_cli_support(self):
        with patch.object(runtime, "catalog", return_value={"catalog-model": {"context": 131072}}), patch.object(sys.stdin, "isatty", return_value=False):
            cfg = runtime.configure({}, self.args(model="catalog-model", context=32768))
        self.assertEqual(cfg["context"], 32768)
        self.assertEqual(cfg["models"]["catalog-model"]["context"], 32768)

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

    def test_instance_downloads_extended_sources_before_launching_local_file(self):
        loader = importlib.machinery.SourceFileLoader("instance_sources", str(ROOT / "pushbutton-instance"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        instance = importlib.util.module_from_spec(spec)
        loader.exec_module(instance)
        sources = [
            "hf:ExampleOrg/Model-GGUF:Q4_K_M",
            "ExampleOrg/Model-GGUF/CaseSensitive-Q4_K_M.gguf",
            "ExampleOrg/Model-GGUF:CaseSensitive-Q4_K_M.gguf",
            "https://example.com/models/CaseSensitive-Q4_K_M.gguf",
        ]
        local = ROOT / ".git" / "pushbutton-test-fixtures" / "Model-Q4_K_M.gguf"
        for source in sources:
            argv = ["pushbutton-instance", "--hf-spec", source, "--alias", "model",
                    "--port", "1234", "--context", "4096", "--slots", "3"]
            with self.subTest(source=source), patch.object(instance.subprocess, "check_output", return_value=str(local) + "\n") as download, patch.dict(instance.os.environ, {"LLAMA_CACHE": str(local.parent)}), patch.object(sys, "argv", argv), patch.object(instance, "ensure_server", return_value=ROOT / "llama-server"), patch.object(instance, "cpu_mode", return_value=False), patch.object(instance.os, "execvpe") as execute:
                instance.main()
            download.assert_called_once()
            self.assertEqual(download.call_args.args[0][-2:], [source, str(local.parent)])
            self.assertEqual(download.call_args.args[0][2], 'source "$1" && download_model_fast "$2" "$3" model_path && printf "%s\\n" "$model_path"')
            command = execute.call_args.args[1]
            self.assertEqual(command[command.index("-m") + 1], str(local))
            self.assertNotIn("-hf", command)
            self.assertNotIn("--trust-remote-code", command)
            self.assertEqual(command[command.index("-c") + 1], "12288")
        self.assertEqual(instance.model_args("ExampleOrg/Model-GGUF:Q4_K_M", str(local.parent)),
                         ["-hf", "ExampleOrg/Model-GGUF:Q4_K_M"])
        with patch.object(instance.subprocess, "check_output", side_effect=runtime.subprocess.CalledProcessError(1, "download")):
            with self.assertRaises(runtime.subprocess.CalledProcessError):
                instance.model_args("https://example.com/model.gguf", str(local.parent))

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
