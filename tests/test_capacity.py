import math
import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
import pushbutton_capacity as capacity
import claude_local_plan as base
import coder_local_plan as coder


class CapacityBudgetTests(unittest.TestCase):
    def test_memory_policy_validation(self):
        self.assertEqual(capacity.parse_options({"margin": "0", "force": "TRUE"}),
                         {"margin": 0, "force": True})
        self.assertEqual(capacity.parse_options({"margin": 512, "force": False}),
                         {"margin": 512, "force": False})
        for value in (-1, True, 1.5, "nan", "inf", "1G", None):
            with self.subTest(margin=value), self.assertRaisesRegex(ValueError, "margin"):
                capacity.parse_options({"margin": value})
        for value in (1, 0, "yes", "", None):
            with self.subTest(force=value), self.assertRaisesRegex(ValueError, "force"):
                capacity.parse_options({"force": value})

    def test_low_margin_allows_reported_qwen_configuration(self):
        spec = "qwen3.6:35b@slots=2,bits=2,context=260000"
        gpus = [base.GPU(0, "test GPU", 16384, 15564)]
        with self.assertRaisesRegex(ValueError, "margin=N"):
            base.plan([spec], gpus, 262144)
        for margin in (0, 1024):
            with self.subTest(margin=margin):
                server = base.plan([spec + f",margin={margin}"], gpus, 262144)["servers"][0]
                estimate = server["capacity"]["memory_estimate"]
                self.assertEqual(server["capacity"]["context"], 260000)
                self.assertEqual(server["capacity"]["slots"], 2)
                self.assertEqual(estimate["overhead_mib"], margin)
                self.assertLessEqual(server["required_mib"], gpus[0].free_mib)
                self.assertGreaterEqual(server["headroom_mib"], 0)

    def test_force_keeps_honest_estimates_and_uses_smallest_fallback(self):
        spec = "qwen3.6:35b@slots=2,bits=2,context=260000"
        gpus = [base.GPU(0, "test GPU", 16384, 15564)]
        plans = [
            (base.plan([spec + ",force=true"], gpus, 262144), "servers"),
            (coder.build_plan([coder.parse_model_spec(spec + ",force=true")],
                              None, gpus, 262144), "workers"),
        ]
        for result, key in plans:
            with self.subTest(planner=key):
                server = result[key][0]
                self.assertEqual(server["profile"]["quant"], "UD-IQ2_M")
                self.assertGreater(server["required_mib"], server["allocated_mib"])
                self.assertLess(server["headroom_mib"], 0)
                self.assertEqual(server["required_mib"], server["capacity"]["memory_estimate"]["required_mib"])
        with self.assertRaises(ValueError):
            base.plan([spec + ",force=false"], gpus, 262144)

    def test_force_prefers_fitting_quant_and_respects_constraints(self):
        spec = "qwen3.8:27b"
        gpus = coder.synthetic_3090(1)
        ordinary = base.plan([spec], gpus, 32768)["servers"][0]
        forced = base.plan([spec + "@force=true"], gpus, 32768)["servers"][0]
        self.assertEqual(ordinary["profile"]["quant"], forced["profile"]["quant"])
        for specs in ([spec + "@force=true,context=9999999"],
                      [spec + "@force=true,gpu=99"],
                      [spec + "@force=true", spec + "@force=true"]):
            with self.subTest(specs=specs), self.assertRaises(ValueError):
                base.plan(specs, gpus, 32768)
        capped = base.plan([spec + "@force=true,vram=1G"], gpus, 32768)["servers"][0]
        self.assertEqual(capped["allocated_mib"], 1024)
        self.assertLess(capped["headroom_mib"], 0)

    def test_margin_is_total_allowance_for_catalog_and_custom_profiles(self):
        from dataclasses import replace
        profiles = [
            base.PROFILES["ornith-1.5:35b-a3b"][0],
            replace(base.PROFILES["qwen3.8:27b"][0],
                    weight_mib=2048, attention_layers=0, kv_width=0),
        ]
        for profile in profiles:
            with self.subTest(model=profile.model):
                estimates = [capacity.memory_estimate(profile, capacity.resolve_options(
                    {"margin": margin, "slots": 2}, 32768)) for margin in (0, 256)]
                self.assertEqual(estimates[0]["overhead_mib"], 0)
                self.assertEqual(estimates[1]["required_mib"] - estimates[0]["required_mib"], 256)

    def test_force_and_margin_work_without_architecture_metadata(self):
        from dataclasses import replace
        model = "test-unknown-architecture"
        profile = replace(base.PROFILES["qwen3.8:27b"][0], model=model)
        with patch.dict(base.PROFILES, {model: (profile,)}):
            ordinary = capacity.memory_estimate(profile, capacity.resolve_options({}, 32768))
            custom = capacity.memory_estimate(profile, capacity.resolve_options({"margin": 256}, 32768))
            self.assertEqual(custom["required_mib"], ordinary["required_mib"] + 256)
            self.assertIsNone(base.best_profile_for_capacity(model, 1, 32768))
            self.assertIsNotNone(base.best_profile_for_capacity(model, 1, 32768, {"force": True}))

    def test_model_policy_defaults_and_cli_override(self):
        model = "qwen3.6:35b"
        defaults = {model: {"margin": 512, "force": True}}
        req = coder.parse_model_spec(model, defaults)
        self.assertEqual(req.capacity, {"margin": 512, "force": True})
        req = coder.parse_model_spec(model + "@margin=0,force=false", defaults)
        self.assertEqual(req.capacity, {"margin": 0, "force": False})

    def test_six_small_presets_resolve_metadata_without_guessed_sizes(self):
        self.assertEqual(len(base.SMALL_MODELS), 6)
        for model, preset in base.SMALL_MODELS.items():
            metadata = dict(repo=preset["repo"], quant="Q4_K_M", context=131072,
                            download_spec=preset["repo"] + ":Q4_K_M",
                            weight_bytes=4 * 2**30, attn_layers=8, kv_dim=512)
            with self.subTest(model=model), patch("gguf_source.resolve_source", return_value=metadata) as resolve:
                try:
                    request = coder.parse_model_spec(model + "@slots=2,bits=4")
                    self.assertEqual(request.model, model)
                    resolve.assert_called_once_with(preset["repo"] + ":Q4_K_M")
                    profile = base.PROFILES[model][0]
                    self.assertEqual(profile.weight_mib, 4096)
                    self.assertEqual(profile.template, "embedded")
                    self.assertEqual(coder.resolve_request_options(request, 262144)["context"], 131072)
                finally:
                    base.PROFILES.pop(model, None)

    def test_source_kv_distinguishes_key_value_width_and_precision(self):
        from dataclasses import replace
        profile = replace(base.PROFILES["qwen3.8:27b"][3],
                          weight_mib=2048, attention_layers=4, kv_width=384,
                          kv_key_width=256, kv_value_width=512)
        settings = capacity.resolve_options(
            {"context": 32768, "kv_k": "q4_0", "kv_v": "f16"}, 262144)
        estimate = capacity.memory_estimate(profile, settings)
        expected = 4 * (256 * capacity.KV_TYPES["q4_0"] + 512 * capacity.KV_TYPES["f16"])
        self.assertEqual(estimate["kv_bytes_per_token"], expected)

    def test_recurrent_source_keeps_per_slot_runtime_budget(self):
        from dataclasses import replace
        profile = replace(base.PROFILES["qwen3.8:27b"][3],
                          weight_mib=2048, attention_layers=0, kv_width=0)
        one = capacity.memory_estimate(profile, capacity.resolve_options({}, 32768))
        four = capacity.memory_estimate(profile, capacity.resolve_options({"slots": 4}, 32768))
        self.assertEqual(four["kv_mib"], 0)
        self.assertEqual(four["required_mib"] - one["required_mib"], 3 * capacity.GPU_RUNTIME_OVERHEAD_MIB)

    def test_custom_source_registers_actual_weights_and_safe_instance_ids(self):
        source = "hf:Example/Small-GGUF:Q4_K_M"
        metadata = dict(repo="Example/Small-GGUF", quant="Q4_K_M", context=32768,
                        download_spec=source, weight_bytes=3 * 2**30,
                        attn_layers=24, kv_dim=256)
        with patch("gguf_source.resolve_source", return_value=metadata):
            try:
                request = coder.parse_model_spec(source + "@gpu=0,slots=2")
                result = base.plan([source + "@gpu=0,slots=2"], coder.synthetic_3090(1), 262144)
                server = result["servers"][0]
                self.assertEqual(server["profile"]["hf_spec"], source)
                self.assertEqual(server["capacity"]["context"], 32768)
                self.assertRegex(server["id"], r"^local-gguf-[a-f0-9]{16}$")
                estimate = server["capacity"]["memory_estimate"]
                self.assertEqual(estimate["weights_mib"], 3072)
                self.assertEqual(estimate["weights_source"], "gguf-metadata")
                self.assertEqual(estimate["kv_mib"], 432)
                with self.assertRaisesRegex(ValueError, "native context"):
                    coder.candidates_for_request(
                        coder.parse_model_spec(source + "@context=65536"),
                        coder.synthetic_3090(1), 262144)
            finally:
                base.PROFILES.pop(source, None)

    def test_bits_validation_and_quant_names(self):
        for bits in (1, 2, 3, 4, 5, 6, 8, 16):
            self.assertEqual(capacity.parse_options({"bits": str(bits)})["bits"], bits)
        for value in (0, -1, True, 4.5, "4K", "four", "", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "bits"):
                capacity.parse_options({"bits": value})
        for quant, bits in (("Q6_K", 6), ("UD-IQ3_XXS", 3), ("UD-Q2_K_XL", 2),
                            ("IQ4_XS", 4), ("MXFP4_MOE", 4), ("native", None)):
            self.assertEqual(capacity.quant_bits(quant), bits)

    def test_standalone_bits_fallback_and_model_override(self):
        self.assertEqual(
            capacity.model_bits_specs(["bits=4", "q38@gpu=0", "q36@bits=3"]),
            ["q38@gpu=0,bits=4", "q36@bits=3"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            capacity.model_bits_specs(["q38", "bits=4", "bits=3"])

    def test_normalized_suffix_settings(self):
        options = capacity.parse_options({
            "slots": "2", "context": "64K", "output": "8K",
            "client_context": "60K", "compact": "80%", "quant": "iq3_xxs",
            "kv_k": "Q8_0", "kv_v": "f16", "min_tps": "12.5", "admission": "1",
        })
        self.assertEqual(options["context"], 65536)
        self.assertEqual(options["quant"], "IQ3_XXS")
        self.assertEqual(options["compact"], 0.8)
        self.assertEqual(options["admission_limit"], 1)
        self.assertEqual(options["min_tps"], 12.5)

    def test_budget_reserves_output_and_safety(self):
        budget = capacity.resolve_budget(32768, 4096, 30000, 0.75, 1000)
        self.assertEqual(budget, {
            "context": 32768, "output_tokens": 4096, "client_context": 30000,
            "compact_trigger": 18678, "input_tokens": 24904, "safety_tokens": 1000,
        })

    def test_default_budget_and_token_compact(self):
        budget = capacity.resolve_budget(262144)
        self.assertEqual(budget["output_tokens"], 32768)
        self.assertEqual(budget["safety_tokens"], 1024)
        self.assertEqual(budget["input_tokens"], 228352)
        self.assertEqual(capacity.resolve_budget(8192, compact="2K")["compact_trigger"], 2048)

    def test_compaction_trigger_stays_below_client_and_hard_context(self):
        for context, slots, compact in (
            (262144, 1, None), (196608, 2, "80%"), (131072, 2, 100000),
            (65536, 1, "0.5"),
        ):
            with self.subTest(context=context, slots=slots, compact=compact):
                options = {"context": context, "slots": slots}
                if compact is not None:
                    options["compact"] = compact
                settings = capacity.resolve_options(options, context)
                self.assertLess(settings["compact_trigger"], settings["input_tokens"])
                self.assertLess(settings["compact_trigger"], settings["client_context"])
                self.assertLess(settings["compact_trigger"], settings["context"])
                self.assertEqual(settings["context"] * settings["slots"],
                                 context * slots)

    def test_admission_defaults_to_one_without_reducing_requested_slots(self):
        settings = capacity.resolve_options({"slots": 4}, 32768)
        self.assertEqual(settings["slots"], 4)
        self.assertEqual(settings["admission_limit"], 1)
        self.assertFalse(settings["admission_explicit"])
        explicit = capacity.resolve_options({"slots": 4, "admission": 3}, 32768)
        self.assertEqual(explicit["admission_limit"], 3)
        self.assertTrue(explicit["admission_explicit"])
        explicit_one = capacity.resolve_options({"slots": 4, "admission": 1}, 32768)
        self.assertEqual(explicit_one["admission_limit"], 1)
        self.assertTrue(explicit_one["admission_explicit"])

    def test_rejects_invalid_scalar_options(self):
        cases = [
            {"slots": 0}, {"slots": True}, {"slots": 1.5}, {"slots": 129},
            {"context": "-1"}, {"output": "unlimited"}, {"client_context": None},
            {"admission": -1}, {"safety": 0}, {"min_tps": "nan"},
            {"min_tps": "inf"}, {"min_tps": True}, {"min_tps": 0},
            {"compact": "0%"}, {"compact": "100%"}, {"compact": 1.5},
            {"compact": "nan"}, {"kv_k": "q2_k"}, {"kv_v": ""},
            {"quant": "bad/value"}, {"quant": None}, {"quant": 123}, {"bogus": 1},
            {"admission_explicit": False},
            {"admission": 1, "admission_limit": 1},
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                capacity.parse_options(options)

    def test_rejects_inconsistent_budget_and_admission(self):
        for kwargs in [
            {"client_context": 8193}, {"output": 8192},
            {"output": 4096, "client_context": 4096},
            {"compact": 8192}, {"compact": 6912}, {"safety": 8192},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                capacity.resolve_budget(8192, **kwargs)
        with self.assertRaisesRegex(ValueError, "admission"):
            capacity.resolve_options({"admission": 3, "slots": 2}, 8192)
        with self.assertRaisesRegex(ValueError, "128"):
            capacity.resolve_options({"slots": 1}, 8192, slots=129)
        self.assertEqual(capacity.parse_options({"slots": 128})["slots"], 128)

    def test_memory_estimate_accounts_for_slots_and_precision(self):
        profile = base.PROFILES["qwen3.8:27b"][-3]
        one = capacity.memory_estimate(profile, capacity.resolve_options({}, 262144))
        two = capacity.memory_estimate(profile, capacity.resolve_options({"slots": 2}, 262144))
        high_precision = capacity.memory_estimate(
            profile, capacity.resolve_options({"kv_k": "f16", "kv_v": "q8_0"}, 262144))
        self.assertEqual(one["weights_source"], "parameters-effective-bpw")
        self.assertGreater(two["required_mib"], one["required_mib"])
        self.assertGreater(high_precision["required_mib"], two["required_mib"])
        self.assertEqual(two["context_tokens_total"], 524288)
        self.assertEqual(two["status"], "ESTIMATED")
        self.assertFalse(two["architecture_exact"])

    def test_memory_estimate_uses_real_weights_and_architecture_kv(self):
        q4 = next(p for p in base.PROFILES["ornith-1.5:35b-a3b"] if p.quant == "Q4_K_M")
        one = capacity.memory_estimate(q4, capacity.resolve_options({}, 262144))
        two = capacity.memory_estimate(q4, capacity.resolve_options({"slots": 2}, 262144))
        self.assertEqual(one["weights_source"], "catalog")
        self.assertEqual(one["weights_mib"], math.ceil(21.7 * 1024))
        self.assertLess(one["required_mib"], q4.required_mib)
        # KV grows with total tokens only; weights/runtime are paid once.
        self.assertAlmostEqual(two["kv_mib"], 2 * one["kv_mib"], delta=1)
        self.assertAlmostEqual(two["required_mib"] - one["required_mib"], one["kv_mib"], delta=2)
        f16 = capacity.memory_estimate(q4, capacity.resolve_options({"kv_k": "f16", "kv_v": "f16"}, 262144))
        q8 = capacity.memory_estimate(q4, capacity.resolve_options({"kv_k": "q8_0", "kv_v": "q8_0"}, 262144))
        self.assertGreater(f16["required_mib"], q8["required_mib"])
        self.assertGreater(q8["required_mib"], one["required_mib"])
        q38 = base.PROFILES["qwen3.8:27b"][0]
        lowered = capacity.memory_estimate(q38, capacity.resolve_options({"context": 65536}, 262144))
        self.assertEqual(lowered["weights_source"], "parameters-effective-bpw")
        self.assertLess(lowered["required_mib"], q38.required_mib)

    def test_uncatalogued_weights_use_effective_bits_not_envelope(self):
        import cpu_platform
        profile = base.PROFILES["qwen3.8:27b"][3]
        settings = capacity.resolve_options({"slots": 4}, 262144)
        estimate = capacity.memory_estimate(profile, settings)
        expected = 27e9 * 4.85 / 8 / 2**20
        self.assertEqual(estimate["weights_mib"], math.ceil(expected))
        self.assertEqual(estimate["kv_mib"], 18 * 1024)
        self.assertEqual(estimate["required_mib"],
                         math.ceil(expected * capacity.WEIGHT_MARGIN + 18 * 1024 + 1536))
        with patch.dict(cpu_platform.MODEL_ARCH, {
                profile.model: {**cpu_platform.MODEL_ARCH[profile.model], "active_b": 1}}):
            self.assertEqual(capacity.memory_estimate(profile, settings), estimate)
        from dataclasses import replace
        self.assertEqual(capacity.memory_estimate(
            replace(profile, required_mib=100 * 1024), settings), estimate)

    def test_two_bit_weight_ballpark_uses_decimal_parameters_and_binary_units(self):
        import cpu_platform
        profile = next(p for p in base.PROFILES["qwen3.6:35b"] if p.quant == "UD-IQ2_M")
        settings = capacity.resolve_options({"context": 260000, "slots": 2}, 262144)
        for params, gb in ((16, 4.0), (35, 8.75)):
            with self.subTest(params=params), patch.dict(cpu_platform.MODEL_ARCH, {
                    profile.model: {**cpu_platform.MODEL_ARCH[profile.model], "total_b": params}}):
                estimate = capacity.memory_estimate(profile, settings)
                self.assertAlmostEqual(cpu_platform.nominal_weight_gib(profile.model, profile.quant),
                                       gb * 1e9 / 2**30)
                self.assertAlmostEqual(estimate["nominal_weights_gb"], gb)
                self.assertEqual(estimate["weights_mib"], math.ceil(params * 1e9 * 2.7 / 8 / 2**20))
                self.assertEqual(estimate["weights_source"], "parameters-effective-bpw")
                self.assertGreater(estimate["quantization_overhead_mib"], 0)
                self.assertGreater(estimate["required_mib"], estimate["weights_mib"])

    def test_weight_fallback_preserves_explicit_bpw_and_multi_digit_precision(self):
        import cpu_platform
        for quant, bits in (("BF16", 16), ("FP32", 32), ("UD-IQ2_M", 2),
                            ("UD-Q4_K_M", 4)):
            with self.subTest(quant=quant):
                self.assertAlmostEqual(cpu_platform.nominal_weight_gib("qwen3.6:35b", quant),
                                       35e9 * bits / 8 / 2**30)
        for quant, bits in (("BF16", 16), ("FP16", 16), ("F16", 16), ("FP32", 32)):
            with self.subTest(quant=quant):
                self.assertEqual(cpu_platform.quant_bpw(quant), bits)
                self.assertAlmostEqual(cpu_platform.weight_gib("qwen3.6:35b", quant),
                                       35e9 * bits / 8 / 2**30)
        mixed = "mixed-Q2_0-Q4_0-2.97BPW"
        self.assertIsNone(cpu_platform.nominal_weight_gib("qwen3.6:35b", mixed))
        self.assertIsNone(cpu_platform.nominal_weight_gib("unknown", "Q4_K_M"))
        self.assertAlmostEqual(cpu_platform.weight_gib("qwen3.6:35b", mixed),
                               35e9 * 2.97 / 8 / 2**30)

    def test_reported_two_slot_model_does_not_fit_from_nominal_bits_alone(self):
        gpu = base.GPU(0, "16 GB GPU", 16384, int(15.2 * 1024))
        spec = "qwen3.6:35b@slots=2,context=260000,quant=UD-IQ2_M"
        with self.assertRaises(ValueError) as error:
            base.plan([spec], [gpu], 262144)
        self.assertIn("nominal packed weights 8.75 GB (8.15 GiB", str(error.exception))
        self.assertIn("quantization/mixed-precision overhead 2.85 GiB", str(error.exception))
        self.assertIn("2.7 effective bits/weight", str(error.exception))

    def test_reported_model_uses_measured_weights_before_estimated_overhead(self):
        from dataclasses import replace
        profile = next(p for p in base.PROFILES["qwen3.6:35b"] if p.quant == "UD-IQ2_M")
        measured = replace(profile, weight_mib=10.73 * 1024)
        gpu = base.GPU(0, "16 GB GPU", 16384, int(15.4 * 1024))
        with patch.dict(base.PROFILES, {profile.model: (measured,)}):
            with self.assertRaisesRegex(ValueError, "weights source: gguf-metadata"):
                base.plan(
                    ["qwen3.6:35b@slots=2,context=260000,quant=UD-IQ2_M"],
                    [replace(gpu, free_mib=int(15.2 * 1024))], 262144)
            result = base.plan(
                ["qwen3.6:35b@slots=2,context=260000,quant=UD-IQ2_M"], [gpu], 262144)
        estimate = result["servers"][0]["capacity"]["memory_estimate"]
        self.assertEqual(estimate["weights_mib"], math.ceil(10.73 * 1024))
        self.assertEqual(estimate["weights_source"], "gguf-metadata")
        self.assertNotIn("quantization_overhead_mib", estimate)
        self.assertLess(estimate["required_mib"], gpu.free_mib)

    def test_four_slot_q4_fit_uses_total_memory_not_just_weights(self):
        gpu = base.GPU(2, "32 GB GPU", 32768, int(31.3 * 1024))
        with self.assertRaisesRegex(ValueError, "parameters-effective-bpw"):
            base.plan(["q38@gpu=2,slots=4,bits=4"], [gpu], 262144)
        result = base.plan(["q38@gpu=2,slots=4,bits=4,context=131072"], [gpu], 262144)
        self.assertEqual(result["servers"][0]["profile"]["quant"], "UD-Q4_K_M")

    def test_unplaceable_pinned_models_report_requirement_breakdown(self):
        free = [31.3, 26.7, 27.0, 27.2, 19.5, 19.3, 25.6, 27.0]
        gpus = [base.GPU(i, "Tesla V100-SXM2-32GB", 32768, int(f * 1024), "7.0")
                for i, f in enumerate(free)]
        with self.assertRaises(ValueError) as ctx:
            base.plan(["ornith-1.5:35b@gpu=4,slots=2,bits=4",
                       "qwen3.8:27b@gpu=5,slots=2,bits=4"], gpus, 262144)
        msg = str(ctx.exception)
        self.assertIn("ornith-1.5:35b-a3b@gpu=4: smallest matching quant Q4_K_M", msg)
        self.assertIn("weights 21.7", msg)
        self.assertIn("GPU(s) [4] have 19.5 GiB free", msg)
        self.assertIn("qwen3.8:27b@gpu=5", msg)
        # Unpinned, the same 4-bit two-slot pair fits on single cards.
        result = base.plan(["ornith-1.5:35b@slots=2,bits=4", "qwen3.8:27b@slots=2,bits=4"],
                           gpus, 262144)
        self.assertEqual([len(s["gpus"]) for s in result["servers"]], [1, 1])

    def test_native_context_cannot_be_exceeded(self):
        profile = base.PROFILES["qwen3.8:27b"][0]
        with self.assertRaisesRegex(ValueError, "native"):
            capacity.memory_estimate(profile, capacity.resolve_options({}, 524288))


class InstanceCapacityTests(unittest.TestCase):
    def test_bits_filters_profiles_without_silent_down_selection(self):
        for bits in (2, 3, 4, 6):
            req = coder.parse_model_spec(f"qwen3.8:27b@bits={bits}")
            candidates = coder.candidates_for_request(req, coder.synthetic_3090(4), 65536)
            self.assertTrue(candidates)
            self.assertTrue(all(capacity.quant_bits(c.profile.quant) == bits for c in candidates))
        req = coder.parse_model_spec("qwen3.8:27b@bits=6,vram=12G,gpu=0")
        self.assertEqual(coder.candidates_for_request(req, coder.synthetic_3090(1), 262144), [])

    def test_bits_selects_lower_same_bitness_profile_when_needed(self):
        from dataclasses import replace
        model = "qwen3.8-flash-next"
        profiles = [replace(p, weight_mib=120000 - i * 10000)
                    for i, p in enumerate(p for p in base.PROFILES[model]
                                          if capacity.quant_bits(p.quant) == 4)]
        self.assertGreater(len(profiles), 1)
        required = capacity.memory_estimate(
            profiles[-1], capacity.resolve_options({}, 262144))["required_mib"]
        with patch.dict(base.PROFILES, {model: profiles}):
            profile, _ = base.best_profile_for_capacity(
                model, required, 262144, {"bits": 4})
        self.assertEqual(profile.quant, profiles[-1].quant)

    def test_bits_rejects_unavailable_and_conflicting_quants(self):
        for suffix in ("bits=7", "bits=4,quant=IQ3_XXS"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "no supported"):
                coder.parse_model_spec("q38@" + suffix)
        req = coder.parse_model_spec("q38@bits=3,quant=IQ3_XXS")
        self.assertEqual(req.capacity["quant"], "IQ3_XXS")

    def test_role_planner_accepts_standalone_bits(self):
        result = base.plan(["q38", "bits=3"], coder.synthetic_3090(1), 65536)
        self.assertEqual(capacity.quant_bits(result["servers"][0]["profile"]["quant"]), 3)

    def test_config_defaults_inline_independent_override(self):
        defaults = {"qwen3.8:27b": {
            "gpu": 0, "vram": "16G", "slots": 2,
            "capacity": {"context": "32K", "output": "4K", "quant": "IQ3_XXS",
                         "kv_k": "q8_0", "admission": 1},
        }}
        request = coder.parse_model_spec(
            "q38@gpu=1,slots=3,context=64K,output=8K,kv_k=q4_0", defaults)
        self.assertEqual(request.gpu_indices, (1,))
        self.assertEqual(request.vram_limit_mib, 16384)
        self.assertEqual(request.capacity["slots"], 3)
        self.assertEqual(request.capacity["context"], 65536)
        self.assertEqual(request.capacity["quant"], "IQ3_XXS")
        self.assertEqual(request.capacity["admission_limit"], 1)
        self.assertEqual(defaults["qwen3.8:27b"]["slots"], 2)

    def test_pinned_quant_does_not_silently_downgrade(self):
        request = coder.parse_model_spec("q38@gpu=0,vram=16G,quant=Q8_0")
        with self.assertRaisesRegex(ValueError, "no fitting quant"):
            coder.build_plan([request], 1, coder.synthetic_3090(1), 262144)
        with self.assertRaisesRegex(ValueError, "unsupported quant"):
            coder.parse_model_spec("q38@quant=made_up")

    def test_slots_and_precision_affect_placement(self):
        for suffix in ["slots=2", "kv_k=f16,kv_v=f16"]:
            request = coder.parse_model_spec("q38@gpu=0,vram=16G,quant=IQ3_XXS," + suffix)
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "no fitting quant"):
                coder.build_plan([request], 1, coder.synthetic_3090(1), 262144)

    def test_repeated_coder_instances_have_independent_budgets(self):
        requests = [
            coder.parse_model_spec("q38@gpu=0,context=32K,slots=2,output=4K,kv_k=q8_0"),
            coder.parse_model_spec("q38@gpu=1,context=64K,output=8K,admission=1"),
        ]
        plan = coder.build_plan(requests, 2, coder.synthetic_3090(2), 262144)
        first, second = [w["capacity"] for w in plan["workers"]]
        self.assertEqual((first["context"], second["context"]), (32768, 65536))
        self.assertEqual((first["slots"], second["slots"]), (2, 1))
        self.assertEqual(first["kv_k"], "q8_0")
        self.assertEqual(plan["workers"][0]["profile"]["kv_k"], "q8_0")
        self.assertEqual(first["memory_estimate"]["required_mib"], plan["workers"][0]["required_mib"])

    def test_global_slots_fallback_and_instance_override(self):
        requests = [coder.parse_model_spec("q38@gpu=0,context=32K"),
                    coder.parse_model_spec("q38@gpu=1,context=32K,slots=1")]
        plan = coder.build_plan(requests, 2, coder.synthetic_3090(2), 262144, slots=3)
        self.assertEqual([w["capacity"]["slots"] for w in plan["workers"]], [3, 1])

    def test_public_selected_request_capacity_helper(self):
        request = coder.parse_model_spec("q38@context=32K,slots=2,output=4K,kv_k=q8_0")
        candidate = coder.candidates_for_request(request, coder.synthetic_3090(1), 262144)[0]
        settings = coder.request_capacity(request, candidate.profile, 262144)
        self.assertEqual(settings["slots"], 2)
        self.assertEqual(settings["context"], 32768)
        self.assertEqual(settings["output_tokens"], 4096)
        self.assertEqual(settings["kv_k"], "q8_0")
        self.assertEqual(settings["memory_estimate"]["required_mib"], candidate.required_mib)

    def test_global_client_context_fallback_and_instance_override(self):
        requests = [coder.parse_model_spec("q38@gpu=0,context=32K"),
                    coder.parse_model_spec("q38@gpu=1,context=32K,client_context=20K")]
        plan = coder.build_plan(requests, 2, coder.synthetic_3090(2), 262144,
                                client_context=24576)
        self.assertEqual([w["capacity"]["client_context"] for w in plan["workers"]],
                         [24576, 20480])
        self.assertNotIn("client_context", requests[0].capacity)

    def test_global_client_context_checked_even_when_overridden(self):
        request = coder.parse_model_spec("q38@context=32K,client_context=20K")
        with self.assertRaisesRegex(ValueError, "global client_context"):
            coder.build_plan([request], 1, coder.synthetic_3090(1), 262144,
                             client_context=65536)

    def test_exact_pin_can_include_more_cards_than_necessary(self):
        request = coder.parse_model_spec("q38@gpu=0+1,context=32K,quant=IQ3_XXS")
        plan = coder.build_plan([request], 1, coder.synthetic_3090(2), 262144)
        self.assertEqual(len(plan["workers"][0]["gpus"]), 2)

    def test_rejects_invalid_or_duplicate_suffix_fields(self):
        for suffix in ["slots=0", "context=", "slots=2,slots=3", "gpu=0,gpus=1",
                       "gpu=0++1", "output=4K,", "", "kv_v=foo",
                       "admission=1,admission_limit=2"]:
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                coder.parse_model_spec("q38@" + suffix)

    def test_planner_rejects_above_native_context(self):
        with self.assertRaisesRegex(ValueError, "native"):
            coder.build_plan([coder.parse_model_spec("q38@context=512K")],
                             1, coder.synthetic_3090(2), 262144)


if __name__ == "__main__":
    unittest.main()
