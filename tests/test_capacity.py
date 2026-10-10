import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
import pushbutton_capacity as capacity
import claude_local_plan as base
import coder_local_plan as coder


class CapacityBudgetTests(unittest.TestCase):
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
        self.assertEqual(one["required_mib"], profile.required_mib)
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
        self.assertEqual(lowered["weights_source"], "profile-envelope")
        self.assertLess(lowered["required_mib"], q38.required_mib)

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
        model = "qwen3.8-flash-next"
        profiles = [p for p in base.PROFILES[model] if capacity.quant_bits(p.quant) == 4]
        self.assertGreater(len(profiles), 1)
        profile, _ = base.best_profile_for_capacity(
            model, profiles[-1].required_mib, 262144, {"bits": 4})
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
