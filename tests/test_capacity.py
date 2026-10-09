import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
import pushbutton_capacity as capacity
import claude_local_plan as base
import coder_local_plan as coder


class CapacityBudgetTests(unittest.TestCase):
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

    def test_rejects_invalid_scalar_options(self):
        cases = [
            {"slots": 0}, {"slots": True}, {"slots": 1.5},
            {"context": "-1"}, {"output": "unlimited"}, {"client_context": None},
            {"admission": -1}, {"safety": 0}, {"min_tps": "nan"},
            {"min_tps": "inf"}, {"min_tps": True}, {"min_tps": 0},
            {"compact": "0%"}, {"compact": "100%"}, {"compact": 1.5},
            {"compact": "nan"}, {"kv_k": "q2_k"}, {"kv_v": ""},
            {"quant": "bad/value"}, {"quant": None}, {"quant": 123}, {"bogus": 1},
            {"admission": 1, "admission_limit": 1},
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                capacity.parse_options(options)

    def test_rejects_inconsistent_budget_and_admission(self):
        for kwargs in [
            {"client_context": 8193}, {"output": 8192},
            {"output": 4096, "client_context": 4096},
            {"compact": 8192}, {"safety": 8192},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                capacity.resolve_budget(8192, **kwargs)
        with self.assertRaisesRegex(ValueError, "admission"):
            capacity.resolve_options({"admission": 3, "slots": 2}, 8192)

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

    def test_native_context_cannot_be_exceeded(self):
        profile = base.PROFILES["qwen3.8:27b"][0]
        with self.assertRaisesRegex(ValueError, "native"):
            capacity.memory_estimate(profile, capacity.resolve_options({}, 524288))


class InstanceCapacityTests(unittest.TestCase):
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
