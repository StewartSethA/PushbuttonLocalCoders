#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


planmod = load("claude_local_plan", ROOT / "lib" / "claude_local_plan.py")


class SchedulerPolicyTests(unittest.TestCase):
    def test_single_model_prefers_most_free_before_link(self):
        gpus = [
            planmod.GPU(0, "RTX 4060 Ti", 16380, 15500, "8.9", "", 4, 8),
            planmod.GPU(1, "RTX 4060 Ti", 16380, 16000, "8.9", "", 2, 8),
        ]
        p = planmod.plan(["q38"], gpus, 262144)
        self.assertEqual(p["servers"][0]["cuda_visible_devices"], "1")
        self.assertEqual(p["placement_policy"], "freest-then-link")

    def test_single_model_breaks_free_vram_tie_by_link(self):
        gpus = [
            planmod.GPU(0, "RTX 4060 Ti", 16380, 16000, "8.9", "", 2, 8),
            planmod.GPU(1, "RTX 4060 Ti", 16380, 16000, "8.9", "", 4, 8),
        ]
        p = planmod.plan(["q38"], gpus, 262144)
        self.assertEqual(p["servers"][0]["cuda_visible_devices"], "1")

    def test_inventory_uses_system_max_link_not_idle_current_link(self):
        responses = [
            ["0, NVIDIA GeForce RTX 4060 Ti, 16380, 16000, 8.9, 00000000:01:00.0"],
            ["1, 8"],  # idle/downshifted current state
            ["4, 8"],  # maximum possible for this GPU in this system path
        ]
        with mock.patch.object(planmod, "_query_nvidia", side_effect=responses):
            gpu = planmod.inventory()[0]
        self.assertEqual((gpu.pcie_gen, gpu.pcie_width), (4, 8))
        self.assertEqual((gpu.pcie_current_gen, gpu.pcie_current_width), (1, 8))
        self.assertEqual(gpu.link_score, 32)

    def test_joint_plan_preserves_large_gpu_for_large_model(self):
        gpus = [
            planmod.GPU(0, "Tesla V100-SXM2-32GB", 32768, 32400, "7.0", "", 3, 16),
            planmod.GPU(1, "RTX 4060 Ti", 16380, 16000, "8.9", "", 4, 8),
        ]
        p = planmod.plan(["q38", "q36"], gpus, 262144)
        by_model = {s["model"]: s for s in p["servers"]}
        self.assertEqual(p["placement_policy"], "joint-global-plan")
        self.assertEqual(by_model["qwen3.6:35b"]["cuda_visible_devices"], "0")
        self.assertEqual(by_model["qwen3.6:35b"]["profile"]["quant"], "UD-Q5_K_M")
        self.assertEqual(by_model["qwen3.8:27b"]["cuda_visible_devices"], "1")
        self.assertEqual(by_model["qwen3.8:27b"]["profile"]["quant"], "IQ3_XXS")

    def test_raw_specs_have_per_server_capacity_and_pins(self):
        gpus = [planmod.GPU(i, "RTX 3090", 24576, 24576) for i in range(2)]
        p = planmod.plan(["q38@gpu=0,context=32K,slots=2,output=4K",
                          "q36@gpu=1,context=64K,quant=UD-IQ3_XXS"], gpus, 262144)
        self.assertEqual([s["cuda_visible_devices"] for s in p["servers"]], ["0", "1"])
        self.assertEqual([s["capacity"]["context"] for s in p["servers"]], [32768, 65536])
        self.assertEqual(p["servers"][0]["capacity"]["slots"], 2)
        self.assertEqual(p["servers"][1]["profile"]["quant"], "UD-IQ3_XXS")

    def test_shared_role_conflicting_capacity_rejected(self):
        gpus = [planmod.GPU(0, "RTX 3090", 24576, 24576)]
        with self.assertRaisesRegex(ValueError, "conflicting shared-role"):
            planmod.plan(["q38@slots=1", "q38@slots=2"], gpus, 262144)
        p = planmod.plan(["q38@slots=1", "q38"], gpus, 262144)
        self.assertEqual(len(p["servers"]), 1)
        p = planmod.plan(["q38@context=32K,output=4K,kv_k=q4_0",
                          "q38@context=32K"], gpus, 262144)
        self.assertEqual(len(p["servers"]), 1)

    def test_shared_role_config_defaults_and_cli_override(self):
        gpus = [planmod.GPU(0, "RTX 3090", 24576, 24576)]
        p = planmod.plan(["q38@slots=1"], gpus, 262144, slots=3,
                         defaults={"qwen3.8:27b": {"context": "32K", "output": "4K"}})
        self.assertEqual(p["servers"][0]["capacity"]["context"], 32768)
        self.assertEqual(p["servers"][0]["capacity"]["output_tokens"], 4096)
        self.assertEqual(p["servers"][0]["capacity"]["slots"], 1)

    def test_global_client_context_is_explicit_fallback(self):
        gpus = [planmod.GPU(0, "RTX 3090", 24576, 24576)]
        p = planmod.plan(["q38@context=32K"], gpus, 262144, client_context=24576)
        self.assertEqual(p["servers"][0]["capacity"]["client_context"], 24576)
        p = planmod.plan(["q38@context=32K,client_context=20K"], gpus, 262144,
                         client_context=24576)
        self.assertEqual(p["servers"][0]["capacity"]["client_context"], 20480)
        with self.assertRaisesRegex(ValueError, "global client_context"):
            planmod.plan(["q38@context=32K,client_context=20K"], gpus, 262144,
                         client_context=65536)


if __name__ == "__main__":
    unittest.main()
