#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
