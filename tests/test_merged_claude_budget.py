"""Merged per-model capacity and documented Claude compaction policy."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("merged_claude_budget", ROOT / "lib/claude_local_budget.py")
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)


class MergedBudgetTests(unittest.TestCase):
    def test_standalone_bits_passes_launcher_budget_preflight(self):
        argv = ["claude_local_budget.py", "--requested", "262144",
                "--model-spec=q38", "--model-spec=bits=3"]
        with patch.object(sys, "argv", argv), patch.object(budget, "validate_managed_settings"), patch.object(budget, "shared_policy") as policy, patch("builtins.print"):
            self.assertEqual(budget.main(), 0)
        self.assertEqual(policy.call_args.args[0]["0"]["capacity"]["bits"], 3)

    def setUp(self):
        self.small = dict(context=131072, slots=3, output_tokens=4096,
                          client_context=120000, compact_trigger=105000,
                          input_tokens=114880, safety_tokens=1024,
                          admission_limit=2, admission_explicit=True)
        self.large = dict(context=262144, slots=2, output_tokens=8192,
                          client_context=200000, compact_trigger=150000,
                          input_tokens=190784, safety_tokens=1024,
                          admission_limit=1, admission_explicit=False)
        self.plan = {"servers":[{"id":"small", "capacity":self.small},
                                {"id":"large", "capacity":self.large}],
                     "role_ids":{"haiku":"small", "sonnet":"large", "opus":"large", "fable":"large"}}
        self.backends = {"small":{"port":1, "capacity":self.small},
                         "large":{"port":2, "capacity":self.large}}
        self.props = {1:{"default_generation_settings":{"n_ctx":131072}, "total_slots":3},
                      2:{"default_generation_settings":{"n_ctx":262144}, "total_slots":2}}

    def request(self, url, body=None):
        port = int(url.split(":")[2].split("/")[0])
        if url.endswith("/props"):
            return self.props[port]
        self.assertTrue(url.endswith("/v1/messages/count_tokens"))
        self.assertIn("tools", body)
        self.assertIn("messages", body)
        return {"input_tokens":20}

    def config(self, client="", env=None):
        with patch.object(budget, "request_json", side_effect=self.request):
            return budget.prepare_config(self.plan, self.backends, 262144, client, env or {})

    def test_shared_roles_use_smallest_verified_context_output_client_and_compact(self):
        config = self.config()
        self.assertEqual(config["budget"], dict(capacity=131072, client_context=118784,
                                               compact_window=105000, max_output_tokens=4096,
                                               prompt_reserve=8192, compact_reserve=8192))
        route = config["roles"]["haiku"]
        self.assertEqual(route["context_capacity"], 131072)
        self.assertEqual(route["prompt_reserve"], config["budget"]["prompt_reserve"])
        self.assertEqual(route["capacity"], {**self.small, "no_context_shift":True})
        self.assertEqual(config["roles"]["sonnet"]["capacity"]["slots"], 2)

    def test_explicit_env_can_only_lower_model_budgets(self):
        config = self.config("115000", {"CLAUDE_CODE_MAX_OUTPUT_TOKENS":"2048",
                                       "CLAUDE_CODE_MAX_CONTEXT_TOKENS":"110000",
                                       "CLAUDE_CODE_AUTO_COMPACT_WINDOW":"100000"})
        self.assertEqual(config["budget"]["max_output_tokens"], 2048)
        self.assertEqual(config["budget"]["client_context"], 110000)
        self.assertEqual(config["budget"]["compact_window"], 100000)
        for client, env in [
            ("", {"CLAUDE_CODE_MAX_OUTPUT_TOKENS":"4097"}),
            ("121000", {}),
            ("", {"CLAUDE_CODE_MAX_CONTEXT_TOKENS":"121000"}),
            ("", {"CLAUDE_CODE_AUTO_COMPACT_WINDOW":"105001"}),
        ]:
            with self.subTest(client=client, env=env), self.assertRaises(budget.BudgetError):
                self.config(client, env)

    def test_documented_default_output_is_capped_not_raised_by_model_budgets(self):
        self.small.update(context=262144, client_context=200000, compact_trigger=150000,
                          output_tokens=32768)
        self.large["output_tokens"] = 32768
        self.props[1]["default_generation_settings"]["n_ctx"] = 262144
        self.assertEqual(self.config()["budget"]["max_output_tokens"], 8192)
        self.assertEqual(self.config(env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS":"16384"})["budget"]["max_output_tokens"],
                         16384)
        with self.assertRaises(budget.BudgetError):
            self.config(env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS":"32769"})

    def test_model_compact_below_supported_minimum_fails_actionably(self):
        self.small["compact_trigger"] = 99999
        with self.assertRaisesRegex(budget.BudgetError, "documented minimum.*100000"):
            self.config()
        self.small["compact_trigger"] = 110000
        self.small["client_context"] = 99999
        with self.assertRaisesRegex(budget.BudgetError, "minimum.*100000"):
            self.config()

    def test_per_model_compact_is_capped_by_physical_reserves(self):
        self.small["compact_trigger"] = 120000
        self.assertEqual(self.config()["budget"]["compact_window"], 110592)

    def test_proof_requires_requested_slots_and_per_slot_context_not_aggregate(self):
        for props in [
            {"default_generation_settings":{"n_ctx":131072}, "total_slots":2},
            {"default_generation_settings":{"n_ctx":131071}, "total_slots":3},
            {"n_ctx":393216, "total_slots":3},
            {"default_generation_settings":{"n_ctx":131072}},
            {"default_generation_settings":{"n_ctx":"131072"}, "total_slots":3},
        ]:
            with self.subTest(props=props):
                self.props[1] = props
                with self.assertRaisesRegex(budget.BudgetError, "Claude was not launched"):
                    self.config()
        self.props[1] = {"n_ctx_slot":131072, "n_parallel":3}
        self.assertEqual(self.config()["budget"]["capacity"], 131072)

    def test_legacy_effective_context_default_still_requires_one_slot(self):
        props = {"default_generation_settings":{"n_ctx":131072}, "total_slots":3}
        with self.assertRaisesRegex(budget.BudgetError, "total_slots=1"):
            budget.effective_context(props)
        self.assertEqual(budget.effective_context(props, expected_slots=3), 131072)

    def test_runtime_larger_than_requested_does_not_relax_hard_model_budget(self):
        self.props[1]["default_generation_settings"]["n_ctx"] = 140000
        config = self.config()
        self.assertEqual(config["roles"]["haiku"]["context_capacity"], 131072)
        self.assertEqual(config["budget"]["capacity"], 131072)

    def test_legacy_registry_preserves_measured_context_policy(self):
        plan = {"role_ids":{"haiku":"small", "sonnet":"large"}}
        with patch.object(budget, "request_json", side_effect=self.request):
            config = budget.prepare_config(plan, {"small":1, "large":2}, 262144, "", {})
        self.assertEqual(config["budget"]["capacity"], 131072)
        self.assertEqual(config["budget"]["compact_window"], 106496)
        self.assertNotIn("capacity", config["roles"]["haiku"])

    def test_registry_accepts_both_formats_and_rejects_malformed_capacity(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as scratch:
            path = Path(scratch) / "backends.tsv"
            path.write_text("small\t1\trepo:quant\nlarge\t2\trepo:quant\t" + json.dumps(self.large) + "\n")
            entries = budget.read_backends(str(path))
            self.assertEqual(entries["small"], {"port":1})
            self.assertEqual(entries["large"], {"port":2, "capacity":self.large})
            for text in ("small\t1\trepo\tnull\n", "small\t1\trepo\t{\n",
                         "small\t1\n", "small\t1\trepo\nsmall\t2\trepo\n"):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(budget.BudgetError):
                        budget.read_backends(str(path))

    def test_planned_preflight_and_live_policy_match(self):
        planned = budget.shared_policy(budget.planned_routes(self.plan), "", {})
        self.assertEqual(planned, self.config()["budget"])
        self.small["compact_trigger"] = 90000
        with self.assertRaisesRegex(budget.BudgetError, "minimum.*100000"):
            budget.shared_policy(budget.planned_routes(self.plan), "", {})

    def test_cli_checks_model_overrides_instead_of_unsafe_global_fallback(self):
        env = {k:v for k, v in os.environ.items()
               if not k.startswith(("CLAUDE", "DISABLE_"))}
        base = [sys.executable, str(ROOT / "lib/claude_local_budget.py"),
                "--requested", "65536", "--slots", "3",
                "--model-spec=q38@context=262144,output=4096,client_context=200000,compact=150000"]
        result = subprocess.run(base, env=env, text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--settings={}", "--model=sonnet[1m]"):
            with self.subTest(flag=flag):
                result = subprocess.run(base + ["--", flag], env=env,
                                        text=True, capture_output=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Context policy error", result.stderr)

    def test_documented_mixed_model_example_preserves_derived_client_and_slots(self):
        self.small.update(output_tokens=8192, client_context=131072, compact_trigger=100000,
                          slots=1)
        self.large.update(context=196608, output_tokens=8192, client_context=196608,
                          compact_trigger=120000, slots=2)
        self.props[1].update(total_slots=1)
        self.props[2]["default_generation_settings"]["n_ctx"] = 196608
        config = self.config()
        self.assertEqual(config["budget"]["client_context"], 114688)
        self.assertEqual(config["budget"]["compact_window"], 100000)
        self.assertEqual(config["budget"]["max_output_tokens"], 8192)
        self.assertEqual(config["roles"]["haiku"]["capacity"]["slots"], 1)
        self.assertEqual(config["roles"]["sonnet"]["capacity"]["slots"], 2)
        with self.assertRaises(budget.BudgetError):
            self.config(env={"CLAUDE_CODE_AUTO_COMPACT_WINDOW":"131000"})

    def test_launcher_unsafe_policy_fails_before_dependencies(self):
        env = {k:v for k, v in os.environ.items()
               if not k.startswith(("CLAUDE", "DISABLE_"))}
        for args in (["--local-context", "65536"],
                     ["--local-context", "131072", "--local-client-context", "200000"],
                     ["--model", "sonnet[1m]"]):
            with self.subTest(args=args):
                result = subprocess.run(["bash", str(ROOT / "claude-local"), "q38", *args],
                                        env=env, text=True, capture_output=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Context policy error", result.stderr)
                self.assertNotIn("Installing", result.stdout)


if __name__ == "__main__":
    unittest.main()
