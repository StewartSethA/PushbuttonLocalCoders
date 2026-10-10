#!/usr/bin/env python3
import importlib.util
import http.client
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


planmod = load("claude_local_plan", ROOT / "lib" / "claude_local_plan.py")
budgetmod = load("claude_local_budget", ROOT / "lib" / "claude_local_budget.py")
gwmod = load("claude_local_gateway", ROOT / "lib" / "claude_local_gateway.py")


class PlannerTests(unittest.TestCase):
    def test_ornith_catalog_aliases_and_profiles(self):
        catalog = json.loads((ROOT / "configs/model-catalog.json").read_text())["models"]
        for model in ("ornith-1.5:9b", "ornith-1.5:35b-a3b"):
            meta = catalog[model]
            for alias in [model, *meta["aliases"]]:
                with self.subTest(alias=alias):
                    self.assertEqual(planmod.canonical_model(alias), model)
            profiles = planmod.PROFILES[model]
            self.assertEqual([p.quant for p in profiles], [q["name"] for q in meta["quants"]])
            self.assertTrue(all(p.repo == meta["gguf_repo"] and
                                p.native_context == meta["context"] and
                                p.template == "embedded" for p in profiles))

    def test_ornith_35b_short_selector(self):
        model = "ornith-1.5:35b-a3b"
        catalog = json.loads((ROOT / "configs/model-catalog.json").read_text())["models"]
        self.assertIn("ornith-1.5:35b", catalog[model]["aliases"])
        for selector in ("ornith-1.5:35b", " ORNITH-1.5:35B "):
            with self.subTest(selector=selector):
                self.assertEqual(planmod.canonical_model(selector), model)
        gpus = [planmod.GPU(0, "Tesla V100", 32768, 32000, "7.0"),
                planmod.GPU(1, "RTX 3090", 24576, 24000, "8.6")]
        self.assertEqual(planmod.plan(["ornith-1.5:35b"], gpus, 262144),
                         planmod.plan([model], gpus, 262144))

    def test_ornith_mixed_role_placement(self):
        gpus = [
            planmod.GPU(0, "Tesla V100", 32768, 32000, "7.0"),
            planmod.GPU(1, "RTX 3090", 24576, 24000, "8.6"),
        ]
        p = planmod.plan(["ornith-9b", "q38"], gpus, 262144)
        self.assertEqual(p["roles"]["haiku"], "ornith-1.5:9b")
        self.assertEqual(p["roles"]["opus"], "qwen3.8:27b")
        self.assertEqual(len(p["servers"]), 2)
        self.assertNotEqual(p["servers"][0]["cuda_visible_devices"],
                            p["servers"][1]["cuda_visible_devices"])
        large = planmod.plan(["ornith-35b"], gpus, 262144)["servers"][0]
        self.assertEqual(large["model"], "ornith-1.5:35b-a3b")
        self.assertEqual(large["profile"]["quant"], "Q8_0")
        self.assertEqual(len(large["gpus"]), 2)

    def test_role_mapping(self):
        self.assertEqual(planmod.role_map(["q38"])["fable"], "qwen3.8:27b")
        r = planmod.role_map(["nemotron", "q36", "q38", "glm53"])
        self.assertEqual(
            [r[x] for x in planmod.ROLES],
            ["nemotron-3.5-lightning", "qwen3.6:35b", "qwen3.8:27b", "glm-5.3-flash"],
        )

    def test_heterogeneous_joint_placement(self):
        gpus = [
            planmod.GPU(0, "Tesla V100-SXM2-32GB", 32768, 32400, "7.0", "", 3, 16),
            planmod.GPU(1, "RTX 4060 Ti", 16380, 16000, "8.9", "", 2, 8),
        ]
        p = planmod.plan(["q38", "q36"], gpus, 262144)
        by_model = {s["model"]: s for s in p["servers"]}
        self.assertEqual(by_model["qwen3.6:35b"]["cuda_visible_devices"], "0")
        self.assertEqual(by_model["qwen3.6:35b"]["profile"]["quant"], "UD-Q5_K_M")
        self.assertEqual(by_model["qwen3.8:27b"]["cuda_visible_devices"], "1")
        self.assertEqual(by_model["qwen3.8:27b"]["profile"]["quant"], "IQ3_XXS")

    def test_full_context_4060ti_profiles(self):
        gpus = [planmod.GPU(0, "RTX 4060 Ti", 16380, 16100, "8.9", "", 4, 8)]
        self.assertEqual(planmod.plan(["q38"], gpus, 262144)["servers"][0]["profile"]["quant"], "IQ3_XXS")
        self.assertEqual(planmod.plan(["q36"], gpus, 262144)["servers"][0]["profile"]["quant"], "UD-IQ3_XXS")


class GatewayTests(unittest.TestCase):
    def setUp(self):
        cfg = {
            "roles": {
                r: {"model_id": "local-" + r, "backend_alias": "local-" + r, "url": "http://127.0.0.1:1",
                    "context_capacity": 131072, "prompt_reserve": 8192}
                for r in ("haiku", "sonnet", "opus", "fable")
            }
        }
        self.router = gwmod.Router(cfg)

    def test_canonical_claude_ids_route_by_family(self):
        self.assertEqual(self.router.resolve("claude-fable-5")["model_id"], "local-fable")
        self.assertEqual(self.router.resolve("claude-opus-5")["model_id"], "local-opus")
        self.assertEqual(self.router.resolve("claude-sonnet-5")["model_id"], "local-sonnet")
        self.assertEqual(self.router.resolve("claude-haiku-4-5")["model_id"], "local-haiku")

    def test_unknown_internal_model_stays_local(self):
        self.assertEqual(self.router.resolve("unexpected-internal-id")["model_id"], "local-sonnet")

    def test_hosted_web_tools_removed_but_client_tools_preserved(self):
        client = {"name": "web_search", "input_schema": {"type": "object"}}
        mcp = {"name": "mcp__pushbutton-web__web_search_exa", "input_schema": {"type": "object"}}
        for hosted in ({"name": "web_search", "type": "web_search_20250305"},
                       {"name": "web_fetch", "type": "web_fetch_20250910"},
                       {"name": "WebSearch", "input_schema": {}},
                       {"name": "WebFetch", "input_schema": {}}):
            body = {"tools": [hosted, client, mcp], "tool_choice": {"type": "auto"}}
            self.assertTrue(gwmod.filter_hosted_web_tools(body))
            self.assertEqual(body["tools"], [client, mcp])
            self.assertEqual(body["tool_choice"], {"type": "auto"})
        self.assertFalse(gwmod.filter_hosted_web_tools({"tools": [client, mcp]}))

    def test_filtered_only_web_tools_clear_choice(self):
        body = {"tools": [{"name": "web_search", "type": "web_search_20250305"}],
                "tool_choice": {"type": "any"}}
        gwmod.filter_hosted_web_tools(body)
        self.assertEqual(body, {"tools": []})

    def test_sse_unsupported_web_call_has_recovery_guidance(self):
        for kind in ("tool_use", "server_tool_use"):
            block = {"type": kind, "name": "web_search", "input": {}}
            payload = ("data: " + json.dumps({"type": "content_block_start",
                                             "index": 0, "content_block": block}) + "\n\n").encode()
            errors = gwmod.validate_backend_payload(payload, "text/event-stream", [])
            self.assertIn("/mcp", errors[0])
            self.assertIn("not a temporary outage", errors[0])
            errors = gwmod.validate_backend_payload(json.dumps({"content": [block]}).encode(),
                                                   "application/json", [])
            self.assertIn("/mcp", errors[0])

    def test_count_tokens_cannot_bypass_smaller_route_capacity(self):
        import json
        import threading
        import urllib.request
        import urllib.error
        from http.server import ThreadingHTTPServer
        cfg = {"roles": {
            "sonnet": {"model_id": "large", "backend_alias": "large", "url": "http://local",
                       "capacity": {"context": 128, "output_tokens": 32, "input_tokens": 80}},
            "haiku": {"model_id": "small", "backend_alias": "small", "url": "http://local",
                      "capacity": {"context": 64, "output_tokens": 16, "input_tokens": 32}},
        }}
        server = ThreadingHTTPServer(("127.0.0.1", 0), gwmod.Handler)
        server.router = gwmod.Router(cfg)
        server.verbose = False
        threading.Thread(target=server.serve_forever, daemon=True).start()
        def backend(endpoint, path, body=None, headers=None):
            return {"n_ctx_per_slot": 128} if path == "/props" else {"input_tokens": 40}
        try:
            with patch.object(gwmod.request_budget, "backend_json", side_effect=backend):
                for model, expected in (("large", 200), ("small", 400)):
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/v1/messages/count_tokens",
                        json.dumps({"model": model, "messages": [], "tools": [{"name": "run"}]}).encode(),
                        {"Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(request, timeout=5) as response:
                            self.assertEqual(response.status, expected)
                            self.assertEqual(json.load(response)["input_tokens"], 40)
                    except urllib.error.HTTPError as exc:
                        self.assertEqual(exc.code, expected)
                        self.assertIn("compact/summarize", exc.read().decode())
        finally:
            server.shutdown()
            server.server_close()


class ContextPolicyTests(unittest.TestCase):
    def test_derived_131072_budget(self):
        p = budgetmod.context_policy(131072, env={})
        self.assertEqual(p["client_context"], 114688)
        self.assertEqual(p["compact_window"], 106496)
        self.assertEqual(p["max_output_tokens"], 8192)
        self.assertLessEqual(p["compact_window"], p["client_context"])
        self.assertLess(p["client_context"], p["capacity"])
        self.assertEqual(sum(p[x] for x in ("compact_window", "max_output_tokens", "prompt_reserve", "compact_reserve")), 131072)
        self.assertEqual(budgetmod.context_policy(262144, env={})["client_context"], 200000)

    def test_safe_lower_overrides(self):
        p = budgetmod.context_policy(131072, "110000", {
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "105000",
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "100000",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096"})
        self.assertEqual((p["client_context"], p["compact_window"], p["max_output_tokens"]), (105000, 100000, 4096))

    def test_shared_policy_combines_global_and_per_model_windows(self):
        routes = {
            "haiku": {"context_capacity": 131072, "capacity": {
                "output_tokens": 4096, "client_context": 110000, "compact_trigger": 105000}},
            "sonnet": {"context_capacity": 262144, "capacity": {
                "output_tokens": 8192, "client_context": 200000, "compact_trigger": 150000}},
        }
        policy = budgetmod.shared_policy(routes, "", {})
        self.assertEqual(policy["capacity"], 131072)
        self.assertEqual((policy["max_output_tokens"], policy["client_context"], policy["compact_window"]),
                         (4096, 110000, 105000))
        self.assertLessEqual(policy["compact_window"], policy["client_context"])
        self.assertLess(policy["client_context"], policy["capacity"])
        for client, env in [
            ("115000", {}), ("", {"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "115000"}),
            ("", {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192"}),
            ("", {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "106000"}),
        ]:
            with self.subTest(client=client, env=env), self.assertRaises(budgetmod.BudgetError):
                budgetmod.shared_policy(routes, client, env)
        routes["haiku"]["capacity"]["compact_trigger"] = 99999
        with self.assertRaisesRegex(budgetmod.BudgetError, "minimum"):
            budgetmod.shared_policy(routes, "", {})

    def test_prepare_config_preserves_model_capacity_and_admission(self):
        capacity = {"context": 131072, "slots": 2, "output_tokens": 4096, "input_tokens": 100000,
                    "client_context": 110000, "compact_trigger": 105000,
                    "safety_tokens": 8192, "admission_limit": 2, "admission_explicit": True}
        plan = {"role_ids": {"sonnet": "local"}, "servers": [{"id": "local", "capacity": capacity}]}
        replies = [{"total_slots": 2, "n_ctx_slot": 262144}, {"input_tokens": 40}]
        with mock.patch.object(budgetmod, "request_json", side_effect=replies):
            config = budgetmod.prepare_config(plan, {"local": 1}, 262144, "", {})
        route = config["roles"]["sonnet"]
        self.assertEqual(route["context_capacity"], 131072)
        for key, value in capacity.items():
            self.assertEqual(route["capacity"][key], value)
        self.assertTrue(route["capacity"]["no_context_shift"])
        router = gwmod.Router(config)
        self.assertTrue(router.gates[(route["url"], route["backend_alias"])].acquire(blocking=False))
        self.assertTrue(router.gates[(route["url"], route["backend_alias"])].acquire(blocking=False))
        self.assertFalse(router.gates[(route["url"], route["backend_alias"])].acquire(blocking=False))
        self.assertEqual(config["budget"]["compact_window"], 105000)
        with mock.patch.object(budgetmod, "request_json", return_value={
                "total_slots": 2, "n_ctx_slot": 65536}), self.assertRaisesRegex(
                    budgetmod.BudgetError, "model requires 131072"):
            budgetmod.prepare_config(plan, {"local": 1}, 262144, "", {})

    def test_unsafe_explicit_overrides(self):
        for client, env in [
            ("131072", {}), ("200000", {}), ("99999", {}),
            ("", {"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "200000"}),
            ("", {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "131072"}),
            ("", {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "99999"}),
            ("", {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "100k"}),
            ("", {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000"}),
            ("", {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "0"}),
        ]:
            with self.subTest(client=client, env=env), self.assertRaises(budgetmod.BudgetError):
                budgetmod.context_policy(131072, client, env)

    def test_disabled_compaction_and_legacy_override(self):
        for name in ("DISABLE_AUTO_COMPACT", "DISABLE_COMPACT", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"):
            with self.subTest(name=name), self.assertRaisesRegex(budgetmod.BudgetError, name):
                budgetmod.context_policy(131072, env={name: "1"})

    def test_below_documented_minimum_fails_without_clamping(self):
        for capacity in (32768, 99999, 100000, 120000):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(budgetmod.BudgetError, "minimum"):
                budgetmod.context_policy(capacity, env={})

    def test_official_per_slot_metadata_not_native_context(self):
        self.assertEqual(budgetmod.effective_context({
            "total_slots": 1, "n_ctx_train": 262144,
            "default_generation_settings": {"n_ctx": 131072}}), 131072)
        self.assertEqual(budgetmod.effective_context({
            "total_slots": 2, "n_ctx_slot": 131072,
            "default_generation_settings": {"n_ctx": 262144}}, expected_slots=2), 131072)
        for props in ({}, {"total_slots": 0, "default_generation_settings": {"n_ctx": 262144}},
                      {"total_slots": 2, "default_generation_settings": {"n_ctx": 262144}},
                      {"total_slots": 1, "default_generation_settings": {"n_ctx": 0}}):
            with self.assertRaises(budgetmod.BudgetError):
                budgetmod.effective_context(props)
        with self.assertRaises(budgetmod.BudgetError):
            budgetmod.effective_context({
                "total_slots": 1, "default_generation_settings": {"n_ctx": 262144}}, expected_slots=2)

    def test_all_roles_use_smallest_runtime_capacity_and_report_mismatch(self):
        plan = {"role_ids": dict(zip(("haiku", "sonnet", "opus", "fable"), ("small", "large", "large", "large")))}

        def backend(url, body=None):
            if url.endswith("/props"):
                return {"total_slots": 1, "default_generation_settings": {"n_ctx": 131072 if ":1/" in url else 262144}}
            self.assertIn("system", body)
            self.assertIn("tools", body)
            return {"input_tokens": 40}

        with mock.patch.object(budgetmod, "request_json", side_effect=backend), mock.patch("sys.stderr") as stderr:
            cfg = budgetmod.prepare_config(plan, {"small": 1, "large": 2}, 262144, "", {})
        self.assertEqual(cfg["budget"]["capacity"], 131072)
        self.assertEqual(cfg["roles"]["fable"]["context_capacity"], 262144)
        self.assertTrue(stderr.write.called)
        with mock.patch.object(budgetmod, "request_json", side_effect=backend), self.assertRaises(budgetmod.BudgetError):
            budgetmod.prepare_config(plan, {"small": 1, "large": 2}, 262144, "200000", {})

    def test_unavailable_metadata_or_tokenizer_refuses_launch(self):
        plan = {"role_ids": {"sonnet": "local"}}
        for replies in ([{}], [{"total_slots": 1, "default_generation_settings": {"n_ctx": 131072}}, {}]):
            with mock.patch.object(budgetmod, "request_json", side_effect=replies), self.assertRaisesRegex(budgetmod.BudgetError, "Claude was not launched"):
                budgetmod.prepare_config(plan, {"local": 1}, 131072, "", {})

    def test_unsafe_frontend_and_provider_flags(self):
        for args, env in [(["--model", "sonnet[1m]"], {}), (["--autocompact=off"], {}),
                          (["--settings", '{"autoCompactEnabled":false}'], {}),
                          (["--setting-sources=user"], {}), ([], {"CLAUDE_CODE_USE_VERTEX": "1"})]:
            with self.subTest(args=args, env=env), self.assertRaises(budgetmod.BudgetError):
                budgetmod.validate_claude_args(args, env)
                budgetmod.frontend_policy_env(args, env)
        budgetmod.validate_claude_args(["--model", "claude-opus-5", "--resume", "SESSION"], {})

    def test_safe_frontend_compaction_window_preserved(self):
        env = budgetmod.frontend_policy_env(["--autocompact", "100000"], {})
        self.assertEqual(budgetmod.context_policy(131072, env=env)["compact_window"], 100000)
        env = budgetmod.frontend_policy_env(["--autocompact=120000"], {})
        with self.assertRaises(budgetmod.BudgetError):
            budgetmod.context_policy(131072, env=env)

    def test_managed_settings_conflicts_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "managed-settings.json"
            for settings in ({"autoCompactEnabled": False}, {"env": {"DISABLE_COMPACT": "1"}},
                             {"env": {"ANTHROPIC_BASE_URL": "https://example.invalid"}}):
                path.write_text(json.dumps(settings))
                with self.assertRaisesRegex(budgetmod.BudgetError, "administrator"):
                    budgetmod.validate_managed_settings(path)
            path.write_text('{"autoCompactEnabled":true}')
            budgetmod.validate_managed_settings(path)
            directory = path.with_suffix(".d")
            directory.mkdir()
            (directory / "10-disable.json").write_text('{"autoCompactEnabled":false}')
            with self.assertRaises(budgetmod.BudgetError):
                budgetmod.validate_managed_settings(path)
            (directory / "20-enable.json").write_text('{"autoCompactEnabled":true}')
            budgetmod.validate_managed_settings(path)
            (directory / "30-provider.json").write_text('{"env":{"ANTHROPIC_BASE_URL":"https://example.invalid"}}')
            with self.assertRaises(budgetmod.BudgetError):
                budgetmod.validate_managed_settings(path)

    def test_installed_claude_feature_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            claude = pathlib.Path(tmp) / "claude"
            claude.write_text("#!/bin/sh\n# CLAUDE_CODE_AUTO_COMPACT_WINDOW CLAUDE_CODE_MAX_CONTEXT_TOKENS CLAUDE_CODE_MAX_OUTPUT_TOKENS\n"
                              "if [ \"$1\" = --help ]; then echo '--autocompact --settings --setting-sources'; else echo '2.1.221 (mock Claude)'; fi\n")
            claude.chmod(0o755)
            with mock.patch.object(budgetmod.shutil, "which", return_value=str(claude)):
                self.assertEqual(budgetmod.claude_capabilities(), "2.1.221 (mock Claude)")
                claude.write_text("#!/bin/sh\necho '2.1.221 (opaque wrapper)'\n")
                with self.assertRaisesRegex(budgetmod.BudgetError, "cannot verify"):
                    budgetmod.claude_capabilities()
                claude.write_text("#!/bin/sh\necho '2.1.220 (old Claude)'\n")
                with self.assertRaisesRegex(budgetmod.BudgetError, "2.1.221"):
                    budgetmod.claude_capabilities()

    def test_mock_claude_launch_exports_policy_and_isolates_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            capture = tmp / "capture.json"
            claude = tmp / "claude"
            claude.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                              "json.dump({'args':sys.argv[1:],'env':dict(os.environ)},open(os.environ['CAPTURE'],'w'))\n")
            claude.chmod(0o755)
            plan = tmp / "plan.json"
            plan.write_text(json.dumps({"role_ids": {r: "local-" + r for r in ("haiku", "sonnet", "opus", "fable")}}))
            config = tmp / "gateway.json"
            config.write_text(json.dumps({"budget": budgetmod.context_policy(131072, env={})}))
            user_config = tmp / ".claude.json"
            user_config.write_text('{"autoCompactEnabled":false}')
            script = tmp / "launch.sh"
            functions = (ROOT / "claude-local").read_text().split('\ncase "${1:-}" in\n', 1)[0]
            functions = functions.replace('ROOT="$(cd "$(dirname "$SELF")" && pwd)"', f"ROOT={shlex.quote(str(ROOT))}")
            script.write_text(functions + f"\nPLAN_FILE={shlex.quote(str(plan))}\n"
                              f"GATEWAY_CONFIG={shlex.quote(str(config))}\nSTATE_DIR={shlex.quote(str(tmp / 'state'))}\n"
                              "GATEWAY_PORT=18180\nCLAUDE_ARGS=(--resume SESSION)\nrun_claude\n")
            env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "DISABLE_"))}
            env.update(HOME=str(tmp), PATH=str(tmp) + ":" + os.environ["PATH"], CAPTURE=str(capture))
            result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            captured = json.loads(capture.read_text())
            self.assertEqual(captured["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "114688")
            self.assertEqual(captured["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "106496")
            self.assertEqual(captured["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "8192")
            self.assertEqual(captured["env"]["CLAUDE_CONFIG_DIR"], str(tmp / "state" / "claude-config"))
            self.assertIn("--resume", captured["args"])
            i = captured["args"].index("--setting-sources")
            self.assertEqual(captured["args"][i + 1], "")
            i = captured["args"].index("--settings")
            settings = json.loads(captured["args"][i + 1])
            self.assertTrue(settings["autoCompactEnabled"])
            self.assertEqual(settings["permissions"]["deny"], ["WebSearch", "WebFetch"])
            i = captured["args"].index("--disallowedTools")
            self.assertEqual(captured["args"][i + 1], "WebSearch,WebFetch")
            i = captured["args"].index("--append-system-prompt")
            self.assertIn("exact advertised names", captured["args"][i + 1])
            for guidance in ("small line ranges", "constrain search results", "dependency directories",
                             "short handoff", "/clear", "Do not clear history automatically",
                             "verified backend capacity"):
                self.assertIn(guidance, captured["args"][i + 1])
            self.assertEqual(captured["args"].count("--append-system-prompt"), 1)
            self.assertIn("save a short handoff, then /clear", result.stdout)
            self.assertEqual(user_config.read_text(), '{"autoCompactEnabled":false}')

    def test_entry_web_config_is_sticky_and_customizable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / "lib").mkdir()
            entry = tmp / "lib" / "claude_local_entry.sh"
            entry.write_text((ROOT / "lib" / "claude_local_entry.sh").read_text())
            (tmp / "lib" / "pushbutton_folders.sh").write_text("initialize_folders() { :; }\n")
            core = tmp / "claude-local"
            core.write_text('#!/bin/bash\n'
                            'if [[ "${CLAUDE_LOCAL_STARTUP_ONLY:-}" == 1 ]]; then\n'
                            '  printf "%s" "${CLAUDE_LOCAL_WEB_CONFIG:-}" > "$WEB_HANDOFF"\n'
                            '  exit 125\nfi\n'
                            'printf "%s\\n" "$@"\n')
            core.chmod(0o755)
            state = tmp / "state"
            env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_LOCAL")}
            env.update(CLAUDE_LOCAL_STATE=str(state), CLAUDE_LOCAL_WEB_MCP="1",
                       CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY="1", WEB_HANDOFF=str(tmp / "handoff"))

            def run(*args):
                return subprocess.run(["bash", str(entry), "q38", *args],
                                      env=env, text=True, capture_output=True)

            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            config = state / "web-mcp.json"
            self.assertEqual(json.loads(config.read_text())["mcpServers"]["pushbutton-web"]["url"],
                             "https://mcp.exa.ai/mcp")
            self.assertIn(str(config), result.stdout)
            self.assertIn("mcp.exa.ai", result.stdout)
            self.assertIn("--local-web-config FILE", result.stdout)
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            custom = '{"mcpServers":{"custom-search":{"type":"http","url":"http://localhost:8080/mcp"}}}'
            config.write_text(custom)
            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(config.read_text(), custom)
            self.assertIn("custom-search", result.stdout)
            config.write_text(json.dumps({"mcpServers": {"custom-search": {
                "type": "http", "url": "https://" + "user:password@" + "example.com/mcp?key=private"}}}))
            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("example.com", result.stdout)
            self.assertNotIn("password", result.stdout)
            self.assertNotIn("private", result.stdout)
            alternate = tmp / "alternate.json"
            alternate.write_text(custom)
            result = run("--local-web-config", str(alternate))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(alternate), result.stdout)
            self.assertEqual((tmp / "handoff").read_text(), str(alternate))
            self.assertNotIn("--local-web-config\n", result.stdout)
            env["CLAUDE_LOCAL_WEB_CONFIG"] = str(alternate)
            self.assertIn(str(alternate), run().stdout)
            result = run("--local-no-web")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("--mcp-config", result.stdout.splitlines()[1:])
            config.write_text("{")
            env.pop("CLAUDE_LOCAL_WEB_CONFIG")
            result = run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid web MCP config", result.stderr)
            self.assertEqual(config.read_text(), "{")

    def test_launcher_rejects_unsafe_flags_before_dependencies(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "DISABLE_"))}
        for args in (["--local-context", "131072", "--local-client-context", "200000"],
                     ["--local-context", "65536"], ["--model", "sonnet[1m]"]):
            result = subprocess.run(["bash", str(ROOT / "claude-local"), "q38", *args], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Context policy error", result.stderr)
            self.assertNotIn("Installing", result.stdout)

    def test_missing_python_bootstraps_before_policy_without_gpu(self):
        launcher = (ROOT / "claude-local").read_text()
        functions = launcher.split('\ncase "${1:-}" in\n', 1)[0]
        functions = functions.replace('ROOT="$(cd "$(dirname "$SELF")" && pwd)"', f"ROOT={shlex.quote(str(ROOT))}")
        startup = launcher[launcher.index("\nif ! have_cmd python3; then"):launcher.index('\nsource "$LIB/pushbutton_folders.sh"\ninitialize_folders')]
        startup += launcher[launcher.index("\nif [[ $DRY_RUN -eq 0 ]]; then install_base_deps; select_accelerator;"):launcher.index('\nmkdir -p "$STATE_DIR" "$CACHE_DIR"; PLAN_FILE=')]
        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "bootstrap.sh"
            script.write_text(functions + """
have_cmd() { [[ "$1" != python3 ]]; }
install_base_deps() { printf 'deps\\n'; }
python3() { printf 'policy\\n'; }
select_accelerator() { printf 'driver\\n'; }
""" + startup)
            result = subprocess.run(["bash", str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["deps", "policy", "deps", "driver"])


class GatewayBudgetHTTPTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.prompt_tokens = 100
        self.inference_status = 200
        self.count_status = 200
        self.count_result = None
        self.count_fail_once = False
        self.inference_fail_once = False
        self.inference_mode = "json"
        self.inference_result = None
        test = self

        class Backend(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def do_GET(self):
                raw = json.dumps({"total_slots": 1, "default_generation_settings": {"n_ctx": 131072}}).encode()
                self.send_response(200)
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                test.requests.append((self.path, body))
                if self.path.endswith("/count_tokens"):
                    status = test.count_status
                    if test.count_fail_once:
                        status = 503
                        test.count_fail_once = False
                    response = {"input_tokens": test.prompt_tokens} if status == 200 else {
                        "type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}}
                    if test.count_result is not None:
                        response = test.count_result
                else:
                    status = test.inference_status
                    if test.inference_fail_once:
                        status = 503
                        test.inference_fail_once = False
                    response = {"type": "message", "content": []} if status == 200 else {
                        "type": "error", "error": {"type": "invalid_request_error", "message": "request exceeds the available context size"}}
                    if test.inference_result is not None:
                        response = test.inference_result
                    if test.inference_mode == "sse":
                        self.send_response(200)
                        self.send_header("content-type", "text/event-stream")
                        self.send_header("connection", "close")
                        self.end_headers()
                        self.wfile.write(b'event: message_start\ndata: {"type":"message_start"}\n\n')
                        self.wfile.flush()
                        self.close_connection = True
                        return
                    if test.inference_mode == "empty-once":
                        test.inference_mode = "sse"
                        self.send_response(200)
                        self.send_header("content-type", "text/event-stream")
                        self.send_header("connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        return
                    if test.inference_mode == "truncated-sse":
                        self.send_response(200)
                        self.send_header("content-type", "text/event-stream")
                        self.send_header("content-length", "10000")
                        self.end_headers()
                        self.wfile.write(b"event: message_start\ndata: {}\n\n")
                        self.wfile.flush()
                        self.close_connection = True
                        return
                raw = json.dumps(response).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        url = f"http://127.0.0.1:{self.backend.server_port}"
        self.gateway = ThreadingHTTPServer(("127.0.0.1", 0), gwmod.Handler)
        self.gateway.router = gwmod.Router({"roles": {
            r: {"model_id": "local-" + r, "backend_alias": "backend-" + r, "url": url,
                "context_capacity": 131072 if r == "haiku" else 262144, "prompt_reserve": 8192}
            for r in ("haiku", "sonnet", "opus", "fable")}})
        self.gateway.verbose = False
        for server in (self.backend, self.gateway):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)

    def request(self, body=None, path="/v1/messages", tools=True):
        body = body or {"model": "claude-haiku-4-5", "max_tokens": 8192,
                        "system": [{"type": "text", "text": "system"}],
                        "tools": [{"name": "test", "input_schema": {"type": "object"}}],
                        "messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "test", "input": {}}]},
                                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x" * 50000}]}]}
        if not tools:
            body.pop("tools", None)
        conn = http.client.HTTPConnection("127.0.0.1", self.gateway.server_port, timeout=5)
        self.addCleanup(conn.close)
        conn.request("POST", path, json.dumps(body), {"content-type": "application/json"})
        resp = conn.getresponse()
        try:
            raw = resp.read()
        except http.client.IncompleteRead as exc:
            raw = exc.partial
        return resp.status, raw, body

    def test_147023_tokens_rejected_before_inference_and_without_truncation(self):
        self.prompt_tokens = 147023
        status, raw, original = self.request()
        error = json.loads(raw)["error"]
        self.assertEqual(status, 400)
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertIn("147023 input tokens", error["message"])
        self.assertIn("131072 tokens", error["message"])
        self.assertIn("resumed transcript", error["message"])
        self.assertEqual([p for p, _ in self.requests], ["/v1/messages/count_tokens"])
        counted = self.requests[0][1]
        self.assertEqual(counted["messages"], original["messages"])
        self.assertEqual(counted["system"], original["system"])
        self.assertEqual(counted["tools"], original["tools"])
        self.assertEqual(counted["max_tokens"], original["max_tokens"])

    def test_output_and_template_reserves_boundary(self):
        self.prompt_tokens = 131072 - 8192 - 8192
        self.assertEqual(self.request()[0], 200)
        self.prompt_tokens += 1
        self.gateway.router.token_counts.clear()
        self.assertEqual(self.request()[0], 400)
        self.assertEqual(sum(p == "/v1/messages" for p, _ in self.requests), 1)

    def test_all_routes_apply_their_own_capacity(self):
        self.prompt_tokens = 147023
        for role in ("haiku", "sonnet", "opus", "fable"):
            status, _, _ = self.request({"model": "claude-" + role + "-5", "max_tokens": 8192, "messages": []})
            self.assertEqual(status, 400 if role == "haiku" else 200)
        self.assertEqual(self.requests[-1][1]["model"], "backend-fable")

    def test_inference_400_is_not_retried(self):
        self.inference_status = 400
        status, raw, _ = self.request()
        self.assertEqual(status, 400)
        self.assertIn(b"exceeds the available context size", raw)
        self.assertEqual(sum(p == "/v1/messages" for p, _ in self.requests), 1)

    def test_tokenizer_400_is_not_retried_or_inferred(self):
        self.count_status = 400
        self.assertEqual(self.request()[0], 400)
        self.assertEqual(len(self.requests), 1)

    def test_invalid_native_counts_fail_closed(self):
        for result in ({}, {"input_tokens": -1}, {"input_tokens": "100"}, {"input_tokens": True}):
            self.count_result = result
            self.assertEqual(self.request()[0], 503)
        self.assertFalse(any(p == "/v1/messages" for p, _ in self.requests))

    def test_capacity_only_route_requires_native_count_even_with_hard_guard(self):
        route = self.gateway.router.roles["haiku"]
        route.pop("context_capacity")
        route.pop("prompt_reserve")
        route["capacity"] = {"context": 131072, "output_tokens": 8192,
                             "input_tokens": 110000, "no_context_shift": True}
        self.count_status = 404
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            with self.subTest(path=path):
                status, raw, _ = self.request(path=path)
                self.assertEqual(status, 503)
                self.assertIn(b"authoritative native token count unavailable", raw)
        self.assertEqual([path for path, _ in self.requests],
                         ["/v1/messages/count_tokens", "/v1/messages/count_tokens"])

    def test_transient_tokenizer_and_inference_5xx_retry_before_commit(self):
        self.count_fail_once = True
        self.inference_fail_once = True
        self.assertEqual(self.request()[0], 200)
        self.assertEqual([p for p, _ in self.requests],
                         ["/v1/messages/count_tokens", "/v1/messages/count_tokens", "/v1/messages", "/v1/messages"])

    def test_count_endpoint_forwards_native_full_payload(self):
        self.prompt_tokens = 147023
        status, raw, body = self.request(path="/v1/messages/count_tokens")
        self.assertEqual((status, json.loads(raw)["input_tokens"]), (200, 147023))
        self.assertEqual(self.requests[0][1]["messages"], body["messages"])

    def test_combined_capacity_uses_smaller_context_and_larger_reserve(self):
        route = self.gateway.router.roles["haiku"]
        route["capacity"] = {"context": 100000, "output_tokens": 8192,
                             "input_tokens": 100000, "safety_tokens": 10000}
        self.prompt_tokens = 100000 - 8192 - 10000
        self.assertEqual(self.request()[0], 200)
        self.prompt_tokens += 1
        self.gateway.router.token_counts.clear()
        status, raw, _ = self.request()
        self.assertEqual(status, 400)
        self.assertIn(b"100000 tokens", raw)
        self.assertEqual(sum(p == "/v1/messages" for p, _ in self.requests), 1)
        route["capacity"]["context"] = 262144
        route["capacity"]["safety_tokens"] = 0
        self.prompt_tokens = 131072 - 8192 - 8192 + 1
        self.gateway.router.token_counts.clear()
        self.assertEqual(self.request()[0], 400)

    def test_combined_capacity_uses_backend_physical_context(self):
        route = self.gateway.router.roles["sonnet"]
        route["capacity"] = {"context": 262144, "output_tokens": 8192,
                             "input_tokens": 240000, "safety_tokens": 0}
        self.prompt_tokens = 147023
        status, raw, _ = self.request({"model": "sonnet", "max_tokens": 8192, "messages": []})
        self.assertEqual(status, 400)
        self.assertIn(b"131072 tokens", raw)
        self.assertFalse(any(p == "/v1/messages" for p, _ in self.requests))

    def test_combined_capacity_enforces_input_output_and_count_budgets(self):
        self.gateway.router.roles["haiku"]["capacity"] = {
            "context": 131072, "output_tokens": 4096, "input_tokens": 80, "safety_tokens": 0}
        self.assertEqual(self.request()[0], 400)
        self.assertEqual(self.requests, [])
        status, raw, _ = self.request({"model": "haiku", "max_tokens": 4096, "messages": []})
        self.assertEqual(status, 400)
        self.assertIn(b"route input budget 80", raw)
        status, raw, _ = self.request(path="/v1/messages/count_tokens")
        self.assertEqual(status, 400)
        self.assertIn(b"route input budget 80", raw)
        self.prompt_tokens = 80
        self.gateway.router.token_counts.clear()
        status, raw, _ = self.request(path="/v1/messages/count_tokens")
        self.assertEqual((status, json.loads(raw)["input_tokens"]), (200, 80))
        self.assertFalse(any(p == "/v1/messages" for p, _ in self.requests))

    def test_combined_count_cache_still_rechecks_route_budgets(self):
        self.gateway.router.roles["haiku"]["capacity"] = {
            "context": 131072, "output_tokens": 8192, "input_tokens": 1000}
        _, _, body = self.request(path="/v1/messages/count_tokens")
        self.gateway.router.roles["haiku"]["capacity"]["input_tokens"] = 80
        self.assertEqual(self.request(body)[0], 400)
        self.assertEqual(sum(p.endswith("/count_tokens") for p, _ in self.requests), 1)
        self.assertFalse(any(p == "/v1/messages" for p, _ in self.requests))

    def test_generation_gate_is_released_after_budget_rejection(self):
        route = self.gateway.router.roles["haiku"]
        gate = mock.Mock()
        released = threading.Event()
        gate.release.side_effect = released.set
        gate.acquire.return_value = True
        self.gateway.router.gates[(route["url"], route["backend_alias"])] = gate
        self.prompt_tokens = 147023
        self.assertEqual(self.request()[0], 400)
        self.assertTrue(released.wait(1))
        gate.acquire.assert_called_once_with()
        gate.release.assert_called_once_with()

    def test_generation_request_queues_instead_of_returning_admission_429(self):
        route = self.gateway.router.roles["haiku"]
        gate = mock.Mock()
        queued = threading.Event()
        release = threading.Event()
        released = threading.Event()
        gate.acquire.side_effect = lambda: (queued.set(), release.wait(2))[1]
        gate.release.side_effect = released.set
        self.gateway.router.gates[(route["url"], route["backend_alias"])] = gate
        responses = []
        request = threading.Thread(target=lambda: responses.append(self.request()))
        request.start()
        self.assertTrue(queued.wait(1))
        self.assertTrue(request.is_alive())
        gate.acquire.assert_called_once_with()
        release.set()
        request.join(2)
        self.assertFalse(request.is_alive())
        self.assertEqual(responses[0][0], 200)
        self.assertEqual([p for p, _ in self.requests], ["/v1/messages/count_tokens", "/v1/messages"])
        self.assertTrue(released.wait(1))
        gate.release.assert_called_once_with()
        gate.acquire.reset_mock()
        self.requests.clear()
        self.assertEqual(self.request(path="/v1/messages/count_tokens")[0], 200)
        gate.acquire.assert_not_called()

    def test_identical_count_then_inference_reuses_digest_only_cache(self):
        _, _, body = self.request(path="/v1/messages/count_tokens")
        self.assertEqual(self.request(body)[0], 200)
        self.assertEqual(sum(p.endswith("/count_tokens") for p, _ in self.requests), 1)
        self.assertEqual(sum(p == "/v1/messages" for p, _ in self.requests), 1)
        keys = list(self.gateway.router.token_counts)
        self.assertEqual(len(keys[0][1]), 32)
        with mock.patch.object(gwmod.time, "monotonic", return_value=10**12):
            self.assertIsNone(self.gateway.router.cached_count(keys[0]))

    def test_stream_not_replayed_after_first_event_or_truncation(self):
        for mode in ("sse", "truncated-sse"):
            self.requests.clear()
            self.inference_mode = mode
            self.assertEqual(self.request(tools=False)[0], 200)
            self.assertEqual(sum(p == "/v1/messages" for p, _ in self.requests), 1)

    def test_empty_stream_can_retry_before_commit(self):
        self.inference_mode = "empty-once"
        status, raw, _ = self.request(tools=False)
        self.assertEqual(status, 200)
        self.assertIn(b"message_start", raw)
        self.assertEqual(sum(p == "/v1/messages" for p, _ in self.requests), 2)
        self.assertEqual(sum(p == "/v1/messages/count_tokens" for p, _ in self.requests), 1)

    def test_preflight_counts_same_policy_payload_as_tool_inference(self):
        self.assertEqual(self.request()[0], 200)
        counted, inferred = (body for _, body in self.requests)
        self.assertEqual(counted, inferred)
        self.assertEqual(counted["temperature"], 0.0)
        self.assertFalse(counted["chat_template_kwargs"]["enable_thinking"])
        self.assertFalse(counted["chat_template_kwargs"]["preserve_thinking"])
        self.assertFalse(counted["chat_template_kwargs"]["preserve_reasoning"])

    def test_malformed_tool_response_is_rejected_before_commit_without_recount(self):
        self.inference_result = {"type": "message", "content": [
            {"type": "tool_use", "id": "t", "name": "test", "input": "not an object"}]}
        status, raw, _ = self.request()
        self.assertEqual(status, 502)
        self.assertIn(b"invalid backend tool arguments", raw)
        self.assertEqual([p for p, _ in self.requests],
                         ["/v1/messages/count_tokens", "/v1/messages", "/v1/messages"])

    def test_hosted_web_tools_filtered_for_count_and_inference(self):
        body = {"model": "haiku", "max_tokens": 8192, "messages": [],
                "tools": [{"name": "web_search", "type": "web_search_20250305"},
                          {"name": "mcp__pushbutton-web__web_search_exa",
                           "input_schema": {"type": "object"}}]}
        self.assertEqual(self.request(body)[0], 200)
        counted, inferred = (b for _, b in self.requests)
        self.assertEqual(counted, inferred)
        self.assertEqual(counted["tools"], body["tools"][1:])
        self.requests.clear()
        self.assertEqual(self.request(body, path="/v1/messages/count_tokens")[0], 200)
        self.assertFalse(any(p == "/v1/messages" for p, _ in self.requests))

    def test_forced_hosted_web_tool_rejected_without_backend_work(self):
        body = {"model": "haiku", "max_tokens": 8192, "messages": [],
                "tools": [{"name": "web_search", "type": "web_search_20250305"}],
                "tool_choice": {"type": "tool", "name": "web_search"}}
        status, raw, _ = self.request(body)
        self.assertEqual(status, 400)
        self.assertIn(b"/mcp", raw)
        self.assertEqual(self.requests, [])

    def test_unsupported_web_call_fails_without_retry_even_after_all_tools_filtered(self):
        self.inference_result = {"type": "message", "content": [
            {"type": "tool_use", "id": "t", "name": "web_search", "input": {}}]}
        for tools in ([{"name": "test", "input_schema": {"type": "object"}}],
                      [{"name": "web_search", "type": "web_search_20250305"}]):
            self.requests.clear()
            body = {"model": "haiku", "max_tokens": 8192, "messages": [], "tools": tools}
            status, raw, _ = self.request(body)
            self.assertEqual(status, 400)
            self.assertIn(b"--local-web-config FILE", raw)
            self.assertIn(b"not a temporary outage", raw)
            self.assertEqual(sum(p == "/v1/messages" for p, _ in self.requests), 1)

    def test_valid_tool_response_preserved_after_budget_check(self):
        self.inference_result = {"type": "message", "content": [
            {"type": "tool_use", "id": "t", "name": "test", "input": {}}]}
        status, raw, _ = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), self.inference_result)
        self.assertEqual([p for p, _ in self.requests], ["/v1/messages/count_tokens", "/v1/messages"])

    def test_startup_cli_reports_effective_capacity_version_and_resume_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            claude = tmp / "claude"
            claude.write_text("#!/bin/sh\n# CLAUDE_CODE_AUTO_COMPACT_WINDOW CLAUDE_CODE_MAX_CONTEXT_TOKENS CLAUDE_CODE_MAX_OUTPUT_TOKENS\n"
                              "if [ \"$1\" = --help ]; then echo '--autocompact --settings --setting-sources'; else echo '2.1.221 (mock Claude)'; fi\n")
            claude.chmod(0o755)
            plan = tmp / "plan.json"
            plan.write_text(json.dumps({"role_ids": {r: "local" for r in ("haiku", "sonnet", "opus", "fable")}}))
            backends = tmp / "backends.tsv"
            backends.write_text(f"local\t{self.backend.server_port}\trepo:quant\n")
            config = tmp / "gateway.json"
            env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "DISABLE_"))}
            env["PATH"] = str(tmp) + ":" + os.environ["PATH"]
            args = [sys.executable, str(ROOT / "lib" / "claude_local_budget.py"), "--requested", "262144",
                    "--check-claude", "--plan", str(plan), "--backends", str(backends), "--config", str(config),
                    "--", "--resume", "SESSION", "--model", "claude-opus-5"]
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for text in ("2.1.221", "requested context=262144", "effective per-slot=131072",
                         "window=114688", "auto-compaction window=106496", "max output=8192",
                         "prompt reserve=8192", "Resume warning", "recognized Claude IDs"):
                self.assertIn(text, result.stdout)
            self.assertIn("Context mismatch", result.stderr)
            cfg = json.loads(config.read_text())
            self.assertEqual(cfg["budget"]["capacity"], 131072)
            self.assertEqual(len(cfg["roles"]), 4)
            self.assertEqual(len(self.requests), 1)
            self.assertEqual(self.requests[0][0], "/v1/messages/count_tokens")


DUAL_3090_SMI = """#!/usr/bin/env bash
case "$*" in
  *index,name,memory.total,memory.free,compute_cap,pci.bus_id*)
    printf '0, NVIDIA GeForce RTX 3090, 24576, 24000, 8.6, 00000000:01:00.0\\n1, NVIDIA GeForce RTX 3090, 24576, 24000, 8.6, 00000000:02:00.0\\n';;
  *pcie.link*) printf '4, 16\\n4, 16\\n';;
  *name,memory.free*) printf 'NVIDIA GeForce RTX 3090, 24000\\nNVIDIA GeForce RTX 3090, 24000\\n';;
  *index*) printf '0\\n1\\n';;
esac
"""


class ReplicaLaunchArgumentTests(unittest.TestCase):
    """--agents/--slots must reach the core launcher intact through every wrapper."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "lib").mkdir()
        (self.root / "bin").mkdir()
        for name in ("lib/claude_local_entry.sh", "lib/pushbutton_folders.sh", "claude-local-safe"):
            (self.root / name).write_text((ROOT / name).read_text())
            (self.root / name).chmod(0o755)
        self.record = self.root / "record.jsonl"
        core = self.root / "claude-local"
        core.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                        "startup=os.environ.get('CLAUDE_LOCAL_STARTUP_ONLY')=='1'\n"
                        f"open({str(self.record)!r},'a').write(json.dumps({{'startup':startup,'argv':sys.argv[1:]}})+'\\n')\n"
                        "sys.exit(125 if startup else 0)\n")
        core.chmod(0o755)
        smi = self.root / "bin" / "nvidia-smi"
        smi.write_text(DUAL_3090_SMI)
        smi.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "DISABLE_"))}
        self.env.update(HOME=str(self.root / "home"), PATH=str(self.root / "bin") + ":" + os.environ["PATH"],
                        CLAUDE_LOCAL_STATE=str(self.root / "state"), CLAUDE_LOCAL_CACHE=str(self.root / "cache"),
                        PUSHBUTTON_CONFIG_DIR=str(self.root / "config"))

    def launched(self, *args):
        result = subprocess.run(["bash", str(self.root / "lib/claude_local_entry.sh"), *args],
                                env=self.env, input="", capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.record.read_text().splitlines()]
        self.record.unlink()
        self.assertTrue(calls[-1]["startup"] is False, calls)
        return calls[-1]["argv"]

    def test_replica_flags_do_not_leak_wrapper_options_or_inject_models(self):
        cases = {
            ("q38@slots=2", "--agents", "2", "--local-no-web"): ["q38@slots=2", "--agents", "2"],
            ("--agents", "2", "q38@slots=2", "--local-no-web"): ["--agents", "2", "q38@slots=2"],
            ("--slots", "2", "--agents", "2", "q38", "--local-no-web"): ["--slots", "2", "--agents", "2", "q38"],
            ("--slots=2", "--agents=2", "q38", "--local-no-web"): ["--slots=2", "--agents=2", "q38"],
            ("--local-context", "131072", "q38@slots=2", "--agents", "2", "--local-no-web"):
                ["--local-context", "131072", "q38@slots=2", "--agents", "2"],
        }
        for args, expected in cases.items():
            with self.subTest(args=args):
                self.assertEqual(self.launched(*args), expected)

    def test_dual_3090_defaults_still_apply_without_model_selector(self):
        self.assertEqual(self.launched("--agents", "2", "--local-no-web"),
                         ["qwen3.8:27b", "qwen3.6:35b", "--agents", "2"])

    def test_json_agents_still_pass_through_to_claude(self):
        argv = self.launched("q38", "--local-no-web", "--agents", '{"a":{}}', "--verbose")
        self.assertEqual(argv, ["q38", "--agents", '{"a":{}}', "--verbose"])

    def test_core_accepts_equals_forms_for_replicas_and_slots(self):
        env = dict(self.env, PATH=str(self.root / "bin") + ":" + os.environ["PATH"])
        result = subprocess.run(["bash", str(ROOT / "claude-local"), "q38", "--slots=2", "--agents=2",
                                 "--local-dry-run", "--quiet"],
                                env=env, input="", capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("local-qwen38-27b-1", result.stdout)
        self.assertIn("local-qwen38-27b-2", result.stdout)
        self.assertEqual(result.stdout.count("slots=2 "), 2)
        result = subprocess.run(["bash", str(ROOT / "claude-local"), "q38", "--agents=9", "--local-dry-run"],
                                env=env, input="", capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--agents must be between 1 and 4", result.stderr)


if __name__ == "__main__":
    unittest.main()
