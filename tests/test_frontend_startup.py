"""Startup routing tests requiring neither GPUs nor installed coding clients."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import pathlib
import pty
import re
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_DIR = pathlib.Path("/tmp/pushbutton-test-fixtures")
FRONTENDS = ("coder-local", "claude-local", "hermes-local")
WRAPPERS = ("qwen-local", "opencode-local", "deepseek-local", "mini-swe-local")


def shell_function(frontend, name):
    text = (ROOT / frontend).read_text()
    match = re.search(r"^" + name + r"\(\)\s*\{.*?^\}", text, re.M | re.S)
    if not match:
        raise AssertionError(f"missing {frontend}:{name}")
    return match.group(0)


class CapacityIntegrationTests(unittest.TestCase):
    def setUp(self):
        FIXTURE_DIR.mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=FIXTURE_DIR)
        self.addCleanup(self.scratch.cleanup)
        self.root = pathlib.Path(self.scratch.name)
        self.capacity = dict(context=8192, slots=3, output_tokens=1024,
                             client_context=7000, compact_trigger=4500,
                             input_tokens=5720, safety_tokens=256,
                             min_tps=None, admission_limit=2)
        self.server = dict(id="model-a", worker=1, model="qwen3.8:27b",
                           cuda_visible_devices="0", multi_gpu=False,
                           gpus=[], required_mib=16384,
                           vram_limit_mib_per_gpu=None, capacity=self.capacity,
                           profile=dict(hf_spec="repo:quant", kv_k="q8_0",
                                        kv_v="q4_0", batch=512, ubatch=256,
                                        flash_attn="on", template="", extra_env={},
                                        quality=1, quant="Q4_K_M"))
        self.plan = self.root / "plan.json"
        self.plan.write_text(json.dumps(dict(context=262144, agents=1, unused_gpus=[],
                                             roles=dict(haiku="qwen3.8:27b", sonnet="qwen3.8:27b",
                                                        opus="qwen3.8:27b", fable="qwen3.8:27b"),
                                             servers=[self.server],
                                             workers=[self.server],
                                             role_ids=dict(haiku="model-a", sonnet="model-a",
                                                           opus="model-a", fable="model-a"))))

    def run_functions(self, frontend, names, body):
        prelude = f"""set -euo pipefail
STATE_DIR={shlex.quote(str(self.root))}
CACHE_DIR="$STATE_DIR/cache"
ROOT={shlex.quote(str(ROOT))}
PLAN_FILE={shlex.quote(str(self.plan))}
BUDGET_PY="$ROOT/lib/claude_local_budget.py"
CLAUDE_ARGS=()
PORT_BASE=18000
CTX=262144
CLIENT_CTX=200000
WEB_SEARCH_BACKEND=exa
WEB_EXTRACT_BACKEND=firecrawl
PIDS=()
ALLOW_OFFLOAD=0
LLAMA_SERVER=llama-server
CAPTURE="$STATE_DIR/capture"
"""
        command = prelude + "\n".join(shell_function(frontend, name) for name in names) + "\n" + body
        result = subprocess.run(["bash", "-c", command], text=True, capture_output=True,
                                timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_all_backends_launch_total_context_and_parallel_slots_and_keep_capacity(self):
        for frontend in FRONTENDS:
            with self.subTest(frontend=frontend):
                row_function = "worker_rows" if frontend == "coder-local" else "server_rows"
                body = """
say(){ :; }; good(){ :; }; die(){ echo "$*" >&2; exit 1; }
validate_download_space(){ :; }
download_model_fast(){ printf -v "$3" '%s' model.gguf; }
free_port(){ echo 18001; }; curl(){ return 0; }
verify_capacity(){ printf '%s' "$2" >"$STATE_DIR/verified"; }
start_log_follower(){ echo ''; }; stop_log_follower(){ :; }
env(){ printf '%s\\n' "$@" >"$CAPTURE"; }
start_backends
wait
"""
                self.run_functions(frontend, [row_function, "start_backends"], body)
                args = (self.root / "capture").read_text().splitlines()
                self.assertEqual(args[args.index("-c") + 1], "24576")
                self.assertEqual(args[args.index("-np") + 1], "3")
                self.assertEqual(args[args.index("--cache-type-k") + 1], "q8_0")
                self.assertIn("--no-context-shift", args)
                self.assertEqual(json.loads((self.root / "verified").read_text()), self.capacity)
                registry = next(self.root.glob("coder-backends.*.tsv")) if frontend == "coder-local" else \
                    next(self.root.glob("backends.*" if frontend == "claude-local" else "hermes-backends.*"))
                self.assertEqual(json.loads(registry.read_text().strip().split("\t")[-1]), self.capacity)

    def test_gateway_routes_carry_capacity(self):
        capacity = {**self.capacity, "context":262144, "client_context":200000,
                    "compact_trigger":150000, "input_tokens":198720}
        plan = json.loads(self.plan.read_text())
        plan["servers"][0]["capacity"] = capacity
        self.plan.write_text(json.dumps(plan))

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.reply(dict(default_generation_settings=dict(n_ctx=262144), total_slots=3))

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.reply(dict(input_tokens=20))

            def reply(self, data):
                raw = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        claude = self.root / "claude"
        claude.write_text(
            "#!/bin/sh\n"
            "# CLAUDE_CODE_AUTO_COMPACT_WINDOW CLAUDE_CODE_MAX_CONTEXT_TOKENS CLAUDE_CODE_MAX_OUTPUT_TOKENS\n"
            'case "$1" in --version) echo "2.1.221";; --help) echo "--autocompact --setting-sources --settings";; esac\n')
        claude.chmod(0o755)
        backends = self.root / "backends"
        backends.write_text(f"model-a\t{server.server_port}\trepo:quant\t" + json.dumps(capacity) + "\n")
        self.run_functions("claude-local", ["write_gateway_config"],
                           f'export PATH={shlex.quote(str(self.root))}:"$PATH"\n'
                           f'BACKENDS_TSV={shlex.quote(str(backends))}\nwrite_gateway_config')
        config = json.loads(next(self.root.glob("gateway.*.json")).read_text())
        self.assertTrue(all(route["capacity"] == {**capacity, "no_context_shift":True}
                            for route in config["roles"].values()))
        self.assertEqual(config["budget"]["max_output_tokens"], 1024)
        self.assertEqual(config["budget"]["compact_window"], 150000)
        self.assertTrue(all(route["context_capacity"] == 262144 and route["prompt_reserve"] == 8192
                            for route in config["roles"].values()))

    def test_startup_requested_throughput_warns_not_proven(self):
        plan = json.loads(self.plan.read_text())
        for row in plan["workers"] + plan["servers"]:
            row["capacity"]["min_tps"] = 25
        self.plan.write_text(json.dumps(plan))
        for frontend in FRONTENDS:
            with self.subTest(frontend=frontend):
                result = self.run_functions(frontend, ["print_plan"], "print_plan")
                self.assertIn("min_tps=25 is requested, not proven", result.stdout)

    def test_qwen_model_context_and_output_are_per_model_and_guarded(self):
        backends = self.root / "backends"
        second = {**self.capacity, "context":16384, "client_context":12000, "output_tokens":2048}
        backends.write_text("1\tmodel-a\t18001\trepo:a\t" + json.dumps(self.capacity) + "\n" +
                            "2\tmodel-b\t18002\trepo:b\t" + json.dumps(second) + "\n")
        self.run_functions("coder-local", ["configure_qwen"],
                           f'BACKENDS={shlex.quote(str(backends))}\nCAPACITY_PROXY_PORT=19000\nsay(){{ :; }}\nconfigure_qwen')
        config = json.loads((self.root / "frontends/qwen/settings.json").read_text())
        models = config["modelProviders"]["openai"]
        self.assertEqual([m["generationConfig"]["contextWindowSize"] for m in models], [7000, 12000])
        self.assertEqual([m["generationConfig"]["samplingParams"]["max_tokens"] for m in models], [1024, 2048])
        for model, capacity in zip(models, (self.capacity, second)):
            self.assertLess(capacity["compact_trigger"], model["generationConfig"]["contextWindowSize"])
            self.assertLess(model["generationConfig"]["contextWindowSize"], capacity["context"])
        self.assertTrue(all(m["baseUrl"] == "http://127.0.0.1:19000/v1" for m in models))
        self.assertEqual(config["context"]["autoCompactThreshold"],
                         min(self.capacity["compact_trigger"]/7000, second["compact_trigger"]/12000))

    def test_hermes_profile_uses_own_context(self):
        result = self.run_functions("hermes-local", ["configure_profile"],
                                    'hermes(){ printf "%s\\n" "$*" >>"$CAPTURE"; }\n'
                                    'configure_profile fast model-a http://127.0.0.1:19000/v1 low')
        self.assertEqual(result.returncode, 0)
        self.assertIn("config set model.context_length 7000", (self.root / "capture").read_text())
        self.assertLess(self.capacity["compact_trigger"], self.capacity["client_context"])
        self.assertLess(self.capacity["client_context"], self.capacity["context"])
        self.assertIn("config set compression.enabled true", (self.root / "capture").read_text())
        self.assertIn("config set compression.threshold_tokens 4500", (self.root / "capture").read_text())
        self.assertIn("config set model.provider custom:pushbutton", (self.root / "capture").read_text())
        self.assertIn("config set providers.pushbutton.base_url http://127.0.0.1:19000/v1",
                      (self.root / "capture").read_text())
        self.assertIn("config set providers.pushbutton.transport chat_completions",
                      (self.root / "capture").read_text())
        self.assertIn("config set providers.pushbutton.extra_body.max_tokens 1024",
                      (self.root / "capture").read_text())

    def test_claude_uses_smallest_shared_output_budget(self):
        plan = json.loads(self.plan.read_text())
        second = {**self.server, "id":"model-b", "capacity":{**self.capacity, "output_tokens":512}}
        plan["servers"].append(second)
        self.plan.write_text(json.dumps(plan))
        config = self.root / "gateway.json"
        config.write_text(json.dumps({"budget":{"client_context":131072,
                                               "compact_window":100000,
                                               "max_output_tokens":512}}))
        result = self.run_functions("claude-local", ["run_claude"],
            f'GATEWAY_CONFIG={shlex.quote(str(config))}\n'
            'GATEWAY_PORT=19000\nENABLE_TEAMS=0\nCLAUDE_ARGS=()\n'
            'say(){ :; }; good(){ :; }; contains_claude_flag(){ return 0; }\n'
            'role_rows(){ printf "sonnet\\tmodel-a\\n"; }\n'
            'claude(){ printf "BUDGET:%s/%s/%s\\n" "$CLAUDE_CODE_MAX_CONTEXT_TOKENS" "$CLAUDE_CODE_MAX_OUTPUT_TOKENS" "$CLAUDE_CODE_AUTO_COMPACT_WINDOW"; }\n'
            'run_claude')
        self.assertIn("BUDGET:131072/512/100000", result.stdout)

    def test_startup_lock_release_preserves_stderr(self):
        result = self.run_functions("claude-local", ["release_startup_lock"],
            'exec {STARTUP_LOCK_FD}>"$STATE_DIR/startup.lock"\n'
            'release_startup_lock\nprintf "visible launch error\\n" >&2')
        self.assertIn("visible launch error", result.stderr)

    def test_silent_claude_failure_reports_status_in_red(self):
        config = self.root / "gateway.json"
        config.write_text(json.dumps({"budget":{"client_context":7000,
                                               "compact_window":4500,
                                               "max_output_tokens":1024}}))
        result = self.run_functions("claude-local",
            ["say", "good", "debug", "warn", "show_runtime_errors", "run_claude"],
            f'GATEWAY_CONFIG={shlex.quote(str(config))}\n'
            'GATEWAY_PORT=19000\nENABLE_TEAMS=0\nVERBOSE=1\n'
            'contains_claude_flag(){ return 0; }\nrole_rows(){ :; }\n'
            'claude(){ return 42; }\n'
            'if run_claude; then exit 1; else rc=$?; [[ "$rc" == 42 ]]; fi')
        self.assertIn("\033[1;31m[claude-local] Claude Code exited with status 42", result.stderr)
        self.assertIn("\033[1;32m[claude-local] All local services ready.", result.stdout)
        self.assertIn("\033[1;33m[claude-local] Claude Code ->", result.stdout)
        self.assertIn("\033[0;37m[claude-local] Runtime logs:", result.stderr)

    def test_readiness_refuses_reduced_or_unproven_per_slot_context(self):
        props = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                raw = json.dumps(props).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        cases = [
            (dict(default_generation_settings=dict(n_ctx=8192), total_slots=3), True),
            (dict(default_generation_settings=dict(n_ctx=4096), total_slots=3), False),
            (dict(default_generation_settings=dict(n_ctx=8192), total_slots=1), False),
            (dict(default_generation_settings=dict(n_ctx=8192)), False),
            (dict(n_ctx=24576, total_slots=3), False),
            ({}, False),
        ]
        for frontend in FRONTENDS:
            for payload, accepted in cases:
                with self.subTest(frontend=frontend, props=payload):
                    props.clear()
                    props.update(payload)
                    command = "ROOT=" + shlex.quote(str(ROOT)) + "\n" + shell_function(frontend, "verify_capacity") + "\nverify_capacity " + \
                        str(server.server_port) + " " + shlex.quote(json.dumps(self.capacity))
                    result = subprocess.run(["bash", "-c", command], text=True,
                                            capture_output=True, timeout=5)
                    self.assertEqual(result.returncode == 0, accepted, result.stderr)

    def test_direct_frontends_start_guard_with_capacity_routes(self):
        (self.root / "logs").mkdir()
        for frontend in ("coder-local", "hermes-local"):
            with self.subTest(frontend=frontend):
                backends = self.root / "backends"
                row = "model-a\t18001\trepo:a\t" + json.dumps(self.capacity) + "\n"
                if frontend == "coder-local":
                    row = "1\t" + row
                backends.write_text(row)
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", 0))
                    port = probe.getsockname()[1]
                registry_var = "BACKENDS" if frontend == "coder-local" else "BACKENDS_TSV"
                result = self.run_functions(frontend, ["free_port", "start_capacity_proxy"],
                    f'{registry_var}={shlex.quote(str(backends))}\nPORT_BASE={port}\n'
                    'say(){ :; }; die(){ echo "$*" >&2; exit 1; }\n'
                    'trap \'for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done\' EXIT\n'
                    'start_capacity_proxy\n' +
                    ('[[ "$PUSHBUTTON_CAPACITY_CONFIG" == "$CAPACITY_PROXY_CONFIG" ]]\n'
                     '[[ "$(python3 -c \'import os; print(os.environ["PUSHBUTTON_CAPACITY_CONFIG"])\')" == "$CAPACITY_PROXY_CONFIG" ]]\n'
                     if frontend == "coder-local" else '') +
                    'curl -fsS "http://127.0.0.1:$CAPACITY_PROXY_PORT/v1/models"')
                models = json.loads(result.stdout)["data"]
                self.assertEqual([model["id"] for model in models], ["model-a"])
                config = json.loads(next(self.root.glob(
                    "coder-capacity.*.json" if frontend == "coder-local" else "hermes-capacity.*.json")).read_text())
                self.assertEqual(config["routes"]["model-a"]["capacity"],
                                 {**self.capacity, "no_context_shift":True})


class StartupTests(unittest.TestCase):
    def setUp(self):
        FIXTURE_DIR.mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=FIXTURE_DIR)
        self.addCleanup(self.scratch.cleanup)
        self.root = pathlib.Path(self.scratch.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("bash", "cat", "dirname", "basename", "readlink", "mkdir", "chmod", "ln", "mv", "rm", "stat", "df"):
            (self.bin / name).symlink_to(shutil.which(name))
        (self.bin / "python3").symlink_to(sys.executable)
        for name in FRONTENDS + WRAPPERS + ("claude-local-safe",):
            shutil.copy2(ROOT / name, self.root / name)
        self.env = dict(os.environ, PATH=str(self.bin), HOME=str(self.root / "home"),
                        CLAUDE_LOCAL_STATE=str(self.root / "state"),
                        CLAUDE_LOCAL_CACHE=str(self.root / "cache"),
                        CLAUDE_LOCAL_CONTEXT="262144", CLAUDE_LOCAL_CLIENT_CONTEXT="200000")
        self.cwd = self.root / "project with spaces"
        self.cwd.mkdir()

    def run_script(self, name, args=()):
        return subprocess.run(["/bin/bash", str(self.root / name), *args],
                              cwd=self.cwd, env=self.env, input="", text=True,
                              capture_output=True, timeout=5)

    def tty_run(self, name, args=(), cancel=False):
        master, slave = pty.openpty()
        proc = subprocess.Popen(["/bin/bash", str(self.root / name), *args],
                                cwd=self.cwd, env=self.env, stdin=slave,
                                stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        output = b""
        deadline = time.monotonic() + 5
        try:
            while time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    output += data
                    if cancel and b"Choose model" in output:
                        os.write(master, b"\x04")
                        cancel = False
                if proc.poll() is not None:
                    break
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            os.close(master)
        return proc.returncode, output.decode()

    def selector_recorder(self):
        (self.root / "pushbutton-select").write_text(
            "import json, os, sys\n"
            "print('RECORDER:' + json.dumps({'argv':sys.argv[1:], 'cwd':os.getcwd(), "
            "'preflight':os.environ.get('CODER_LOCAL_STARTUP_ONLY'), "
            "'claude_preflight':os.environ.get('CLAUDE_LOCAL_STARTUP_ONLY'), "
            "'web_mcp':os.environ.get('CLAUDE_LOCAL_WEB_MCP'), "
            "'disable_telemetry':os.environ.get('DISABLE_TELEMETRY'), "
            "'disable_error_reporting':os.environ.get('DISABLE_ERROR_REPORTING'), "
            "'do_not_track':os.environ.get('DO_NOT_TRACK'), "
            "'otel_disabled':os.environ.get('OTEL_SDK_DISABLED'), "
            "'swarm':os.environ.get('MINI_SWE_SWARM_TASK')}))\n")

    def recorded(self, name, args=()):
        rc, out = self.tty_run(name, args)
        self.assertEqual(rc, 0, out)
        return json.loads(out.split("RECORDER:", 1)[1].splitlines()[0])

    def test_non_tty_no_models_and_help_need_no_assets_or_toolchain(self):
        for name in FRONTENDS + WRAPPERS:
            with self.subTest(frontend=name):
                result = self.run_script(name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Usage:", result.stderr)
                self.assertIn("interactive terminal", result.stderr)
                self.assertFalse((self.root / "state").exists())
                result = self.run_script(name, ["--help"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--select", result.stdout)
                if name in WRAPPERS:
                    result = self.run_script(name, ["q38", "--help"])
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Usage:", result.stdout)
                    self.assertNotIn("Installing", result.stdout)
                result = self.run_script(name, ["--select", "q38"])
                self.assertIn("interactive terminal", result.stderr)
                self.assertNotIn("required", result.stderr)
                if name in ("coder-local", *WRAPPERS):
                    result = self.run_script(name, ["--plan-only"])
                    self.assertIn("interactive terminal", result.stderr)
                    self.assertNotIn("Installing", result.stdout)

    def test_no_model_tty_routes_before_provisioning(self):
        self.selector_recorder()
        for name in FRONTENDS + WRAPPERS:
            with self.subTest(frontend=name):
                record = self.recorded(name)
                self.assertEqual(record["cwd"], str(self.cwd))
                self.assertEqual(record["argv"][:3], ["--frontend", name, "--menu"])
                self.assertIsNone(record["preflight"])
                self.assertFalse((self.root / "state").exists())

    def test_no_telemetry_flag_reaches_every_frontend_without_disabling_it(self):
        self.selector_recorder()
        for name in FRONTENDS + WRAPPERS:
            with self.subTest(frontend=name):
                record = self.recorded(name, ["--no-telemetry"])
                self.assertEqual(record["disable_telemetry"], "1")
                self.assertEqual(record["disable_error_reporting"], "1")
                self.assertEqual(record["do_not_track"], "1")
                self.assertEqual(record["otel_disabled"], "true")
                self.assertIn("--launch-arg=--no-telemetry", record["argv"])

    def test_default_client_context_is_left_to_selector_clamping(self):
        self.selector_recorder()
        self.env.pop("CLAUDE_LOCAL_CLIENT_CONTEXT")
        for name in FRONTENDS + WRAPPERS:
            with self.subTest(frontend=name):
                argv = self.recorded(name, ["--local-context", "32768"])["argv"]
                self.assertNotIn("--client-context", argv)
                self.assertIn("32768", argv)
                argv = self.recorded(name, ["--local-context", "32768",
                                           "--local-client-context", "16000"])["argv"]
                self.assertEqual(argv[argv.index("--client-context") + 1], "16000")
        self.env["CLAUDE_LOCAL_CLIENT_CONTEXT"] = "12000"
        for name in FRONTENDS + WRAPPERS:
            with self.subTest(environment_frontend=name):
                argv = self.recorded(name, ["--local-context", "32768"])["argv"]
                self.assertEqual(argv[argv.index("--client-context") + 1], "12000")

    def test_controlled_contexts_and_individual_client_argv(self):
        self.selector_recorder()
        cases = {
            "coder-local": (["--agents", "3", "--placement-config", "placement with spaces.json",
                             "--resume", "--", "--prompt", "a b", ""],
                            ["--agents", "3", "--placement-config", "placement with spaces.json"],
                            ["--resume", "--", "--prompt", "a b", ""]),
            "claude-local": (["--local-no-teams", "--", "--resume", "session with spaces",
                              "--agents", '{"role": "shared"}', "a b", ""],
                             [], ["--local-no-teams", "--", "--resume", "session with spaces",
                                  "--agents", '{"role": "shared"}', "a b", ""]),
            "hermes-local": (["--tui", "--resume", "--bot-prefix", "two words"],
                             [], ["--tui", "--resume", "--bot-prefix", "two words"]),
        }
        for name, (extras, controlled, forwarded) in cases.items():
            with self.subTest(frontend=name):
                args = ["--select", "q38", "--local-context", "32768",
                        "--local-client-context", "16000", *extras]
                record = self.recorded(name, args)
                self.assertEqual(record["argv"],
                                 ["--frontend", name, "--menu", "--context", "32768",
                                  "--client-context", "16000", *controlled, "q38",
                                  *["--launch-arg=" + x for x in forwarded]])

    def test_per_model_capacity_suffix_and_global_slots_survive_selection(self):
        self.selector_recorder()
        model = "q38@context=8192,slots=3,output=1024,client_context=7000,compact=4000,kv_k=q8_0"
        for frontend in FRONTENDS:
            with self.subTest(frontend=frontend):
                argv = self.recorded(frontend, ["--select", model, "--slots", "2"])["argv"]
                self.assertIn(model, argv)
                self.assertEqual(argv[argv.index("--slots") + 1], "2")

    def test_claude_numeric_agents_survives_selection(self):
        self.selector_recorder()
        for frontend in ("claude-local", "claude-local-safe"):
            argv = self.recorded(frontend, ["--select", "q38", "--agents", "2"])["argv"]
            self.assertEqual(argv[argv.index("--agents") + 1], "2")
            self.assertNotIn("--launch-arg=--agents", argv)

    def test_qwen_alias_preserves_frontend(self):
        self.selector_recorder()
        self.assertEqual(self.recorded("qwen-local")["argv"][:2],
                         ["--frontend", "qwen-local"])
        self.assertEqual(self.recorded("coder-local", ["--frontend", "qwen"])["argv"][:2],
                         ["--frontend", "qwen-local"])

    def test_wrapper_select_preserves_replica_controls_and_client_argv(self):
        self.selector_recorder()
        for name in WRAPPERS:
            with self.subTest(frontend=name):
                record = self.recorded(name, ["--select", "q38", "--agents", "2",
                                             "--local-context", "32768",
                                             "--local-client-context", "16000",
                                             "--resume", "--", "--prompt", "a b", ""])
                self.assertEqual(record["argv"],
                                 ["--frontend", name, "--menu", "--context", "32768",
                                  "--client-context", "16000", "--agents", "2", "q38",
                                  "--launch-arg=--resume",
                                  "--launch-arg=--", "--launch-arg=--prompt",
                                  "--launch-arg=a b", "--launch-arg="])
                self.assertIsNone(record["preflight"])
                self.assertFalse((self.root / "state").exists())

    def test_mini_swarm_survives_menu_without_installing(self):
        self.selector_recorder()
        record = self.recorded("mini-swe-local", ["--swarm", "task with spaces"])
        self.assertEqual(record["swarm"], "task with spaces")
        self.assertEqual(record["argv"][:2], ["--frontend", "mini-swe-local"])

    def test_no_parallel_survives_selection_on_all_frontends(self):
        self.selector_recorder()
        for name in FRONTENDS + WRAPPERS:
            with self.subTest(frontend=name):
                record = self.recorded(name, ["--no-parallel"])
                self.assertIn("--launch-arg=--no-parallel", record["argv"])
                self.assertFalse((self.root / "state").exists())

    def test_storage_options_survive_selection_without_creating_folders(self):
        self.selector_recorder()
        cache = self.root / "chosen model cache"
        for name in FRONTENDS + WRAPPERS:
            with self.subTest(frontend=name):
                record = self.recorded(name, ["--select", "q38", "--quiet",
                                             "--local-cache", str(cache)])
                for arg in ("--quiet", "--local-cache", str(cache)):
                    self.assertIn("--launch-arg=" + arg, record["argv"])
                self.assertFalse(cache.exists())
                self.assertFalse((self.root / "state").exists())

    def test_selector_uses_relocated_placement_without_writing_state(self):
        self.selector_recorder()
        (self.root / "lib").mkdir()
        shutil.copy2(ROOT / "lib/pushbutton_folders.sh",
                     self.root / "lib/pushbutton_folders.sh")
        config = self.root / "custom config"
        config.mkdir()
        folders = config / "folders.json"
        original = json.dumps({
            "schema_version": 1,
            "folders": {"state_dir": str(self.root / "state"),
                        "cache_dir": str(self.root / "cache"),
                        "config_dir": str(config)},
            "created_at": "2026-10-09T00:00:00+00:00",
        })
        folders.write_text(original)
        self.env["PUSHBUTTON_CONFIG_DIR"] = str(config)
        for name in ("coder-local", "qwen-local"):
            with self.subTest(frontend=name):
                argv = self.recorded(name, ["--select", "q38"])["argv"]
                self.assertEqual(argv[argv.index("--placement-config") + 1],
                                 str(config / "placement.json"))
                argv = self.recorded(name, ["--select", "q38", "--placement-config",
                                           "explicit-placement.json"])["argv"]
                self.assertEqual(argv[argv.index("--placement-config") + 1],
                                 "explicit-placement.json")
                self.assertEqual(folders.read_text(), original)
                self.assertFalse((self.root / "state").exists())
                self.assertFalse((self.root / "cache").exists())

    def test_safe_claude_wrapper_has_menu_and_help_before_gpu_probe(self):
        self.selector_recorder()
        nvidia = self.bin / "nvidia-smi"
        nvidia.write_text("#!/bin/bash\necho UNEXPECTED_GPU_PROBE >&2\nexit 91\n")
        nvidia.chmod(0o755)
        for args in ([], ["--resume"], ["--select", "q38"]):
            with self.subTest(args=args):
                record = self.recorded("claude-local-safe", args)
                self.assertEqual(record["argv"][:3], ["--frontend", "claude-local", "--menu"])
                result = self.run_script("claude-local-safe", args)
                self.assertIn("interactive terminal", result.stderr)
                self.assertNotIn("UNEXPECTED_GPU_PROBE", result.stderr)
        result = self.run_script("claude-local-safe", ["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("--select", result.stdout)
        self.assertNotIn("UNEXPECTED_GPU_PROBE", result.stderr)

    def test_explicit_wrapper_model_argv_is_unchanged(self):
        coder = self.root / "coder-local"
        coder.write_text(
            "#!" + sys.executable + "\nimport json, os, sys\n"
            "if os.environ.get('CODER_LOCAL_STARTUP_ONLY') == '1': sys.exit(125)\n"
            "print('RECORDER:' + json.dumps({'argv':sys.argv[1:], 'cwd':os.getcwd(), "
            "'swarm':os.environ.get('MINI_SWE_SWARM_TASK')}))\n")
        coder.chmod(0o755)
        (self.root / "lib").mkdir()
        shutil.copy2(ROOT / "lib/pushbutton_folders.sh",
                     self.root / "lib/pushbutton_folders.sh")
        (self.root / "lib" / "claude_local_hostcc.sh").write_text(
            "claude_local_prepare_hostcc() { :; }\n")
        for tool in ("cmake", "opencode", "npm", "dsh", "pnpm", "mini", "grep"):
            executable = self.bin / tool
            executable.write_text("#!/bin/bash\nexit 0\n")
            executable.chmod(0o755)
        args = ["q38", "--agents", "2", "--local-context", "32768",
                "--local-client-context", "16000", "--", "--prompt", "a b", ""]
        for name in WRAPPERS:
            with self.subTest(frontend=name):
                result = self.run_script(name, args)
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(result.stdout.split("RECORDER:", 1)[1])
                self.assertEqual(record["argv"], ["--frontend", "qwen", *args])
                self.assertEqual(record["cwd"], str(self.cwd))

    def test_explicit_models_do_not_enter_selector(self):
        self.selector_recorder()
        for name in FRONTENDS + WRAPPERS:
            with self.subTest(frontend=name):
                result = self.run_script(name, ["q38"])
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("RECORDER:", result.stdout)
                self.assertNotIn("interactive terminal", result.stderr)

    def test_wrapper_plan_only_does_not_install_clients(self):
        for name in WRAPPERS:
            with self.subTest(frontend=name):
                result = self.run_script(name, ["q38", "--plan-only"])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("git is required", result.stderr)
                self.assertNotIn("Installing", result.stdout)
                self.assertFalse((self.root / "state").exists())

    def test_menu_preview_maps_to_selector_plan_only(self):
        self.selector_recorder()
        for name in ("coder-local", *WRAPPERS):
            with self.subTest(frontend=name):
                argv = self.recorded(name, ["--select", "q38", "--plan-only"])["argv"]
                self.assertIn("--plan-only", argv)
                self.assertNotIn("--launch-arg=--plan-only", argv)

    def test_no_model_tty_plan_only_never_prompts_or_provisions(self):
        shutil.copy2(ROOT / "pushbutton-select", self.root / "pushbutton-select")
        shutil.copytree(ROOT / "lib", self.root / "lib", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "configs", self.root / "configs")
        for name in ("coder-local", *WRAPPERS):
            with self.subTest(frontend=name):
                rc, out = self.tty_run(name, ["--plan-only"])
                self.assertEqual(rc, 0, out)
                self.assertNotIn("Action [", out)
                self.assertNotIn("Launch this", out)
                self.assertNotIn("Installing", out)
                self.assertFalse((self.root / "state").exists())

    def test_source_launchers_are_executable(self):
        for name in ("hermes-local", "mini-swe-local", "qwen-local", "claude-local-safe"):
            with self.subTest(frontend=name):
                result = subprocess.run([str(ROOT / name), "--help"], cwd=self.cwd,
                                        env=self.env, text=True, capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Usage:", result.stdout)

    def prepare_claude_entry(self):
        (self.root / "lib").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "lib" / "claude_local_entry.sh",
                     self.root / "lib" / "claude_local_entry.sh")
        return "lib/claude_local_entry.sh"

    def test_claude_entry_non_tty_startup_exits_before_policy_side_effects(self):
        entry = self.prepare_claude_entry()
        for args in ([], ["--resume"], ["--local-context", "32768", "--select"], ["--help"]):
            with self.subTest(args=args):
                result = self.run_script(entry, args)
                if args == ["--help"]:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("interactive terminal", result.stderr)
                self.assertFalse((self.root / "state").exists())

    def test_claude_entry_menu_preserves_resume_state_and_web_preferences(self):
        entry = self.prepare_claude_entry()
        self.selector_recorder()
        state = self.root / "custom state"
        record = self.recorded(entry, ["--local-state", str(state), "--local-no-web", "--resume"])
        self.assertEqual(record["argv"][-2:], ["--launch-arg=--", "--launch-arg=--resume"])
        self.assertEqual(record["web_mcp"], "0")
        self.assertIsNone(record["claude_preflight"])
        self.assertFalse(state.exists())

    def test_claude_entry_confirmed_models_keep_hardened_policy(self):
        entry = self.prepare_claude_entry()
        shutil.copy2(ROOT / "lib/pushbutton_folders.sh",
                     self.root / "lib/pushbutton_folders.sh")
        frontend = self.root / "claude-local"
        frontend.write_text(
            "#!" + sys.executable + "\nimport json, os, sys\n"
            "if os.environ.get('CLAUDE_LOCAL_STARTUP_ONLY') == '1': sys.exit(125)\n"
            "print('RECORDER:' + json.dumps({'argv':sys.argv[1:], "
            "'api_timeout':os.environ.get('API_TIMEOUT_MS'), "
            "'idle_timeout':os.environ.get('CLAUDE_STREAM_IDLE_TIMEOUT_MS')}))\n")
        frontend.chmod(0o755)
        self.env.pop("API_TIMEOUT_MS", None)
        self.env.pop("CLAUDE_STREAM_IDLE_TIMEOUT_MS", None)
        self.env.pop("CLAUDE_LOCAL_WEB_MCP", None)
        args = ["q38", "--resume", "session with spaces", "--", "prompt with spaces", ""]
        result = self.run_script(entry, args)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout.split("RECORDER:", 1)[1])
        self.assertEqual(record["argv"][:len(args)], args)
        self.assertEqual(record["argv"][len(args)], "--mcp-config")
        self.assertEqual(record["api_timeout"], "1800000")
        self.assertEqual(record["idle_timeout"], "900000")
        config = self.root / "state" / "web-mcp.json"
        self.assertTrue(config.is_file())
        config.unlink()
        result = self.run_script(entry, ["--local-no-web", *args])
        record = json.loads(result.stdout.split("RECORDER:", 1)[1])
        self.assertEqual(record["argv"], args)
        self.assertFalse(config.exists())

    def test_wrapper_plan_only_success_with_mocked_gpu_never_builds(self):
        shutil.copytree(ROOT / "lib", self.root / "lib", ignore=shutil.ignore_patterns("__pycache__"))
        (self.bin / "flock").symlink_to(shutil.which("flock"))
        for tool in ("git", "cmake", "ninja", "curl", "tar"):
            executable = self.bin / tool
            executable.write_text("#!/bin/bash\necho UNEXPECTED_PROVISIONING >&2\nexit 91\n")
            executable.chmod(0o755)
        nvidia = self.bin / "nvidia-smi"
        nvidia.write_text(
            "#!/bin/bash\ncase \"$*\" in\n"
            "  *index,name*) echo '0, NVIDIA GeForce RTX 3090, 24576, 24000, 8.6, 0000:01:00.0';;\n"
            "  *) echo '4, 16';;\nesac\n")
        nvidia.chmod(0o755)
        for name in WRAPPERS:
            with self.subTest(frontend=name):
                result = self.run_script(name, ["q38", "--plan-only"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Coding agents:", result.stdout)
                self.assertNotIn("UNEXPECTED_PROVISIONING", result.stderr)
                self.assertNotIn("Installing", result.stdout)
                self.assertFalse((self.root / "state" / "frontends").exists())

    def test_real_selector_pty_cancel_is_before_provisioning(self):
        shutil.copy2(ROOT / "pushbutton-select", self.root / "pushbutton-select")
        shutil.copytree(ROOT / "lib", self.root / "lib", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "configs", self.root / "configs")
        for name in FRONTENDS + WRAPPERS:
            with self.subTest(frontend=name):
                rc, out = self.tty_run(name, cancel=True)
                self.assertEqual(rc, 0, out)
                self.assertIn("Choose model", out)
                self.assertIn("Cancelled", out)
                self.assertNotIn("Traceback", out)
                self.assertFalse((self.root / "state").exists())

    def prepare_installer(self, name):
        installer = "install-" + name + ".sh"
        shutil.copy2(ROOT / installer, self.root / installer)
        template = self.root / "template"
        template.mkdir(exist_ok=True)
        for frontend in FRONTENDS:
            shutil.copy2(ROOT / frontend, template / frontend)
        shutil.copy2(ROOT / "qwen-local", template / "qwen-local")
        shutil.copy2(ROOT / "claude-local-safe", template / "claude-local-safe")
        for extra in ("opencode-local", "deepseek-local", "mini-swe-local",
                      "pushbutton", "pushbutton-instance", "pushbutton-broker", "pushbutton-proxy",
                      "pushbutton-backend", "pushbutton-bench", "pushbutton-observe", "pushbutton-select"):
            (template / extra).write_text("#!/bin/bash\necho UNEXPECTED_TOOL\nexit 91\n")
        (template / "lib").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "lib/pushbutton_folders.sh",
                     template / "lib/pushbutton_folders.sh")
        for asset in ("pushbutton_metrics.py", "cpu_platform.py", "coder_local_plan.py", "claude_local_plan.py",
                      "pushbutton_capacity.py", "pushbutton_request_budget.py", "pushbutton_capacity_proxy.py"):
            (template / "lib" / asset).touch()
        entry = template / "lib" / "claude_local_entry.sh"
        shutil.copy2(ROOT / "lib" / "claude_local_entry.sh", entry)
        (template / "configs").mkdir(exist_ok=True)
        (template / "configs" / "backend-registry.json").write_text("{}")
        (template / ".git").mkdir(exist_ok=True)
        git = self.bin / "git"
        git.write_text("#!" + sys.executable + "\nimport os, pathlib, shutil, sys\n"
                       "args = sys.argv[1:]\n"
                       "if 'clone' in args: shutil.copytree(os.environ['TEMPLATE'], args[-1])\n"
                       "elif 'checkout' in args:\n"
                       "  if '-f' not in args: sys.exit(1)\n"
                       "  dest = pathlib.Path(args[args.index('-C') + 1])\n"
                       "  shutil.copy2(pathlib.Path(os.environ['TEMPLATE']) / 'lib/claude_local_entry.sh', "
                       "dest / 'lib/claude_local_entry.sh')\n")
        git.chmod(0o755)
        self.env.update(PUSHBUTTON_DIR=str(self.root / "installed"), TEMPLATE=str(template))
        return installer

    def test_claude_installer_update_replaces_local_edits(self):
        installer = self.prepare_installer("claude-local")
        result = self.run_script(installer)
        self.assertEqual(result.returncode, 0, result.stderr)

        installed_entry = self.root / "installed/PushbuttonLocalCoders/lib/claude_local_entry.sh"
        installed_entry.write_text("local edit\n")
        result = self.run_script(installer)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(installed_entry.read_text(),
                         (self.root / "template/lib/claude_local_entry.sh").read_text())

    def test_piped_installer_noargs_installs_and_prints_help(self):
        for name in FRONTENDS:
            with self.subTest(frontend=name):
                installer = self.prepare_installer(name)
                result = self.run_script(installer)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Usage:", result.stdout)
                self.assertNotIn("UNEXPECTED_TOOL", result.stdout)
                self.assertFalse((self.root / "state/web-mcp.json").exists())
                self.assertFalse((self.root / "state/frontends").exists())
                if name != "hermes-local":
                    self.assertTrue((self.root / "home/.config/pushbutton-local/folders.json").exists())
                shutil.rmtree(self.root / "installed")

    def test_coder_installer_install_only_retains_runtime_installation(self):
        installer = self.prepare_installer("coder-local")
        result = self.run_script(installer, ["--install-only"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Installed unified runtime", result.stdout)
        self.assertNotIn("UNEXPECTED_TOOL", result.stdout)
        self.assertTrue((self.root / "home/.local/bin/pushbutton").exists())

    def test_installers_refuse_missing_capacity_runtime_modules(self):
        for frontend in FRONTENDS:
            for module in ("pushbutton_capacity.py", "pushbutton_request_budget.py", "pushbutton_capacity_proxy.py"):
                with self.subTest(frontend=frontend, module=module):
                    installer = self.prepare_installer(frontend)
                    (self.root / "template/lib" / module).unlink()
                    result = self.run_script(installer)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(module, result.stderr)
                    self.assertNotIn("UNEXPECTED_TOOL", result.stdout)
                    shutil.rmtree(self.root / "installed")

    def test_installed_claude_shim_options_preflight_before_mcp_state(self):
        installer = self.prepare_installer("claude-local")
        result = self.run_script(installer)
        self.assertEqual(result.returncode, 0, result.stderr)
        shim = "home/.local/bin/claude-local"
        for args in (["--resume"], ["--local-context", "32768", "--select"]):
            with self.subTest(args=args):
                result = self.run_script(shim, args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("interactive terminal", result.stderr)
                self.assertFalse((self.root / "state/web-mcp.json").exists())

    def test_piped_installer_select_and_help_exit_before_git(self):
        for name in FRONTENDS:
            with self.subTest(frontend=name):
                installer = "install-" + name + ".sh"
                shutil.copy2(ROOT / installer, self.root / installer)
                result = self.run_script(installer, ["--select"])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("interactive terminal", result.stderr)
                self.assertNotIn("git", result.stderr)
                result = self.run_script(installer, ["--help"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Usage:", result.stdout)

    def test_piped_installer_noargs_does_not_install_missing_git(self):
        for name in FRONTENDS:
            with self.subTest(frontend=name):
                installer = "install-" + name + ".sh"
                shutil.copy2(ROOT / installer, self.root / installer)
                result = self.run_script(installer)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("git is required", result.stderr)
                self.assertNotIn("sudo", result.stderr)
                self.assertNotIn("Installing", result.stdout)
                self.assertFalse((self.root / "home").exists())


if __name__ == "__main__":
    unittest.main()
