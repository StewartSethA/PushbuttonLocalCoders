"""Startup routing tests requiring neither GPUs nor installed coding clients."""
import json
import os
import pathlib
import pty
import select
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTENDS = ("coder-local", "claude-local", "hermes-local")
WRAPPERS = ("qwen-local", "opencode-local", "deepseek-local", "mini-swe-local")


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir="/tmp")
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
        for asset in ("pushbutton_metrics.py", "coder_local_plan.py", "claude_local_plan.py"):
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
