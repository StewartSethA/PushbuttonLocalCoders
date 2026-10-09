import json
import os
import pathlib
import shlex
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class StorageLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "lib").mkdir()
        self.env = dict(os.environ, HOME=str(self.root / "home"))
        self.helper = self.root / "lib/pushbutton_folders.sh"
        self.helper.write_text("""
load_folder_config() {
  STATE_DIR="$HOME/state"; CACHE_DIR="${CLAUDE_LOCAL_CACHE:-$HOME/cache}"
  CONFIG_DIR="$HOME/config"
}
check_folders_exist() { mkdir -p "$STATE_DIR" "$CACHE_DIR" "$CONFIG_DIR"; }
print_folder_summary() { printf 'State: %s\\nCache: %s (100 GiB free)\\nConfig: %s\\n' "$STATE_DIR" "$CACHE_DIR" "$CONFIG_DIR"; }
initialize_folders() { load_folder_config; check_folders_exist; }
validate_download_space() { echo 'Error: Model requires 89 GiB but only 0 GiB free on /test; use --local-cache /bigger/disk' >&2; return 2; }
""")
        (self.root / "lib/claude_local_plan.py").touch()
        (self.root / "lib/claude_local_gateway.py").touch()
        (self.root / "lib/pushbutton_download.sh").write_text(
            (ROOT / "lib/pushbutton_download.sh").read_text())

    def launcher(self, name):
        target = self.root / name
        target.write_text((ROOT / name).read_text())
        return target

    def test_folder_flags_exit_without_hardware_or_servers(self):
        for name in ("claude-local", "coder-local", "hermes-local"):
            for flag in ("--folders", "--system-info"):
                with self.subTest(name=name, flag=flag):
                    result = subprocess.run(
                        ["bash", str(self.launcher(name)), flag,
                         "--local-cache", str(self.root / "custom cache")],
                        env=self.env, text=True, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(str(self.root / "custom cache"), result.stdout)
                    self.assertIn("100 GiB free", result.stdout)
                    if name == "claude-local":
                        self.assertIn(str(self.root / "home/state/claude-config"), result.stdout)
                        self.assertIn("isolated local settings/sessions", result.stdout)

    def test_disk_full_stops_before_backend_and_prints_quoted_log_hint(self):
        for name, marker, rows in (
            ("claude-local", '\ncase "${1:-}" in', "server_rows"),
            ("coder-local", "\nwhile (($#));do", "worker_rows"),
            ("hermes-local", "\nwhile (($#)); do", "server_rows"),
        ):
            with self.subTest(name=name):
                launcher = self.launcher(name)
                definitions = launcher.read_text().split(marker)[0]
                called = self.root / "server-called"
                server = self.root / "fake-server"
                server.write_text(f"#!/bin/bash\ntouch {shlex.quote(str(called))}\n")
                server.chmod(0o755)
                row = ["local-test", "owner/model:Q4_K_M", "0", "0",
                       "q4_0", "q4_0", "512", "256", "on", "embedded", "", ""]
                if rows == "worker_rows":
                    row = ["1"] + row + [""]
                capacity = dict(context=262144, slots=1, output_tokens=8192,
                                client_context=200000, compact_trigger=150000,
                                input_tokens=190784, safety_tokens=1024,
                                admission_limit=1)
                row += [str(capacity["context"]), str(capacity["slots"]), json.dumps(capacity)]
                shell = self.root / f"test-{name}"
                shell.write_text(definitions + f"""
source "$ROOT/lib/pushbutton_folders.sh"
STATE_DIR={shlex.quote(str(self.root / "state with spaces"))}
CACHE_DIR={shlex.quote(str(self.root / "cache"))}
LLAMA_SERVER={shlex.quote(str(server))}
download_model_fast() {{ touch {shlex.quote(str(self.root / "download-called"))}; return 1; }}
free_port() {{ echo 20181; }}
{rows}() {{ printf '%s\\n' {shlex.quote(chr(31).join(row))}; }}
start_backends
""")
                result = subprocess.run(["bash", str(shell)], env=self.env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(called.exists())
                self.assertFalse((self.root / "download-called").exists())
                self.assertIn("89 GiB", result.stderr)
                self.assertIn("--local-cache /bigger/disk", result.stderr)
                # Once validation passes, the backend still prints a quoted log hint.
                shell.write_text(shell.read_text().replace("start_backends\n", """
validate_download_space() { return 0; }
download_model_fast() { printf -v "$3" '%s' '/mock/model.gguf'; }
curl() { return 0; }
verify_capacity() { return 0; }
start_backends
wait "${PIDS[@]}"
"""))
                result = subprocess.run(["bash", str(shell)], env=self.env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                command = next(line.strip() for line in result.stderr.splitlines()
                               if line.strip().startswith("tail -n 50"))
                args = shlex.split(command)
                self.assertEqual(args[:5], ["tail", "-n", "50", "-F", "--"])
                self.assertIn("state with spaces/logs/", args[5])
                called.unlink()

    def test_entry_keeps_frontend_arguments_and_runtime_policy(self):
        entry = self.root / "lib/claude_local_entry.sh"
        entry.write_text((ROOT / "lib/claude_local_entry.sh").read_text())
        launcher = self.root / "claude-local"
        launcher.write_text(
            '#!/bin/bash\n[[ "${CLAUDE_LOCAL_STARTUP_ONLY:-0}" != 1 ]] || exit 125\n'
            'printf "TIMEOUT=%s\\n" "${API_TIMEOUT_MS:-}"\n'
            'printf "ARG=%s\\n" "$@"\n')
        launcher.chmod(0o755)
        for args in (
            ["qwen3.8:27b", "--", "-p", "help"],
            ["qwen3.8:27b", "--", "-p", "--local-cache"],
            ["qwen3.8:27b", "-p", "--local-cache"],
        ):
            with self.subTest(args=args):
                env = dict(self.env)
                env.pop("API_TIMEOUT_MS", None)
                result = subprocess.run(["bash", str(entry), *args], env=env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("TIMEOUT=1800000", result.stdout)
                for arg in args:
                    self.assertIn("ARG=" + arg + "\n", result.stdout)
                self.assertIn("ARG=--mcp-config", result.stdout)

    def test_installed_shims_use_per_user_defaults_and_remember_custom_config(self):
        tree = self.root / "installed"
        (tree / "lib").mkdir(parents=True)
        entry = tree / "lib/claude_local_entry.sh"
        entry.write_text('#!/bin/bash\nprintf "%s\\n" "$PUSHBUTTON_CONFIG_DIR"\n')
        entry.chmod(0o755)
        for installer, function in (
            ("install-claude-local.sh", "write_shim() {"),
            ("install-coder-local.sh", "install_wrapper(){"),
        ):
            text = (ROOT / installer).read_text()
            start = text.index(function)
            definition = text[start:text.index("\n}\n", start) + 3]
            for custom in (False, True):
                with self.subTest(installer=installer, custom=custom):
                    config = (self.root / "custom config's $literal}" if custom
                              else pathlib.Path(self.env["HOME"]) / ".config/pushbutton-local")
                    shim = self.root / "command"
                    code = definition + "\n"
                    for name, value in (("INSTALL_ROOT", self.root), ("ROOT", self.root),
                                        ("DEST", tree), ("CONFIG_DIR", config)):
                        code += f"{name}={shlex.quote(str(value))}\n"
                    if installer == "install-claude-local.sh":
                        code += f"write_shim {shlex.quote(str(shim))}\n"
                    else:
                        code += (f"install_wrapper {shlex.quote(str(shim))} "
                                 f"{shlex.quote(shlex.quote(str(entry)))}\n")
                    built = subprocess.run(["bash", "-eu", "-c", code], env=self.env,
                                           text=True, capture_output=True)
                    self.assertEqual(built.returncode, 0, built.stderr)
                    runtime_env = dict(self.env, HOME=str(self.root / "another-user"))
                    runtime_env.pop("PUSHBUTTON_CONFIG_DIR", None)
                    result = subprocess.run(["bash", str(shim)], env=runtime_env,
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = (config if custom else self.root / "another-user"
                                / ".config/pushbutton-local")
                    self.assertEqual(result.stdout.strip(), str(expected))
                    runtime_env["PUSHBUTTON_CONFIG_DIR"] = str(self.root / "override")
                    result = subprocess.run(["bash", str(shim)], env=runtime_env,
                                            text=True, capture_output=True)
                    self.assertEqual(result.stdout.strip(), runtime_env["PUSHBUTTON_CONFIG_DIR"])


if __name__ == "__main__":
    unittest.main()
