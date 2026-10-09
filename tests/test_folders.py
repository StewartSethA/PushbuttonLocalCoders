"""Folder helper integration tests."""
import http.server
import json
import os
import pathlib
import pty
import select
import shutil
import subprocess
import tempfile
import threading
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER = ROOT / "lib" / "pushbutton_folders.sh"
GIB = 1024 ** 3


class FolderTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix=".folder-tests-", dir="/tmp")
        self.work = pathlib.Path(self.fixture.name).resolve()
        self.addCleanup(self.cleanup_fixture)
        self.home = self.work / "home"
        self.home.mkdir()
        self.state = self.home / ".local/share/pushbutton/claude-local"
        self.cache = self.home / ".cache/pushbutton/llama"
        self.config = self.home / ".config/pushbutton-local"
        self.env = os.environ.copy()
        for name in ("CLAUDE_LOCAL_STATE", "CLAUDE_LOCAL_CACHE", "PUSHBUTTON_CONFIG_DIR",
                     "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_ENDPOINT", "HF_HUB_OFFLINE",
                     "PUSHBUTTON_HF_OFFLINE"):
            self.env.pop(name, None)
        self.env.update(HOME=str(self.home), HF_ENDPOINT="http://127.0.0.1:1")

    def cleanup_fixture(self):
        # Tests intentionally remove write/search permissions, including as root.
        for root, dirs, files in os.walk(self.work):
            os.chmod(root, 0o700)
            for name in dirs:
                path = pathlib.Path(root) / name
                if not path.is_symlink():
                    os.chmod(path, 0o700)
        self.fixture.cleanup()

    def run_bash(self, code, env=None, check=True):
        environment = self.env.copy()
        if env:
            environment.update(env)
        cp = subprocess.run(["bash", "-c", 'set -eu; source "$1"; ' + code,
                             "folder-tests", str(HELPER)], cwd=ROOT, env=environment,
                            text=True, capture_output=True)
        if check and cp.returncode:
            self.fail(f"bash failed ({cp.returncode})\n{cp.stdout}\n{cp.stderr}")
        return cp

    def save_config(self, paths=None, destination=None):
        paths = paths or [self.work / "saved-state", self.work / "saved-cache", self.config]
        destination = destination or self.config
        destination.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 1,
                "folders": dict(zip(("state_dir", "cache_dir", "config_dir"), map(str, paths))),
                "created_at": "2026-01-01T00:00:00+00:00"}
        (destination / "folders.json").write_text(json.dumps(data))
        return data

    def metadata(self, entries=None, sha="b" * 40, repo="owner/repo"):
        entries = entries if entries is not None else [
            {"rfilename": "model-Q4_K_M.gguf", "size": 3 * GIB,
             "lfs": {"size": 3 * GIB, "sha256": "a" * 64}}]
        data = {"siblings": entries, "sha": sha}
        root = self.cache / ("models--" + repo.replace("/", "--"))
        root.mkdir(parents=True, exist_ok=True)
        (root / "metadata.json").write_text(json.dumps(data))
        return data, root

    def save_plan(self, data):
        plan_file = self.work / "plan.json"
        plan_file.write_text(json.dumps(data))
        return plan_file

    def mock_space(self, gib=20, mode="linux"):
        bin_dir = self.work / "bin"
        bin_dir.mkdir(exist_ok=True)
        stat_script = {
            "linux": f'if [ "$2" = "-c" ]; then echo "{gib * 1024} 1048576"; else exit 1; fi',
            "mac": f'if [ "$2" = "-c" ]; then exit 1; else echo "{gib * 1024} 1048576"; fi',
            "df": "exit 1",
        }[mode]
        # $1=-f, $2=-c on Linux; BSD's $2 is the format string.
        (bin_dir / "stat").write_text("#!/bin/sh\n" + stat_script + "\n")
        (bin_dir / "df").write_text(
            '#!/bin/sh\nprintf "Filesystem 1024-blocks Used Available Capacity Mounted on\\n'
            f'/dev/test 999999999 0 {gib * 1024 * 1024} 0%% /fixture-mount\\n"\n')
        for name in ("stat", "df"):
            (bin_dir / name).chmod(0o755)
        return {"PATH": str(bin_dir) + os.pathsep + self.env["PATH"]}

    def test_defaults_canonical_and_exported(self):
        cp = self.run_bash(
            'load_folder_config; printf "%s\\n" "$STATE_DIR" "$CACHE_DIR" "$CONFIG_DIR"; '
            'python3 -c \'import os; print(os.environ["CLAUDE_LOCAL_STATE"]); '
            'print(os.environ["CLAUDE_LOCAL_CACHE"]); print(os.environ["PUSHBUTTON_CONFIG_DIR"])\'')
        self.assertEqual(cp.stdout.splitlines(), list(map(str, [self.state, self.cache, self.config] * 2)))
        self.assertEqual(cp.stderr, "")
        self.assertFalse(self.config.exists())

    def test_saved_config_and_original_override_precedence(self):
        self.save_config()
        cp = self.run_bash('load_folder_config; printf "%s\\n" "$STATE_DIR" "$CACHE_DIR"',
                           {"CLAUDE_LOCAL_STATE": "~/env-state/../env-state"})
        self.assertEqual(cp.stdout.splitlines(),
                         [str(self.home / "env-state"), str(self.work / "saved-cache")])

    def test_exported_defaults_do_not_mask_config_on_reload(self):
        code = '''
load_folder_config
mkdir -p "$CONFIG_DIR"
python3 -c 'import json,os; p=os.environ["PUSHBUTTON_CONFIG_DIR"]; json.dump(
{"schema_version":1,"folders":{"state_dir":os.environ["HOME"]+"/new-state",
"cache_dir":os.environ["HOME"]+"/new-cache","config_dir":p},"created_at":"now"},
open(p+"/folders.json","w"))'
load_folder_config
printf "%s\\n" "$STATE_DIR" "$CACHE_DIR"
'''
        cp = self.run_bash(code)
        self.assertEqual(cp.stdout.splitlines(), [str(self.home / "new-state"), str(self.home / "new-cache")])

    def test_original_override_survives_reload(self):
        self.save_config()
        cp = self.run_bash('load_folder_config; load_folder_config; echo "$STATE_DIR"',
                           {"CLAUDE_LOCAL_STATE": "~/override"})
        self.assertEqual(cp.stdout.strip(), str(self.home / "override"))

    def test_initialize_persists_changed_original_environment_overrides(self):
        original = self.save_config()
        override = self.work / "runtime-override"
        env = dict(self.mock_space(100), CLAUDE_LOCAL_CACHE=str(override))
        cp = self.run_bash("initialize_folders 1", env)
        self.assertEqual(cp.stdout, "")
        self.assertEqual(cp.stderr, "")
        saved = json.loads((self.config / "folders.json").read_text())
        self.assertEqual(saved["folders"]["cache_dir"], str(override))
        self.assertEqual(saved["created_at"], original["created_at"])
        cp = self.run_bash('load_folder_config; echo "$CACHE_DIR"')
        self.assertEqual(cp.stdout.strip(), str(override))

    def test_initialize_preserves_corrupt_config_despite_environment_override(self):
        self.config.mkdir(parents=True)
        corrupt = '{"schema_version":1,"folders":'
        (self.config / "folders.json").write_text(corrupt)
        env = dict(self.mock_space(100), CLAUDE_LOCAL_CACHE=str(self.work / "override"))
        cp = self.run_bash("initialize_folders 1", env)
        self.assertIn("corrupt folder config", cp.stderr)
        self.assertEqual((self.config / "folders.json").read_text(), corrupt)

    def test_initialize_does_not_rewrite_unchanged_valid_config(self):
        self.save_config()
        path = self.config / "folders.json"
        before = path.stat()
        self.run_bash("initialize_folders 1", self.mock_space(100))
        after = path.stat()
        self.assertEqual(before.st_ino, after.st_ino)
        self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)

    def test_deliberate_environment_change_after_load(self):
        cp = self.run_bash('load_folder_config; export CLAUDE_LOCAL_CACHE="$HOME/changed"; '
                           'load_folder_config; echo "$CACHE_DIR"')
        self.assertEqual(cp.stdout.strip(), str(self.home / "changed"))

    def test_unsetting_config_override_restores_default_lookup(self):
        self.save_config()
        custom = self.work / "custom-config"
        self.save_config([self.work / "custom-state", self.work / "custom-cache", custom],
                         destination=custom)
        cp = self.run_bash('load_folder_config; unset PUSHBUTTON_CONFIG_DIR; '
                           'load_folder_config; echo "$STATE_DIR"; echo "$CONFIG_DIR"',
                           {"PUSHBUTTON_CONFIG_DIR": str(custom)})
        self.assertEqual(cp.stdout.splitlines(), [str(self.work / "saved-state"), str(self.config)])

    def test_corrupt_json_and_invalid_schema_warn_and_default(self):
        self.config.mkdir(parents=True)
        for text in ('not json', '[]', '{"schema_version":2}', '{"schema_version":true}',
                     '{"schema_version":1,"folders":null}'):
            with self.subTest(text=text):
                (self.config / "folders.json").write_text(text)
                cp = self.run_bash('load_folder_config; echo "$STATE_DIR"')
                self.assertIn("corrupt folder config", cp.stderr)
                self.assertEqual(cp.stdout.strip(), str(self.state))

    def test_json_is_never_shell_evaluated(self):
        marker = self.work / "executed"
        self.save_config([f"{self.work}/$(touch {marker})", self.cache, self.config])
        cp = self.run_bash('load_folder_config; echo "$STATE_DIR"')
        self.assertIn("$(touch", cp.stdout)
        self.assertFalse(marker.exists())

    def test_control_character_path_is_rejected(self):
        cp = self.run_bash('load_folder_config', {"CLAUDE_LOCAL_STATE": "bad\npath"}, check=False)
        self.assertNotEqual(cp.returncode, 0)
        self.assertIn("control characters", cp.stderr)

    def test_overlapping_env_paths_and_symlinks(self):
        (self.work / "target").mkdir()
        (self.work / "alias").symlink_to(self.work / "target", target_is_directory=True)
        for cache in (self.work / "target", self.work / "target/child", self.work / "alias/child"):
            with self.subTest(cache=cache):
                cp = self.run_bash('load_folder_config',
                                   {"CLAUDE_LOCAL_STATE": str(self.work / "target"),
                                    "CLAUDE_LOCAL_CACHE": str(cache)}, check=False)
                self.assertNotEqual(cp.returncode, 0)
                self.assertIn("overlapping", cp.stderr)
                self.assertIn("unset CLAUDE_LOCAL_STATE", cp.stderr)

    def test_config_overlaps_cache_rejected(self):
        cp = self.run_bash('load_folder_config',
                           {"PUSHBUTTON_CONFIG_DIR": str(self.work / "cache/config"),
                            "CLAUDE_LOCAL_CACHE": str(self.work / "cache")}, check=False)
        self.assertNotEqual(cp.returncode, 0)
        self.assertIn("PUSHBUTTON_CONFIG_DIR", cp.stderr)

    def test_saved_overlap_is_corrupt_config(self):
        self.save_config([self.work / "same", self.work / "same/child", self.config])
        cp = self.run_bash('load_folder_config; echo "$STATE_DIR"')
        self.assertIn("corrupt folder config", cp.stderr)
        self.assertEqual(cp.stdout.strip(), str(self.state))

    def test_check_creates_directories(self):
        self.run_bash("load_folder_config; check_folders_exist")
        for path in (self.state, self.cache, self.config):
            self.assertTrue(path.is_dir())
            self.assertEqual(list(path.iterdir()), [])

    def test_existing_file_not_directory(self):
        path = self.work / "file"
        path.write_text("not a folder")
        cp = self.run_bash("load_folder_config; check_folders_exist",
                           {"CLAUDE_LOCAL_CACHE": str(path)}, check=False)
        self.assertNotEqual(cp.returncode, 0)

    def test_permission_checks_even_under_root(self):
        readonly = self.work / "readonly"
        readonly.mkdir()
        for mode in (0o555, 0o444, 0o666):
            readonly.chmod(mode)
            for path in (readonly, readonly / "new"):
                with self.subTest(mode=oct(mode), path=path):
                    cp = self.run_bash("load_folder_config; check_folders_exist",
                                       {"CLAUDE_LOCAL_CACHE": str(path)}, check=False)
                    self.assertNotEqual(cp.returncode, 0)
            readonly.chmod(0o700)

    def test_save_is_atomic_and_preserves_creation_time(self):
        original = self.save_config()
        cp = self.run_bash('load_folder_config; STATE_DIR="$HOME/chosen/../chosen"; '
                           'save_folder_config; echo "$CLAUDE_LOCAL_STATE"')
        saved = json.loads((self.config / "folders.json").read_text())
        self.assertEqual(saved["schema_version"], 1)
        self.assertEqual(saved["created_at"], original["created_at"])
        self.assertEqual(saved["folders"]["state_dir"], str(self.home / "chosen"))
        self.assertEqual(cp.stdout.strip(), str(self.home / "chosen"))
        self.assertEqual((self.config / "folders.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual([p.name for p in self.config.iterdir()], ["folders.json"])

    def test_noninteractive_configure_is_silent_and_saves(self):
        cp = self.run_bash("configure_folders")
        self.assertEqual(cp.stdout, "")
        self.assertEqual(cp.stderr, "")
        data = json.loads((self.config / "folders.json").read_text())
        self.assertEqual(data["folders"]["cache_dir"], str(self.cache))

    def test_noninteractive_configure_persists_environment(self):
        override = self.work / "override-cache"
        self.run_bash("configure_folders", {"CLAUDE_LOCAL_CACHE": str(override)})
        data = json.loads((self.config / "folders.json").read_text())
        self.assertEqual(data["folders"]["cache_dir"], str(override))

    def test_direct_custom_config_locator_survives_fresh_shell_and_actual_edits(self):
        custom = self.work / "direct-config"
        chosen_cache = self.work / "direct-cache"
        self.run_bash('load_folder_config; '
                      f'CONFIG_DIR="{custom}"; CACHE_DIR="{chosen_cache}"; save_folder_config')
        locator = json.loads((self.config / "folders.json").read_text())
        self.assertEqual(locator["schema_version"], 1)
        self.assertTrue(locator["config_locator"])
        self.assertEqual(locator["folders"]["config_dir"], str(custom))
        cp = self.run_bash('load_folder_config; printf "%s\\n" "$CONFIG_DIR" "$CACHE_DIR"')
        self.assertEqual(cp.stdout.splitlines(), [str(custom), str(chosen_cache)])
        actual_file = custom / "folders.json"
        actual = json.loads(actual_file.read_text())
        changed_cache = self.work / "later-edited-cache"
        actual["folders"]["cache_dir"] = str(changed_cache)
        actual_file.write_text(json.dumps(actual))
        cp = self.run_bash('load_folder_config; printf "%s\\n" "$CONFIG_DIR" "$CACHE_DIR"; '
                           'print_folder_summary')
        self.assertEqual(cp.stdout.splitlines()[:2], [str(custom), str(changed_cache)])
        self.assertIn("Config locator:", cp.stdout)

    def test_edited_config_directory_materializes_without_discarding_loaded_paths(self):
        relocated = self.work / "edited-config-target"
        chosen_state = self.work / "edited-state"
        chosen_cache = self.work / "edited-cache"
        original = self.save_config([chosen_state, chosen_cache, relocated])
        cp = self.run_bash('initialize_folders 1; printf "%s\\n" "$STATE_DIR" "$CACHE_DIR" "$CONFIG_DIR"',
                           self.mock_space(100))
        self.assertEqual(cp.stdout.splitlines(), list(map(str, [chosen_state, chosen_cache, relocated])))
        self.assertEqual(cp.stderr, "")
        actual_file = relocated / "folders.json"
        self.assertTrue(actual_file.is_file())
        actual = json.loads(actual_file.read_text())
        self.assertEqual(actual["folders"], original["folders"])
        self.assertEqual(actual["created_at"], original["created_at"])
        locator = json.loads((self.config / "folders.json").read_text())
        self.assertTrue(locator["config_locator"])
        cp = self.run_bash('load_folder_config; printf "%s\\n" "$STATE_DIR" "$CACHE_DIR" "$CONFIG_DIR"; '
                           'print_folder_summary')
        self.assertEqual(cp.stdout.splitlines()[:3], list(map(str, [chosen_state, chosen_cache, relocated])))
        self.assertIn("Config locator:", cp.stdout)
        self.assertIn(f"To customize locations, edit {relocated}/folders.json or set env vars.", cp.stdout)

    def test_edited_custom_config_relocation_updates_source_locator_chain(self):
        custom = self.work / "custom-source"
        relocated = self.work / "custom-target"
        chosen_state = self.work / "custom-state"
        chosen_cache = self.work / "custom-cache"
        self.run_bash('load_folder_config; '
                      f'CONFIG_DIR="{custom}"; save_folder_config')
        source_file = custom / "folders.json"
        actual = json.loads(source_file.read_text())
        actual["folders"] = {"state_dir": str(chosen_state), "cache_dir": str(chosen_cache),
                             "config_dir": str(relocated)}
        source_file.write_text(json.dumps(actual))
        self.run_bash("initialize_folders 1", self.mock_space(100))
        self.assertTrue((relocated / "folders.json").is_file())
        self.assertTrue(json.loads(source_file.read_text())["config_locator"])
        cp = self.run_bash('load_folder_config; printf "%s\\n" "$STATE_DIR" "$CACHE_DIR" "$CONFIG_DIR"')
        self.assertEqual(cp.stdout.splitlines(), list(map(str, [chosen_state, chosen_cache, relocated])))

    def test_explicit_config_override_does_not_create_default_locator(self):
        custom = self.work / "explicit-config"
        self.run_bash("configure_folders", {"PUSHBUTTON_CONFIG_DIR": str(custom)})
        self.assertTrue((custom / "folders.json").is_file())
        self.assertFalse((self.config / "folders.json").exists())

    def test_broken_or_cyclic_locator_warns_and_is_not_overwritten(self):
        target = self.work / "locator-target"
        self.save_config([self.state, self.cache, target])
        default_file = self.config / "folders.json"
        data = json.loads(default_file.read_text())
        data["config_locator"] = True
        default_file.write_text(json.dumps(data))
        original = default_file.read_text()
        cp = self.run_bash("initialize_folders 1", self.mock_space(100))
        self.assertIn("locator target is missing", cp.stderr)
        self.assertEqual(default_file.read_text(), original)
        self.save_config([self.state, self.cache, self.config], destination=target)
        target_file = target / "folders.json"
        cycle = json.loads(target_file.read_text())
        cycle["config_locator"] = True
        target_file.write_text(json.dumps(cycle))
        cp = self.run_bash("initialize_folders 1", self.mock_space(100))
        self.assertIn("cyclic folder config locator", cp.stderr)
        self.assertEqual(default_file.read_text(), original)

    def test_initialize_quiet_and_normal_banner(self):
        cp = self.run_bash('initialize_folders --quiet; initialize_folders 1', self.mock_space(100))
        self.assertEqual(cp.stdout, "")
        self.assertEqual(cp.stderr, "")
        self.assertTrue((self.config / "folders.json").is_file())
        cp = self.run_bash("initialize_folders 0", self.mock_space(100))
        self.assertIn("[pushbutton] System folders", cp.stdout)
        self.assertIn(str(self.state), cp.stdout)

    def test_low_cache_startup_warning_including_quiet_mode(self):
        for quiet in ("0", "1"):
            with self.subTest(quiet=quiet):
                cp = self.run_bash(f"initialize_folders {quiet}", self.mock_space(49))
                self.assertIn("less than 50 GiB", cp.stderr)
                self.assertIn("--local-cache", cp.stderr)
                self.assertIn("/fixture-mount", cp.stderr)
                if quiet == "1":
                    self.assertEqual(cp.stdout, "")
        cp = self.run_bash("initialize_folders 1", self.mock_space(50))
        self.assertEqual(cp.stderr, "")

    def test_verbose_summary(self):
        self.run_bash("initialize_folders --quiet")
        (self.state / "logs").mkdir()
        (self.state / "logs/test.log").write_text("log")
        cp = self.run_bash('load_folder_config; print_folder_summary --verbose',
                           {"CLAUDE_LOCAL_CACHE": str(self.work / "override")})
        for text in ("State:", "Cache:", "Config:", "GiB free", "used:", "Logs:",
                     "Environment overrides: CLAUDE_LOCAL_CACHE=", "Code:"):
            self.assertIn(text, cp.stdout)

    def test_code_summary_prefers_installed_repository_over_installer_root(self):
        installed = self.work / "install-root/PushbuttonLocalCoders"
        cp = self.run_bash('load_folder_config; ROOT="$HOME/install-root"; '
                           f'PUSHBUTTON_CODE_DIR="{installed}"; print_folder_summary')
        self.assertIn("Code: " + str(installed) + " (ROOT)", cp.stdout)
        cp = self.run_bash('load_folder_config; ROOT="$HOME/repository"; print_folder_summary')
        self.assertIn("Code: " + str(self.home / "repository") + " (ROOT)", cp.stdout)

    def test_normal_summary_includes_logs_customization_and_install_override(self):
        cp = self.run_bash('load_folder_config; print_folder_summary',
                           {"PUSHBUTTON_DIR": str(self.work / "install")})
        for text in ("[pushbutton] System folders", "Code:", "State:", "Cache:",
                     "Config:", "Logs:", "Environment overrides: PUSHBUTTON_DIR=",
                     "Customize folders with", "--local-cache"):
            self.assertIn(text, cp.stdout)
        self.assertIn(f"[pushbutton] To customize locations, edit {self.config}/folders.json "
                      "or set env vars.", cp.stdout)

    def test_linux_mac_and_df_free_space_with_missing_path(self):
        for mode in ("linux", "mac", "df"):
            with self.subTest(mode=mode):
                cp = self.run_bash('load_folder_config; get_free_space "$HOME/not/yet/created"',
                                   self.mock_space(20, mode))
                self.assertEqual(int(cp.stdout), 20 * GIB)

    def test_real_free_space(self):
        cp = self.run_bash('load_folder_config; get_free_space')
        self.assertGreater(int(cp.stdout), 0)

    def test_cache_space_threshold_and_guidance(self):
        env = self.mock_space(20)
        self.run_bash(f"load_folder_config; validate_cache_space {20 * GIB}", env)
        cp = self.run_bash(f"load_folder_config; validate_cache_space {20 * GIB + 1}", env, check=False)
        self.assertEqual(cp.returncode, 2)
        for text in ("--local-cache", "smaller quant", "free files", "/dev/test", "/fixture-mount"):
            self.assertIn(text, cp.stderr)
        for invalid in ("NaN", "-1", "bad", "1.5"):
            cp = self.run_bash(f"load_folder_config; validate_cache_space {invalid}", env, check=False)
            self.assertEqual(cp.returncode, 2)

    def test_estimate_uses_planner_envelope(self):
        cp = self.run_bash('estimate_model_size qwen3.8:27b UD-Q4_K_M; '
                           'estimate_model_size unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M')
        for value in cp.stdout.splitlines():
            self.assertEqual(int(value), (22 * GIB * 11 + 9) // 10)
        cp = self.run_bash("estimate_model_size unknown", check=False)
        self.assertEqual(cp.returncode, 2)

    def test_download_space_offline_metadata_and_reserve(self):
        self.metadata()
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           self.mock_space(13))
        self.assertIn("3.00 GiB remaining", cp.stderr)
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           self.mock_space(12), check=False)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("13.00 GiB required", cp.stderr)

    def test_complete_blobs_subtracted_partial_not_subtracted(self):
        _, root = self.metadata()
        blobs = root / "blobs"
        blobs.mkdir()
        digest = blobs / ("a" * 64)
        with digest.open("wb") as stream:
            stream.truncate(3 * GIB)
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           self.mock_space(10))
        self.assertIn("0.00 GiB remaining", cp.stderr)
        digest.unlink()
        partial = blobs / (digest.name + ".downloadInProgress")
        with partial.open("wb") as stream:
            stream.truncate(3 * GIB)
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           self.mock_space(10), check=False)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("3.00 GiB remaining", cp.stderr)
        digest.symlink_to(partial)
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           self.mock_space(10), check=False)
        self.assertEqual(cp.returncode, 2)

    def test_snapshot_and_flat_llama_cache(self):
        _, root = self.metadata()
        snapshot = root / "snapshots" / ("b" * 40)
        snapshot.mkdir(parents=True)
        cached = snapshot / "model-Q4_K_M.gguf"
        with cached.open("wb") as stream:
            stream.truncate(3 * GIB)
        self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                      self.mock_space(10))
        cached.unlink()
        flat = self.cache / "owner_repo_model-Q4_K_M.gguf"
        with flat.open("wb") as stream:
            stream.truncate(3 * GIB)
        self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                      self.mock_space(10))

    def test_wrong_size_cache_is_not_subtracted(self):
        _, root = self.metadata()
        (root / "blobs").mkdir()
        (root / "blobs" / ("a" * 64)).write_text("partial")
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           self.mock_space(10), check=False)
        self.assertEqual(cp.returncode, 2)

    def test_revision_download_cache_excludes_aria2_partial_files(self):
        _, root = self.metadata()
        directory = root / ("b" * 40)
        directory.mkdir()
        cached = directory / "model-Q4_K_M.gguf"
        with cached.open("wb") as stream:
            stream.truncate(3 * GIB)
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           self.mock_space(10))
        self.assertIn("0.00 GiB remaining", cp.stderr)
        pathlib.Path(str(cached) + ".aria2").touch()
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           self.mock_space(10), check=False)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("3.00 GiB remaining", cp.stderr)

    def test_exact_quant_matching_and_ud_distinction(self):
        self.metadata([
            {"rfilename": "model-Q4_K_M.gguf", "size": GIB},
            {"rfilename": "model-UD-Q4_K_M.gguf", "size": 20 * GIB},
            {"rfilename": "model-Q4_K_M_XL.gguf", "size": 20 * GIB},
            {"rfilename": "model-Q4_K_S.gguf", "size": 20 * GIB},
        ])
        self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                      self.mock_space(11))
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4",
                           self.mock_space(100), check=False)
        self.assertEqual(cp.returncode, 2)

    def test_split_files_and_explicit_path_with_quant(self):
        self.metadata([
            {"rfilename": f"Q4_K_M/model-Q4_K_M-{i:05d}-of-00002.gguf", "size": 2 * GIB}
            for i in (1, 2)])
        for spec in ("owner/repo:Q4_K_M",
                     "owner/repo/Q4_K_M/model-Q4_K_M-00001-of-00002.gguf:Q4_K_M",
                     "hf.co/owner/repo/Q4_K_M/model-Q4_K_M-00002-of-00002.gguf"):
            with self.subTest(spec=spec):
                cp = self.run_bash(f"load_folder_config; validate_download_space {spec}",
                                   self.mock_space(14))
                self.assertIn("4.00 GiB remaining", cp.stderr)

    def test_ambiguous_quant_rejected_explicit_file_allowed(self):
        self.metadata([{"rfilename": f"{name}-Q4_K_M.gguf", "size": GIB}
                       for name in ("model", "other")])
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           self.mock_space(100), check=False)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("ambiguous", cp.stderr)
        self.run_bash("load_folder_config; validate_download_space owner/repo/model-Q4_K_M.gguf",
                      self.mock_space(11))

    def test_missing_unknown_and_incomplete_metadata_fail_closed(self):
        entries = [
            [{"rfilename": "model-Q4_K_M.gguf"}],
            [{"rfilename": "model-Q4_K_M.gguf", "size": 0}],
            [{"rfilename": "model-Q4_K_M.gguf", "size": "100"}],
            [{"rfilename": "model-Q4_K_M-00001-of-00002.gguf", "size": GIB}],
            [{"rfilename": "../model-Q4_K_M.gguf", "size": GIB}],
            [],
        ]
        for inventory in entries:
            with self.subTest(inventory=inventory):
                self.metadata(inventory)
                cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                                   self.mock_space(100), check=False)
                self.assertEqual(cp.returncode, 2)

    def test_offline_without_metadata_returns_two(self):
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           check=False)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("metadata.json", cp.stderr)

    def test_invalid_spec_and_explicit_quant_mismatch(self):
        self.metadata()
        for spec in ("owner/repo", "owner/repo/../bad.gguf", "owner/repo:bad/quant",
                     "owner/repo/model-Q4_K_M.gguf:Q8_0"):
            with self.subTest(spec=spec):
                cp = self.run_bash(f"load_folder_config; validate_download_space {spec}", check=False)
                self.assertEqual(cp.returncode, 2)

    def test_download_launcher_three_argument_signature(self):
        self.metadata()
        cp = self.run_bash('load_folder_config; '
                           'validate_download_space "$CACHE_DIR" owner/repo:Q4_K_M Q4_K_M',
                           self.mock_space(13))
        self.assertIn("3.00 GiB remaining", cp.stderr)
        cp = self.run_bash('load_folder_config; '
                           'validate_download_space "$CACHE_DIR" owner/repo:Q4_K_M Q8_0',
                           check=False)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("conflicts with HF spec", cp.stderr)
        self.run_bash(f'validate_download_space "{self.cache}" owner/repo:Q4_K_M Q4_K_M',
                      self.mock_space(13))
        for spec in ("owner/repo", "owner/repo/model-Q4_K_M.gguf",
                     "https://huggingface.co/owner/repo"):
            with self.subTest(spec=spec):
                self.run_bash(f'validate_download_space "{self.cache}" "{spec}" Q4_K_M',
                              self.mock_space(13))
        cp = self.run_bash(f'validate_download_space "{self.cache}" owner/repo Q8_0',
                           self.mock_space(100), check=False)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("no exact GGUF match", cp.stderr)

    def test_validation_usage_failures_return_two(self):
        commands = (
            "unset CACHE_DIR; validate_cache_space 1",
            "load_folder_config; validate_cache_space",
            "load_folder_config; validate_cache_space 1 extra",
            "unset CACHE_DIR; validate_download_space owner/repo:Q4_K_M",
            "load_folder_config; validate_download_space",
            "validate_download_space '' owner/repo Q4_K_M",
            "validate_download_space cache owner/repo Q4_K_M extra",
            "unset CACHE_DIR; validate_plan_download_space",
            "load_folder_config; validate_plan_download_space",
            "validate_plan_download_space missing.json cache extra",
        )
        for command in commands:
            with self.subTest(command=command):
                cp = self.run_bash(command, check=False)
                self.assertEqual(cp.returncode, 2)

    def test_real_launcher_folder_flags_use_config_and_cli_override(self):
        self.save_config()
        env = self.env.copy()
        bin_dir = self.work / "no-provisioning"
        bin_dir.mkdir()
        provisioned = self.work / "provisioning-attempt"
        for command in ("curl", "npm", "uv", "nvidia-smi", "git", "cmake", "sudo"):
            blocked = bin_dir / command
            blocked.write_text('#!/bin/sh\nprintf "%s\\n" "$0" >> "$PROVISIONING_ATTEMPT"\n'
                               'echo "summary flag attempted provisioning: $0" >&2\nexit 89\n')
            blocked.chmod(0o755)
        env.update(PATH=str(bin_dir) + os.pathsep + self.env["PATH"],
                   PROVISIONING_ATTEMPT=str(provisioned))
        cli_cache = self.work / "CLI cache with spaces"
        for name in ("claude-local", "coder-local", "hermes-local", "qwen-local",
                     "lib/claude_local_entry.sh"):
            for flag in ("--folders", "--system-info"):
                with self.subTest(name=name, flag=flag):
                    if provisioned.exists():
                        provisioned.unlink()
                    cp = subprocess.run(["bash", str(ROOT / name), flag, "--local-cache",
                                         str(cli_cache)], cwd=ROOT, env=env,
                                        text=True, capture_output=True, timeout=15)
                    self.assertEqual(cp.returncode, 0, cp.stderr)
                    for path in (self.work / "saved-state", cli_cache, self.config):
                        self.assertIn(str(path), cp.stdout)
                    self.assertIn("Code:", cp.stdout)
                    self.assertIn("GiB free", cp.stdout)
                    self.assertIn(f"To customize locations, edit {self.config}/folders.json", cp.stdout)
                    self.assertNotIn("nvidia-smi", cp.stderr)
                    self.assertNotIn("Pre-install", cp.stdout)
                    self.assertFalse(provisioned.exists(), cp.stderr)

    def test_noninteractive_installers_clone_configure_fetch_and_remember_config(self):
        fixture = self.work / "fixture-repo"
        (fixture / "lib").mkdir(parents=True)
        (fixture / "configs").mkdir()
        (fixture / ".git").mkdir()
        files = (
            "claude-local", "coder-local", "opencode-local", "deepseek-local", "mini-swe-local",
            "pushbutton-backend", "pushbutton-bench", "pushbutton-select", "pushbutton-observe",
            "pushbutton", "pushbutton-instance", "pushbutton-broker", "pushbutton-proxy",
            "claude-local-safe", "qwen-local",
            "lib/pushbutton_download.sh",
            "lib/pushbutton_folders.sh", "lib/claude_local_entry.sh", "lib/claude_local_plan.py",
            "lib/claude_local_gateway.py", "lib/coder_local_plan.py",
            "configs/qwen-local-defaults.json",
            "configs/backend-registry.json", "lib/pushbutton_metrics.py",
        )
        for name in files:
            shutil.copy2(ROOT / name, fixture / name)
        bin_dir = self.work / "installer-bin"
        bin_dir.mkdir()
        git = bin_dir / "git"
        git.write_text("""#!/usr/bin/env python3
import json, os, pathlib, shutil, sys
args = sys.argv[1:]
with open(os.environ["GIT_LOG"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[0] == "clone":
    shutil.copytree(os.environ["FIXTURE_REPO"], args[-1])
elif args[0] != "-C":
    sys.exit("unexpected git invocation")
""")
        git.chmod(0o755)
        git_log = self.work / "git-calls.jsonl"
        config = self.work / "installer config with spaces"
        state = self.work / "installer-state"
        cache = self.work / "installer-cache"
        env = dict(self.env, PATH=str(bin_dir) + os.pathsep + self.env["PATH"],
                   FIXTURE_REPO=str(fixture), GIT_LOG=str(git_log),
                   PUSHBUTTON_CONFIG_DIR=str(config), CLAUDE_LOCAL_STATE=str(state),
                   CLAUDE_LOCAL_CACHE=str(cache), PUSHBUTTON_REPO_URL="fixture://repo")
        for installer, shim in (("install-claude-local.sh", "claude-local"),
                                ("install-coder-local.sh", "qwen-local")):
            with self.subTest(installer=installer):
                env["PUSHBUTTON_DIR"] = str(self.work / (installer + "-install"))
                for _ in range(2):
                    args = ["--install-only"] if installer == "install-coder-local.sh" else []
                    cp = subprocess.run(["bash", str(ROOT / installer), *args], cwd=ROOT, env=env,
                                        text=True, capture_output=True, timeout=20)
                    self.assertEqual(cp.returncode, 0, cp.stderr)
                    self.assertNotIn("Customize state/cache/config", cp.stdout)
                    self.assertNotIn("Pre-install folder summary", cp.stdout)
                    self.assertIn("Configuration saved at", cp.stdout)
                data = json.loads((config / "folders.json").read_text())
                self.assertEqual(data["folders"],
                                 {"state_dir": str(state), "cache_dir": str(cache),
                                  "config_dir": str(config)})
                # A fresh shell without overrides must discover the installer's
                # chosen config directory through the installed command shim.
                fresh = self.env.copy()
                cp = subprocess.run(["bash", str(self.home / ".local/bin" / shim), "--folders"],
                                    cwd=ROOT, env=fresh, text=True, capture_output=True, timeout=15)
                self.assertEqual(cp.returncode, 0, cp.stderr)
                for path in (state, cache, config):
                    self.assertIn(str(path), cp.stdout)
        calls = [json.loads(line) for line in git_log.read_text().splitlines()]
        self.assertEqual(sum(call[0] == "clone" for call in calls), 2)
        self.assertEqual(sum("fetch" in call for call in calls), 2)

    def test_plan_aggregate_space_prevents_individually_fitting_downloads(self):
        self.metadata()
        self.metadata([{"rfilename": "second-Q8_0.gguf", "size": 5 * GIB}], repo="owner/second")
        plan = self.save_plan({"servers": [
            {"profile": {"hf_spec": "owner/repo:Q4_K_M"}},
            {"profile": {"hf_spec": "owner/second:Q8_0"}},
        ]})
        env = self.mock_space(17)
        self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M; "
                      "validate_download_space owner/second:Q8_0", env)
        cp = self.run_bash(f'load_folder_config; validate_plan_download_space "{plan}"',
                           env, check=False)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("18.00 GiB required", cp.stderr)
        self.assertIn("8.00 GiB remaining + 10 GiB safety reserve", cp.stderr)
        self.run_bash(f'load_folder_config; validate_plan_download_space "{plan}"',
                      self.mock_space(18))

    def test_plan_deduplicates_workers_servers_and_reserves_once(self):
        self.metadata()
        row = {"profile": {"hf_spec": "owner/repo:Q4_K_M"}}
        plan = self.save_plan({"workers": [row, row], "servers": [row]})
        cp = self.run_bash(f'load_folder_config; validate_plan_download_space "{plan}"',
                           self.mock_space(13))
        self.assertIn("1 distinct model spec(s)", cp.stderr)
        self.assertEqual(cp.stderr.count("GGUF download:"), 1)
        self.assertIn("3.00 GiB remaining + 10 GiB safety reserve", cp.stderr)

    def test_plan_explicit_cache_parameter_and_complete_cache_subtraction(self):
        _, root = self.metadata()
        (root / "blobs").mkdir()
        with (root / "blobs" / ("a" * 64)).open("wb") as stream:
            stream.truncate(3 * GIB)
        self.metadata([{"rfilename": "second-Q8_0.gguf", "size": 5 * GIB}], repo="owner/second")
        alternate = self.work / "alternate cache"
        shutil.move(self.cache, alternate)
        plan = self.save_plan({"workers": [
            {"profile": {"hf_spec": "owner/repo:Q4_K_M"}},
            {"profile": {"hf_spec": "owner/second:Q8_0"}},
        ]})
        cp = self.run_bash(f'load_folder_config; validate_plan_download_space "{plan}" "{alternate}"',
                           self.mock_space(15))
        self.assertIn("5.00 GiB remaining + 10 GiB safety reserve", cp.stderr)
        self.assertFalse(self.cache.exists())

    def test_plan_invalid_schema_and_missing_metadata_return_two(self):
        self.metadata()
        cases = [
            [],
            {},
            {"servers": []},
            {"servers": "not a list"},
            {"workers": [None]},
            {"workers": [{}]},
            {"workers": [{"profile": []}]},
            {"workers": [{"profile": {"hf_spec": 1}}]},
            {"workers": [{"profile": {"hf_spec": " "}}]},
            {"workers": [{"profile": {"hf_spec": "owner/unknown:Q4_K_M"}}]},
        ]
        for data in cases:
            with self.subTest(data=data):
                plan = self.save_plan(data)
                cp = self.run_bash(f'load_folder_config; validate_plan_download_space "{plan}"',
                                   self.mock_space(100), check=False)
                self.assertEqual(cp.returncode, 2)
        plan.write_text("invalid JSON")
        cp = self.run_bash(f'load_folder_config; validate_plan_download_space "{plan}"',
                           check=False)
        self.assertEqual(cp.returncode, 2)

    def test_plan_top_level_hf_spec(self):
        self.metadata()
        plan = self.save_plan({"servers": [{"hf_spec": "owner/repo:Q4_K_M"}]})
        self.run_bash(f'load_folder_config; validate_plan_download_space "{plan}"',
                      self.mock_space(13))

    def test_http_metadata_requests_blob_sizes(self):
        data, root = self.metadata()
        # A stale small inventory must not undercount a newer online artifact.
        cached_metadata = root / "metadata.json"
        stale = {"sha": "c" * 40, "siblings": [
            {"rfilename": "model-Q4_K_M.gguf", "size": GIB}]}
        cached_metadata.write_text(json.dumps(stale))
        requested = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                requested.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            env = self.mock_space(12)
            env["HF_ENDPOINT"] = f"http://127.0.0.1:{server.server_port}"
            cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                               env, check=False)
            self.assertEqual(cp.returncode, 2)
            self.assertIn("13.00 GiB required", cp.stderr)
            self.assertEqual(json.loads(cached_metadata.read_text()), data)
            self.assertEqual(cached_metadata.stat().st_mode & 0o777, 0o600)
            self.assertFalse(list(root.glob(".metadata-*.json")))
            env.update(self.mock_space(13), HF_HUB_OFFLINE="1")
            self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M", env)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(requested, ["/api/models/owner/repo?blobs=true"])

    def test_explicit_offline_requires_complete_local_inventory(self):
        env = dict(self.mock_space(100), PUSHBUTTON_HF_OFFLINE="1")
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           env, check=False)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("offline mode", cp.stderr)
        self.metadata([{"rfilename": "model-Q4_K_M.gguf"}])
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           env, check=False)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("unknown GGUF size", cp.stderr)

    def test_cached_metadata_must_belong_to_requested_repository(self):
        data, root = self.metadata()
        data["id"] = "other/repo"
        (root / "metadata.json").write_text(json.dumps(data))
        cp = self.run_bash("load_folder_config; validate_download_space owner/repo:Q4_K_M",
                           {"HF_HUB_OFFLINE": "1"}, check=False)
        self.assertEqual(cp.returncode, 2)

    def test_interactive_configuration_with_pty(self):
        self.check_interactive_configuration("configure_folders")

    def test_quiet_first_initialize_still_offers_interactive_selection(self):
        self.check_interactive_configuration("initialize_folders 1")

    def check_interactive_configuration(self, command):
        master, slave = pty.openpty()
        process = subprocess.Popen(["bash", "-c", 'set -eu; source "$1"; ' + command + '; '
                                    'load_folder_config; echo "RELOADED:$CONFIG_DIR"',
                                    "folder-tests", str(HELPER)],
                                   cwd=ROOT, env=self.env, stdin=slave, stdout=slave, stderr=slave)
        os.close(slave)
        transcript = bytearray()
        new_config = self.work / "chosen-config"
        responses = [
            (b"Customize state/cache/config folders? [y/N] ", b"y\n"),
            (b"State directory [", b"\n"),
            (b"Cache directory [", b"~/chosen-cache\n"),
            (b"Config directory [", str(new_config).encode() + b"\n"),
        ]
        deadline = time.monotonic() + 15
        try:
            while time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    transcript.extend(chunk)
                    if responses and responses[0][0] in transcript:
                        _, response = responses.pop(0)
                        os.write(master, response)
                if process.poll() is not None:
                    break
            process.wait(timeout=2)
            self.assertEqual(process.returncode, 0, transcript.decode())
            self.assertFalse(responses, transcript.decode())
            self.assertIn(b"Pre-install folder summary", transcript)
            data = json.loads((new_config / "folders.json").read_text())
            self.assertEqual(data["folders"]["cache_dir"], str(self.home / "chosen-cache"))
            self.assertEqual(data["folders"]["config_dir"], str(new_config))
            self.assertIn(("RELOADED:" + str(new_config)).encode(), transcript)
            fresh = self.run_bash('load_folder_config; printf "%s\\n" "$CONFIG_DIR" "$CACHE_DIR"')
            self.assertEqual(fresh.stdout.splitlines(),
                             [str(new_config), str(self.home / "chosen-cache")])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            os.close(master)


if __name__ == "__main__":
    unittest.main()
