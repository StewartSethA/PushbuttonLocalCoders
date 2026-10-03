#!/usr/bin/env python3
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pushbutton_download", ROOT / "lib/pushbutton_download.py")
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)
REVISION = "a" * 40
METADATA = {"sha": REVISION, "siblings": [
    {"rfilename": "Qwen-UD-Q4_K_XL-00001-of-00002.gguf", "size": 32},
    {"rfilename": "Qwen-UD-Q4_K_XL-00002-of-00002.gguf", "lfs": {"size": 48}},
    {"rfilename": "Qwen-Q4_K_M.gguf", "size": 16},
]}


class MetadataTests(unittest.TestCase):
    def test_quant_and_shards(self):
        self.assertEqual(len(download.select_files(METADATA, "UD-Q4_K_XL")), 2)
        self.assertEqual(len(download.select_files(METADATA, "Q4_K_M")), 1)
        self.assertEqual(len(download.select_files(METADATA, "UD-Q4_K_XL,Q4_K_M")), 3)
        self.assertEqual(len(download.select_files(METADATA, METADATA["siblings"][0]["rfilename"])), 2)
        with self.assertRaises(ValueError):
            download.select_files(METADATA, "Q4_K")

    def test_unknown_size_and_unsafe_paths(self):
        for sibling in ({"rfilename": "../Q4_K.gguf", "size": 16},
                        {"rfilename": "Q4_K.gguf"},
                        {"rfilename": "/Q4_K.gguf", "size": 16}):
            with self.subTest(sibling=sibling), self.assertRaises(ValueError):
                download.select_files({"siblings": [sibling]}, "Q4_K")

    def test_incomplete_or_ambiguous_shards_rejected(self):
        for files in ([METADATA["siblings"][0]], [
                {"rfilename": "one-Q4_K.gguf", "size": 16},
                {"rfilename": "two-Q4_K.gguf", "size": 16}]):
            # An incomplete set must include at least one shard with a missing peer.
            with self.assertRaises(ValueError):
                download.select_files({"siblings": files}, "UD-Q4_K_XL" if len(files) == 1 else "Q4_K")

    def test_display_name(self):
        self.assertEqual(download.display_name("short"), "short")
        self.assertEqual(download.display_name("x" * 100), "x" * 69 + "...")

    def test_disk_reserve_cached_and_preallocated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "model.gguf"
            path.write_bytes(b"x" * 4096)
            manifest = {"directory": tmp, "files": [{"name": path.name, "size": 4096}]}
            disk = shutil._ntuple_diskusage(100 * 1024**3, 0, 10 * 1024**3)
            with mock.patch.object(download.shutil, "disk_usage", return_value=disk):
                download.validate_space(manifest)
                pathlib.Path(str(path) + ".aria2").touch()
                with self.assertRaisesRegex(ValueError, "Insufficient disk space"):
                    download.validate_space(manifest)
                download.validate_space(manifest, reuse_partial=True)
                self.assertFalse(download.complete(path, 4096))

    def test_metadata_size_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp) / "work"
            work.mkdir()
            output = io.StringIO()
            with mock.patch.object(download.urllib.request, "urlopen",
                                   return_value=io.BytesIO(json.dumps(METADATA).encode())), \
                    contextlib.redirect_stderr(output):
                manifest = download.prepare("owner/repo:UD-Q4_K_XL", tmp, str(work))
            self.assertIn("0.0 GiB total", output.getvalue())
            self.assertEqual(manifest["download_bytes"], 80)
            self.assertIn(REVISION, (work / "urls").read_text())

    def test_existing_blob_reused_without_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp) / REVISION
            directory.mkdir()
            digest = "b" * 64
            blob = pathlib.Path(tmp) / "blobs" / digest
            blob.parent.mkdir()
            blob.write_bytes(b"x" * 16)
            with contextlib.redirect_stderr(io.StringIO()):
                download.reuse_blobs(directory, [{"name": "model.gguf", "size": 16, "sha256": digest}])
            self.assertEqual(blob.stat().st_ino, (directory / "model.gguf").stat().st_ino)

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp) / "cache"
            directory.mkdir()
            (directory / "escape").symlink_to(tmp, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes cache"):
                download.reuse_blobs(directory, [{"name": "escape/model.gguf", "size": 16}])


class ShellTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = pathlib.Path(self.tmp.name)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.cache = self.base / "model cache"
        self.calls = self.base / "calls"
        self.metadata = self.base / "metadata.json"
        self.metadata.write_text(json.dumps(METADATA))
        self.env = {**os.environ, "PATH": str(self.bin) + ":" + os.environ["PATH"],
                    "TEST_METADATA": str(self.metadata), "TEST_CALLS": str(self.calls),
                    "HOME": str(self.base), "HF_TOKEN": "", "HF_TOKEN_PATH": str(self.base / "no-token")}
        self.executable("python3", f"""#!{sys.executable}
import io,json,os,runpy,shutil,sys,urllib.request
urllib.request.urlopen=lambda *a,**k: io.BytesIO(open(os.environ['TEST_METADATA'],'rb').read())
if os.environ.get('TEST_DISK_FULL'):
    shutil.disk_usage=lambda p: shutil._ntuple_diskusage(100*1024**3,100*1024**3,0)
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
""")
        self.executable("aria2c", f"""#!{sys.executable}
import json,os,pathlib,sys
with open(os.environ['TEST_CALLS'],'a') as s: s.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('TEST_ARIA_FAIL'): sys.exit(int(os.environ['TEST_ARIA_FAIL']))
urls=pathlib.Path(next(a.split('=',1)[1] for a in sys.argv if a.startswith('--input-file=')))
manifest=json.loads((urls.parent/'manifest.json').read_text())
for f in manifest['files']:
    path=pathlib.Path(manifest['directory'])/f['name']
    path.write_bytes(b'x'*f['size'])
    pathlib.Path(str(path)+'.aria2').unlink(missing_ok=True)
print('[parallel chunks] 45% ETA: 3m42s',file=sys.stderr)
""")
        self.executable("hf", f"""#!{sys.executable}
import json,os,pathlib,sys
with open(os.environ['TEST_CALLS'],'a') as s: s.write('hf '+json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('TEST_HF_FAIL'): sys.exit(1)
directory=pathlib.Path(sys.argv[sys.argv.index('--local-dir')+1])
item=next(f for f in json.load(open(os.environ['TEST_METADATA']))['siblings'] if f['rfilename']==sys.argv[3])
path=directory/item['rfilename']
path.write_bytes(b'x'*(item.get('size') or item['lfs']['size']))
print('huggingface_hub: 100%',file=sys.stderr)
""")

    def executable(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def shell(self, body):
        return subprocess.run(["bash", "-c", 'set -euo pipefail; source "$1"; ' + body, "test",
                               str(ROOT / "lib/pushbutton_download.sh"), str(self.cache)],
                              env=self.env, capture_output=True, text=True, timeout=10)

    def fast(self, prefix=""):
        return self.shell(prefix + 'download_model_fast owner/repo:UD-Q4_K_XL "$2" MODEL_PATH; printf "%s\\n" "$MODEL_PATH"')

    def test_parallel_progress_arguments_log_and_cached_reuse(self):
        proc = self.fast()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(pathlib.Path(proc.stdout.strip()).is_file())
        self.assertIn("45% ETA", proc.stderr)
        args = self.calls.read_text()
        for flag in ("--split=3", "--continue=true", "--max-concurrent-downloads=4",
                     "--max-connection-per-server=3", "--summary-interval=1",
                     "--file-allocation=prealloc", "--retry-wait=10"):
            self.assertIn(flag, args)
        event = json.loads((self.cache / ".download-log").read_text())
        self.assertEqual(event["size"], 80)
        self.assertEqual(event["method"], "aria2c")
        self.assertGreater(event["speed_bytes_per_second"], 0)
        self.assertEqual(self.fast().returncode, 0)
        self.assertEqual(self.calls.read_text(), args)

    def test_fallback_on_parallel_failure(self):
        self.env["TEST_ARIA_FAIL"] = "1"
        proc = self.fast()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Parallel download failed", proc.stderr)
        self.assertEqual(self.calls.read_text().count("hf "), 2)
        self.assertIn("100%", proc.stderr)

    def test_no_parallel_and_unavailable_fallback(self):
        proc = self.fast("PUSHBUTTON_NO_PARALLEL=1; ")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("--split=3", self.calls.read_text())
        self.assertNotIn("Enabling fast", proc.stderr)
        shutil.rmtree(self.cache)
        self.calls.unlink()
        proc = self.fast("ensure_aria2c() { return 1; }; ")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.calls.read_text().count("hf "), 2)

    def test_disk_full_stops_before_install_or_download(self):
        self.env["TEST_DISK_FULL"] = "1"
        proc = self.fast("ensure_aria2c() { echo INSTALL_ATTEMPT >&2; return 1; }; ")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Insufficient disk space", proc.stderr)
        self.assertNotIn("INSTALL_ATTEMPT", proc.stderr)
        self.assertFalse(self.calls.exists())

    def test_resume_and_sequential_removes_aria_marker(self):
        directory = self.cache / "models--owner--repo" / REVISION
        directory.mkdir(parents=True)
        path = directory / METADATA["siblings"][0]["rfilename"]
        path.write_bytes(b"partial")
        marker = pathlib.Path(str(path) + ".aria2")
        marker.touch()
        proc = self.fast()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Resuming", proc.stderr)
        self.assertFalse(marker.exists())
        marker.touch()
        proc = self.fast("PUSHBUTTON_NO_PARALLEL=1; ")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(marker.exists())

    def test_download_failure_not_logged(self):
        self.env["TEST_HF_FAIL"] = "1"
        proc = self.fast("PUSHBUTTON_NO_PARALLEL=1; ")
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse((self.cache / ".download-log").exists())

    def test_interruption_does_not_start_fallback(self):
        self.env["TEST_ARIA_FAIL"] = "130"
        proc = self.fast()
        self.assertEqual(proc.returncode, 130, proc.stderr)
        self.assertNotIn("hf ", self.calls.read_text())
        self.assertFalse((self.cache / ".download-log").exists())

    def test_folder_validation_hook(self):
        proc = self.fast('validate_download_space() { echo "folder policy rejected" >&2; return 1; }; ')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("folder policy rejected", proc.stderr)
        self.assertFalse(self.calls.exists())

    def test_install_linux_noninteractive_once_and_nonlinux_skip(self):
        (self.bin / "aria2c").unlink()
        prelude = """
command() {
  if [[ "$1" == -v && "$2" == aria2c ]]; then [[ -n "${INSTALLED:-}" ]]; return; fi
  if [[ "$1" == -v && "$2" == apt-get ]]; then return 0; fi
  builtin command "$@"
}
uname() { echo Linux; }
sudo() { [[ "$1" == -n ]]; shift; "$@"; }
timeout() {
  printf '%s\\n' "$*" >> "$TEST_CALLS"
  [[ "$(cat)" == "" ]]
  INSTALLED=1
}
ensure_aria2c; ensure_aria2c
"""
        proc = self.shell(prelude)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr.count("Enabling fast"), 1)
        args = self.calls.read_text()
        self.assertIn("DEBIAN_FRONTEND=noninteractive", args)
        self.assertIn("install -y aria2", args)
        self.assertIn("30s", args)
        self.assertEqual(len(args.splitlines()), 1)
        self.calls.unlink()
        proc = self.shell(prelude.replace("echo Linux", "echo Darwin").replace(
            "ensure_aria2c; ensure_aria2c", "if ensure_aria2c; then exit 9; fi"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Using sequential", proc.stderr)
        self.assertFalse(self.calls.exists())

    def test_failed_install_and_dnf_fallback(self):
        (self.bin / "aria2c").unlink()
        body = """
command() {
  if [[ "$1" == -v && "$2" == aria2c ]]; then return 1; fi
  if [[ "$1" == -v && "$2" == apt-get ]]; then return 1; fi
  if [[ "$1" == -v && "$2" == dnf ]]; then return 0; fi
  builtin command "$@"
}
uname() { echo Linux; }
timeout() { printf '%s\\n' "$*" >> "$TEST_CALLS"; echo 'permission denied'; return 1; }
if ensure_aria2c; then exit 9; fi
if ensure_aria2c; then exit 9; fi
"""
        proc = self.shell(body)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr.count("Using sequential"), 1)
        self.assertNotIn("permission denied", proc.stdout + proc.stderr)
        self.assertIn("dnf install -y aria2", self.calls.read_text())
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)

    def test_launchers_use_local_paths_and_document_flag(self):
        for launcher in ("claude-local", "coder-local", "hermes-local"):
            with self.subTest(launcher=launcher):
                text = (ROOT / launcher).read_text()
                self.assertIn('download_model_fast "$hf" "$CACHE_DIR" model_path', text)
                self.assertIn('"$LLAMA_SERVER" -m "$model_path"', text)
                self.assertNotIn('"$LLAMA_SERVER" -hf', text)
                proc = subprocess.run(["bash", str(ROOT / launcher), "--help"],
                                      capture_output=True, text=True, timeout=10)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("--no-parallel", proc.stdout)


if __name__ == "__main__":
    unittest.main()
