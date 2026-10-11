#!/usr/bin/env python3
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pushbutton_download", ROOT / "lib/pushbutton_download.py")
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)
source = download.gguf_source
tempfile.tempdir = str(ROOT)
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

    def test_quant_selects_standard_build_unless_mtp_requested(self):
        for quant in ("IQ3_XXS", "IQ2_S", "IQ2_XS"):
            standard = f"Qwen3.8-27B-GSQ-RCO-{quant}.gguf"
            mtp = f"Qwen3.8-27B-GSQ-RCO-{quant}-mtp.gguf"
            metadata = {"siblings": [
                {"rfilename": mtp, "size": 48},
                {"rfilename": standard, "size": 32},
            ]}
            for selector, expected in ((quant, standard), (quant.lower(), standard),
                                       (standard, standard), (quant + "-mtp", mtp),
                                       (mtp, mtp)):
                with self.subTest(selector=selector):
                    self.assertEqual(download.select_files(metadata, selector)[0]["name"], expected)
            self.assertEqual(len(download.select_files(metadata, quant + "," + mtp)), 2)
            with self.assertRaisesRegex(ValueError, "No GGUF files"):
                download.select_files({"siblings": metadata["siblings"][:1]}, quant)

    def test_standard_and_mtp_shards_selected_separately(self):
        metadata = {"siblings": [
            {"rfilename": f"Qwen-IQ3_XXS{suffix}-{index:05d}-of-00002.gguf", "size": 32}
            for suffix in ("", "-mtp") for index in (1, 2)
        ]}
        for selector, start in (("IQ3_XXS", 0), ("IQ3_XXS-mtp", 2),
                                (metadata["siblings"][2]["rfilename"], 2)):
            with self.subTest(selector=selector):
                self.assertEqual(
                    [f["name"] for f in download.select_files(metadata, selector)],
                    [f["rfilename"] for f in metadata["siblings"][start:start + 2]])
        metadata["siblings"].pop(1)
        with self.assertRaisesRegex(ValueError, "incomplete GGUF shard set"):
            download.select_files(metadata, "IQ3_XXS")

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
            with mock.patch.object(source, "safe_open",
                                   return_value=io.BytesIO(json.dumps(METADATA).encode())), \
                    contextlib.redirect_stderr(output):
                manifest = download.prepare("owner/repo:UD-Q4_K_XL", tmp, str(work))
            self.assertIn("0.0 GiB total", output.getvalue())
            self.assertEqual(manifest["download_bytes"], 80)
            self.assertIn(REVISION, (work / "urls").read_text())

    def test_reported_large_disk_uses_fixed_reserve_for_both_models(self):
        gib = 1024**3
        free = int(36.30 * gib)
        with tempfile.TemporaryDirectory() as tmp:
            for size in (int(7.04 * gib), int(10.73 * gib)):
                manifest = {"directory": tmp, "files": [{"name": "model.gguf", "size": size}]}
                disk = shutil._ntuple_diskusage(959218776 * 1024, 0, free)
                with mock.patch.object(download.shutil, "disk_usage", return_value=disk):
                    download.validate_space(manifest)
                free -= size
            self.assertGreater(free, 10 * gib)

    def test_fixed_reserve_boundary_on_large_and_small_disks(self):
        gib = 1024**3
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {"directory": tmp, "files": [{"name": "model.gguf", "size": 7 * gib}]}
            for total in (100 * gib, 1024 * gib):
                for reuse_partial in (False, True):
                    with self.subTest(total=total, reuse_partial=reuse_partial):
                        disk = shutil._ntuple_diskusage(total, total - 17 * gib, 17 * gib)
                        with mock.patch.object(download.shutil, "disk_usage", return_value=disk):
                            download.validate_space(manifest, reuse_partial=reuse_partial)
                        disk = disk._replace(free=disk.free - 1)
                        with mock.patch.object(download.shutil, "disk_usage", return_value=disk):
                            with self.assertRaisesRegex(ValueError, r"\+ 10.0 GiB reserve"):
                                download.validate_space(manifest, reuse_partial=reuse_partial)

    def test_existing_blob_reused_without_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp) / REVISION
            directory.mkdir()
            digest = "b" * 64
            blob = pathlib.Path(tmp) / "blobs" / digest
            blob.parent.mkdir()
            blob.write_bytes(b"x" * 16)
            with contextlib.redirect_stderr(io.StringIO()), \
                    mock.patch.object(pathlib.Path, "is_relative_to",
                                      side_effect=AttributeError("not available"), create=True):
                download.reuse_blobs(directory, [{"name": "model.gguf", "size": 16, "sha256": digest}])
            self.assertEqual(blob.stat().st_ino, (directory / "model.gguf").stat().st_ino)

    def test_checksum_and_private_url_file(self):
        metadata = {"sha": REVISION, "siblings": [
            {"rfilename": "Q4_K.gguf", "size": 16, "lfs": {"sha256": "b" * 64}}]}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(source, "safe_open",
                                  return_value=io.BytesIO(json.dumps(metadata).encode())), \
                contextlib.redirect_stderr(io.StringIO()):
            download.prepare("owner/repo:Q4_K", tmp, tmp)
            urls = pathlib.Path(tmp) / "urls"
            self.assertIn("checksum=sha-256=" + "b" * 64, urls.read_text())
            self.assertEqual(urls.stat().st_mode & 0o777, 0o600)

    def test_custom_and_xdg_token_paths(self):
        for custom in (True, False):
            with self.subTest(custom=custom), tempfile.TemporaryDirectory() as tmp:
                token = pathlib.Path(tmp) / "huggingface" / "token"
                token.parent.mkdir()
                token.write_text("example-auth-value")
                env = {"XDG_CACHE_HOME": tmp}
                if custom:
                    env["HF_TOKEN_PATH"] = "$XDG_CACHE_HOME/huggingface/token"
                with mock.patch.dict(os.environ, env, clear=True), \
                        mock.patch.object(source, "safe_open",
                                          return_value=io.BytesIO(json.dumps(METADATA).encode())) as api, \
                        contextlib.redirect_stderr(io.StringIO()):
                    download.prepare("owner/repo:UD-Q4_K_XL", tmp, tmp)
                self.assertEqual(api.call_args.args[0].get_header("Authorization"),
                                 "Bearer " + "example-auth-value")
                self.assertIn("header=Authorization: " + "Bearer " + "example-auth-value",
                              (pathlib.Path(tmp) / "urls").read_text())

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp) / "cache"
            directory.mkdir()
            (directory / "escape").symlink_to(tmp, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes cache"), \
                    mock.patch.object(pathlib.Path, "is_relative_to",
                                      side_effect=AttributeError("not available"), create=True):
                download.reuse_blobs(directory, [{"name": "escape/model.gguf", "size": 16}])


def gguf_fixture(arch="llama", **extra):
    metadata = {"general.architecture": arch, arch + ".context_length": 32768,
                arch + ".block_count": 32, arch + ".embedding_length": 4096,
                arch + ".attention.head_count": 32, arch + ".attention.head_count_kv": 8,
                **extra}

    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
    for key, value in metadata.items():
        data += string(key)
        data += struct.pack("<I", 8 if isinstance(value, str) else 4)
        data += string(value) if isinstance(value, str) else struct.pack("<I", value)
    return data


class SourceTests(unittest.TestCase):
    def response(self, data, status=200, **headers):
        response = io.BytesIO(data)
        response.status = status
        response.headers = {"Content-Length": str(len(data)), **headers}
        return response

    def test_supported_syntax_and_pinned_shards(self):
        for spec in ("hf:owner/repo:Q4_K_M", "owner/repo:Q4_K_M",
                     "owner/repo/Qwen-Q4_K_M.gguf",
                     f"https://huggingface.co/owner/repo/resolve/{REVISION}/Qwen-Q4_K_M.gguf",
                     "https://models.example/model.gguf"):
            self.assertTrue(source.is_source(spec), spec)
        with mock.patch.object(source, "fetch_metadata", return_value=METADATA) as api:
            result = source.resolve_files(
                f"https://huggingface.co/owner/repo/resolve/{REVISION}/"
                + METADATA["siblings"][0]["rfilename"])
        self.assertEqual(result["weight_bytes"], 80)
        self.assertEqual(len(result["files"]), 2)
        api.assert_called_once_with("owner/repo", REVISION)
        self.assertIn(REVISION, result["download_spec"])

    def test_unsafe_sources(self):
        for spec in ("http://example.com/a.gguf", "https://localhost/a.gguf",
                     "https://127.0.0.1/a.gguf", "https://[::1]/a.gguf",
                     "https://169.254.169.254/a.gguf", "******example.com/a.gguf",
                     "https://example.com/a.gguf?token=secret", "https://example.com/a.gguf\n",
                     "owner/repo/../a.gguf", "hf:owner/repo:Q4\n"):
            self.assertFalse(source.is_source(spec), spec)
        with mock.patch.object(source.socket, "getaddrinfo",
                               return_value=[(2, 1, 6, "", ("10.0.0.5", 443))]):
            with self.assertRaisesRegex(ValueError, "non-public"):
                source.validate_url("https://models.example/a.gguf")

    def test_redirect_revalidates_and_strips_token(self):
        handler = source._SafeRedirect()
        req = download.urllib.request.Request(
            "https://huggingface.co/owner/repo/resolve/main/a.gguf",
            headers={"Authorization": "******"})
        with mock.patch.object(source, "validate_url") as validate:
            redirect = handler.redirect_request(req, None, 302, "Found", {},
                                                "https://cdn.example/a.gguf?Signature=abc")
        validate.assert_called_once()
        self.assertIsNone(redirect.get_header("Authorization"))
        with mock.patch.object(source.socket, "getaddrinfo",
                               return_value=[(2, 1, 6, "", ("127.0.0.1", 443))]):
            with self.assertRaises(ValueError):
                handler.redirect_request(req, None, 302, "Found", {},
                                         "https://private.example/a.gguf")

    def test_actual_weight_and_unequal_kv(self):
        data = gguf_fixture(**{"llama.attention.key_length": 128,
                               "llama.attention.value_length": 64})
        with mock.patch.object(source, "fetch_metadata", return_value={
                "sha": REVISION, "siblings": [{"rfilename": "model-Q4_K_M.gguf", "size": len(data)}]}), \
                mock.patch.object(source, "safe_open", side_effect=lambda *a, **k: self.response(data)):
            result = source.resolve_source("owner/repo:Q4_K_M")
        self.assertEqual(result["weight_bytes"], len(data))
        self.assertEqual(result["context"], 32768)
        self.assertEqual(result["kv_dim"], 768)
        self.assertEqual(result["kv_key_dim"], 1024)
        self.assertEqual(result["kv_value_dim"], 512)
        self.assertEqual(result["quant"], "Q4_K_M")

    def test_hybrid_and_unsupported_architecture(self):
        data = gguf_fixture("qwen35", **{"qwen35.full_attention_interval": 4,
                                       "qwen35.attention.key_length": 256,
                                       "qwen35.attention.value_length": 256})
        result = source.architecture(source.read_gguf_metadata(data))
        self.assertEqual(result["attn_layers"], 8)
        for metadata in ({"general.architecture": "unknown"},
                         source.read_gguf_metadata(gguf_fixture("qwen35")),
                         {"general.architecture": "llama"}):
            with self.assertRaises(ValueError):
                source.architecture(metadata)
        with self.assertRaises(ValueError):
            source.read_gguf_metadata(b"not a GGUF")

    def test_public_url_size_and_cache_identity(self):
        data = gguf_fixture()
        with mock.patch.object(source, "validate_url"), \
                mock.patch.object(source, "safe_open", side_effect=lambda *a, **k: self.response(
                    data[:1], 206, **{"Content-Range": f"bytes 0-0/{len(data)}", "ETag": '"one"'})):
            result = source.resolve_files("https://models.example/a.gguf")
        self.assertEqual(result["weight_bytes"], len(data))
        self.assertIsNone(result["repo"])
        self.assertEqual(len(result["revision"]), 64)

    def test_resumable_safe_download_and_no_token_to_public_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "model.gguf"
            partial = pathlib.Path(str(path) + ".part")
            partial.write_bytes(b"abc")
            manifest = {"directory": tmp, "repo": None, "files": [
                {"name": path.name, "size": 6, "url": "https://models.example/model.gguf",
                 "etag": '"stable"'}]}
            with mock.patch.object(source, "safe_open", return_value=self.response(
                    b"def", 206, **{"Content-Range": "bytes 3-5/6"})) as api, \
                    mock.patch.dict(os.environ, {"HF_TOKEN": "example-token"}):
                download.download_safe(manifest)
            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertEqual(api.call_args.args[0].get_header("Range"), "bytes=3-")
            self.assertIsNone(api.call_args.args[0].get_header("Authorization"))
            self.assertFalse(partial.exists())

    def test_bounded_prefix_local_cache_and_missing_metadata(self):
        data = gguf_fixture()
        metadata = {"sha": REVISION, "siblings": [
            {"rfilename": "model-Q4_K_M.gguf", "size": len(data)}]}
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp) / "models--owner--repo" / REVISION
            directory.mkdir(parents=True)
            (directory / "model-Q4_K_M.gguf").write_bytes(data)
            with mock.patch.dict(os.environ, {"PUSHBUTTON_LOCAL_CACHE": tmp}), \
                    mock.patch.object(source, "fetch_metadata", return_value=metadata), \
                    mock.patch.object(source, "safe_open", side_effect=AssertionError("network")):
                self.assertEqual(source.resolve_source("owner/repo:Q4_K_M")["context"], 32768)
        with mock.patch.object(source, "fetch_metadata", return_value=metadata), \
                mock.patch.object(source, "safe_open", side_effect=lambda *a, **k: self.response(
                    data[:24], **{"Content-Length": str(len(data))})):
            with self.assertRaisesRegex(ValueError, "bounded"):
                source.resolve_source("owner/repo:Q4_K_M")

    def test_pinned_revision_mismatch_and_exact_path(self):
        with mock.patch.object(source, "fetch_metadata", return_value=METADATA):
            with self.assertRaisesRegex(ValueError, "pinned revision"):
                source.resolve_files("https://huggingface.co/owner/repo/resolve/"
                                     + "b" * 40 + "/Qwen-Q4_K_M.gguf")
        metadata = {"siblings": [{"rfilename": "one/model-Q4.gguf", "size": 16},
                                  {"rfilename": "two/model-Q4.gguf", "size": 16}]}
        self.assertEqual(download.select_files(metadata, "one/model-Q4.gguf"),
                         [{"name": "one/model-Q4.gguf", "size": 16, "sha256": None}])

    def test_public_download_wrapper_returns_local_path(self):
        data = gguf_fixture()
        resolved = {"repo": None, "revision": "a" * 64, "files": [
            {"name": "model.gguf", "size": len(data), "url": "https://models.example/model.gguf"}]}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(source, "resolve_files", return_value=resolved), \
                mock.patch.object(source, "safe_open", side_effect=lambda *a, **k: self.response(data)):
            path = source.download("https://models.example/model.gguf", cache_dir=tmp)
            self.assertIsInstance(path, pathlib.Path)
            self.assertEqual(path.read_bytes(), data)

    def test_default_quant_and_recurrent_zero_kv(self):
        self.assertEqual(source.parse_source("hf:owner/repo")["selector"], "Q4_K_M")
        with mock.patch.object(source, "fetch_metadata", return_value=METADATA):
            self.assertEqual(source.resolve_files("owner/repo")["files"][0]["name"],
                             "Qwen-Q4_K_M.gguf")
        for arch in ("mamba", "mamba2", "rwkv6", "rwkv7"):
            result = source.architecture(source.read_gguf_metadata(gguf_fixture(arch)))
            self.assertEqual(result["attn_layers"], 0)
            self.assertEqual(result["kv_dim"], 0)
            self.assertEqual(result["context"], 32768)


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
import io,json,os,pathlib,runpy,shutil,sys,urllib.request,socket
sys.path.insert(0,{str(ROOT / 'lib')!r})
import gguf_source
class Response(io.BytesIO):
    status=200
def safe_open(req,**kwargs):
    metadata=json.load(open(os.environ['TEST_METADATA']))
    if '/api/models/' in req.full_url:
        return io.BytesIO(json.dumps(metadata).encode())
    if os.environ.get('TEST_HF_FAIL'): raise OSError('test sequential failure')
    name=urllib.parse.unquote(req.full_url.split('/resolve/')[1].split('/',1)[1])
    item=next(f for f in metadata['siblings'] if f['rfilename']==name)
    size=item.get('size') or item['lfs']['size']
    with open(os.environ['TEST_CALLS'],'a') as stream: stream.write('hf '+json.dumps([name])+'\\n')
    print('sequential: 100%',file=sys.stderr)
    response=Response(b'x'*size)
    response.headers={{'Content-Length':str(size)}}
    return response
gguf_source.safe_open=safe_open
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
    with path.open('wb') as stream: stream.truncate(f['size'])
    pathlib.Path(str(path)+'.aria2').unlink(missing_ok=True)
    if os.environ.get('TEST_ARIA_PARTIAL_FAILURE'): sys.exit(1)
print('[parallel chunks] 45% ETA: 3m42s',file=sys.stderr)
""")
        self.executable("hf", f"""#!{sys.executable}
import json,os,pathlib,sys
with open(os.environ['TEST_CALLS'],'a') as s: s.write('hf '+json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('TEST_HF_FAIL'): sys.exit(1)
directory=pathlib.Path(sys.argv[sys.argv.index('--local-dir')+1])
item=next(f for f in json.load(open(os.environ['TEST_METADATA']))['siblings'] if f['rfilename']==sys.argv[3])
path=directory/item['rfilename']
with path.open('wb') as stream: stream.truncate(item.get('size') or item['lfs']['size'])
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

    def test_standard_build_download_with_optional_mtp(self):
        standard = "Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"
        mtp = "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
        self.metadata.write_text(json.dumps({"sha": REVISION, "siblings": [
            {"rfilename": standard, "size": 32},
            {"rfilename": mtp, "size": 48},
        ]}))
        for sequential in (False, True):
            with self.subTest(sequential=sequential):
                self.env["PUSHBUTTON_NO_PARALLEL"] = "1" if sequential else "0"
                proc = self.shell(
                    'download_model_fast ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_XXS '
                    '"$2" MODEL_PATH; printf "%s\\n" "$MODEL_PATH"')
                self.assertEqual(proc.returncode, 0, proc.stderr)
                path = pathlib.Path(proc.stdout.strip())
                self.assertEqual(path.name, standard)
                self.assertEqual(path.stat().st_size, 32)
                self.assertFalse((path.parent / mtp).exists())
                shutil.rmtree(self.cache)

    def test_fallback_skips_completed_shards(self):
        self.env["TEST_ARIA_PARTIAL_FAILURE"] = "1"
        proc = self.fast()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        hf_calls = [line for line in self.calls.read_text().splitlines() if line.startswith("hf ")]
        self.assertEqual(len(hf_calls), 1)
        self.assertIn("00002-of-00002.gguf", hf_calls[0])

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

    def test_plan_and_download_agree_on_reported_disk_space(self):
        gib = 1024**3
        models = (
            ("ornith-ai/Ornith-1.5-9B-GGUF", "Q6_K", "Ornith-Q6_K.gguf", int(7.04 * gib)),
            ("unsloth/Qwen3.6-35B-A3B-GGUF", "UD-IQ2_M", "Qwen-UD-IQ2_M.gguf", int(10.73 * gib)),
        )
        metadata = {"sha": REVISION, "siblings": [
            {"rfilename": name, "size": size} for _, _, name, size in models]}
        self.metadata.write_text(json.dumps(metadata))
        plan = self.base / "plan.json"
        plan.write_text(json.dumps({"servers": [
            {"profile": {"hf_spec": repo + ":" + quant}} for repo, quant, _, _ in models]}))
        self.env.update(TEST_CACHE=str(self.cache), HF_HUB_OFFLINE="1")
        self.executable("python3", f"""#!{sys.executable}
import io,json,os,pathlib,runpy,shutil,sys,urllib.request
sys.path.insert(0,{str(ROOT / 'lib')!r})
import gguf_source,pushbutton_download
gguf_source.safe_open=lambda *a,**k: io.BytesIO(open(os.environ['TEST_METADATA'],'rb').read())
def download_safe(manifest):
    for f in manifest['files']:
        path=pathlib.Path(manifest['directory'])/f['name']
        if pushbutton_download.complete(path,f['size']): continue
        if os.environ.get('TEST_HF_FAIL'): raise OSError('test failure')
        with path.open('wb') as stream: stream.truncate(f['size'])
        pathlib.Path(str(path)+'.aria2').unlink(missing_ok=True)
        with open(os.environ['TEST_CALLS'],'a') as stream: stream.write('hf '+f['name']+'\\n')
pushbutton_download.download_safe=download_safe
def disk_usage(path):
    total=959218776*1024
    used=sum(p.stat().st_size for p in pathlib.Path(os.environ['TEST_CACHE']).rglob('*.gguf'))
    free=int(os.environ['TEST_FREE_BYTES'])-used
    return shutil._ntuple_diskusage(total,total-free,free)
shutil.disk_usage=disk_usage
sys.argv=sys.argv[1:]
if sys.argv[0]=='-':
    exec(compile(sys.stdin.read(),'<stdin>','exec'))
else:
    pushbutton_download.main()
""")
        self.executable("stat", f"""#!{sys.executable}
import os,pathlib
used=sum(p.stat().st_size for p in pathlib.Path(os.environ['TEST_CACHE']).rglob('*.gguf'))
print(int(os.environ['TEST_FREE_BYTES'])-used,1)
""")
        for mode in ("parallel", "sequential", "fallback", "insufficient"):
            with self.subTest(mode=mode):
                shutil.rmtree(self.cache, ignore_errors=True)
                self.calls.unlink(missing_ok=True)
                for repo, _, _, _ in models:
                    root = self.cache / ("models--" + repo.replace("/", "--"))
                    root.mkdir(parents=True)
                    (root / "metadata.json").write_text(json.dumps(metadata))
                self.env["TEST_FREE_BYTES"] = str(int(36.30 * gib) if mode != "insufficient" else 27 * gib)
                self.env["PUSHBUTTON_NO_PARALLEL"] = "1" if mode == "sequential" else "0"
                self.env["TEST_ARIA_FAIL"] = "1" if mode == "fallback" else ""
                commands = [
                    f'source "{ROOT / "lib/pushbutton_folders.sh"}"',
                    f'validate_plan_download_space "{plan}" "$2" || exit $?',
                ]
                for repo, quant, _, _ in models:
                    commands.append(f'download_model_fast {repo}:{quant} "$2" MODEL_PATH || exit $?')
                    commands.append('printf "%s\\n" "$MODEL_PATH"')
                proc = self.shell("; ".join(commands))
                self.assertIn("17.77 GiB remaining + 10 GiB safety reserve", proc.stderr)
                if mode == "insufficient":
                    self.assertEqual(proc.returncode, 2, proc.stderr)
                    self.assertIn("Insufficient cache space", proc.stderr)
                    self.assertFalse(self.calls.exists())
                    continue
                self.assertEqual(proc.returncode, 0, proc.stderr)
                paths = [pathlib.Path(line) for line in proc.stdout.splitlines()]
                self.assertEqual(len(paths), 2)
                for path, (_, _, name, size) in zip(paths, models):
                    self.assertEqual(path.name, name)
                    self.assertEqual(path.stat().st_size, size)
                calls = self.calls.read_text()
                self.assertEqual(calls.count("hf "), 0 if mode == "parallel" else 2)
                self.assertEqual(len((self.cache / ".download-log").read_text().splitlines()), 2)

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
        self.env["TEST_CACHE"] = str(self.cache)
        proc = self.fast('validate_download_space() { '
                         '[[ "$1" == "$TEST_CACHE" && "$2" == owner/repo:UD-Q4_K_XL ]] || return 99; '
                         'echo "folder policy rejected" >&2; return 2; }; ')
        self.assertEqual(proc.returncode, 2)
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
