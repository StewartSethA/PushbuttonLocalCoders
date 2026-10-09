import contextlib
import importlib.machinery
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from claude_local_resources import Host


def load(name, filename):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


backend = load("flash_backend", "pushbutton-backend")
bench = load("flash_bench", "pushbutton-bench")


def capacity(engine="sglang-v100", **overrides):
    entry = {"backend": engine, "model": "qwen3.8-flash-next", "profile": "mtp",
             "artifact": backend.FLASH_ARTIFACTS[engine], "source": "mock fixture, NOT a real measurement",
             "runtime_revision": "fixture-runtime", "host_mib": 120000, "gpu_mib": [31800] * 4,
             "max_context": 65536, "max_concurrency": 4, "cpu_threads": 4}
    entry.update(overrides)
    return {"version": 1, "placements": [entry]}


class BackendTests(unittest.TestCase):
    def setUp(self):
        backend.DRY = False

    def tearDown(self):
        backend.DRY = False

    def test_four_cards_required_even_force_and_dry_run(self):
        for dry in (True, False):
            backend.DRY = dry
            for engine in backend.FLASH_BACKENDS:
                for selected in ([0, 1], [0, 1, 2, 2], [0, 1, 2, 3, 4]):
                    with self.subTest(engine=engine, selected=selected, dry=dry), self.assertRaises(SystemExit):
                        backend.validate(engine, "qwen3.8-flash-next", selected, True)

    def test_live_specialized_requires_capacity_even_force(self):
        with self.assertRaisesRegex(SystemExit, "calibrated"):
            backend.validate_capacity("sglang-v100", "qwen3.8-flash-next", [0, 1, 2, 3],
                                      65536, 2, None, None)

    def check_capacity(self, data, *, context=65536, concurrency=2, host=None, rows=None):
        engine = data["placements"][0]["backend"]
        host = host or Host(256000, None, 256000, tuple(range(8)), 8, ())
        rows = rows or [{"index": i, "compute_cap": "7.0" if engine == "sglang-v100" else "8.6",
                         "memory_total_mib": 32768, "memory_free_mib": 32700} for i in range(4)]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory) / "capacity.json"
            path.write_text(json.dumps(data))
            with mock.patch("claude_local_resources.host_inventory", return_value=host), \
                    mock.patch.object(backend, "gpu_rows", return_value=rows), \
                    mock.patch.object(backend, "runtime_revision", return_value="fixture-runtime"):
                return backend.validate_capacity(engine, "qwen3.8-flash-next", [0, 1, 2, 3],
                                                 context, concurrency, None, path)

    def test_capacity_context_slots_ram_gpu_and_artifact_fail_closed(self):
        self.assertEqual(self.check_capacity(capacity())["max_concurrency"], 4)
        for overrides in ({"max_context": 32768}, {"max_concurrency": 1},
                          {"artifact": "unverified"}, {"host_mib": 999999}, {"gpu_mib": [32768] * 4},
                          {"source": ""}, {"cpu_threads": 9}, {"gpu_mib": [1000] * 4},
                          {"runtime_revision": "different-runtime"}):
            with self.subTest(overrides=overrides), self.assertRaises(SystemExit):
                self.check_capacity(capacity(**overrides))
        with self.assertRaises(SystemExit):
            self.check_capacity(capacity(), host=Host(256000, 6000, 6000, tuple(range(8)), 8, ()))

    def test_3090_recipe_known_host_floor_and_context_ceiling(self):
        self.assertEqual(self.check_capacity(capacity("vllm-flashnext-3090"))["profile"], "mtp")
        with self.assertRaisesRegex(SystemExit, "97 GiB"):
            self.check_capacity(capacity("vllm-flashnext-3090", host_mib=1000))
        with self.assertRaisesRegex(SystemExit, "context limit"):
            self.check_capacity(capacity("vllm-flashnext-3090"), context=131072)

    def launch(self, engine, concurrency=2, profile=None):
        backend.DRY = True
        with mock.patch.object(backend, "exec_cmd") as execute, \
                mock.patch.object(backend, "run", return_value="mock-snapshot"), \
                mock.patch.object(backend, "CACHE", ROOT / ".backend-test-cache"):
            # Existing parent cache avoids creating dry-run artifacts.
            with mock.patch.object(pathlib.Path, "mkdir"):
                backend.serve(engine, "qwen3.8-flash-next", [2, 6, 3, 5], 20880,
                              65536, profile, concurrency)
        return execute.call_args

    def test_sglang_concurrency_is_not_silently_one(self):
        cmd = self.launch("sglang-v100").args[0]
        self.assertEqual(cmd[cmd.index("--max-running-requests") + 1], "2")
        self.assertIn("--disable-cuda-graph", cmd)
        self.assertNotIn("--cuda-graph-bs", cmd)
        cmd = self.launch("sglang-v100", concurrency=1).args[0]
        self.assertEqual(cmd[cmd.index("--cuda-graph-bs") + 1], "1")

    def test_docker_physical_group_is_one_csv_device_field(self):
        with mock.patch.object(backend.os, "execvp") as execute, mock.patch.object(backend, "say"):
            backend.exec_cmd(["docker", "run", "--gpus", "device=0,1,2,3", "image"])
        self.assertEqual(execute.call_args.args[1][3], '"device=0,1,2,3"')

    def test_3090_foreground_recipe_preserves_gpu_order_context_and_mtp(self):
        call = self.launch("vllm-flashnext-3090", concurrency=4)
        cmd, env = call.args[0], call.kwargs["env"]
        self.assertEqual(cmd[1], "serve")
        self.assertTrue(cmd[0].endswith("/bin/vllm"))
        self.assertNotIn("serve.sh", cmd)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,6,3,5")
        self.assertEqual(env["VLLM_PLE_CPU_OFFLOAD"], "1")
        self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "65536")
        self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "4")
        self.assertEqual(json.loads(cmd[cmd.index("--speculative-config") + 1]),
                         {"method": "mtp", "num_speculative_tokens": 4})
        self.assertNotIn("--speculative-config", self.launch("vllm-flashnext-3090", profile="bf16").args[0])

    def test_llama_context_is_per_slot_and_flash_next_is_iq4_xs(self):
        cmd = self.launch("llama.cpp", concurrency=2).args[0]
        self.assertEqual(cmd[cmd.index("-np") + 1], "2")
        self.assertEqual(cmd[cmd.index("-c") + 1], "131072")
        self.assertIn("unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS", cmd)

    def test_live_serve_rejects_occupied_port_before_launch_or_download(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        try:
            for engine in backend.FLASH_BACKENDS:
                with self.subTest(engine=engine), \
                        mock.patch.object(backend, "exec_cmd") as execute, \
                        mock.patch.object(backend, "ensure_hf") as download, \
                        self.assertRaisesRegex(RuntimeError, "occupied"):
                    backend.serve(engine, "qwen3.8-flash-next", [0, 1, 2, 3],
                                  server.server_port, 65536, None)
                execute.assert_not_called()
                download.assert_not_called()
        finally:
            server.server_close()

    def test_flash_adapters_forward_launch_identity_including_docker(self):
        with mock.patch.dict(backend.os.environ, {"PUSHBUTTON_BENCH_MODEL_ID": "unique-launch"}):
            for engine in backend.FLASH_BACKENDS:
                cmd = self.launch(engine).args[0]
                self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "unique-launch")


class BenchmarkTests(unittest.TestCase):
    def test_both_stream_readers_measure_reasoning_only_and_mixed(self):
        for reader in (bench.request_stream, bench.request_stream_compat):
            for content in ("", "final answer"):
                for reported in (True, False):
                    chunks = [{"choices": [{"delta": {"reasoning_content": "think carefully "}}]},
                              {"choices": [{"delta": {"reasoning_content": "more reasoning",
                                                       "content": content}}]}]
                    if reported:
                        chunks.append({"usage": {"prompt_tokens": 20, "completion_tokens": 8}})
                    response = mock.Mock(status=200)
                    response.readline.side_effect = [
                        b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks
                    ] + [b"data: [DONE]\n\n", b""]
                    connection = mock.Mock()
                    connection.getresponse.return_value = response
                    with self.subTest(reader=reader.__name__, content=content, reported=reported), \
                            mock.patch.object(bench.http.client, "HTTPConnection", return_value=connection), \
                            mock.patch.object(bench.time, "perf_counter", side_effect=[10, 11, 14]):
                        result = reader(1234, "model", "prompt", 8)
                    self.assertEqual(result["ttft_s"], 1)
                    self.assertEqual(result["chars"], len(content))
                    self.assertEqual(result["reasoning_chars"], len("think carefully more reasoning"))
                    self.assertEqual(result["completion_tokens_estimated"], not reported)
                    tokens = 8 if reported else round((4 + len(content.split())) * 1.33)
                    self.assertEqual(result["completion_tokens"], tokens)
                    self.assertEqual(result["decode_tok_s"], (tokens - 1) / 3)

    def test_empty_stream_rejected_by_both_readers(self):
        for reader in (bench.request_stream, bench.request_stream_compat):
            connection = mock.Mock()
            connection.getresponse.return_value.status = 200
            connection.getresponse.return_value.readline.side_effect = [b"data: [DONE]\n\n", b""]
            with mock.patch.object(bench.http.client, "HTTPConnection", return_value=connection), \
                    self.assertRaisesRegex(RuntimeError, "synthetic"):
                reader(1234, "model", "prompt", 8)

    def test_stream_options_retry_preserves_reasoning_metrics(self):
        rejected = mock.Mock()
        rejected.getresponse.return_value.status = 400
        rejected.getresponse.return_value.read.return_value = b"unsupported stream_options"
        accepted = mock.Mock()
        accepted.getresponse.return_value.status = 200
        accepted.getresponse.return_value.readline.side_effect = [
            b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n',
            b'data: {"usage":{"completion_tokens":8}}\n\n', b""]
        with mock.patch.object(bench.http.client, "HTTPConnection", side_effect=[rejected, accepted]), \
                mock.patch.object(bench.time, "perf_counter", side_effect=[9, 10, 11, 14]):
            result = bench.request_stream(1234, "model", "prompt", 8)
        self.assertEqual(result["ttft_s"], 1)
        self.assertEqual(result["completion_tokens"], 8)
        self.assertFalse(result["completion_tokens_estimated"])
        self.assertEqual(result["reasoning_chars"], 8)
        retry_body = json.loads(accepted.request.call_args.kwargs["body"])
        self.assertNotIn("stream_options", retry_body)

    def test_wait_ready_rejects_early_exit_and_exit_during_http(self):
        for polls in ([1], [None, 1, 1]):
            proc = mock.Mock(returncode=1)
            proc.poll.side_effect = polls
            response = mock.MagicMock()
            response.__enter__.return_value.status = 200
            with mock.patch.object(bench.urllib.request, "urlopen", return_value=response) as request, \
                    mock.patch.object(bench.time, "sleep"), \
                    self.assertRaisesRegex(RuntimeError, "exited"):
                bench.wait_ready(1234, proc)
            if polls == [1]: request.assert_not_called()

    def test_wait_ready_rejects_foreign_models_even_with_live_child(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        response = mock.MagicMock()
        response.__enter__.return_value.status = 200
        with mock.patch.object(bench.urllib.request, "urlopen", return_value=response), \
                mock.patch.object(bench.json, "load", return_value={"data": [{"id": "foreign"}]}), \
                mock.patch.object(bench.time, "time", side_effect=[0, 0, 2]), \
                mock.patch.object(bench.time, "sleep"), \
                self.assertRaisesRegex(TimeoutError, "does not match"):
            bench.wait_ready(1234, proc, timeout=1, expected_model="this-launch")

    def test_occupied_port_never_launches_or_benchmarks_foreign_server(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        try:
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                argv = ["pushbutton-bench", "flash-next", "--backend", "sglang-v100",
                        "--port-base", str(server.server_port), "--output-dir", directory]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(bench, "hardware", return_value={}), \
                        mock.patch.object(bench, "backend_check", return_value=(True, "{}")), \
                        mock.patch.object(bench.subprocess, "Popen") as launch, \
                        mock.patch.object(bench, "run_case") as measure, \
                        self.assertRaisesRegex(RuntimeError, "occupied"):
                    bench.main()
                launch.assert_not_called()
                measure.assert_not_called()
                self.assertFalse(list(pathlib.Path(directory).glob("*.json")))
        finally:
            server.server_close()

    def test_empty_stream_cannot_generate_synthetic_metrics(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            bench.finish_metrics(1, None, 2, {}, "")

    def test_requested_context_and_concurrency_include_c1(self):
        cases = bench.cases("standard", context=131072, concurrency=2)
        self.assertEqual({c["concurrency"] for c in cases}, {1, 2})
        context_cases = [c for c in cases if c["name"].startswith("context-131072-")]
        self.assertEqual({c["concurrency"] for c in context_cases}, {1, 2})
        self.assertTrue(all(c["words"] + c["output"] < 131072 for c in context_cases))

    def test_custom_prompt_scale_and_output(self):
        cases = bench.cases("quick", context=65536, concurrency=3, prompt_words=30000, output_tokens=128)
        self.assertEqual([(c["words"], c["output"], c["concurrency"]) for c in cases],
                         [(30000, 128, 1), (30000, 128, 3)])

    def test_check_receives_actual_capacity_context_and_concurrency(self):
        with mock.patch.object(bench.subprocess, "run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "{}"
            bench.backend_check("sglang-v100", "qwen3.8-flash-next", "0,1,2,3",
                                context=131072, concurrency=2, capacity="capacity.json")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--context") + 1], "131072")
        self.assertEqual(cmd[cmd.index("--concurrency") + 1], "2")
        self.assertEqual(cmd[cmd.index("--capacity") + 1], "capacity.json")

    def test_dry_run_does_not_start_servers_or_write_metrics(self):
        argv = ["pushbutton-bench", "flash-next", "--backend", "sglang-v100", "--gpus", "0,1,2,3",
                "--context", "65536", "--concurrency", "2", "--dry-run"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(bench, "hardware", return_value={}), \
                mock.patch.object(bench, "backend_check", return_value=(True, "{}")), \
                mock.patch.object(bench.subprocess, "Popen") as popen, \
                mock.patch.object(bench, "summarize") as summarize, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(bench.main(), 0)
        result = json.loads(output.getvalue())
        self.assertFalse(result["capacity_verified"])
        self.assertEqual(result["server_concurrency"], 2)
        self.assertEqual({c["concurrency"] for c in result["cases"]}, {1, 2})
        popen.assert_not_called()
        summarize.assert_not_called()

    def test_mock_openai_server_full_benchmark_lifecycle_and_c1_c2_report(self):
        lock = threading.Lock()
        active = 0
        peak = 0
        launch_identity = "mock-flash-next"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                data = json.dumps({"data": [{"id": launch_identity}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(data)

            def do_POST(self):
                nonlocal active, peak
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.assert_path = self.path
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(.05)
                    chunks = [
                        {"choices": [{"delta": {"content": "mock generated content"}}]},
                        {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": payload["max_tokens"]}},
                    ]
                    self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for chunk in chunks:
                        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                finally:
                    with lock: active -= 1

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                output = pathlib.Path(directory) / "results"
                argv = ["pushbutton-bench", "flash-next", "--backend", "sglang-v100",
                        "--gpus", "0,1,2,3", "--context", "4096", "--concurrency", "2",
                        "--prompt-words", "16", "--output-tokens", "8", "--no-prepare",
                        "--port-base", str(server.server_port), "--output-dir", str(output)]
                process = mock.Mock(pid=123456)
                process.poll.return_value = None
                def launch(*args, **kwargs):
                    nonlocal launch_identity
                    launch_identity = kwargs["env"]["PUSHBUTTON_BENCH_MODEL_ID"]
                    return process
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.dict(bench.os.environ, {"PUSHBUTTON_BACKEND_STATE": directory}), \
                        mock.patch.object(bench, "hardware", return_value={"gpus": []}), \
                        mock.patch.object(bench, "backend_check", return_value=(True, "{}")), \
                        mock.patch.object(bench, "fingerprint", return_value={"id": "mock, not live engine"}), \
                        mock.patch.object(bench, "require_free_port"), \
                        mock.patch.object(bench.subprocess, "Popen", side_effect=launch) as popen, \
                        mock.patch.object(bench.os, "killpg") as stop, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(bench.main(), 0)
                reports = list(output.glob("*.json"))
                self.assertEqual(len(reports), 1)
                report = json.loads(reports[0].read_text())
                self.assertEqual(report["server_concurrency"], 2)
                self.assertEqual([c["concurrency"] for c in report["cases"]], [1, 2])
                self.assertEqual(report["served_model"], launch_identity)
                self.assertEqual(peak, 2)
                self.assertTrue(all(not r["completion_tokens_estimated"]
                                    for c in report["cases"] for r in c["requests"]))
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index("--concurrency") + 1], "2")
                stop.assert_called_once()
                process.wait.assert_called_once()
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
