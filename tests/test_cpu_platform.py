import json, os, pathlib, subprocess, sys, tempfile, unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
import cpu_platform as cp  # noqa: E402
import claude_local_plan as base  # noqa: E402
import coder_local_plan as workers  # noqa: E402
import pushbutton_capacity as capacity  # noqa: E402

GIB_KB = 1024 * 1024


def cpuinfo(n_logical, model_name, family, model, flags, sockets=1, cores_per_socket=None, vendor='GenuineIntel',
            smt=2):
    cores_per_socket = cores_per_socket or n_logical // sockets // smt
    blocks = []
    for i in range(n_logical):
        sock = (i // (cores_per_socket * smt)) % sockets if sockets > 1 else 0
        core = i % cores_per_socket
        blocks.append(f"processor\t: {i}\nvendor_id\t: {vendor}\ncpu family\t: {family}\nmodel\t\t: {model}\n"
                      f"model name\t: {model_name}\nphysical id\t: {sock}\ncore id\t\t: {core}\nflags\t\t: {flags}\n")
    return "\n".join(blocks)


def meminfo(total_gib, avail_gib):
    return f"MemTotal: {total_gib * GIB_KB} kB\nMemAvailable: {avail_gib * GIB_KB} kB\n"


KNL_FLAGS = "fpu sse4_2 avx avx2 fma f16c bmi2 avx512f avx512cd avx512er avx512pf"
CLX_FLAGS = ("fpu sse4_2 avx avx2 fma f16c bmi2 avx512f avx512cd avx512bw avx512dq avx512vl "
             "avx512_vnni")
SPR_FLAGS = CLX_FLAGS + " avx512_bf16 avx512vbmi avx_vnni amx_tile amx_int8 amx_bf16"
ZEN4_FLAGS = CLX_FLAGS + " avx512_bf16 avx512vbmi"


def knl(flat=True):
    nodes = [(0, "0-271", 192 * GIB_KB, 180 * GIB_KB)]
    if flat:
        nodes.append((1, "", 16 * GIB_KB, 16 * GIB_KB - 200_000))
    return cp.parse_linux(cpuinfo(272, "Intel(R) Xeon Phi(TM) CPU 7250 @ 1.40GHz", 6, 87, KNL_FLAGS, smt=4),
                          nodes, meminfo(208 if flat else 192, 190), arch="x86_64")


def cascade_lake_2s():
    nodes = [(0, "0-27,56-83", 192 * GIB_KB, 180 * GIB_KB), (1, "28-55,84-111", 192 * GIB_KB, 180 * GIB_KB)]
    return cp.parse_linux(cpuinfo(112, "Intel(R) Xeon(R) Platinum 8280 CPU @ 2.70GHz", 6, 85, CLX_FLAGS,
                                  sockets=2, cores_per_socket=28),
                          nodes, meminfo(384, 360), arch="x86_64")


class DetectionTests(unittest.TestCase):
    def test_knl_flat_mode_detects_mcdram_node(self):
        info = knl(flat=True)
        self.assertEqual((info.family, info.tier), ("knl", "knl"))
        self.assertEqual(info.physical_cores, 68)
        self.assertEqual(info.fast_mem_mode, "flat")
        self.assertEqual([n.kind for n in info.numa], ["dram", "mcdram"])

    def test_knl_cache_mode_without_cpuless_node(self):
        with mock.patch.dict(os.environ, {"PUSHBUTTON_MCDRAM_GIB": "16"}):
            info = knl(flat=False)
        self.assertEqual(info.fast_mem_mode, "cache")
        self.assertEqual(info.fast_mem_mib, 16 * 1024)

    def test_cascade_lake_vnni_two_sockets(self):
        info = cascade_lake_2s()
        self.assertEqual(info.family, "cascadelake")
        self.assertEqual(info.tier, "avx512-vnni")
        self.assertEqual((info.sockets, info.physical_cores, len(info.compute_nodes)), (2, 56, 2))

    def test_sapphire_rapids_amx_and_zen4(self):
        spr = cp.parse_linux(cpuinfo(16, "Intel(R) Xeon(R) Platinum 8480+", 6, 143, SPR_FLAGS),
                             [(0, "0-15", 64 * GIB_KB, 60 * GIB_KB)], meminfo(64, 60), arch="x86_64")
        self.assertEqual(spr.tier, "amx")
        zen4 = cp.parse_linux(cpuinfo(16, "AMD Ryzen 9 7950X 16-Core Processor", 25, 97, ZEN4_FLAGS,
                                      vendor="AuthenticAMD"),
                              [(0, "0-15", 64 * GIB_KB, 60 * GIB_KB)], meminfo(64, 60), arch="x86_64")
        self.assertEqual(zen4.family, "zen4")
        self.assertEqual(zen4.tier, "avx512-vnni-bf16")

    def test_detect_runs_on_this_host(self):
        info = cp.detect(refresh=True)
        self.assertGreaterEqual(info.physical_cores, 1)
        self.assertTrue(info.signature)


class BuildFlagTests(unittest.TestCase):
    def flags(self, info):
        return dict(f[2:].split("=", 1) for f in cp.cmake_flags(info, portable=False) if f.startswith("-DGGML_"))

    def test_knl_never_enables_avx512_bw_vl_build(self):
        f = self.flags(knl())
        self.assertEqual(f["GGML_AVX512"], "OFF")
        self.assertEqual(f["GGML_AVX2"], "ON")
        self.assertEqual(f["GGML_NATIVE"], "OFF")
        self.assertIn("-DCMAKE_C_FLAGS=-mtune=knl", cp.cmake_flags(knl(), portable=False))

    def test_cascade_lake_enables_vnni(self):
        f = self.flags(cascade_lake_2s())
        self.assertEqual((f["GGML_AVX512"], f["GGML_AVX512_VNNI"], f["GGML_AMX_INT8"]), ("ON", "ON", "OFF"))

    def test_amx_enabled_when_present(self):
        spr = cp.parse_linux(cpuinfo(16, "Intel(R) Xeon(R) Platinum 8480+", 6, 143, SPR_FLAGS),
                             [(0, "0-15", 64 * GIB_KB, 60 * GIB_KB)], meminfo(64, 60), arch="x86_64")
        f = self.flags(spr)
        self.assertEqual((f["GGML_AMX_TILE"], f["GGML_AMX_INT8"], f["GGML_AVX512_BF16"]), ("ON", "ON", "ON"))

    def test_portable_build_uses_runtime_dispatch(self):
        f = self.flags(knl())
        self.assertNotIn("GGML_BACKEND_DL", f)
        portable = cp.cmake_flags(knl(), portable=True)
        self.assertIn("-DGGML_CPU_ALL_VARIANTS=ON", portable)
        self.assertIn("-DGGML_NATIVE=OFF", portable)

    def test_build_tag_differs_per_isa(self):
        a, b = knl(), cascade_lake_2s()
        self.assertNotEqual(cp.build_tag(a, cp.cmake_flags(a, False)), cp.build_tag(b, cp.cmake_flags(b, False)))


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"PUSHBUTTON_ASSUME_NUMACTL": "1"})
        self.env.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.results = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_knl_flat_quant_is_sized_to_mcdram_and_bound(self):
        info = knl(flat=True)
        cap = cp.fast_capacity_mib(info)
        for model in ("ornith-1.5:9b", "qwen3.6:35b", "qwen3.8:27b"):
            c = cp.choose_profile(model, capacity.resolve_options({}, 32768), info, results_dir=self.results)
            self.assertLessEqual(c.required_mib, cap, model)
            self.assertEqual(c.strategy.name, "mcdram-bind")
            self.assertEqual(c.strategy.numactl, ("numactl", "--membind=1"))
            self.assertTrue(c.strategy.no_mmap)
            self.assertEqual(c.strategy.threads, 68)
        # The best-quality quant that fits 16 GB MCDRAM, not the smallest one.
        c = cp.choose_profile("ornith-1.5:9b", capacity.resolve_options({}, 32768), info, results_dir=self.results)
        self.assertEqual(c.profile.quant, "Q8_0")

    def test_knl_cache_mode_sizes_weights_to_cache(self):
        info = knl(flat=False)
        c = cp.choose_profile("qwen3.6:35b", capacity.resolve_options({}, 32768), info, results_dir=self.results)
        self.assertEqual(c.strategy.name, "mcdram-cache")
        self.assertLessEqual(cp.weight_gib("qwen3.6:35b", c.profile.quant) * 1024 * 1.08, cp.fast_capacity_mib(info))

    def test_two_instances_get_disjoint_numa_nodes(self):
        info = cascade_lake_2s()
        items = [("ornith-1.5:9b", capacity.resolve_options({}, 32768))] * 2
        a, b = cp.place_requests(items, info)
        self.assertEqual(a.strategy.name, "numa-partition")
        self.assertIn("--cpunodebind=0", a.strategy.numactl)
        self.assertIn("--cpunodebind=1", b.strategy.numactl)
        self.assertEqual(a.strategy.threads, 28)

    def test_single_instance_on_two_sockets_uses_numa(self):
        c = cp.place_requests([("ornith-1.5:9b", capacity.resolve_options({}, 32768))], cascade_lake_2s())[0]
        self.assertIn(c.strategy.name, {"numa-distribute", "numa-isolate"})
        self.assertIn("--numa", c.strategy.llama_args())
        self.assertEqual(c.strategy.llama_args()[:2], ["-ngl", "0"])

    def test_measured_result_overrides_heuristic_strategy(self):
        info = cascade_lake_2s()
        c = cp.choose_profile("ornith-1.5:9b", capacity.resolve_options({}, 32768), info, results_dir=self.results)
        self.assertNotEqual(c.strategy.name, "physical-cores")
        (self.results / "r.json").write_text(json.dumps({
            "schema": "pushbutton.cpu-bench.v1", "cpu": {"signature": info.signature},
            "model": "ornith-1.5:9b", "quant": c.profile.quant, "strategy": {"name": "physical-cores"},
            "depths": [{"depth": 0, "pp_tps": 300.0, "tg_tps": 99.0}]}))
        m = cp.choose_profile("ornith-1.5:9b", capacity.resolve_options({}, 32768), info, results_dir=self.results)
        self.assertEqual(m.strategy.name, "physical-cores")
        self.assertEqual(m.as_row()["evidence"], "LOCAL MEASURED")
        self.assertEqual(m.as_row()["tg"], 99.0)

    def test_does_not_fit_raises_clear_error(self):
        info = cp.parse_linux(cpuinfo(8, "Intel(R) Core(TM) i7-4770", 6, 60, "sse4_2 avx avx2 fma f16c bmi2"),
                              [(0, "0-7", 16 * GIB_KB, 14 * GIB_KB)], meminfo(16, 14), arch="x86_64")
        with self.assertRaisesRegex(ValueError, "no CPU quant"):
            cp.place_requests([("glm-5.3-flash", capacity.resolve_options({}, 32768))], info)


class EstimateTests(unittest.TestCase):
    def test_every_reference_cpu_has_an_estimate_per_model(self):
        rows = cp.reference_estimates()
        self.assertEqual(len(rows), len(cp.REFERENCE_CPUS) * len(cp.MODEL_ARCH))
        for r in rows:
            if r["fits"]:
                self.assertEqual([d["depth"] for d in r["depths"]], list(cp.DEFAULT_DEPTHS))
                tgs = [d["tg_tps"] for d in r["depths"]]
                self.assertEqual(tgs, sorted(tgs, reverse=True), r)
                self.assertGreater(r["load_time_s"], 0)

    def test_mcdram_beats_ddr_for_same_quant(self):
        spec = next(s for s in cp.REFERENCE_CPUS if s.key == "xeon-phi-7250")
        fast = cp.estimate(spec, "ornith-1.5:9b", "Q4_K_M", (0,), memory_tier="mcdram")["depths"][0]["tg_tps"]
        ddr = cp.estimate(spec, "ornith-1.5:9b", "Q4_K_M", (0,), memory_tier="dram")["depths"][0]["tg_tps"]
        self.assertGreater(fast, 2 * ddr)

    def test_estimates_markdown_is_in_sync(self):
        self.assertEqual((ROOT / "benchmarks/cpu/ESTIMATES.md").read_text(), cp.estimates_markdown(),
                         "regenerate with: python3 lib/cpu_platform.py estimates --markdown > benchmarks/cpu/ESTIMATES.md")


class PlannerFallbackTests(unittest.TestCase):
    def setUp(self):
        self.big = cascade_lake_2s()

    def test_claude_local_plans_cpu_servers_without_gpus(self):
        with mock.patch.object(cp, "detect", return_value=self.big):
            p = base.plan(["ornith-1.5:9b"], [], 65536)
        self.assertEqual(p["device"], "cpu")
        s = p["servers"][0]
        self.assertEqual((s["gpus"], s["cuda_visible_devices"], s["device"]), ([], "", "cpu"))
        self.assertIn("-ngl", s["cpu"]["strategy"]["llama_args"])
        self.assertEqual(set(p["role_ids"].values()), {s["id"]})

    def test_smart_defaults_pick_cpu_model(self):
        with mock.patch.object(cp, "detect", return_value=self.big):
            self.assertEqual(base.smart_defaults([], 65536), ["qwen3.6:35b"])

    def test_cpu_fallback_can_be_disabled(self):
        with mock.patch.dict(os.environ, {"PUSHBUTTON_DISABLE_CPU_FALLBACK": "1"}):
            with self.assertRaises(ValueError):
                base.smart_defaults([])

    def test_coder_local_plans_cpu_workers_and_launch_args(self):
        with mock.patch.object(cp, "detect", return_value=self.big), \
                mock.patch.dict(os.environ, {"PUSHBUTTON_ASSUME_NUMACTL": "1"}):
            p = workers.build_plan([workers.parse_model_spec("ornith-1.5:9b")], 2, [], 32768)
        self.assertEqual(p["device"], "cpu")
        self.assertEqual(len(p["workers"]), 2)
        with tempfile.TemporaryDirectory() as td:
            plan = pathlib.Path(td) / "plan.json"
            plan.write_text(json.dumps(p))
            self.assertTrue(cp.plan_is_cpu(str(plan)))
            wid = p["workers"][0]["id"]
            out = subprocess.run([sys.executable, str(ROOT / "lib/cpu_platform.py"), "launch-args", "--plan", str(plan),
                                  "--id", wid, "prefix"], stdout=subprocess.PIPE, check=True).stdout
            self.assertEqual(out.split(b"\0")[0], b"numactl")
            out = subprocess.run([sys.executable, str(ROOT / "lib/cpu_platform.py"), "launch-args", "--plan", str(plan),
                                  "--id", wid, "args"], stdout=subprocess.PIPE, check=True).stdout
            self.assertEqual(out.split(b"\0")[:2], [b"-ngl", b"0"])

    def test_gpu_plans_are_not_cpu(self):
        p = base.plan(["ornith-1.5:9b"], workers.synthetic_3090(1), 65536)
        self.assertNotEqual(p.get("device"), "cpu")


if __name__ == "__main__":
    unittest.main()
