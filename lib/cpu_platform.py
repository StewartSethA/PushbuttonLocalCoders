#!/usr/bin/env python3
"""CPU-only deployment planner for llama.cpp.

Responsibilities (all dependency-free):

* detect the CPU, ISA tier, sockets, physical cores, NUMA layout and high-bandwidth
  memory (Xeon Phi MCDRAM in flat/cache mode, Xeon Max HBM);
* produce llama.cpp CMake flags that are safe for the detected target (e.g. never
  ``GGML_AVX512`` on Knights Landing, which lacks AVX512BW/VL/DQ) while enabling
  AVX512-VNNI, AVX-VNNI, AVX512-BF16 and AMX where present;
* choose a runtime strategy (threads, NUMA mode, numactl MCDRAM/HBM binding,
  ``--no-mmap``/``--mlock``);
* pick the GGUF quant for a model, sized to MCDRAM/HBM when that memory exists;
* estimate model weight load time, prompt processing (PP) and token generation
  (TG) throughput at several context depths for detected and reference CPUs.

All throughput figures from this module are *planning estimates*. Measured
strategy benchmarks in ``benchmarks/cpu/results`` override them automatically.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CPU_RESULTS = ROOT / "benchmarks/cpu/results"
CATALOG = ROOT / "configs/model-catalog.json"
DEFAULT_DEPTHS = (0, 4096, 16384, 32768)

# ── ISA tiers ────────────────────────────────────────────────────────────────
# Effective peak per core per cycle. int8 ops drive quantized matmuls; fp32
# flops drive CPU attention. These are coarse and only used for estimates.
INT8_OPS_PER_CYCLE = {
    "generic": 8, "sse4": 16, "avx": 16, "avx2": 64, "avx2-vnni": 128,
    "knl": 64, "avx512": 128, "avx512-vnni": 256, "avx512-vnni-bf16": 256,
    "amx": 1024, "arm-neon": 32, "arm-dotprod": 64, "arm-i8mm": 128,
    "arm-sve": 128, "apple": 128,
}
FP32_FLOPS_PER_CYCLE = {
    "generic": 4, "sse4": 8, "avx": 16, "avx2": 32, "avx2-vnni": 32,
    "knl": 64, "avx512": 64, "avx512-vnni": 64, "avx512-vnni-bf16": 64,
    "amx": 64, "arm-neon": 16, "arm-dotprod": 16, "arm-i8mm": 32,
    "arm-sve": 32, "apple": 32,
}
TIER_ORDER = ("generic", "sse4", "avx", "avx2", "avx2-vnni", "knl", "avx512",
              "avx512-vnni", "avx512-vnni-bf16", "amx")
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625,
            "q4_1": 0.625, "q5_0": 0.6875, "q5_1": 0.75}

# Bits per weight for the GGUF artifacts used by the planner profiles.
QUANT_BPW = {
    "Q8_0": 8.5, "Q6_K": 6.56, "Q5_K_M": 5.69, "Q5_0": 5.5, "Q4_K_M": 4.85,
    "Q4_K_XL": 4.9, "IQ4_XS": 4.25, "MXFP4_MOE": 4.25, "Q3_K_XL": 3.9,
    "IQ3_XXS": 3.06, "IQ2_M": 2.7, "IQ2_S": 2.5, "IQ2_XS": 2.31,
    "IQ2_XXS": 2.06, "Q2_K_XL": 2.9, "IQ1_M": 1.75, "IQ1_S": 1.56,
    "MIXED-Q2_0-Q4_0-2.97BPW": 2.97,
}

# Planning architecture per model. Total/active parameters (billions), number of
# layers with full KV attention, and per-layer query/KV widths. Hybrid models
# (linear attention / Mamba) only pay KV for their attention layers. These are
# planning assumptions; measured strategy benchmarks supersede them.
MODEL_ARCH = {
    "ornith-1.5:9b": dict(total_b=9.0, active_b=9.0, attn_layers=36, q_dim=4096, kv_dim=1024),
    "ornith-1.5:35b-a3b": dict(total_b=35.0, active_b=3.0, attn_layers=10, q_dim=4096, kv_dim=512),
    "qwen3.8:27b": dict(total_b=27.0, active_b=27.0, attn_layers=16, q_dim=6144, kv_dim=1024),
    "qwen3.8-flash-next": dict(total_b=235.0, active_b=12.0, attn_layers=15, q_dim=8192, kv_dim=512),
    "qwen3.6:35b": dict(total_b=35.0, active_b=3.0, attn_layers=10, q_dim=4096, kv_dim=512),
    "nemotron-3.5-lightning": dict(total_b=30.0, active_b=3.0, attn_layers=6, q_dim=4096, kv_dim=256),
    "glm-5.3-flash": dict(total_b=400.0, active_b=32.0, attn_layers=78, q_dim=8192, kv_dim=576),
    "deepseek-v4-flash": dict(total_b=390.0, active_b=20.0, attn_layers=61, q_dim=7168, kv_dim=576),
}


# ── Detected / reference hardware ────────────────────────────────────────────
@dataclass(frozen=True)
class NumaNode:
    id: int
    cpus: tuple[int, ...]
    total_mib: int
    free_mib: int
    kind: str = "dram"  # dram | mcdram | hbm | memory-only


@dataclass(frozen=True)
class CPUSpec:
    """Hardware description used for strategy and throughput estimation."""
    key: str
    name: str
    family: str
    tier: str
    sockets: int
    cores_per_socket: int
    threads_per_core: int
    ghz: float                      # sustained all-core vector clock
    dram_bw_gbs: float              # sustained (STREAM-like) per socket
    ram_gib: float                  # total installed RAM (reference configs)
    fast_mem_kind: str = ""         # mcdram | hbm
    fast_mem_gib: float = 0.0       # per socket
    fast_mem_bw_gbs: float = 0.0    # sustained per socket
    fast_mem_mode: str = ""         # flat | cache | hybrid
    note: str = ""

    @property
    def cores(self) -> int:
        return self.sockets * self.cores_per_socket


@dataclass(frozen=True)
class CPUInfo:
    arch: str
    vendor: str
    model_name: str
    family_id: int
    model_id: int
    flags: frozenset
    sockets: int
    physical_cores: int
    logical_cpus: int
    numa: tuple[NumaNode, ...]
    ram_total_mib: int
    ram_available_mib: int
    family: str = "generic"
    tier: str = "generic"
    fast_mem_kind: str = ""
    fast_mem_mode: str = ""
    fast_mem_mib: int = 0
    performance_cores: int = 0

    @property
    def is_xeon_phi(self) -> bool:
        return self.family in {"knl", "knm"}

    @property
    def compute_nodes(self) -> list[NumaNode]:
        return [n for n in self.numa if n.cpus]

    @property
    def fast_nodes(self) -> list[NumaNode]:
        return [n for n in self.numa if n.kind in {"mcdram", "hbm"}]

    @property
    def signature(self) -> str:
        layout = [(n.kind, bool(n.cpus), round(n.total_mib / 1024)) for n in self.numa]
        raw = json.dumps([self.model_name, self.sockets, self.physical_cores,
                          self.logical_cpus, layout, self.fast_mem_mode], sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def summary(self) -> dict:
        return {
            "model_name": self.model_name, "arch": self.arch, "vendor": self.vendor,
            "family": self.family, "tier": self.tier, "sockets": self.sockets,
            "physical_cores": self.physical_cores, "logical_cpus": self.logical_cpus,
            "numa_nodes": [{"id": n.id, "kind": n.kind, "cpus": len(n.cpus),
                            "total_mib": n.total_mib} for n in self.numa],
            "fast_mem_kind": self.fast_mem_kind, "fast_mem_mode": self.fast_mem_mode,
            "fast_mem_mib": self.fast_mem_mib, "ram_total_mib": self.ram_total_mib,
            "isa": sorted(f for f in self.flags if f in INTERESTING_FLAGS),
            "signature": self.signature,
        }


INTERESTING_FLAGS = {
    "sse4_2", "avx", "avx2", "fma", "f16c", "bmi2", "avx_vnni", "avx512f",
    "avx512cd", "avx512er", "avx512pf", "avx512bw", "avx512dq", "avx512vl",
    "avx512vbmi", "avx512_vnni", "avx512_bf16", "avx512_4vnniw", "avx512_4fmaps",
    "amx_tile", "amx_int8", "amx_bf16", "asimd", "asimddp", "i8mm", "sve", "sve2",
    "bf16",
}

# Sustained bandwidth (GB/s per socket) and clocks by microarchitecture. Used for
# detected CPUs; the reference table below is the documented estimate set.
FAMILY_DEFAULTS = {
    "knl": dict(ghz=1.3, dram_bw=82.0, fast_bw=420.0, fast_gib=16.0),
    "knm": dict(ghz=1.4, dram_bw=88.0, fast_bw=440.0, fast_gib=16.0),
    "haswell": dict(ghz=3.0, dram_bw=20.0), "broadwell": dict(ghz=2.4, dram_bw=60.0),
    "skylake": dict(ghz=3.5, dram_bw=30.0), "skylake-sp": dict(ghz=2.0, dram_bw=100.0),
    "cascadelake": dict(ghz=2.2, dram_bw=110.0), "cooperlake": dict(ghz=2.3, dram_bw=115.0),
    "icelake-sp": dict(ghz=2.3, dram_bw=160.0), "icelake": dict(ghz=3.5, dram_bw=40.0),
    "sapphirerapids": dict(ghz=2.0, dram_bw=230.0, fast_bw=600.0, fast_gib=64.0),
    "emeraldrapids": dict(ghz=2.1, dram_bw=250.0), "graniterapids": dict(ghz=2.0, dram_bw=450.0),
    "alderlake": dict(ghz=4.0, dram_bw=65.0), "raptorlake": dict(ghz=4.2, dram_bw=70.0),
    "zen": dict(ghz=2.8, dram_bw=110.0), "zen2": dict(ghz=2.8, dram_bw=150.0),
    "zen3": dict(ghz=3.0, dram_bw=155.0), "zen4": dict(ghz=3.0, dram_bw=360.0),
    "zen5": dict(ghz=3.2, dram_bw=450.0), "neoverse-n1": dict(ghz=3.0, dram_bw=160.0),
    "neoverse-v1": dict(ghz=2.6, dram_bw=250.0), "neoverse-v2": dict(ghz=3.1, dram_bw=420.0),
    "apple": dict(ghz=3.5, dram_bw=100.0), "generic": dict(ghz=2.5, dram_bw=40.0),
}

INTEL_MODELS = {
    87: "knl", 133: "knm", 60: "haswell", 63: "haswell", 69: "haswell", 70: "haswell",
    61: "broadwell", 71: "broadwell", 79: "broadwell", 86: "broadwell",
    78: "skylake", 94: "skylake", 142: "skylake", 158: "skylake", 165: "skylake",
    106: "icelake-sp", 108: "icelake-sp", 125: "icelake", 126: "icelake",
    143: "sapphirerapids", 207: "emeraldrapids", 173: "graniterapids", 174: "graniterapids",
    151: "alderlake", 154: "alderlake", 183: "raptorlake", 186: "raptorlake", 191: "raptorlake",
}
ARM_PARTS = {0xD0C: "neoverse-n1", 0xD40: "neoverse-v1", 0xD4F: "neoverse-v2", 0xD49: "neoverse-n1"}


def classify_family(vendor: str, family_id: int, model_id: int, flags: frozenset,
                    name: str, arch: str, cpu_part: int = 0) -> str:
    low = name.lower()
    if arch in {"arm64", "aarch64"}:
        if "apple" in low:
            return "apple"
        return ARM_PARTS.get(cpu_part, "generic")
    if "xeon phi" in low:
        return "knm" if model_id == 133 or "avx512_4vnniw" in flags else "knl"
    if vendor == "GenuineIntel" and family_id == 6:
        if model_id == 85:  # Skylake-SP / Cascade Lake / Cooper Lake share the model id
            if "avx512_bf16" in flags:
                return "cooperlake"
            if "avx512_vnni" in flags:
                return "cascadelake"
            return "skylake-sp"
        return INTEL_MODELS.get(model_id, "generic")
    if vendor == "AuthenticAMD":
        if family_id == 23:
            return "zen2" if model_id >= 0x30 else "zen"
        if family_id == 25:
            return "zen4" if "avx512f" in flags else "zen3"
        if family_id >= 26:
            return "zen5"
    return "generic"


def classify_tier(flags: frozenset, arch: str, family: str) -> str:
    if arch in {"arm64", "aarch64"}:
        if family == "apple":
            return "apple"
        if "sve" in flags:
            return "arm-sve"
        if "i8mm" in flags:
            return "arm-i8mm"
        if "asimddp" in flags:
            return "arm-dotprod"
        return "arm-neon"
    avx512_full = {"avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512cd"} <= flags
    if "avx512f" in flags and not avx512_full:
        return "knl"  # AVX512F/CD/ER/PF only: ggml must use its AVX2 kernels
    if avx512_full:
        if {"amx_tile", "amx_int8"} <= flags:
            return "amx"
        if "avx512_bf16" in flags and "avx512_vnni" in flags:
            return "avx512-vnni-bf16"
        if "avx512_vnni" in flags:
            return "avx512-vnni"
        return "avx512"
    if "avx2" in flags:
        return "avx2-vnni" if "avx_vnni" in flags else "avx2"
    if "avx" in flags:
        return "avx"
    if "sse4_2" in flags:
        return "sse4"
    return "generic"


def _parse_cpulist(text: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(out)


def _read(path: str) -> str:
    try:
        return pathlib.Path(path).read_text()
    except OSError:
        return ""


def _sysfs_nodes() -> list[tuple[int, str, int, int]]:
    nodes = []
    for d in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
        nid = int(re.sub(r"\D", "", os.path.basename(d)))
        mem = _read(os.path.join(d, "meminfo"))
        total = re.search(r"MemTotal:\s+(\d+)", mem)
        free = re.search(r"MemFree:\s+(\d+)", mem)
        nodes.append((nid, _read(os.path.join(d, "cpulist")).strip(),
                      int(total.group(1)) if total else 0, int(free.group(1)) if free else 0))
    return nodes


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], text=True, capture_output=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""


def parse_linux(cpuinfo: str, nodes: list[tuple[int, str, int, int]], meminfo: str,
                arch: str | None = None) -> CPUInfo:
    """Build CPUInfo from /proc/cpuinfo, NUMA node tuples and /proc/meminfo text."""
    arch = arch or platform.machine().lower()
    blocks = [b for b in cpuinfo.split("\n\n") if b.strip()]
    first: dict[str, str] = {}
    cores: set[tuple[str, str]] = set()
    packages: set[str] = set()
    logical = 0
    for block in blocks:
        kv = {}
        for line in block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip().lower()] = v.strip()
        if "processor" not in kv:
            continue
        logical += 1
        if not first:
            first = kv
        pkg = kv.get("physical id", "0")
        packages.add(pkg)
        if "core id" in kv:
            cores.add((pkg, kv["core id"]))
    flags = frozenset((first.get("flags") or first.get("features") or "").split())
    vendor = first.get("vendor_id", "ARM" if arch in {"arm64", "aarch64"} else "unknown")
    name = first.get("model name") or first.get("cpu model") or "Unknown CPU"
    try:
        family_id = int(first.get("cpu family", "0"))
        model_id = int(first.get("model", "0"))
    except ValueError:
        family_id = model_id = 0
    try:
        cpu_part = int(first.get("cpu part", "0"), 0)
    except ValueError:
        cpu_part = 0
    logical = max(1, logical)
    physical = len(cores) or logical
    family = classify_family(vendor, family_id, model_id, flags, name, arch, cpu_part)
    tier = classify_tier(flags, arch, family)
    total = re.search(r"MemTotal:\s+(\d+)", meminfo)
    avail = re.search(r"MemAvailable:\s+(\d+)", meminfo) or re.search(r"MemFree:\s+(\d+)", meminfo)
    ram_total = int(total.group(1)) // 1024 if total else 0
    ram_avail = int(avail.group(1)) // 1024 if avail else ram_total
    numa_nodes = []
    for nid, cpulist, total_kb, free_kb in nodes:
        cpus = _parse_cpulist(cpulist) if cpulist else ()
        kind = "dram"
        if not cpus and total_kb > 0:
            if family in {"knl", "knm"}:
                kind = "mcdram"
            elif family == "sapphirerapids" and "max" in name.lower():
                kind = "hbm"
            else:
                kind = "memory-only"
        numa_nodes.append(NumaNode(nid, cpus, total_kb // 1024, free_kb // 1024, kind))
    if not numa_nodes:
        numa_nodes = [NumaNode(0, tuple(range(logical)), ram_total, ram_avail, "dram")]
    fast = [n for n in numa_nodes if n.kind in {"mcdram", "hbm"}]
    fast_kind = fast_mode = ""
    fast_mib = 0
    env_mode = os.environ.get("PUSHBUTTON_MCDRAM_MODE", "").strip().lower()
    if fast:
        fast_kind = fast[0].kind
        fast_mode = "flat"
        fast_mib = sum(n.total_mib for n in fast)
    elif family in {"knl", "knm"}:
        # No CPU-less node on a Phi means MCDRAM is configured as a memory-side cache.
        fast_kind, fast_mode = "mcdram", "cache"
        fast_mib = int(float(os.environ.get("PUSHBUTTON_MCDRAM_GIB", "16")) * 1024)
    if env_mode in {"flat", "cache", "hybrid"} and fast_kind:
        fast_mode = env_mode
    return CPUInfo(arch=arch, vendor=vendor, model_name=name, family_id=family_id,
                   model_id=model_id, flags=flags, sockets=max(1, len(packages)),
                   physical_cores=physical, logical_cpus=logical, numa=tuple(numa_nodes),
                   ram_total_mib=ram_total, ram_available_mib=ram_avail, family=family,
                   tier=tier, fast_mem_kind=fast_kind, fast_mem_mode=fast_mode,
                   fast_mem_mib=fast_mib, performance_cores=physical)


def detect_mac() -> CPUInfo:
    name = _sysctl("machdep.cpu.brand_string") or "Apple CPU"
    arch = platform.machine().lower()
    physical = int(_sysctl("hw.physicalcpu") or 1)
    logical = int(_sysctl("hw.logicalcpu") or physical)
    perf = int(_sysctl("hw.perflevel0.physicalcpu") or physical)
    mem = int(_sysctl("hw.memsize") or 0) // 1048576
    feats = (_sysctl("machdep.cpu.features") + " " + _sysctl("machdep.cpu.leaf7_features")).lower()
    flags = set(feats.replace(".", "_").split())
    if arch == "arm64":
        flags |= {"asimd", "asimddp"}
        if _sysctl("hw.optional.arm.FEAT_I8MM") == "1":
            flags.add("i8mm")
    flags = frozenset(flags)
    vendor = "Apple" if arch == "arm64" else "GenuineIntel"
    family = classify_family(vendor, 0, 0, flags, name if arch != "arm64" else "apple " + name, arch)
    tier = classify_tier(flags, arch, family)
    node = NumaNode(0, tuple(range(logical)), mem, int(mem * 0.7), "dram")
    return CPUInfo(arch=arch, vendor=vendor, model_name=name, family_id=0, model_id=0,
                   flags=flags, sockets=1, physical_cores=physical, logical_cpus=logical,
                   numa=(node,), ram_total_mib=mem, ram_available_mib=int(mem * 0.7),
                   family=family, tier=tier, performance_cores=perf)


_DETECTED: CPUInfo | None = None


def detect(refresh: bool = False) -> CPUInfo:
    global _DETECTED
    if _DETECTED is not None and not refresh:
        return _DETECTED
    if platform.system() == "Darwin":
        info = detect_mac()
    else:
        info = parse_linux(_read("/proc/cpuinfo"), _sysfs_nodes(), _read("/proc/meminfo"))
    _DETECTED = info
    return info


def detected_spec(info: CPUInfo, cores: int | None = None) -> CPUSpec:
    d = FAMILY_DEFAULTS.get(info.family, FAMILY_DEFAULTS["generic"])
    bw = float(os.environ.get("PUSHBUTTON_CPU_MEM_BW_GBS", 0) or 0) / max(1, info.sockets) or d["dram_bw"]
    if info.family in {"haswell", "skylake", "icelake", "alderlake", "raptorlake", "apple"} or (
            info.family in {"zen2", "zen3", "zen4", "zen5"} and info.physical_cores <= 24):
        # Desktop parts: two memory channels regardless of microarchitecture.
        bw = min(bw, d["dram_bw"] if info.family != "zen4" else 70.0)
        if info.family in {"zen2", "zen3"}:
            bw = 45.0
        elif info.family == "zen5":
            bw = 75.0
    per_socket = max(1, info.physical_cores // max(1, info.sockets))
    if cores:
        per_socket = max(1, min(per_socket, cores // max(1, info.sockets) or cores))
    fast_gib = info.fast_mem_mib / 1024 / max(1, info.sockets)
    return CPUSpec(
        key="detected", name=info.model_name, family=info.family, tier=info.tier,
        sockets=info.sockets, cores_per_socket=per_socket,
        threads_per_core=max(1, info.logical_cpus // max(1, info.physical_cores)),
        ghz=d["ghz"], dram_bw_gbs=bw, ram_gib=info.ram_total_mib / 1024,
        fast_mem_kind=info.fast_mem_kind, fast_mem_gib=fast_gib,
        fast_mem_bw_gbs=d.get("fast_bw", 0.0) if info.fast_mem_kind else 0.0,
        fast_mem_mode=info.fast_mem_mode,
    )


# Reference CPUs for the published estimate table (benchmarks/cpu/ESTIMATES.md).
REFERENCE_CPUS: tuple[CPUSpec, ...] = (
    CPUSpec("xeon-phi-7210", "Intel Xeon Phi 7210 (KNL, flat MCDRAM)", "knl", "knl", 1, 64, 4, 1.2, 80.0, 192, "mcdram", 16, 400.0, "flat"),
    CPUSpec("xeon-phi-7250", "Intel Xeon Phi 7250 (KNL, flat MCDRAM)", "knl", "knl", 1, 68, 4, 1.3, 82.0, 192, "mcdram", 16, 420.0, "flat"),
    CPUSpec("xeon-phi-7250-cache", "Intel Xeon Phi 7250 (KNL, cache-mode MCDRAM)", "knl", "knl", 1, 68, 4, 1.3, 82.0, 192, "mcdram", 16, 330.0, "cache"),
    CPUSpec("xeon-phi-7295", "Intel Xeon Phi 7295 (KNM, flat MCDRAM)", "knm", "knl", 1, 72, 4, 1.4, 88.0, 192, "mcdram", 16, 440.0, "flat"),
    CPUSpec("core-i7-4770", "Intel Core i7-4770 (Haswell, 2ch DDR3)", "haswell", "avx2", 1, 4, 2, 3.5, 20.0, 32),
    CPUSpec("xeon-8180-2s", "2x Intel Xeon Platinum 8180 (Skylake-SP)", "skylake-sp", "avx512", 2, 28, 2, 2.0, 100.0, 384),
    CPUSpec("xeon-8280-2s", "2x Intel Xeon Platinum 8280 (Cascade Lake, VNNI)", "cascadelake", "avx512-vnni", 2, 28, 2, 2.2, 110.0, 384),
    CPUSpec("xeon-6248r-1s", "Intel Xeon Gold 6248R (Cascade Lake, VNNI)", "cascadelake", "avx512-vnni", 1, 24, 2, 2.6, 110.0, 192),
    CPUSpec("xeon-8380-2s", "2x Intel Xeon Platinum 8380 (Ice Lake-SP)", "icelake-sp", "avx512-vnni", 2, 40, 2, 2.3, 160.0, 512),
    CPUSpec("xeon-8480-2s", "2x Intel Xeon Platinum 8480+ (Sapphire Rapids, AMX)", "sapphirerapids", "amx", 2, 56, 2, 2.0, 230.0, 1024),
    CPUSpec("xeon-max-9480-1s", "Intel Xeon Max 9480 (SPR + 64 GB HBM, flat)", "sapphirerapids", "amx", 1, 56, 2, 1.9, 230.0, 512, "hbm", 64, 600.0, "flat"),
    CPUSpec("xeon-6980p-1s", "Intel Xeon 6980P (Granite Rapids, AMX)", "graniterapids", "amx", 1, 128, 2, 2.0, 450.0, 768),
    CPUSpec("core-i9-13900k", "Intel Core i9-13900K (Raptor Lake, AVX-VNNI)", "raptorlake", "avx2-vnni", 1, 8, 2, 5.0, 70.0, 64, note="P-cores only"),
    CPUSpec("epyc-7742-1s", "AMD EPYC 7742 (Zen 2)", "zen2", "avx2", 1, 64, 2, 2.6, 150.0, 512),
    CPUSpec("epyc-7763-2s", "2x AMD EPYC 7763 (Zen 3)", "zen3", "avx2", 2, 64, 2, 2.8, 155.0, 1024),
    CPUSpec("epyc-9654-1s", "AMD EPYC 9654 (Zen 4, AVX-512)", "zen4", "avx512-vnni-bf16", 1, 96, 2, 2.6, 360.0, 768),
    CPUSpec("epyc-9755-1s", "AMD EPYC 9755 (Zen 5, AVX-512)", "zen5", "avx512-vnni-bf16", 1, 128, 2, 2.7, 450.0, 1152),
    CPUSpec("ryzen-7950x", "AMD Ryzen 9 7950X (Zen 4, 2ch DDR5)", "zen4", "avx512-vnni-bf16", 1, 16, 2, 5.0, 70.0, 128),
    CPUSpec("graviton3", "AWS Graviton3 (Neoverse V1, SVE)", "neoverse-v1", "arm-sve", 1, 64, 1, 2.6, 250.0, 256),
    CPUSpec("altra-q80", "Ampere Altra Q80-30 (Neoverse N1)", "neoverse-n1", "arm-dotprod", 1, 80, 1, 3.0, 160.0, 256),
    CPUSpec("apple-m2-ultra", "Apple M2 Ultra (CPU only)", "apple", "apple", 1, 16, 1, 3.5, 300.0, 192, note="P-cores only"),
)


# ── llama.cpp build flags ─────────────────────────────────────────────────────
def cmake_flags(info: CPUInfo, portable: bool | None = None) -> list[str]:
    """CMake flags for a CPU-only llama.cpp build tuned for this host.

    x86 builds enable exactly the ISA extensions the host reports, so a build can
    never contain instructions the CPU lacks. ``portable`` builds all ggml CPU
    variants as loadable backends and selects the best one at runtime, which is
    the right choice for binaries shared across heterogeneous nodes.
    """
    if portable is None:
        portable = os.environ.get("PUSHBUTTON_CPU_BUILD", "").lower() == "portable"
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF", "-DGGML_OPENMP=ON"]
    if portable:
        return flags + ["-DGGML_NATIVE=OFF", "-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON"]
    if info.arch in {"x86_64", "amd64", "i686", "x86"}:
        f = info.flags
        avx512 = {"avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512cd"} <= f
        want = {
            "GGML_SSE42": "sse4_2" in f, "GGML_AVX": "avx" in f, "GGML_AVX2": "avx2" in f,
            "GGML_FMA": "fma" in f, "GGML_F16C": "f16c" in f, "GGML_BMI2": "bmi2" in f,
            "GGML_AVX_VNNI": "avx_vnni" in f, "GGML_AVX512": avx512,
            "GGML_AVX512_VBMI": avx512 and "avx512vbmi" in f,
            "GGML_AVX512_VNNI": avx512 and "avx512_vnni" in f,
            "GGML_AVX512_BF16": avx512 and "avx512_bf16" in f,
            "GGML_AMX_TILE": avx512 and "amx_tile" in f,
            "GGML_AMX_INT8": avx512 and "amx_int8" in f,
            "GGML_AMX_BF16": avx512 and "amx_bf16" in f,
        }
        flags.append("-DGGML_NATIVE=OFF")
        flags += [f"-D{k}={'ON' if v else 'OFF'}" for k, v in want.items()]
        if info.family in {"knl", "knm"}:
            # KNL/KNM: AVX2 kernels on the 512-bit VPUs; tune scheduling for the
            # Silvermont-derived core rather than a big-core default.
            flags.append("-DCMAKE_C_FLAGS=-mtune=knl")
            flags.append("-DCMAKE_CXX_FLAGS=-mtune=knl")
        return flags
    if info.arch in {"arm64", "aarch64"} and platform.system() == "Darwin":
        return flags + ["-DGGML_NATIVE=ON", "-DGGML_METAL=OFF", "-DGGML_ACCELERATE=ON"]
    return flags + ["-DGGML_NATIVE=ON"]


def build_tag(info: CPUInfo, flags: list[str]) -> str:
    digest = hashlib.sha256("\n".join(flags).encode()).hexdigest()[:8]
    return f"cpu-{info.tier}-{digest}"


# ── Runtime strategies ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Strategy:
    name: str
    threads: int
    threads_batch: int
    numa: str = ""                  # "", distribute, isolate, numactl
    numactl: tuple[str, ...] = ()
    no_mmap: bool = False
    mlock: bool = False
    memory_tier: str = "dram"       # dram | mcdram | hbm | mcdram-cache | spill
    note: str = ""

    def llama_args(self) -> list[str]:
        args = ["-ngl", "0", "--threads", str(self.threads), "--threads-batch", str(self.threads_batch)]
        if self.numa:
            args += ["--numa", self.numa]
        if self.no_mmap:
            args.append("--no-mmap")
        if self.mlock:
            args.append("--mlock")
        return args

    def bench_args(self, threads: int) -> list[str]:
        args = ["-ngl", "0", "-t", str(threads)]
        if self.numa:
            args += ["--numa", self.numa]
        if self.no_mmap:
            args += ["-mmp", "0"]
        return args

    def to_dict(self) -> dict:
        d = asdict(self)
        d["numactl"] = list(self.numactl)
        d["llama_args"] = self.llama_args()
        return d


def strategy_from_dict(d: dict) -> Strategy:
    allowed = {k: d[k] for k in Strategy.__dataclass_fields__ if k in d}
    allowed["numactl"] = tuple(str(x) for x in allowed.get("numactl", ()))
    return Strategy(**allowed)


def _mlock_allowed(required_mib: int, info: CPUInfo) -> bool:
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        unlimited = soft == resource.RLIM_INFINITY or soft >= required_mib * 1048576
    except Exception:
        return False
    return unlimited and info.ram_available_mib >= required_mib * 1.5


def _ids(nodes: list[NumaNode]) -> str:
    return ",".join(str(n.id) for n in nodes)


def fast_capacity_mib(info: CPUInfo) -> int:
    """Usable MCDRAM/HBM for model placement; 0 when no such memory exists."""
    if not info.fast_mem_kind:
        return 0
    if info.fast_mem_mode == "flat":
        nodes = info.fast_nodes
        free = sum(n.free_mib for n in nodes) if nodes else info.fast_mem_mib
        return int(free * 0.94)
    # Cache mode: the memory-side cache is most effective when the working set
    # (weights + KV that are touched every token) fits with some slack.
    return int(info.fast_mem_mib * 0.90)


def candidate_strategies(info: CPUInfo, required_mib: int, threads: int | None = None,
                         nodes: list[NumaNode] | None = None) -> list[Strategy]:
    """Applicable strategies, best heuristic default first."""
    phys = threads or (info.performance_cores if info.family == "apple" else info.physical_cores)
    phys = max(1, phys)
    smt = max(1, info.logical_cpus // max(1, info.physical_cores))
    batch = min(phys * smt, phys * 2) if info.is_xeon_phi else phys
    have_numactl = bool(shutil.which("numactl")) or os.environ.get("PUSHBUTTON_ASSUME_NUMACTL") == "1"
    out: list[Strategy] = []
    compute = nodes if nodes is not None else info.compute_nodes
    if nodes is not None and len(nodes) < len(info.compute_nodes):
        # Partitioned multi-instance placement: bind this instance to its nodes.
        ids = _ids(nodes)
        prefix = ("numactl", f"--cpunodebind={ids}", f"--membind={ids}") if have_numactl else ()
        out.append(Strategy("numa-partition", phys, batch, "numactl" if prefix else "",
                            prefix, memory_tier="dram",
                            note="instance bound to its own NUMA node(s)"))
    fast = info.fast_nodes
    cap = fast_capacity_mib(info)
    if info.fast_mem_mode == "flat" and fast and have_numactl:
        ids = _ids(fast)
        if required_mib <= cap:
            policy = f"--membind={ids}" if len(fast) == 1 else f"--interleave={ids}"
            out.append(Strategy(f"{info.fast_mem_kind}-bind", phys, batch, "", ("numactl", policy),
                                no_mmap=True, memory_tier=info.fast_mem_kind,
                                note=f"weights and KV pinned to {info.fast_mem_kind.upper()} nodes {ids}; "
                                     "--no-mmap keeps page-cache copies out of DDR"))
        out.append(Strategy(f"{info.fast_mem_kind}-preferred", phys, batch, "",
                            ("numactl", f"--preferred={fast[0].id}"), no_mmap=True, memory_tier="spill",
                            note=f"fill {info.fast_mem_kind.upper()} first, spill the remainder to DDR"))
    elif info.fast_mem_mode == "cache":
        out.append(Strategy("mcdram-cache", phys, batch, "", (), memory_tier="mcdram-cache",
                            note="MCDRAM in cache mode; quant sized to keep the hot set cache-resident"))
    if len(compute) > 1:
        out.append(Strategy("numa-distribute", phys, batch, "distribute",
                            note="spread threads and pages evenly over NUMA nodes; drop page cache before first load"))
        node0 = compute[0]
        node0_cores = max(1, len(node0.cpus) // smt)
        if required_mib <= node0.free_mib * 0.94 and have_numactl:
            out.append(Strategy("numa-isolate", node0_cores, node0_cores, "numactl",
                                ("numactl", f"--cpunodebind={node0.id}", f"--membind={node0.id}"),
                                note="single NUMA node; avoids cross-socket traffic"))
    out.append(Strategy("physical-cores", phys, batch, note="one thread per physical core"))
    if smt > 1:
        out.append(Strategy("all-threads", phys * smt, phys * smt, note="SMT threads included"))
    out.append(Strategy("no-mmap", phys, batch, no_mmap=True, mlock=_mlock_allowed(required_mib, info),
                        note="anonymous pre-faulted weights (mlock when RLIMIT_MEMLOCK allows)"))
    seen, unique = set(), []
    for s in out:
        if s.name not in seen:
            seen.add(s.name)
            unique.append(s)
    return unique


# ── Estimation ────────────────────────────────────────────────────────────────
def _catalog() -> dict:
    try:
        return json.loads(CATALOG.read_text()).get("models", {})
    except Exception:
        return {}


def quant_bpw(quant: str) -> float:
    q = quant.upper()
    if q.startswith("UD-"):
        q = q[3:]
    if q in QUANT_BPW:
        return QUANT_BPW[q]
    m = re.search(r"([0-9.]+)BPW", q)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:IQ|Q|FP)([0-9])", q)
    return float(m.group(1)) + 0.5 if m else 4.5


def weight_gib(model: str, quant: str) -> float:
    for q in _catalog().get(model, {}).get("quants", []) or []:
        if str(q.get("name", "")).upper() == quant.upper():
            return float(q["weight_gib"])
    arch = MODEL_ARCH.get(model)
    if not arch:
        raise ValueError(f"no CPU planning architecture for {model}")
    return arch["total_b"] * 1e9 * quant_bpw(quant) / 8 / 2**30


def cpu_required_mib(model: str, quant: str, context_tokens: int, kv_k: str = "q4_0",
                     kv_v: str = "q4_0") -> int:
    """Host-RAM envelope: weights + full KV for all slots + compute buffers, +5%.

    The GPU planner's envelope includes CUDA buffers and padding that a CPU
    process does not allocate, so CPU placement uses this architecture estimate.
    """
    arch = MODEL_ARCH[model]
    kv = context_tokens * arch["attn_layers"] * arch["kv_dim"] * (KV_BYTES.get(kv_k, 2.0) + KV_BYTES.get(kv_v, 2.0))
    compute = 768 * 2**20 + context_tokens * 4096  # graph/compute buffers grow with ubatch×ctx
    total = (weight_gib(model, quant) * 2**30 + kv + compute) * 1.05
    return int(total / 2**20) + 1


def _ops(spec: CPUSpec) -> tuple[float, float]:
    int8 = INT8_OPS_PER_CYCLE.get(spec.tier, 16)
    fp32 = FP32_FLOPS_PER_CYCLE.get(spec.tier, 8)
    if spec.family == "zen4" and spec.tier.startswith("avx512"):
        int8, fp32 = int8 / 2, fp32 / 2  # 256-bit datapath, double-pumped AVX-512
    return int8, fp32


def estimate(spec: CPUSpec, model: str, quant: str, depths=DEFAULT_DEPTHS,
             kv_type: str = "q4_0", memory_tier: str | None = None, required_mib: int | None = None,
             storage_gbs: float = 1.5) -> dict:
    """Estimate load time and PP/TG tok/s at each context depth."""
    arch = MODEL_ARCH.get(model)
    if not arch:
        raise ValueError(f"no CPU planning architecture for {model}")
    w_gib = weight_gib(model, quant)
    w_bytes = w_gib * 2**30
    active_frac = arch["active_b"] / arch["total_b"]
    active_bytes = w_bytes * active_frac
    active_params = arch["active_b"] * 1e9
    kv_b = KV_BYTES.get(kv_type, 0.5625)
    knl = spec.tier == "knl"
    fast_total_gib = spec.fast_mem_gib * spec.sockets
    # Flat mode binds weights+KV (the whole allocation); cache mode only needs the
    # per-token hot set (weights plus a little KV) to stay cache-resident.
    working_gib = (required_mib / 1024) if required_mib and spec.fast_mem_mode != "cache" else w_gib * 1.08
    if memory_tier is None:
        memory_tier = "dram"
        if spec.fast_mem_kind and working_gib <= fast_total_gib * (0.94 if spec.fast_mem_mode == "flat" else 0.90):
            memory_tier = spec.fast_mem_kind if spec.fast_mem_mode == "flat" else "mcdram-cache"
        elif spec.fast_mem_kind and spec.fast_mem_mode == "flat":
            memory_tier = "spill"
    dram = spec.dram_bw_gbs * spec.sockets
    if memory_tier in {"mcdram", "hbm", "mcdram-cache"}:
        bw = spec.fast_mem_bw_gbs * spec.sockets
    elif memory_tier == "spill" and fast_total_gib > 0:
        # Fraction of each token's bytes served from fast memory vs DDR.
        f = min(1.0, fast_total_gib * 0.94 / max(working_gib, 1e-6))
        bw = 1.0 / (f / (spec.fast_mem_bw_gbs * spec.sockets) + (1 - f) / dram)
    else:
        bw = dram
    bw_eff = 0.35 if knl else 0.62
    if spec.sockets > 1:
        bw_eff *= 0.8  # cross-socket traffic with --numa distribute
    bw_bytes = bw * 1e9 * bw_eff
    int8, fp32 = _ops(spec)
    cores = spec.cores
    cmp_eff = 0.18 if knl else 0.30
    if spec.tier == "amx":
        cmp_eff = 0.12  # AMX only accelerates a subset of quant types in ggml
    int8_ops = cores * spec.ghz * 1e9 * int8 * cmp_eff
    fp32_ops = cores * spec.ghz * 1e9 * fp32 * cmp_eff
    moe_penalty = 0.6 if active_frac < 0.5 else 1.0
    rows = []
    for d in depths:
        kv_bytes = d * arch["attn_layers"] * 2 * arch["kv_dim"] * kv_b
        attn_tg = 4 * arch["attn_layers"] * arch["q_dim"] * max(d, 1)
        tg_mem = (active_bytes + kv_bytes) / bw_bytes
        tg_cmp = 2 * active_params / (int8_ops * moe_penalty) + attn_tg / fp32_ops
        # Fixed per-token graph/thread-barrier cost; large on KNL's slow cores.
        tg = 1.0 / (max(tg_mem, tg_cmp) + (0.008 if knl else 0.0015))
        attn_pp = 4 * arch["attn_layers"] * arch["q_dim"] * (d + 256)
        pp_s = 2 * active_params / (int8_ops * moe_penalty) + attn_pp / fp32_ops
        pp = 1.0 / pp_s
        rows.append({"depth": int(d), "pp_tps": round(pp, 1), "tg_tps": round(tg, 2)})
    load = w_bytes / (storage_gbs * 1e9) + 1.5
    if memory_tier in {"mcdram", "hbm", "spill"}:
        load += w_bytes / (0.5 * dram * 1e9)  # --no-mmap copy into bound pages
    return {"model": model, "quant": quant, "weight_gib": round(w_gib, 2),
            "memory_tier": memory_tier, "load_time_s": round(load, 1), "depths": rows,
            "status": "ESTIMATED", "method": "bandwidth/compute roofline (cpu_platform.estimate)"}


# ── Measured strategy benchmarks ─────────────────────────────────────────────
def load_results(directory: pathlib.Path = CPU_RESULTS) -> list[dict]:
    out = []
    for p in sorted(pathlib.Path(directory).glob("*.json")):
        try:
            x = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(x, dict) and x.get("schema") == "pushbutton.cpu-bench.v1":
            out.append(x)
    return out


def best_measured(info: CPUInfo, model: str, quant: str,
                  directory: pathlib.Path = CPU_RESULTS) -> dict | None:
    """Fastest measured strategy (by shallowest-depth TG) for this exact host layout."""
    best = None
    for r in load_results(directory):
        if (r.get("cpu") or {}).get("signature") != info.signature:
            continue
        if r.get("model") != model or str(r.get("quant", "")).upper() != quant.upper():
            continue
        rows = [x for x in r.get("depths", []) if x.get("tg_tps")]
        if not rows:
            continue
        tg = min(rows, key=lambda x: x["depth"])["tg_tps"]
        if best is None or tg > best[0]:
            best = (tg, r)
    return best[1] if best else None


# ── Planning ─────────────────────────────────────────────────────────────────
def ram_capacity_mib(info: CPUInfo) -> int:
    headroom = max(2048, int(info.ram_total_mib * 0.08))
    return max(0, info.ram_available_mib - headroom)


def min_tps(settings: dict | None) -> float:
    if settings and settings.get("min_tps"):
        return float(settings["min_tps"])
    env = os.environ.get("PUSHBUTTON_CPU_MIN_TPS")
    if env:
        return float(env)
    import pushbutton_policy
    return pushbutton_policy.MIN_DECODE_TOK_S


@dataclass
class CPUChoice:
    model: str
    profile: object
    required_mib: int
    strategy: Strategy
    estimate: dict
    memory_cap_mib: int
    measured: dict | None = None
    alternatives: list = field(default_factory=list)

    def as_row(self) -> dict:
        tg = pp = None
        evidence = "ESTIMATED"
        if self.measured:
            rows = sorted(self.measured.get("depths", []), key=lambda x: x["depth"])
            if rows:
                tg, pp, evidence = rows[0].get("tg_tps"), rows[0].get("pp_tps"), "LOCAL MEASURED"
        if tg is None:
            first = self.estimate["depths"][0]
            tg, pp = first["tg_tps"], first["pp_tps"]
        return {"tg": tg, "pp": pp, "evidence": evidence}


def choose_profile(model: str, settings: dict, info: CPUInfo, ram_cap_mib: int | None = None,
                   fast_cap_mib: int | None = None, threads: int | None = None,
                   nodes: list[NumaNode] | None = None, results_dir: pathlib.Path = CPU_RESULTS) -> CPUChoice | None:
    """Best GGUF profile and strategy for a CPU host.

    With MCDRAM/HBM, the best-quality quant that fits that memory wins (the fast
    tier is worth more decode speed than one quant step). Otherwise the
    best-quality quant meeting the decode target is chosen, falling back to the
    fastest quant with quality >= 90.
    """
    import claude_local_plan as base
    import pushbutton_capacity as capacity
    if model not in base.PROFILES or model not in MODEL_ARCH:
        return None
    ram_cap = ram_capacity_mib(info) if ram_cap_mib is None else ram_cap_mib
    fast_cap = fast_capacity_mib(info) if fast_cap_mib is None else fast_cap_mib
    spec = detected_spec(info, threads)
    depth = min(int(settings["context"]), 8192)
    cands = []
    for p in base.PROFILES[model]:
        if not capacity.matches_quant(p.quant, settings) or settings["context"] > p.native_context:
            continue
        p2 = replace(p, kv_k=settings.get("kv_k", p.kv_k), kv_v=settings.get("kv_v", p.kv_v))
        req = cpu_required_mib(model, p.quant, settings["context"] * settings["slots"], p2.kv_k, p2.kv_v)
        if req > ram_cap:
            continue
        w_mib = int(weight_gib(model, p.quant) * 1024)
        in_fast = bool(fast_cap) and (req if info.fast_mem_mode == "flat" else int(w_mib * 1.08)) <= fast_cap
        est = estimate(spec, model, p.quant, (depth,), p2.kv_k, required_mib=req)
        cands.append((p2, req, in_fast, est["depths"][0]["tg_tps"]))
    if not cands:
        return None
    if fast_cap and any(c[2] and c[0].quality >= 80 for c in cands):
        pick = max((c for c in cands if c[2] and c[0].quality >= 80), key=lambda c: c[0].quality)
    else:
        target = min_tps(settings)
        meeting = [c for c in cands if c[3] >= target]
        if meeting:
            pick = max(meeting, key=lambda c: c[0].quality)
        else:
            floor = [c for c in cands if c[0].quality >= 90] or cands
            pick = max(floor, key=lambda c: (c[3], c[0].quality))
    profile, req, _, _ = pick
    strategies = candidate_strategies(info, req, threads, nodes)
    strategy = strategies[0]
    measured = best_measured(info, model, profile.quant, results_dir)
    if measured:
        by_name = {s.name: s for s in strategies}
        chosen = by_name.get((measured.get("strategy") or {}).get("name"))
        if chosen:
            strategy = chosen
        else:
            measured = None
    est = estimate(spec, model, profile.quant, sorted({0, depth, int(settings["context"])}),
                   profile.kv_k, memory_tier=strategy.memory_tier if strategy.memory_tier != "dram" else None,
                   required_mib=req)
    return CPUChoice(model, profile, req, strategy, est, ram_cap, measured,
                     [{"quant": c[0].quant, "required_mib": c[1], "fits_fast_memory": c[2],
                       "tg_estimate": c[3], "quality": c[0].quality} for c in cands])


def place_requests(items: list[tuple[str, dict]], info: CPUInfo | None = None,
                   memory_caps: list[int | None] | None = None) -> list[CPUChoice]:
    """Place one or more model instances on a CPU-only host.

    With enough DRAM NUMA nodes, instances get disjoint node sets; otherwise they
    share the machine with an even split of physical cores and memory.
    """
    info = info or detect()
    n = len(items)
    if n == 0:
        return []
    compute = info.compute_nodes
    caps = memory_caps or [None] * n
    choices = []
    if n > 1 and len(compute) >= n:
        per = len(compute) // n
        groups = [compute[i * per:(i + 1) * per] for i in range(n)]
        for i, extra in enumerate(compute[n * per:]):
            groups[i].append(extra)
        smt = max(1, info.logical_cpus // max(1, info.physical_cores))
        for (model, settings), group, cap in zip(items, groups, caps):
            node_cap = int(sum(g.free_mib for g in group) * 0.94)
            threads = max(1, sum(len(g.cpus) for g in group) // smt)
            c = choose_profile(model, settings, info, min(node_cap, cap or node_cap),
                               fast_cap_mib=0 if info.fast_mem_mode == "flat" else None,
                               threads=threads, nodes=group)
            if c is None:
                raise ValueError(f"no CPU quant of {model} fits NUMA node(s) {[g.id for g in group]} "
                                 f"({node_cap:,} MiB)")
            choices.append(c)
        return choices
    ram = ram_capacity_mib(info) // n
    fast = fast_capacity_mib(info) // n
    threads = max(1, (info.performance_cores if info.family == "apple" else info.physical_cores) // n)
    for (model, settings), cap in zip(items, caps):
        c = choose_profile(model, settings, info, min(ram, cap or ram), fast, threads if n > 1 else None)
        if c is None:
            raise ValueError(f"no CPU quant of {model} fits {ram:,} MiB of available RAM per instance "
                             f"at context {settings['context']:,}")
        choices.append(c)
    return choices


def server_fields(choice: CPUChoice, info: CPUInfo) -> dict:
    """Extra plan fields that mark a server/worker as CPU-hosted."""
    return {
        "device": "cpu",
        "cpu": {
            "strategy": choice.strategy.to_dict(),
            "estimate": choice.estimate,
            "measured": bool(choice.measured),
            "memory_cap_mib": choice.memory_cap_mib,
            "host": info.summary(),
            "alternatives": choice.alternatives,
        },
    }


# ── Reports ──────────────────────────────────────────────────────────────────
def reference_estimates(context: int = 32768, depths=DEFAULT_DEPTHS) -> list[dict]:
    """Best fitting quant per reference CPU × model with MCDRAM/HBM-aware sizing."""
    import claude_local_plan as base
    import pushbutton_capacity as capacity
    out = []
    for spec in REFERENCE_CPUS:
        ram_cap = int(spec.ram_gib * 1024 * 0.92) - 2048
        fast_cap = int(spec.fast_mem_gib * spec.sockets * 1024 * (0.94 if spec.fast_mem_mode == "flat" else 0.90))
        for model in MODEL_ARCH:
            settings = capacity.resolve_options({}, context)
            cands = []
            for p in base.PROFILES.get(model, ()):
                if context > p.native_context:
                    continue
                req = cpu_required_mib(model, p.quant, context, p.kv_k, p.kv_v)
                if req > ram_cap:
                    continue
                w_mib = weight_gib(model, p.quant) * 1024
                in_fast = bool(fast_cap) and (req if spec.fast_mem_mode == "flat" else w_mib * 1.08) <= fast_cap
                cands.append((p, req, in_fast))
            if not cands:
                out.append({"cpu": spec.key, "cpu_name": spec.name, "model": model, "fits": False})
                continue
            fast_fit = [c for c in cands if c[2] and c[0].quality >= 80]
            if fast_fit:
                p, req, _ = max(fast_fit, key=lambda c: c[0].quality)
            else:
                scored = [(c, estimate(spec, model, c[0].quant, (0,), c[0].kv_k, required_mib=c[1])["depths"][0]["tg_tps"]) for c in cands]
                good = [s for s in scored if s[1] >= 10.0]
                if good:
                    (p, req, _), _ = max(good, key=lambda s: s[0][0].quality)
                else:
                    floor = [s for s in scored if s[0][0].quality >= 90] or scored
                    (p, req, _), _ = max(floor, key=lambda s: (s[1], s[0][0].quality))
            est = estimate(spec, model, p.quant, depths, p.kv_k, required_mib=req)
            out.append({"cpu": spec.key, "cpu_name": spec.name, "model": model, "fits": True,
                        "quant": p.quant, "required_mib": req, **{k: est[k] for k in ("weight_gib", "memory_tier", "load_time_s", "depths")}})
    return out


def estimates_markdown(context: int = 32768, depths=DEFAULT_DEPTHS) -> str:
    rows = reference_estimates(context, depths)
    lines = [
        "# CPU throughput estimates",
        "",
        "Generated by `python3 lib/cpu_platform.py estimates --markdown`; do not edit by hand.",
        "CI checks that this file matches the generator.",
        "",
        f"Planning context: {context:,} tokens per slot, one slot. Quant choice per CPU is automatic:",
        "the best-quality quant that fits MCDRAM/HBM when present, otherwise the best-quality quant",
        "estimated at >=10 tok/s TG, otherwise the fastest quant with quality >= 90.",
        "",
        "These are roofline **estimates** (memory bandwidth for TG, int8/fp32 compute for PP),",
        "not measurements. `pushbutton-cpu-bench` records measured numbers per strategy in",
        "`benchmarks/cpu/results/`, and the planner prefers measured results automatically.",
        "Model architectures for unreleased/vendor-only checkpoints are planning assumptions (see",
        "`MODEL_ARCH` in `lib/cpu_platform.py`).",
        "",
        "Memory tiers: `mcdram`/`hbm` = bound to on-package memory (flat mode), `mcdram-cache` =",
        "working set fits the MCDRAM cache, `spill` = partly in fast memory, `dram` = DDR only.",
        "",
        "## Reference CPUs",
        "",
        "| key | CPU | ISA tier | cores | GHz | DRAM GB/s | fast memory |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for s in REFERENCE_CPUS:
        fast = f"{s.fast_mem_gib * s.sockets:.0f} GB {s.fast_mem_kind.upper()} {s.fast_mem_mode}, {s.fast_mem_bw_gbs * s.sockets:.0f} GB/s" if s.fast_mem_kind else "-"
        lines.append(f"| {s.key} | {s.name} | {s.tier} | {s.cores} | {s.ghz:.1f} | {s.dram_bw_gbs * s.sockets:.0f} | {fast} |")
    head = " | ".join(f"PP@{d // 1024}K / TG@{d // 1024}K" if d else "PP@0 / TG@0" for d in depths)
    for model in MODEL_ARCH:
        lines += ["", f"## {model}", "", f"| CPU | quant | weights GiB | tier | load s | {head} |",
                  "|---|---|---:|---|---:|" + "---:|" * len(depths)]
        for r in rows:
            if r["model"] != model:
                continue
            if not r["fits"]:
                lines.append(f"| {r['cpu']} | does not fit RAM | - | - | - |" + " - |" * len(depths))
                continue
            cells = " | ".join(f"{x['pp_tps']:.0f} / {x['tg_tps']:.1f}" for x in r["depths"])
            lines.append(f"| {r['cpu']} | {r['quant']} | {r['weight_gib']:.1f} | {r['memory_tier']} | {r['load_time_s']:.0f} | {cells} |")
    return "\n".join(lines) + "\n"


def launch_args(plan_path: str, server_id: str, part: str) -> list[str]:
    obj = json.loads(pathlib.Path(plan_path).read_text())
    for s in obj.get("servers", []) + obj.get("workers", []):
        if s.get("id") != server_id:
            continue
        if s.get("device") != "cpu":
            return []
        strat = s["cpu"]["strategy"]
        return list(strat.get("numactl", [])) if part == "prefix" else list(strat.get("llama_args", []))
    raise ValueError(f"server {server_id} not found in plan")


def plan_is_cpu(plan_path: str) -> bool:
    obj = json.loads(pathlib.Path(plan_path).read_text())
    rows = obj.get("servers", []) + obj.get("workers", [])
    return bool(rows) and all(r.get("device") == "cpu" for r in rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cpu_platform")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect")
    cf = sub.add_parser("cmake-flags")
    cf.add_argument("--portable", action="store_true")
    pp = sub.add_parser("plan")
    pp.add_argument("model")
    pp.add_argument("--context", type=int, default=32768)
    pp.add_argument("--quant")
    es = sub.add_parser("estimates")
    es.add_argument("--markdown", action="store_true")
    es.add_argument("--context", type=int, default=32768)
    la = sub.add_parser("launch-args")
    la.add_argument("--plan", required=True)
    la.add_argument("--id", required=True)
    la.add_argument("part", choices=["prefix", "args"])
    ic = sub.add_parser("plan-is-cpu")
    ic.add_argument("plan")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "detect":
            info = detect()
            print(json.dumps({**info.summary(), "cmake_flags": cmake_flags(info),
                              "ram_capacity_mib": ram_capacity_mib(info),
                              "fast_capacity_mib": fast_capacity_mib(info)}, indent=2))
        elif a.cmd == "cmake-flags":
            print("\n".join(cmake_flags(detect(), a.portable or None)))
        elif a.cmd == "plan":
            import claude_local_plan as base
            import pushbutton_capacity as capacity
            model = base.canonical_model(a.model)
            opts = {"quant": a.quant} if a.quant else {}
            choice = place_requests([(model, capacity.resolve_options(opts, a.context))])[0]
            print(json.dumps({"model": model, "quant": choice.profile.quant,
                              "required_mib": choice.required_mib, **server_fields(choice, detect())}, indent=2))
        elif a.cmd == "estimates":
            if a.markdown:
                sys.stdout.write(estimates_markdown(a.context))
            else:
                print(json.dumps(reference_estimates(a.context), indent=2))
        elif a.cmd == "launch-args":
            sys.stdout.write("".join(x + "\0" for x in launch_args(a.plan, a.id, a.part)))
        elif a.cmd == "plan-is-cpu":
            return 0 if plan_is_cpu(a.plan) else 1
        return 0
    except ValueError as exc:
        print(f"cpu_platform: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
