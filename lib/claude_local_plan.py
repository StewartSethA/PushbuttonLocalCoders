#!/usr/bin/env python3
"""Hardware-aware model and GPU planner for claude-local.

No third-party Python dependencies. The planner treats VRAM as a hard resource:
concurrently served model instances receive disjoint GPU sets. A single model is
placed on the freest viable GPU set, with system-aware maximum PCIe link
capability as the next placement tie-breaker. Multiple requested models are
planned jointly before any server is started so an early assignment cannot
strand a later model.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Iterable

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import pushbutton_capacity as capacity

MIB_PER_GIB = 1024


@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    total_mib: int
    free_mib: int
    compute_cap: str = ""
    pci_bus: str = ""
    # pcie_gen/width are the maximum possible link for this GPU in this system,
    # not the instantaneous idle link state. NVML documents max generation as
    # system-aware (e.g. a Gen2 GPU in a Gen1 slot reports Gen1).
    pcie_gen: int = 0
    pcie_width: int = 0
    pcie_current_gen: int = 0
    pcie_current_width: int = 0

    @property
    def free_gib(self) -> float:
        return self.free_mib / MIB_PER_GIB

    @property
    def link_score(self) -> int:
        return max(1, self.pcie_gen) * max(1, self.pcie_width)

    @property
    def speed_score(self) -> int:
        # Memory bandwidth dominates single-token GGUF decode. This table is
        # deliberately coarse and is only a late placement tie-breaker.
        n = self.name.lower()
        known = [
            ("b200", 8000), ("h200", 4800), ("h100", 3350), ("a100", 1900),
            ("4090", 1008), ("3090", 936), ("v100", 900), ("4080", 717),
            ("4070 ti", 504), ("4070", 504), ("4060 ti", 288), ("4060", 272),
            ("p40", 346),
        ]
        for token, bandwidth in known:
            if token in n:
                return bandwidth
        try:
            major, minor = (int(x) for x in self.compute_cap.split(".", 1))
            return 150 + major * 40 + minor * 5
        except Exception:
            return 200


@dataclass(frozen=True)
class Profile:
    model: str
    display: str
    repo: str
    quant: str
    required_mib: int
    quality: int
    native_context: int = 262144
    kv_k: str = "q4_0"
    kv_v: str = "q4_0"
    batch: int = 512
    ubatch: int = 256
    flash_attn: str = "on"
    template: str = "embedded"  # embedded | qwen-fixed
    build: str = "upstream"     # upstream | glm53
    extra_env: tuple[tuple[str, str], ...] = ()
    note: str = ""

    @property
    def hf_spec(self) -> str:
        return f"{self.repo}:{self.quant}"


ALIASES = {
    "ornith-1.5:9b": "ornith-1.5:9b", "ornith": "ornith-1.5:9b",
    "ornith-1.5": "ornith-1.5:9b", "ornith-9b": "ornith-1.5:9b",
    "ornith-1.5-9b": "ornith-1.5:9b",
    "ornith-1.5:35b-a3b": "ornith-1.5:35b-a3b", "ornith-35b": "ornith-1.5:35b-a3b",
    "ornith-1.5:35b": "ornith-1.5:35b-a3b",
    "ornith-1.5-35b": "ornith-1.5:35b-a3b", "ornith-1.5-35b-a3b": "ornith-1.5:35b-a3b",
    "qwen3.8:27b": "qwen3.8:27b", "qwen3.8-27b": "qwen3.8:27b", "q38": "qwen3.8:27b",
    "qwen3.8": "qwen3.8:27b", "qwen38": "qwen3.8:27b",
    "qwen3.8-flash-next": "qwen3.8-flash-next", "qwen3.8:flash-next": "qwen3.8-flash-next",
    "qwen38-flash-next": "qwen3.8-flash-next", "qwen38next": "qwen3.8-flash-next", "q38next": "qwen3.8-flash-next",
    "qwen3.6:35b": "qwen3.6:35b", "qwen3.6-35b": "qwen3.6:35b", "q36": "qwen3.6:35b",
    "qwen3.6": "qwen3.6:35b", "qwen36": "qwen3.6:35b",
    "nemotron-3.5-lightning": "nemotron-3.5-lightning", "nemotron3.5-lightning": "nemotron-3.5-lightning",
    "nemotron": "nemotron-3.5-lightning", "nemotron-lightning": "nemotron-3.5-lightning",
    "glm-5.3-flash": "glm-5.3-flash", "glm5.3-flash": "glm-5.3-flash", "glm53": "glm-5.3-flash",
    "glm": "glm-5.3-flash",
    "deepseek-v4-flash": "deepseek-v4-flash", "deepseek-v4-flash-0731": "deepseek-v4-flash",
    "deepseek-v4": "deepseek-v4-flash", "deepseek": "deepseek-v4-flash", "dsv4": "deepseek-v4-flash",
}

# required_mib includes model weights, full native context at the listed KV
# quant, compute buffers, and a modest CUDA safety margin. Profiles are ordered
# best quality first. Values are conservative planning envelopes, not promises.
PROFILES: dict[str, tuple[Profile, ...]] = {
    # Catalog weights plus context/buffer overhead, rounded up to whole GiB.
    # These envelopes are planning estimates, not local allocation/speed evidence.
    "ornith-1.5:9b": (
        Profile("ornith-1.5:9b", "Ornith 1.5 9B", "ornith-ai/Ornith-1.5-9B-GGUF", "Q8_0", 17*1024, 100),
        Profile("ornith-1.5:9b", "Ornith 1.5 9B", "ornith-ai/Ornith-1.5-9B-GGUF", "Q6_K", 15*1024, 98),
        Profile("ornith-1.5:9b", "Ornith 1.5 9B", "ornith-ai/Ornith-1.5-9B-GGUF", "Q5_K_M", 14*1024, 97),
        Profile("ornith-1.5:9b", "Ornith 1.5 9B", "ornith-ai/Ornith-1.5-9B-GGUF", "Q4_K_M", 13*1024, 95),
    ),
    "ornith-1.5:35b-a3b": (
        Profile("ornith-1.5:35b-a3b", "Ornith 1.5 35B-A3B", "ornith-ai/Ornith-1.5-35B-A3B-GGUF", "Q8_0", 49*1024, 100),
        Profile("ornith-1.5:35b-a3b", "Ornith 1.5 35B-A3B", "ornith-ai/Ornith-1.5-35B-A3B-GGUF", "Q6_K", 40*1024, 98),
        Profile("ornith-1.5:35b-a3b", "Ornith 1.5 35B-A3B", "ornith-ai/Ornith-1.5-35B-A3B-GGUF", "Q5_K_M", 36*1024, 97),
        Profile("ornith-1.5:35b-a3b", "Ornith 1.5 35B-A3B", "ornith-ai/Ornith-1.5-35B-A3B-GGUF", "Q4_K_M", 33*1024, 95),
    ),
    "qwen3.8:27b": (
        Profile("qwen3.8:27b", "Qwen3.8 27B", "unsloth/Qwen3.8-27B-GGUF", "Q8_0", 35*1024, 100, template="qwen-fixed"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "unsloth/Qwen3.8-27B-GGUF", "UD-Q6_K", 28*1024, 98, template="qwen-fixed"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "unsloth/Qwen3.8-27B-GGUF", "UD-Q5_K_M", 25*1024, 97, template="qwen-fixed"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "unsloth/Qwen3.8-27B-GGUF", "UD-Q4_K_M", 22*1024, 95, template="qwen-fixed"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", "IQ3_XXS", 15000, 91, template="qwen-fixed", note="full-context 16 GB profile"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", "IQ2_S", 14200, 86, template="qwen-fixed", note="lower-bit fallback with extra VRAM margin"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", "IQ2_XS", 13500, 80, template="qwen-fixed"),
    ),
    "qwen3.8-flash-next": (
        Profile("qwen3.8-flash-next", "Qwen3.8 Flash Next", "unsloth/Qwen3.8-Flash-Next-GGUF", "UD-Q4_K_XL", 136*1024, 100, batch=256, ubatch=128, note="quality tier for >128 GB practical free VRAM; stock llama.cpp, no MTP"),
        Profile("qwen3.8-flash-next", "Qwen3.8 Flash Next", "unsloth/Qwen3.8-Flash-Next-GGUF", "UD-IQ4_XS", 116*1024, 97, batch=256, ubatch=128, note="recommended stable 4x32 GB V100 profile at 256K context; MTP disabled"),
        Profile("qwen3.8-flash-next", "Qwen3.8 Flash Next", "unsloth/Qwen3.8-Flash-Next-GGUF", "UD-Q3_K_XL", 112*1024, 94, batch=256, ubatch=128, note="high-headroom 4x32 GB profile"),
        Profile("qwen3.8-flash-next", "Qwen3.8 Flash Next", "unsloth/Qwen3.8-Flash-Next-GGUF", "UD-IQ3_XXS", 104*1024, 91, batch=256, ubatch=128, note="balanced fallback"),
        Profile("qwen3.8-flash-next", "Qwen3.8 Flash Next", "unsloth/Qwen3.8-Flash-Next-GGUF", "UD-Q2_K_XL", 99*1024, 86, batch=256, ubatch=128, note="maximum-headroom 4x32 GB fallback"),
        Profile("qwen3.8-flash-next", "Qwen3.8 Flash Next", "unsloth/Qwen3.8-Flash-Next-GGUF", "UD-IQ1_M", 94*1024, 80, batch=256, ubatch=128, note="low-bit fallback"),
    ),
    "qwen3.6:35b": (
        Profile("qwen3.6:35b", "Qwen3.6 35B-A3B", "unsloth/Qwen3.6-35B-A3B-GGUF", "UD-Q6_K", 35*1024, 100, template="qwen-fixed"),
        Profile("qwen3.6:35b", "Qwen3.6 35B-A3B", "unsloth/Qwen3.6-35B-A3B-GGUF", "UD-Q5_K_M", 31800, 99, template="qwen-fixed"),
        Profile("qwen3.6:35b", "Qwen3.6 35B-A3B", "unsloth/Qwen3.6-35B-A3B-GGUF", "UD-Q4_K_M", 27*1024, 97, template="qwen-fixed"),
        Profile("qwen3.6:35b", "Qwen3.6 35B-A3B", "unsloth/Qwen3.6-35B-A3B-GGUF", "UD-IQ3_XXS", 15800, 91, template="qwen-fixed", note="measured full-context 16 GB class profile"),
        Profile("qwen3.6:35b", "Qwen3.6 35B-A3B", "unsloth/Qwen3.6-35B-A3B-GGUF", "UD-IQ2_M", 14500, 84, template="qwen-fixed"),
    ),
    "nemotron-3.5-lightning": (
        Profile("nemotron-3.5-lightning", "Nemotron 3.5 Lightning 30B-A3B", "ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "Q8_0", 40*1024, 100),
        Profile("nemotron-3.5-lightning", "Nemotron 3.5 Lightning 30B-A3B", "vcruz305/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "Q5_0", 29*1024, 98),
        Profile("nemotron-3.5-lightning", "Nemotron 3.5 Lightning 30B-A3B", "vcruz305/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "MXFP4_MOE", 24*1024, 96, note="agentic-quality 24 GB profile"),
        Profile("nemotron-3.5-lightning", "Nemotron 3.5 Lightning 30B-A3B", "vcruz305/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "MIXED-Q2_0-Q4_0-2.97BPW", 15800, 78, note="experimental 16 GB fallback; prefer Qwen for critical agent work"),
    ),
    "glm-5.3-flash": (
        Profile("glm-5.3-flash", "GLM-5.3-Flash", "unsloth/GLM-5.3-Flash-GGUF", "UD-IQ2_XXS", 120*1024, 94, flash_attn="off", build="glm53", extra_env=(("NVIDIA_TF32_OVERRIDE", "0"),), note="requires current GLM5Next llama.cpp fork"),
        Profile("glm-5.3-flash", "GLM-5.3-Flash", "unsloth/GLM-5.3-Flash-GGUF", "UD-IQ1_M", 112*1024, 87, flash_attn="off", build="glm53", extra_env=(("NVIDIA_TF32_OVERRIDE", "0"),), note="requires current GLM5Next llama.cpp fork"),
    ),
    "deepseek-v4-flash": (
        Profile("deepseek-v4-flash", "DeepSeek V4 Flash 0731", "unsloth/DeepSeek-V4-Flash-0731-GGUF", "UD-IQ3_XXS", 132*1024, 95, batch=256, ubatch=128, note="quality tier; requires >128 GB practical free VRAM at 256K context"),
        Profile("deepseek-v4-flash", "DeepSeek V4 Flash 0731", "unsloth/DeepSeek-V4-Flash-0731-GGUF", "UD-IQ2_XXS", 112*1024, 91, batch=256, ubatch=128, note="recommended 4x32 GB V100 baseline at 256K context"),
        Profile("deepseek-v4-flash", "DeepSeek V4 Flash 0731", "unsloth/DeepSeek-V4-Flash-0731-GGUF", "UD-IQ1_M", 106*1024, 85, batch=256, ubatch=128, note="extra-headroom fallback"),
        Profile("deepseek-v4-flash", "DeepSeek V4 Flash 0731", "unsloth/DeepSeek-V4-Flash-0731-GGUF", "UD-IQ1_S", 101*1024, 80, batch=256, ubatch=128, note="maximum-headroom fallback"),
    ),
}

ROLES = ("haiku", "sonnet", "opus", "fable")


def canonical_model(name: str) -> str:
    key = name.strip().lower()
    if key not in ALIASES:
        raise ValueError(f"unknown model '{name}'")
    return ALIASES[key]


def role_map(models: list[str]) -> dict[str, str]:
    if not models:
        raise ValueError("no models supplied")
    if len(models) > 4:
        raise ValueError("at most four model selectors may be supplied")
    m = [canonical_model(x) for x in models]
    if len(m) == 1:
        m = m * 4
    elif len(m) == 2:
        m = [m[0], m[1], m[1], m[1]]
    elif len(m) == 3:
        m = [m[0], m[1], m[2], m[2]]
    return dict(zip(ROLES, m))


def _query_nvidia(fields: str) -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True, capture_output=True,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _parse_link(line: str) -> tuple[int, int]:
    parts = [x.strip() for x in line.split(",")]
    if len(parts) < 2:
        return 0, 0
    return (
        int(re.sub(r"\D", "", parts[0]) or 0),
        int(re.sub(r"\D", "", parts[1]) or 0),
    )


def inventory() -> list[GPU]:
    base = _query_nvidia("index,name,memory.total,memory.free,compute_cap,pci.bus_id")
    if not base:
        # Older nvidia-smi may not expose compute_cap.
        base = _query_nvidia("index,name,memory.total,memory.free,pci.bus_id")
        has_cc = False
    else:
        has_cc = True

    current_links = _query_nvidia("pcie.link.gen.current,pcie.link.width.current")
    max_links = _query_nvidia("pcie.link.gen.max,pcie.link.width.max")
    # Some old drivers do not expose max fields. In that case retain previous
    # behavior rather than dropping all PCIe information.
    if not max_links:
        max_links = current_links

    out: list[GPU] = []
    for i, line in enumerate(base):
        parts = [x.strip() for x in line.split(",")]
        try:
            if has_cc:
                idx, name, total, free, cc, bus = parts[:6]
            else:
                idx, name, total, free, bus = parts[:5]
                low = name.lower()
                if "v100" in low: cc = "7.0"
                elif "p40" in low: cc = "6.1"
                elif "p100" in low: cc = "6.0"
                elif "a100" in low: cc = "8.0"
                elif "3090" in low: cc = "8.6"
                elif any(x in low for x in ("4090", "4080", "4070", "4060")): cc = "8.9"
                else: cc = ""

            max_gen, max_width = _parse_link(max_links[i]) if i < len(max_links) else (0, 0)
            cur_gen, cur_width = _parse_link(current_links[i]) if i < len(current_links) else (0, 0)
            out.append(
                GPU(
                    int(idx), name, int(float(total)), int(float(free)), cc, bus,
                    max_gen, max_width, cur_gen, cur_width,
                )
            )
        except (ValueError, IndexError):
            continue
    return out


def cpu_fallback_enabled() -> bool:
    return os.environ.get("PUSHBUTTON_DISABLE_CPU_FALLBACK", "0") != "1"


def cpu_plan(requests, context: int, slots: int, rm: dict) -> dict:
    """Plan every requested instance on the CPU when no GPU is visible."""
    import cpu_platform
    from coder_local_plan import request_capacity
    info = cpu_platform.detect()
    items = [(r.model, capacity.resolve_options(r.capacity, context, slots)) for r in requests]
    choices = cpu_platform.place_requests(items, info, [r.vram_limit_mib for r in requests])
    servers = []
    for i, (request, c) in enumerate(zip(requests, choices)):
        settings = request_capacity(request, c.profile, context, slots)
        server_id = f"local-{c.model.replace(':', '-').replace('.', '').replace('_', '-')}"
        if sum(r.model == request.model for r in requests) > 1:
            server_id += f"-{i + 1}"
        servers.append({
            "id": server_id,
            "model": c.model,
            "profile": {**asdict(c.profile), "extra_env": dict(c.profile.extra_env),
                        "hf_spec": c.profile.hf_spec},
            "gpus": [],
            "cuda_visible_devices": "",
            "multi_gpu": False,
            "allocated_mib": c.memory_cap_mib,
            "required_mib": c.required_mib,
            "headroom_mib": c.memory_cap_mib - c.required_mib,
            "placement_policy": "cpu",
            "capacity": settings,
            "requested_gpus": None,
            "vram_limit_mib_per_gpu": request.vram_limit_mib,
            "placement_source": request.source,
            **cpu_platform.server_fields(c, info),
        })
    role_ids = {role: servers[min(i, len(servers) - 1)]["id"] for i, role in enumerate(ROLES)}
    return {"context": context, "roles": rm, "role_ids": role_ids, "servers": servers,
            "unused_gpus": [], "placement_policy": "cpu", "device": "cpu"}


def smart_defaults(gpus: list[GPU], context: int = 262144) -> list[str]:
    if not gpus:
        if not cpu_fallback_enabled():
            raise ValueError("no NVIDIA GPU detected")
        # Small-active-parameter MoE models decode fastest on CPUs; take the
        # first preference that fits this host's RAM at the requested context.
        import cpu_platform
        info = cpu_platform.detect()
        for model in ("qwen3.6:35b", "ornith-1.5:35b-a3b", "nemotron-3.5-lightning", "ornith-1.5:9b"):
            settings = capacity.resolve_options({}, min(context, PROFILES[model][0].native_context))
            if cpu_platform.choose_profile(model, settings, info) is not None:
                return [model]
        raise ValueError("no model fits the available host RAM at this context; lower --context")
    max_free = max(g.free_mib for g in gpus)
    total_free = sum(g.free_mib for g in gpus)
    if max_free >= 15800:
        if total_free >= 42000 and len(gpus) >= 2:
            return ["nemotron-3.5-lightning", "qwen3.6:35b"]
        return ["qwen3.6:35b"]
    if max_free >= 14500:
        return ["qwen3.8:27b"]
    raise ValueError("no supported full-context profile fits the currently free VRAM")


@dataclass(frozen=True)
class Candidate:
    model: str
    profile: Profile
    required_mib: int
    gpu_positions: tuple[int, ...]
    mask: int
    free_mib: int
    link_score: int
    speed_score: int

    @property
    def card_count(self) -> int:
        return len(self.gpu_positions)

    @property
    def headroom_mib(self) -> int:
        return self.free_mib - self.required_mib


def scaled_required_mib(profile: Profile, context: int) -> int:
    return capacity.memory_estimate(profile, capacity.resolve_options({}, context))["required_mib"]


def gpu_subsets(gpus: list[GPU]) -> Iterable[tuple[int, ...]]:
    positions = range(len(gpus))
    for n in range(1, len(gpus) + 1):
        yield from itertools.combinations(positions, n)


def best_profile_for_capacity(
    model: str, capacity_mib: int, context: int, settings: dict | None = None
) -> tuple[Profile, int] | None:
    settings = capacity.resolve_options(
        {k: v for k, v in (settings or {}).items() if k in capacity.OPTION_KEYS and v is not None},
        context)
    if settings.get("quant") and settings["quant"] not in {p.quant.upper() for p in PROFILES[model]}:
        raise ValueError(f"unsupported quant '{settings['quant']}' for {model}")
    if settings["context"] > max(p.native_context for p in PROFILES[model]):
        raise ValueError(f"context exceeds supported native context for {model}")
    for profile in PROFILES[model]:
        if not capacity.matches_quant(profile.quant, settings):
            continue
        if settings["context"] > profile.native_context:
            continue
        req = capacity.memory_estimate(profile, settings)["required_mib"]
        if req <= capacity_mib:
            return replace(profile, kv_k=settings.get("kv_k", profile.kv_k),
                           kv_v=settings.get("kv_v", profile.kv_v)), req
    return None


def placement_candidates(model: str, gpus: list[GPU], context: int,
                         settings: dict | None = None, exact_gpus=None) -> list[Candidate]:
    raw: list[Candidate] = []
    for positions in gpu_subsets(gpus):
        group = [gpus[p] for p in positions]
        if exact_gpus is not None and {g.index for g in group} != set(exact_gpus):
            continue
        free_mib = sum(g.free_mib for g in group)
        picked = best_profile_for_capacity(model, free_mib, context, settings)
        if picked is None:
            continue
        profile, req = picked
        raw.append(
            Candidate(
                model=model,
                profile=profile,
                required_mib=req,
                gpu_positions=positions,
                mask=sum(1 << p for p in positions),
                free_mib=free_mib,
                link_score=sum(g.link_score for g in group),
                speed_score=sum(g.speed_score for g in group),
            )
        )

    # Drop a larger GPU subset when a strict subset of those same GPUs already
    # fits an equal-or-better quant. This bounds the global search without
    # discarding useful placements.
    useful: list[Candidate] = []
    for c in raw:
        dominated = any(
            other is not c
            and other.card_count < c.card_count
            and other.profile.quality >= c.profile.quality
            and (other.mask & c.mask) == other.mask
            for other in raw
        )
        if not dominated:
            useful.append(c)
    return useful


def single_model_choice(
    model: str, gpus: list[GPU], context: int
) -> Candidate | None:
    """Pick the freest viable remaining GPU set; max link speed breaks ties."""
    candidates = placement_candidates(model, gpus, context)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (
            c.card_count,             # do not split unless necessary
            -c.free_mib,              # most free remaining first
            -c.link_score,            # then max system-aware PCIe capability
            -c.profile.quality,       # then best quant on that placement
            -c.speed_score,           # late tie: local memory bandwidth
            tuple(gpus[p].index for p in c.gpu_positions),
        ),
    )


def _candidate_score(c: Candidate, sonnet_model: str) -> tuple[int, ...]:
    """Additive lexicographic contribution; larger totals are better."""
    is_sonnet = c.model == sonnet_model
    return (
        -c.card_count,                         # preserve concurrency
        c.profile.quality,                    # maximize aggregate quant quality
        c.free_mib if is_sonnet else 0,       # roomy daily-driver placement
        c.link_score if is_sonnet else 0,     # fast daily-driver PCIe path
        c.headroom_mib,                       # allocator robustness
        c.free_mib,                           # prefer freer overall layout
        c.link_score,                         # then better links overall
        c.speed_score if is_sonnet else 0,
        c.speed_score,
    )


def _score_add(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(x + y for x, y in zip(a, b))


def joint_model_choices(
    models: list[str] | list[int], gpus: list[GPU], context: int, sonnet_model: str,
    candidate_map: dict | None = None,
) -> list[Candidate] | None:
    """Find the best complete disjoint model/GPU plan before launching."""
    if not models:
        return []

    candidates = candidate_map if candidate_map is not None else {
        m: placement_candidates(m, gpus, context) for m in models}
    if any(not candidates[m] for m in models):
        return None

    if len(gpus) > 16:
        # Avoid an exponential search on unusually large clusters. Place the
        # hardest models first, with the same freest/link-aware local policy.
        remaining = list(gpus)
        result: list[Candidate] = []
        order = sorted(
            models,
            key=lambda m: min(c.required_mib for c in candidates[m]),
            reverse=True,
        )
        for model in order:
            available = {gpus.index(g) for g in remaining}
            viable = [c for c in candidates[model] if set(c.gpu_positions) <= available]
            local = min(viable, key=lambda c: (c.card_count, -c.free_mib, -c.link_score,
                                               -c.profile.quality, -c.speed_score)) if viable else None
            if local is None:
                return None
            chosen = [gpus[p] for p in local.gpu_positions]
            physical_positions = tuple(gpus.index(g) for g in chosen)
            result.append(
                Candidate(
                    model=local.model,
                    profile=local.profile,
                    required_mib=local.required_mib,
                    gpu_positions=physical_positions,
                    mask=sum(1 << p for p in physical_positions),
                    free_mib=local.free_mib,
                    link_score=local.link_score,
                    speed_score=local.speed_score,
                )
            )
            remaining = [g for g in remaining if g not in chosen]
        by_model = dict(zip(order, result))
        return [by_model[m] for m in models]

    # Hardest-first ordering reduces branching only; the global score chooses
    # the resulting assignment. The bitmask makes disjointness explicit.
    order = sorted(
        models,
        key=lambda m: min(c.required_mib for c in candidates[m]),
        reverse=True,
    )
    zero_score = (0,) * 9

    @lru_cache(maxsize=None)
    def search(
        i: int, used_mask: int
    ) -> tuple[tuple[int, ...], tuple[Candidate, ...]] | None:
        if i == len(order):
            return zero_score, ()
        model = order[i]
        best = None
        local = sorted(
            candidates[model],
            key=lambda c: (
                c.card_count,
                -c.profile.quality,
                -c.free_mib,
                -c.link_score,
                -c.headroom_mib,
            ),
        )
        for c in local:
            if c.mask & used_mask:
                continue
            tail = search(i + 1, used_mask | c.mask)
            if tail is None:
                continue
            tail_score, tail_choices = tail
            score = _score_add(_candidate_score(c, sonnet_model), tail_score)
            proposal = (score, (c,) + tail_choices)
            if best is None or score > best[0]:
                best = proposal
        return best

    found = search(0, 0)
    if found is None:
        return None
    _, choices = found
    by_model = dict(zip(order, choices))
    return [by_model[m] for m in models]


def ordered_group_for_layer_split(candidate: Candidate, gpus: list[GPU]) -> list[GPU]:
    group = [gpus[p] for p in candidate.gpu_positions]
    if len(group) <= 1:
        return group
    # Keep the slowest endpoint at an edge of the layer pipeline rather than
    # between two faster GPU partitions.
    slowest = min(group, key=lambda g: (g.link_score, g.speed_score))
    rest = sorted(
        (g for g in group if g != slowest),
        key=lambda g: (-g.link_score, -g.speed_score),
    )
    return [slowest] + rest


def unplaceable_detail(request, gpus: list[GPU], context: int, slots: int = 1) -> str:
    """Explain why one request has no placement: smallest fitting need vs. free VRAM."""
    settings = capacity.resolve_options(request.capacity, context, slots)
    allowed = [g for g in gpus if request.gpu_indices is None or g.index in request.gpu_indices]
    limit = request.vram_limit_mib
    free_mib = sum(min(g.free_mib, limit) if limit else g.free_mib for g in allowed)
    where = (f"GPU(s) {[g.index for g in allowed]}" if request.gpu_indices is not None
             else f"all {len(allowed)} GPU(s)")
    if limit:
        where += f" capped at {limit / MIB_PER_GIB:.1f} GiB each"
    label = request.model + (f"@gpu={'+'.join(map(str, request.gpu_indices))}"
                             if request.gpu_indices is not None else "")
    estimates = [(capacity.memory_estimate(p, settings), p) for p in PROFILES[request.model]
                 if capacity.matches_quant(p.quant, settings) and settings["context"] <= p.native_context]
    if not estimates:
        return f"{label}: no profile matches the requested quant/bits/context"
    est, profile = min(estimates, key=lambda x: x[0]["required_mib"])
    text = f"{label}: smallest matching quant {profile.quant} needs {est['required_mib'] / MIB_PER_GIB:.1f} GiB"
    if "weights_mib" in est:
        text += (f" (weights {est['weights_mib'] / MIB_PER_GIB:.1f} + KV {est['kv_mib'] / MIB_PER_GIB:.1f} "
                 f"for {settings['slots']} slot(s) x {settings['context']:,} tokens at "
                 f"{est['kv_k']}/{est['kv_v']} + runtime {est['overhead_mib'] / MIB_PER_GIB:.1f})")
    return text + f"; {where} have {free_mib / MIB_PER_GIB:.1f} GiB free"


def plan(models: list[str], gpus: list[GPU], context: int, slots: int = 1,
         defaults: dict | None = None, client_context: int | None = None,
         agents: int | None = None) -> dict:
    # Lazy import keeps the existing coder -> base API import cycle harmless.
    from coder_local_plan import (parse_model_spec, candidates_for_request,
                                  request_capacity, apply_client_context, expand_workers)
    capacity.resolve_options({}, context, slots)
    requests = expand_workers([parse_model_spec(m, defaults) for m in capacity.model_bits_specs(models)], agents)
    requests = apply_client_context(requests, context, client_context)
    rm = role_map([r.model for r in requests])
    if not gpus and cpu_fallback_enabled():
        return cpu_plan(requests, context, slots, rm)
    instances = list(range(len(requests)))
    candidate_map = {i: candidates_for_request(requests[i], gpus, context, slots,
                                              quality_filter=False) for i in instances}
    if len(instances) == 1:
        candidates = candidate_map[0]
        picked = min(candidates, key=lambda c: (c.card_count, -c.free_mib, -c.link_score,
                     -c.profile.quality, -c.speed_score,
                     tuple(gpus[p].index for p in c.gpu_positions))) if candidates else None
        choices = [picked] if picked else None
        policy = "freest-then-link"
    else:
        choices = joint_model_choices(instances, gpus, context, rm["sonnet"], candidate_map)
        policy = "joint-global-plan"

    if not choices or any(c is None for c in choices):
        unplaceable = [i for i in instances if not candidate_map[i]]
        if unplaceable:
            raise ValueError(
                "cannot place requested model set: "
                + "; ".join(unplaceable_detail(requests[i], gpus, context, slots) for i in unplaceable)
                + ". Lower context/slots, use smaller KV types (kv_k/kv_v), fewer bits, or more GPUs"
            )
        raise ValueError(
            f"cannot place requested model set at context {context:,} without "
            f"sharing/overcommitting GPUs; free VRAM is "
            f"{[round(g.free_gib, 1) for g in gpus]} GiB"
        )

    servers = []
    used_mask = 0
    for i, (request, c) in enumerate(zip(requests, choices)):
        assert c is not None
        used_mask |= c.mask
        group = ordered_group_for_layer_split(c, gpus)
        settings = request_capacity(request, c.profile, context, slots)
        server_id = f"local-{c.model.replace(':', '-').replace('.', '').replace('_', '-')}"
        if sum(r.model == request.model for r in requests) > 1:
            server_id += f"-{i + 1}"
        servers.append(
            {
                "id": server_id,
                "model": c.model,
                "profile": {
                    **asdict(c.profile),
                    "extra_env": dict(c.profile.extra_env),
                    "hf_spec": c.profile.hf_spec,
                },
                "gpus": [asdict(g) for g in group],
                "cuda_visible_devices": ",".join(str(g.index) for g in group),
                "multi_gpu": len(group) > 1,
                "allocated_mib": c.free_mib,
                "required_mib": c.required_mib,
                "headroom_mib": c.headroom_mib,
                "placement_policy": policy,
                "capacity": settings,
                "requested_gpus": list(request.gpu_indices) if request.gpu_indices is not None else None,
                "vram_limit_mib_per_gpu": request.vram_limit_mib,
                "placement_source": request.source,
            }
        )

    role_ids = {
        role: servers[min(i, len(servers) - 1)]["id"]
        for i, role in enumerate(ROLES)
    }
    return {
        "context": context,
        "roles": rm,
        "role_ids": role_ids,
        "servers": servers,
        "unused_gpus": [
            asdict(g)
            for pos, g in enumerate(gpus)
            if not (used_mask & (1 << pos))
        ],
        "placement_policy": policy,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory")
    lp = sub.add_parser("plan")
    lp.add_argument("models", nargs="*")
    lp.add_argument("--context", type=int, default=262144)
    lp.add_argument("--slots", type=int, default=1)
    lp.add_argument("--agents", type=int)
    lp.add_argument("--client-context", type=int)
    lp.add_argument("--placement-config", default=os.path.expanduser(
        os.environ.get("PUSHBUTTON_PLACEMENT_CONFIG", "~/.config/pushbutton-local/placement.json")))
    lp.add_argument("--smart", action="store_true")
    sub.add_parser("catalogue")
    args = ap.parse_args()
    try:
        if args.cmd == "inventory":
            print(json.dumps([asdict(x) for x in inventory()], indent=2))
            return 0
        if args.cmd == "catalogue":
            data = {
                m: [
                    {**asdict(p), "extra_env": dict(p.extra_env), "hf_spec": p.hf_spec}
                    for p in ps
                ]
                for m, ps in PROFILES.items()
            }
            print(json.dumps(data, indent=2))
            return 0
        gpus = inventory()
        selected = os.environ.get("PUSHBUTTON_SELECTOR_GPU_INDICES")
        if selected is not None:
            try:
                indices = {int(x) for x in selected.split(",") if x}
            except ValueError as exc:
                raise ValueError("invalid selector GPU pool") from exc
            if not indices or not indices <= {g.index for g in gpus}:
                raise ValueError("selector GPU pool is empty or no longer visible")
            gpus = [g for g in gpus if g.index in indices]
        models = args.models
        if args.smart or not models:
            models = smart_defaults(gpus, args.context)
        from coder_local_plan import load_placement_config
        defaults = load_placement_config(args.placement_config)
        print(json.dumps(plan(models, gpus, args.context, args.slots, defaults,
                              args.client_context, args.agents), indent=2))
        return 0
    except ValueError as exc:
        print(f"claude-local planner: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())