#!/usr/bin/env python3
"""Hardware-aware model planner for claude-local.

No third-party Python dependencies. The planner deliberately treats VRAM as a
hard resource: each concurrently served model receives an exclusive GPU set.
That avoids accidental CUDA unified-memory spill and makes agent fan-out
predictable on heterogeneous NVIDIA hosts.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Iterable

MIB_PER_GIB = 1024


@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    total_mib: int
    free_mib: int
    compute_cap: str = ""
    pci_bus: str = ""
    pcie_gen: int = 0
    pcie_width: int = 0

    @property
    def free_gib(self) -> float:
        return self.free_mib / MIB_PER_GIB

    @property
    def link_score(self) -> int:
        return max(1, self.pcie_gen) * max(1, self.pcie_width)

    @property
    def speed_score(self) -> int:
        # Memory bandwidth dominates single-token GGUF decode. This table is
        # intentionally coarse and only breaks placement ties; it is not a
        # performance benchmark.
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
    "qwen3.8:27b": "qwen3.8:27b", "qwen3.8-27b": "qwen3.8:27b", "q38": "qwen3.8:27b",
    "qwen3.8": "qwen3.8:27b", "qwen38": "qwen3.8:27b",
    "qwen3.6:35b": "qwen3.6:35b", "qwen3.6-35b": "qwen3.6:35b", "q36": "qwen3.6:35b",
    "qwen3.6": "qwen3.6:35b", "qwen36": "qwen3.6:35b",
    "nemotron-3.5-lightning": "nemotron-3.5-lightning", "nemotron3.5-lightning": "nemotron-3.5-lightning",
    "nemotron": "nemotron-3.5-lightning", "nemotron-lightning": "nemotron-3.5-lightning",
    "glm-5.3-flash": "glm-5.3-flash", "glm5.3-flash": "glm-5.3-flash", "glm53": "glm-5.3-flash",
    "glm": "glm-5.3-flash",
}

# required_mib includes model weights, full native context at the listed KV
# quant, compute buffers, and a modest CUDA safety margin. Profiles are ordered
# best quality first. Values are conservative planning envelopes, not promises.
PROFILES: dict[str, tuple[Profile, ...]] = {
    "qwen3.8:27b": (
        Profile("qwen3.8:27b", "Qwen3.8 27B", "unsloth/Qwen3.8-27B-GGUF", "Q8_0", 35*1024, 100, template="qwen-fixed"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "unsloth/Qwen3.8-27B-GGUF", "UD-Q6_K", 28*1024, 98, template="qwen-fixed"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "unsloth/Qwen3.8-27B-GGUF", "UD-Q5_K_M", 25*1024, 97, template="qwen-fixed"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "unsloth/Qwen3.8-27B-GGUF", "UD-Q4_K_M", 22*1024, 95, template="qwen-fixed"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", "IQ3_XXS", 15000, 91, template="qwen-fixed", note="full-context 16 GB profile"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", "IQ2_S", 14200, 86, template="qwen-fixed", note="lower-bit fallback with extra VRAM margin"),
        Profile("qwen3.8:27b", "Qwen3.8 27B", "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", "IQ2_XS", 13500, 80, template="qwen-fixed"),
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


def inventory() -> list[GPU]:
    base = _query_nvidia("index,name,memory.total,memory.free,compute_cap,pci.bus_id")
    if not base:
        # Older nvidia-smi may not expose compute_cap.
        base = _query_nvidia("index,name,memory.total,memory.free,pci.bus_id")
        has_cc = False
    else:
        has_cc = True
    links = _query_nvidia("pcie.link.gen.current,pcie.link.width.current")
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
            gen = width = 0
            if i < len(links):
                lp = [x.strip() for x in links[i].split(",")]
                gen = int(re.sub(r"\D", "", lp[0]) or 0)
                width = int(re.sub(r"\D", "", lp[1]) or 0)
            out.append(GPU(int(idx), name, int(float(total)), int(float(free)), cc, bus, gen, width))
        except (ValueError, IndexError):
            continue
    return out


def smart_defaults(gpus: list[GPU]) -> list[str]:
    if not gpus:
        raise ValueError("no NVIDIA GPU detected")
    max_free = max(g.free_mib for g in gpus)
    total_free = sum(g.free_mib for g in gpus)
    # Bias the everyday Sonnet role toward Qwen3.6 when a 16 GB-class device
    # can hold its full-context IQ3 profile; otherwise Qwen3.8 is the robust
    # full-context fallback. On roomy multi-GPU hosts, use Nemotron as Haiku.
    if max_free >= 15800:
        if total_free >= 42000 and len(gpus) >= 2:
            return ["nemotron-3.5-lightning", "qwen3.6:35b", "qwen3.6:35b", "qwen3.6:35b"]
        return ["qwen3.6:35b"]
    if max_free >= 14500:
        return ["qwen3.8:27b"]
    raise ValueError("no supported full-context profile fits the currently free VRAM")


def gpu_subsets(gpus: list[GPU]) -> Iterable[tuple[GPU, ...]]:
    # Exhaustive subsets are fine for workstation-sized GPU counts and let a
    # large model greedily span several heterogeneous cards.
    import itertools
    for n in range(1, len(gpus) + 1):
        for combo in itertools.combinations(gpus, n):
            yield combo


def choose_group(gpus: list[GPU], required_mib: int, prefer_fast: bool) -> tuple[GPU, ...] | None:
    viable = [c for c in gpu_subsets(gpus) if sum(g.free_mib for g in c) >= required_mib]
    if not viable:
        return None
    def key(c: tuple[GPU, ...]):
        waste = sum(g.free_mib for g in c) - required_mib
        speed = sum(g.speed_score for g in c)
        # Fewer cards first (less PCIe traffic); for the daily driver, prefer
        # faster bandwidth before minimizing waste.
        return (len(c), -speed if prefer_fast else waste, waste if prefer_fast else -speed)
    return min(viable, key=key)


def plan(models: list[str], gpus: list[GPU], context: int) -> dict:
    rm = role_map(models)
    # Serve duplicate model selectors once and map multiple Claude roles to it.
    unique = []
    for role in ROLES:
        m = rm[role]
        if m not in unique:
            unique.append(m)

    available = list(gpus)
    servers = []
    # Allocate the Sonnet model first: it is the interactive daily driver.
    order = sorted(unique, key=lambda m: (0 if rm["sonnet"] == m else 1, -max(p.required_mib for p in PROFILES[m])))
    for model in order:
        chosen = None
        group = None
        choices = []
        for profile in PROFILES[model]:
            scaled_req = profile.required_mib
            if context < profile.native_context:
                # Only the context-dependent portion shrinks; keep this simple
                # and conservative by allowing at most 15% reduction.
                frac = max(0.0, min(1.0, context / profile.native_context))
                scaled_req = int(profile.required_mib * (0.85 + 0.15 * frac))
            candidate = choose_group(available, scaled_req, prefer_fast=(rm["sonnet"] == model))
            if candidate:
                choices.append((profile, candidate, scaled_req))
        if choices:
            # Interactive performance and concurrency win over a tiny quant
            # quality bump: first minimize GPU count, then maximize quant
            # quality, then prefer the faster group. Large models still span
            # cards automatically when no one-card profile exists.
            choices.sort(key=lambda x: (len(x[1]), -x[0].quality, -sum(g.speed_score for g in x[1]), sum(g.free_mib for g in x[1]) - x[2]))
            chosen, group, _ = choices[0]
        if chosen is None or group is None:
            raise ValueError(
                f"cannot place {model} at context {context:,} without sharing/overcommitting GPUs; "
                f"free VRAM is {[round(g.free_gib, 1) for g in available]} GiB"
            )
        # Put the slowest PCIe endpoint first for layer splitting, avoiding a
        # slow card between two faster layer partitions when possible.
        ordered_group = sorted(group, key=lambda g: (g.link_score, -g.speed_score))
        for g in group:
            available.remove(g)
        servers.append({
            "id": f"local-{model.replace(':', '-').replace('.', '').replace('_', '-')}",
            "model": model,
            "profile": {**asdict(chosen), "extra_env": dict(chosen.extra_env), "hf_spec": chosen.hf_spec},
            "gpus": [asdict(g) for g in ordered_group],
            "cuda_visible_devices": ",".join(str(g.index) for g in ordered_group),
            "multi_gpu": len(group) > 1,
            "allocated_mib": sum(g.free_mib for g in group),
        })

    role_ids = {}
    for role, model in rm.items():
        role_ids[role] = next(s["id"] for s in servers if s["model"] == model)
    return {
        "context": context,
        "roles": rm,
        "role_ids": role_ids,
        "servers": servers,
        "unused_gpus": [asdict(g) for g in available],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory")
    lp = sub.add_parser("plan")
    lp.add_argument("models", nargs="*")
    lp.add_argument("--context", type=int, default=262144)
    lp.add_argument("--smart", action="store_true")
    sub.add_parser("catalogue")
    args = ap.parse_args()
    try:
        if args.cmd == "inventory":
            print(json.dumps([asdict(x) for x in inventory()], indent=2))
            return 0
        if args.cmd == "catalogue":
            data = {m: [{**asdict(p), "extra_env": dict(p.extra_env), "hf_spec": p.hf_spec} for p in ps] for m, ps in PROFILES.items()}
            print(json.dumps(data, indent=2))
            return 0
        gpus = inventory()
        models = args.models
        if args.smart or not models:
            models = smart_defaults(gpus)
        print(json.dumps(plan(models, gpus, args.context), indent=2))
        return 0
    except ValueError as exc:
        print(f"claude-local planner: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
