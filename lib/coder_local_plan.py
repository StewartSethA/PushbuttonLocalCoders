#!/usr/bin/env python3
"""Replica-aware worker planner for local coding frontends.

Model selectors may carry placement constraints:
  qwen3.8:27b@gpu=0,vram=16G
  qwen3.8-flash-next@gpu=0+1+2+3,vram=30G

`vram` is a per-selected-GPU planning lease.  The planner behaves as if each
selected GPU had at most that much free VRAM, so quant selection automatically
falls to the best profile that fits the lease.  `gpu=` is exact: when present,
the worker must use exactly those physical GPU indices.
"""
from __future__ import annotations
import argparse, json, os, re, sys
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import claude_local_plan as base

DEFAULT_PLACEMENT_CONFIG = os.path.expanduser(
    os.environ.get("PUSHBUTTON_PLACEMENT_CONFIG", "~/.config/pushbutton-local/placement.json")
)


@dataclass(frozen=True)
class WorkerRequest:
    model: str
    gpu_indices: tuple[int, ...] | None = None
    vram_limit_mib: int | None = None
    source: str = "cli"


@dataclass(frozen=True)
class WorkerCandidate:
    worker: int
    model: str
    candidate: base.Candidate


def parse_memory_mib(value: str | int | float | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError("VRAM limit must be positive")
        # Numeric config values are MiB to avoid an implicit unit surprise.
        return int(value)
    text = str(value).strip().lower().replace(" ", "")
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(gib|gb|g|mib|mb|m)?", text)
    if not m:
        raise ValueError(f"invalid VRAM size '{value}' (examples: 16G, 15360MiB)")
    number = float(m.group(1)); unit = m.group(2) or "mib"
    if number <= 0:
        raise ValueError("VRAM limit must be positive")
    if unit in {"g", "gb", "gib"}:
        return int(number * 1024)
    return int(number)


def parse_gpu_indices(value) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        vals = value
    elif isinstance(value, tuple):
        vals = list(value)
    else:
        vals = [x for x in re.split(r"[+,;]", str(value)) if x != ""]
    try:
        out = tuple(int(x) for x in vals)
    except Exception as exc:
        raise ValueError(f"invalid GPU list '{value}'") from exc
    if not out or any(x < 0 for x in out):
        raise ValueError(f"invalid GPU list '{value}'")
    if len(set(out)) != len(out):
        raise ValueError(f"GPU list contains duplicates: {out}")
    return out


def load_placement_config(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        return {}
    try:
        obj = json.load(open(p))
    except Exception as exc:
        raise ValueError(f"cannot read placement config {p}: {exc}") from exc
    raw = obj.get("models", obj)
    if not isinstance(raw, dict):
        raise ValueError("placement config must be an object or contain a 'models' object")
    out: dict[str, dict] = {}
    for name, spec in raw.items():
        model = base.canonical_model(name)
        if not isinstance(spec, dict):
            raise ValueError(f"placement config for {name} must be an object")
        out[model] = spec
    return out


def parse_model_spec(text: str, defaults: dict[str, dict] | None = None) -> WorkerRequest:
    defaults = defaults or {}
    model_text, sep, suffix = text.partition("@")
    model = base.canonical_model(model_text)
    d = defaults.get(model, {})
    gpu_indices = parse_gpu_indices(d.get("gpus", d.get("gpu")))
    vram_limit = parse_memory_mib(d.get("vram_limit", d.get("vram")))
    source = "config" if d else "cli"
    if sep:
        source = "cli"
        for field in suffix.split(","):
            if not field.strip():
                continue
            if "=" not in field:
                raise ValueError(
                    f"invalid model placement '{field}' in '{text}'; use gpu=0+1 or vram=16G"
                )
            key, value = (x.strip() for x in field.split("=", 1))
            key = key.lower()
            if key in {"gpu", "gpus"}:
                gpu_indices = parse_gpu_indices(value)
            elif key in {"vram", "vram_limit", "memory", "mem"}:
                vram_limit = parse_memory_mib(value)
            else:
                raise ValueError(f"unknown placement key '{key}' in '{text}'")
    return WorkerRequest(model, gpu_indices, vram_limit, source)


def expand_workers(requests: list[WorkerRequest], agents: int | None) -> list[WorkerRequest]:
    if not requests:
        raise ValueError("at least one model is required")
    n = agents if agents is not None else len(requests)
    if n < 1 or n > 8:
        raise ValueError("--agents must be between 1 and 8")
    if len(requests) > n:
        raise ValueError("more models were supplied than --agents")
    if len(requests) == 1:
        return requests * n
    if len(requests) < n:
        requests = requests + [requests[-1]] * (n - len(requests))
    return requests


def score(c: base.Candidate) -> tuple[int, ...]:
    return (
        -c.card_count,
        c.profile.quality,
        c.headroom_mib,
        c.free_mib,
        c.link_score,
        c.speed_score,
    )


def _near_best(candidates: list[base.Candidate], quality_slack: int = 3) -> list[base.Candidate]:
    if not candidates:
        return candidates
    best = max(c.profile.quality for c in candidates)
    kept = [c for c in candidates if c.profile.quality >= best - quality_slack]
    return kept or candidates


def candidates_for_request(req: WorkerRequest, gpus: list[base.GPU], context: int) -> list[base.Candidate]:
    physical = {g.index for g in gpus}
    if req.gpu_indices is not None:
        missing = [x for x in req.gpu_indices if x not in physical]
        if missing:
            raise ValueError(f"{req.model} requests GPU(s) {missing}, which are not visible")

    # Preserve list positions/masks but cap free VRAM for this worker.  The cap
    # is per GPU, which makes `vram=16G` mean exactly that on a single 24 GB 3090.
    effective = [
        replace(g, free_mib=min(g.free_mib, req.vram_limit_mib))
        if req.vram_limit_mib is not None else g
        for g in gpus
    ]
    candidates = base.placement_candidates(req.model, effective, context)
    if req.gpu_indices is not None:
        exact_positions = tuple(i for i, g in enumerate(gpus) if g.index in set(req.gpu_indices))
        exact_set = set(exact_positions)
        candidates = [c for c in candidates if set(c.gpu_positions) == exact_set]
    return _near_best(candidates)


def choose_workers(requests: list[WorkerRequest], gpus: list[base.GPU], context: int) -> list[base.Candidate]:
    candidates = [candidates_for_request(r, gpus, context) for r in requests]
    if any(not x for x in candidates):
        details = []
        for r, cs in zip(requests, candidates):
            if not cs:
                pin = f" GPUs={list(r.gpu_indices)}" if r.gpu_indices is not None else ""
                cap = f" cap={r.vram_limit_mib/1024:.1f}GiB/GPU" if r.vram_limit_mib else ""
                details.append(f"{r.model}{pin}{cap}")
        raise ValueError("no fitting quant/placement for: " + "; ".join(details))

    @lru_cache(maxsize=None)
    def search(i: int, used: int):
        if i == len(requests):
            return (0, 0, 0, 0, 0, 0), ()
        best = None
        for c in candidates[i]:
            if c.mask & used:
                continue
            tail = search(i + 1, used | c.mask)
            if tail is None:
                continue
            s0 = score(c)
            s = tuple(a + b for a, b in zip(s0, tail[0]))
            proposal = (s, (c,) + tail[1])
            if best is None or s > best[0]:
                best = proposal
        return best

    found = search(0, 0)
    if found is None:
        raise ValueError(
            f"cannot place {len(requests)} independent workers at context {context:,}; "
            "pinned workers may overlap, or the requested VRAM leases may be too small"
        )
    return list(found[1])


def build_plan(requests: list[WorkerRequest], agents: int | None, gpus: list[base.GPU], context: int) -> dict:
    workers = expand_workers(requests, agents)
    choices = choose_workers(workers, gpus, context)
    out = []
    used = 0
    for i, (req, c) in enumerate(zip(workers, choices), 1):
        used |= c.mask
        group = base.ordered_group_for_layer_split(c, gpus)
        alias = f"local-coder-{i}-{req.model.replace(':','-').replace('.','').replace('_','-')}"
        out.append({
            "worker": i,
            "id": alias,
            "model": req.model,
            "profile": {**asdict(c.profile), "extra_env": dict(c.profile.extra_env), "hf_spec": c.profile.hf_spec},
            "gpus": [asdict(g) for g in group],
            "cuda_visible_devices": ",".join(str(g.index) for g in group),
            "multi_gpu": len(group) > 1,
            "required_mib": c.required_mib,
            "allocated_mib": c.free_mib,
            "headroom_mib": c.headroom_mib,
            "requested_gpus": list(req.gpu_indices) if req.gpu_indices is not None else None,
            "vram_limit_mib_per_gpu": req.vram_limit_mib,
            "placement_source": req.source,
        })
    return {
        "context": context,
        "agents": len(workers),
        "workers": out,
        "unused_gpus": [asdict(g) for p, g in enumerate(gpus) if not (used & (1 << p))],
    }


def synthetic_v100(n: int) -> list[base.GPU]:
    return [base.GPU(i, "Tesla V100-SXM2-32GB", 32768, 32768, "7.0", f"0000:{i:02x}:00.0", 3, 16, 1, 16) for i in range(n)]


def synthetic_3090(n: int) -> list[base.GPU]:
    return [base.GPU(i, "NVIDIA GeForce RTX 3090", 24576, 24576, "8.6", f"0000:{i:02x}:00.0", 4, 16, 1, 16) for i in range(n)]


def self_test() -> None:
    req = [WorkerRequest("qwen3.8-flash-next")]
    p = build_plan(req, 2, synthetic_v100(8), 262144)
    assert [len(w["gpus"]) for w in p["workers"]] == [4, 4], p
    assert all(w["profile"]["quant"] == "UD-IQ4_XS" for w in p["workers"]), p

    req = [WorkerRequest(x) for x in ["qwen3.8-flash-next", "qwen3.8:27b", "qwen3.6:35b", "nemotron-3.5-lightning"]]
    p = build_plan(req, 4, synthetic_v100(8), 262144)
    assert [len(w["gpus"]) for w in p["workers"]] == [4, 1, 1, 1], p
    assert p["workers"][0]["profile"]["quant"] == "UD-IQ4_XS", p
    assert len(p["unused_gpus"]) == 1, p

    # Requested dual-3090 use case: leave 8 GiB free on each card.  The cap
    # forces automatic quant selection independently for the two models.
    defaults = {}
    r1 = parse_model_spec("qwen3.8:27b@gpu=0,vram=16G", defaults)
    r2 = parse_model_spec("qwen3.6:35b@gpu=1,vram=16G", defaults)
    p = build_plan([r1, r2], 2, synthetic_3090(2), 262144)
    assert p["workers"][0]["cuda_visible_devices"] == "0", p
    assert p["workers"][1]["cuda_visible_devices"] == "1", p
    assert p["workers"][0]["profile"]["quant"] == "IQ3_XXS", p
    assert p["workers"][1]["profile"]["quant"] == "UD-IQ3_XXS", p
    assert p["workers"][0]["required_mib"] <= 16384, p
    assert p["workers"][1]["required_mib"] <= 16384, p
    assert all(w["vram_limit_mib_per_gpu"] == 16384 for w in p["workers"]), p

    # A cap below every known full-context profile must fail cleanly.
    try:
        build_plan([parse_model_spec("qwen3.8:27b@gpu=0,vram=12G")], 1, synthetic_3090(1), 262144)
    except ValueError as exc:
        assert "no fitting quant" in str(exc), exc
    else:
        raise AssertionError("12G cap unexpectedly fit Qwen3.8-27B full-context profile")
    print("coder-local planner self-test: PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--agents", type=int)
    ap.add_argument("--context", type=int, default=262144)
    ap.add_argument("--placement-config", default=DEFAULT_PLACEMENT_CONFIG)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    try:
        if args.self_test:
            self_test(); return 0
        defaults = load_placement_config(args.placement_config)
        requests = [parse_model_spec(x, defaults) for x in args.models]
        gpus = base.inventory()
        print(json.dumps(build_plan(requests, args.agents, gpus, args.context), indent=2))
        return 0
    except ValueError as exc:
        print(f"coder-local planner: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
