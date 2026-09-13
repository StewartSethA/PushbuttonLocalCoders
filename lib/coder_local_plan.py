#!/usr/bin/env python3
"""Replica-aware worker planner for local coding frontends."""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import asdict, dataclass
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import claude_local_plan as base

@dataclass(frozen=True)
class WorkerCandidate:
    worker: int
    model: str
    candidate: base.Candidate


def expand_workers(models: list[str], agents: int | None) -> list[str]:
    if not models:
        raise ValueError("at least one model is required")
    canon = [base.canonical_model(m) for m in models]
    n = agents if agents is not None else len(canon)
    if n < 1 or n > 4:
        raise ValueError("--agents must be between 1 and 4")
    if len(canon) > n:
        raise ValueError("more models were supplied than --agents")
    if len(canon) == 1:
        return canon * n
    if len(canon) < n:
        canon.extend([canon[-1]] * (n - len(canon)))
    return canon


def score(c: base.Candidate) -> tuple[int, ...]:
    # After candidates are trimmed to a near-best quality band, preserve GPUs
    # first. This prevents a tiny quant-quality bump from consuming an extra
    # whole accelerator that could host another worker or remain as headroom.
    return (
        -c.card_count,
        c.profile.quality,
        c.headroom_mib,
        c.free_mib,
        c.link_score,
        c.speed_score,
    )


def _near_best(candidates: list[base.Candidate], quality_slack: int = 3) -> list[base.Candidate]:
    """Keep candidates within a small quality band of the best feasible tier.

    A three-point band intentionally treats Flash-Next IQ4_XS (97) as a peer of
    Q4_K_XL (100), allowing the planner to prefer 4 GPUs over 5 on 32 GB V100s,
    while still rejecting materially lower-bit 3-GPU fallbacks unless required.
    """
    if not candidates:
        return candidates
    best = max(c.profile.quality for c in candidates)
    kept = [c for c in candidates if c.profile.quality >= best - quality_slack]
    return kept or candidates


def choose_workers(models: list[str], gpus: list[base.GPU], context: int) -> list[base.Candidate]:
    candidates = [_near_best(base.placement_candidates(m, gpus, context)) for m in models]
    if any(not x for x in candidates):
        raise ValueError("at least one requested worker model cannot fit current free VRAM")

    @lru_cache(maxsize=None)
    def search(i: int, used: int):
        if i == len(models):
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
            f"cannot place {len(models)} independent workers at context {context:,}; "
            f"free VRAM is {[round(g.free_gib, 1) for g in gpus]} GiB"
        )
    return list(found[1])


def build_plan(models: list[str], agents: int | None, gpus: list[base.GPU], context: int) -> dict:
    workers = expand_workers(models, agents)
    choices = choose_workers(workers, gpus, context)
    out = []
    used = 0
    for i, (model, c) in enumerate(zip(workers, choices), 1):
        used |= c.mask
        group = base.ordered_group_for_layer_split(c, gpus)
        alias = f"local-coder-{i}-{model.replace(':','-').replace('.','').replace('_','-')}"
        out.append({
            "worker": i,
            "id": alias,
            "model": model,
            "profile": {**asdict(c.profile), "extra_env": dict(c.profile.extra_env), "hf_spec": c.profile.hf_spec},
            "gpus": [asdict(g) for g in group],
            "cuda_visible_devices": ",".join(str(g.index) for g in group),
            "multi_gpu": len(group) > 1,
            "required_mib": c.required_mib,
            "allocated_mib": c.free_mib,
            "headroom_mib": c.headroom_mib,
        })
    return {
        "context": context,
        "agents": len(workers),
        "workers": out,
        "unused_gpus": [asdict(g) for p, g in enumerate(gpus) if not (used & (1 << p))],
    }


def synthetic_v100(n: int) -> list[base.GPU]:
    return [base.GPU(i, "Tesla V100-SXM2-32GB", 32768, 32768, "7.0", f"0000:{i:02x}:00.0", 3, 16, 1, 16) for i in range(n)]


def self_test() -> None:
    p = build_plan(["qwen3.8-flash-next"], 2, synthetic_v100(8), 262144)
    assert [len(w["gpus"]) for w in p["workers"]] == [4, 4], p
    assert all(w["profile"]["quant"] == "UD-IQ4_XS" for w in p["workers"]), p
    assert len(p["unused_gpus"]) == 0, p

    p = build_plan(
        ["qwen3.8-flash-next", "qwen3.8:27b", "qwen3.6:35b", "nemotron-3.5-lightning"],
        4, synthetic_v100(8), 262144,
    )
    assert [len(w["gpus"]) for w in p["workers"]] == [4, 1, 1, 1], p
    assert p["workers"][0]["profile"]["quant"] == "UD-IQ4_XS", p
    assert len(p["unused_gpus"]) == 1, p
    assert len({g["index"] for w in p["workers"] for g in w["gpus"]}) == 7, p
    print("coder-local planner self-test: PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--agents", type=int)
    ap.add_argument("--context", type=int, default=262144)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    try:
        if args.self_test:
            self_test(); return 0
        gpus = base.inventory()
        print(json.dumps(build_plan(args.models, args.agents, gpus, args.context), indent=2))
        return 0
    except ValueError as exc:
        print(f"coder-local planner: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
