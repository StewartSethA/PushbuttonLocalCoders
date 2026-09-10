#!/usr/bin/env python3
"""Joint GPU placement policy for claude-local.

The original catalogue/inventory implementation lives in
claude_local_plan_legacy.py. This module replaces only placement:

* one unique model -> fewest GPUs, then most free VRAM, then fastest PCIe link;
* multiple unique models -> plan the entire disjoint placement globally before
  any llama-server starts, so a greedy early choice cannot strand later models.

This remains deliberately conservative: models never share a GPU in one plan,
and CUDA unified-memory spill is still disabled by the launcher unless the user
explicitly opts in.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable

import claude_local_plan_legacy as legacy
from claude_local_plan_legacy import *  # re-export public API used by tests/CLI


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
    if context >= profile.native_context:
        return profile.required_mib
    # Only the context-dependent portion shrinks. Cap the reduction at 15%
    # because most of required_mib is immutable model weights.
    frac = max(0.0, min(1.0, context / profile.native_context))
    return int(profile.required_mib * (0.85 + 0.15 * frac))


def gpu_subsets(gpus: list[GPU]) -> Iterable[tuple[int, ...]]:
    positions = range(len(gpus))
    for n in range(1, len(gpus) + 1):
        yield from itertools.combinations(positions, n)


def best_profile_for_capacity(
    model: str, capacity_mib: int, context: int
) -> tuple[Profile, int] | None:
    # Catalogue order is quality-descending.
    for profile in PROFILES[model]:
        req = scaled_required_mib(profile, context)
        if req <= capacity_mib:
            return profile, req
    return None


def placement_candidates(model: str, gpus: list[GPU], context: int) -> list[Candidate]:
    raw: list[Candidate] = []
    for positions in gpu_subsets(gpus):
        group = [gpus[p] for p in positions]
        free_mib = sum(g.free_mib for g in group)
        picked = best_profile_for_capacity(model, free_mib, context)
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

    # A larger subset is pointless if a strict subset of the same cards already
    # fits an equal-or-better quant. Removing those candidates keeps the global
    # search small while preserving every useful placement.
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
    """Pick the freest viable remaining GPU; link speed is the tiebreaker."""
    candidates = placement_candidates(model, gpus, context)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (
            c.card_count,             # do not split unless necessary
            -c.free_mib,              # MOST FREE REMAINING FIRST
            -c.link_score,            # then highest negotiated PCIe link
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
        c.profile.quality,                    # maximize quant quality
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
    models: list[str], gpus: list[GPU], context: int, sonnet_model: str
) -> list[Candidate] | None:
    """Find the best complete disjoint model/GPU plan before launching.

    For normal workstations (<=16 GPUs), dynamic programming over a GPU bitmask
    evaluates the whole requested model set. Thus, for example, a 16 GB model
    will not consume the only 32 GB GPU if another requested model actually
    needs the 32 GB card.
    """
    if not models:
        return []

    candidates = {m: placement_candidates(m, gpus, context) for m in models}
    if any(not candidates[m] for m in models):
        return None

    if len(gpus) > 16:
        # Rare large clusters: largest minimum requirement first, while each
        # step still uses the freest/link-aware single-model policy.
        remaining = list(gpus)
        result: list[Candidate] = []
        for model in sorted(
            models,
            key=lambda m: min(c.required_mib for c in candidates[m]),
            reverse=True,
        ):
            local = single_model_choice(model, remaining, context)
            if local is None:
                return None
            chosen = [remaining[p] for p in local.gpu_positions]
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
        by_model = {c.model: c for c in result}
        return [by_model[m] for m in models]

    # Hardest models first only reduces branching; the score decides placement.
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
    by_model = {c.model: c for c in choices}
    return [by_model[m] for m in models]


def ordered_group_for_layer_split(candidate: Candidate, gpus: list[GPU]) -> list[GPU]:
    group = [gpus[p] for p in candidate.gpu_positions]
    if len(group) <= 1:
        return group
    # Keep the slowest endpoint at an edge of the layer pipeline rather than
    # between two faster GPUs.
    slowest = min(group, key=lambda g: (g.link_score, g.speed_score))
    rest = sorted(
        (g for g in group if g != slowest),
        key=lambda g: (-g.link_score, -g.speed_score),
    )
    return [slowest] + rest


def plan(models: list[str], gpus: list[GPU], context: int) -> dict:
    rm = role_map(models)
    unique: list[str] = []
    for role in ROLES:
        model = rm[role]
        if model not in unique:
            unique.append(model)

    if len(unique) == 1:
        picked = single_model_choice(unique[0], gpus, context)
        choices = [picked] if picked else None
        policy = "freest-then-link"
    else:
        choices = joint_model_choices(unique, gpus, context, rm["sonnet"])
        policy = "joint-global-plan"

    if not choices or any(c is None for c in choices):
        raise ValueError(
            f"cannot place requested model set at context {context:,} without "
            f"sharing/overcommitting GPUs; free VRAM is "
            f"{[round(g.free_gib, 1) for g in gpus]} GiB"
        )

    servers = []
    used_mask = 0
    for c in choices:
        assert c is not None
        used_mask |= c.mask
        group = ordered_group_for_layer_split(c, gpus)
        servers.append(
            {
                "id": f"local-{c.model.replace(':', '-').replace('.', '').replace('_', '-')}",
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
            }
        )

    role_ids = {
        role: next(s["id"] for s in servers if s["model"] == model)
        for role, model in rm.items()
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


# legacy.main() owns CLI argument parsing, inventory and catalogue output.
# Patch its global plan symbol so every CLI invocation uses this scheduler.
legacy.plan = plan

if __name__ == "__main__":
    raise SystemExit(legacy.main())
