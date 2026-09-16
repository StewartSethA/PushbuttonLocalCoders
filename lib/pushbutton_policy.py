#!/usr/bin/env python3
"""Pushbutton runtime policy: context, concurrency, SLA and harmless admission.

The policy is deliberately conservative. Unknown backends start at C1; measured
capacity may raise concurrency, but never beyond the serving framework's known
hard limit.  A decode rate below MIN_DECODE_TOK_S is always surfaced as slow.
"""
from __future__ import annotations

from dataclasses import dataclass

MIN_DECODE_TOK_S = 25.0
SLOW_SEVERE_TOK_S = 15.0
DEFAULT_CONTEXT = 65536


@dataclass(frozen=True)
class BackendLimits:
    max_context: int
    max_concurrency: int
    dynamic_batching: bool
    queue_safe: bool = True
    notes: str = ""


# These are Pushbutton launch-policy ceilings, not claims that every artifact can
# actually reach them. The artifact/model memory envelope may reduce either one.
BACKEND_LIMITS: dict[str, BackendLimits] = {
    "llama.cpp": BackendLimits(262144, 8, True, notes="parallel slots; exact safe count depends on KV/state budget"),
    "llamampere": BackendLimits(262144, 1, True, notes="3090 tuned recipe defaults to a single long-context slot"),
    "volta-llama": BackendLimits(262144, 1, True, notes="conservative until measured on the selected Volta artifact"),
    "vllm-qwen38-3090": BackendLimits(262144, 8, True, notes="continuous batching; memory envelope may force C1"),
    "vllm-flashnext-3090": BackendLimits(262144, 8, True, notes="continuous batching; TP layout and KV pool constrain capacity"),
    "sglang-v100": BackendLimits(262144, 8, True, notes="continuous batching; four-GPU recipe may still be context-bound"),
    "ninfer-3090": BackendLimits(262144, 8, True, notes="native max-concurrency is fixed at engine startup"),
    "resident": BackendLimits(262144, 1, False, notes="unknown resident endpoint: C1 until proven"),
}


def backend_limits(backend: str, declared_context: int | None = None, declared_concurrency: int | None = None) -> BackendLimits:
    base = BACKEND_LIMITS.get(backend, BackendLimits(262144, 1, False, notes="unknown backend: conservative C1"))
    ctx = min(base.max_context, declared_context) if declared_context else base.max_context
    conc = min(base.max_concurrency, declared_concurrency) if declared_concurrency else base.max_concurrency
    return BackendLimits(ctx, max(1, conc), base.dynamic_batching, base.queue_safe, base.notes)


def clamp_context(requested: int | None, *, model_context: int | None = None, backend: str = "resident", declared_context: int | None = None) -> tuple[int, str | None]:
    lim = backend_limits(backend, declared_context=declared_context).max_context
    if model_context:
        lim = min(lim, int(model_context))
    requested = int(requested or DEFAULT_CONTEXT)
    if requested <= lim:
        return requested, None
    return lim, f"requested context {requested:,} exceeds safe limit {lim:,}; clamped"


def speed_label(tg: float | None, *, measured: bool) -> tuple[str, str | None]:
    if tg is None:
        return "UNKNOWN", "no decode SLA yet; conservative scheduling until normal use measures it"
    source = "measured" if measured else "estimated"
    if tg < SLOW_SEVERE_TOK_S:
        return "VERY_SLOW", f"{source} decode {tg:.1f} tok/s is far below the {MIN_DECODE_TOK_S:.0f} tok/s Pushbutton floor"
    if tg < MIN_DECODE_TOK_S:
        return "SLOW", f"{source} decode {tg:.1f} tok/s is below the {MIN_DECODE_TOK_S:.0f} tok/s Pushbutton floor"
    return "OK", None


def proven_concurrency(*, backend: str, requested_context: int, framework_max: int | None = None,
                       measured_envelopes: list[dict] | None = None, conservative_default: int = 1) -> int:
    """Largest concurrency proven safe at this context, capped by framework max.

    Each envelope may contain concurrency, max_context and safe=True.  A proof at
    a larger context is valid for a smaller request; a proof at a smaller context
    is not promoted upward. Unknown configurations remain C1.
    """
    hard = backend_limits(backend, declared_concurrency=framework_max).max_concurrency
    best = max(1, min(conservative_default, hard))
    for e in measured_envelopes or []:
        if not e.get("safe", True):
            continue
        try:
            c = int(e.get("concurrency") or 1)
            ctx = int(e.get("max_context") or e.get("context") or 0)
        except Exception:
            continue
        if ctx >= requested_context:
            best = max(best, min(c, hard))
    return max(1, min(best, hard))


def admission_decision(*, active: int, capacity: int, queue_depth: int, queue_limit: int = 64) -> str:
    if active < max(1, capacity):
        return "ADMIT"
    if queue_depth < max(0, queue_limit):
        return "QUEUE"
    return "REJECT_BUSY"
