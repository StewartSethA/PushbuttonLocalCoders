#!/usr/bin/env python3
"""Pushbutton runtime policy: context, concurrency, SLA and harmless admission.

Unknown configurations start at C1. Measured capacity may raise concurrency but
never beyond the actual launched instance/framework ceiling. A decode rate below
MIN_DECODE_TOK_S is always surfaced as slow.
"""
from __future__ import annotations
from dataclasses import dataclass

MIN_DECODE_TOK_S = 25.0
SLOW_SEVERE_TOK_S = 15.0
DEFAULT_CONTEXT = 65536
GENERIC_CONTEXT_CEILING = 4_194_304


@dataclass(frozen=True)
class BackendLimits:
    max_context: int
    max_concurrency: int
    dynamic_batching: bool
    queue_safe: bool = True
    notes: str = ""


# Generic framework entries are intentionally broad: the model and launched
# instance provide the real context/concurrency ceilings. Specialized recipes are
# narrower where Pushbutton controls a known configuration.
BACKEND_LIMITS: dict[str, BackendLimits] = {
    "llama.cpp": BackendLimits(GENERIC_CONTEXT_CEILING, 128, True, notes="parallel slots; launched --parallel and KV budget are the real ceiling"),
    "vllm": BackendLimits(GENERIC_CONTEXT_CEILING, 1024, True, notes="continuous batching; max-num-seqs/KV pool and model context are the real ceiling"),
    "sglang": BackendLimits(GENERIC_CONTEXT_CEILING, 1024, True, notes="continuous batching; launch settings and memory pool are the real ceiling"),
    "ollama": BackendLimits(GENERIC_CONTEXT_CEILING, 64, True, notes="runtime model/context settings remain authoritative"),
    "mlx": BackendLimits(GENERIC_CONTEXT_CEILING, 64, True, notes="Apple unified-memory/runtime settings remain authoritative"),
    "llamampere": BackendLimits(262144, 1, True, notes="3090 tuned recipe currently launches one long-context slot"),
    "volta-llama": BackendLimits(262144, 1, True, notes="current recipe is conservative C1"),
    "vllm-qwen38-3090": BackendLimits(262144, 8, True, notes="continuous batching; exact launched image/KV pool may reduce this"),
    "vllm-flashnext-3090": BackendLimits(262144, 8, True, notes="TP layout and KV pool constrain actual capacity"),
    "sglang-v100": BackendLimits(262144, 8, True, notes="four-GPU recipe; memory envelope may reduce concurrency"),
    "ninfer-3090": BackendLimits(262144, 8, True, notes="native max-concurrency is fixed at engine startup"),
    "resident": BackendLimits(65536, 1, False, notes="unknown resident endpoint: C1/64K until declared or proven"),
}


def backend_limits(backend: str, declared_context: int | None = None, declared_concurrency: int | None = None) -> BackendLimits:
    base = BACKEND_LIMITS.get(backend, BackendLimits(65536, 1, False, notes="unknown backend: conservative C1/64K"))
    ctx = min(base.max_context, int(declared_context)) if declared_context else base.max_context
    conc = min(base.max_concurrency, int(declared_concurrency)) if declared_concurrency else base.max_concurrency
    return BackendLimits(ctx, max(1, conc), base.dynamic_batching, base.queue_safe, base.notes)


def clamp_context(requested: int | None, *, model_context: int | None = None, backend: str = "resident", declared_context: int | None = None) -> tuple[int, str | None]:
    lim = backend_limits(backend, declared_context=declared_context).max_context
    if model_context:
        lim = min(lim, int(model_context))
    requested = int(requested or DEFAULT_CONTEXT)
    if requested <= lim:
        return requested, None
    return lim, f"requested context {requested:,} exceeds safe model/framework/instance limit {lim:,}; clamped"


def speed_label(tg: float | None, *, measured: bool) -> tuple[str, str | None]:
    if tg is None:
        return "UNKNOWN", "no decode SLA yet; conservative scheduling until normal use measures it"
    source = "measured" if measured else "estimated"
    if tg < SLOW_SEVERE_TOK_S:
        return "VERY_SLOW", f"{source} decode {tg:.1f} tok/s is far below the {MIN_DECODE_TOK_S:.0f} tok/s Pushbutton floor"
    if tg < MIN_DECODE_TOK_S:
        return "SLOW", f"{source} decode {tg:.1f} tok/s is below the {MIN_DECODE_TOK_S:.0f} tok/s Pushbutton floor"
    return "OK", None


def envelope_client_tg(e: dict) -> float | None:
    """Return a conservative per-client decode SLA from a measured envelope.

    Prefer explicit p10/client values. A median/client value is accepted only
    when no percentile was captured. Aggregate throughput is divided by
    concurrency as a last-resort approximation; it never counts as faster than
    the explicitly reported per-client number.
    """
    for k in ("tg_per_client_p10", "decode_per_client_p10", "tg_client_p10"):
        if e.get(k) is not None:
            try:return float(e[k])
            except Exception:return None
    for k in ("tg_per_client", "decode_per_client", "tg_client"):
        if e.get(k) is not None:
            try:return float(e[k])
            except Exception:return None
    agg=e.get("aggregate_tg") or e.get("aggregate_output_tok_s")
    if agg is not None:
        try:return float(agg)/max(1,int(e.get("concurrency") or 1))
        except Exception:return None
    return None


def proven_concurrency(*, backend: str, requested_context: int, framework_max: int | None = None,
                       measured_envelopes: list[dict] | None = None, conservative_default: int = 1,
                       min_client_tg: float = MIN_DECODE_TOK_S) -> int:
    """Largest proven-safe and acceptably fast concurrency at this context.

    A proof at a larger context is valid for a smaller request; a proof at a
    smaller context is not promoted upward. C>1 also requires a measured
    per-client decode SLA at or above `min_client_tg`; high aggregate throughput
    alone cannot hide slow individual subagents. Unknown configurations remain C1.
    """
    hard = backend_limits(backend, declared_concurrency=framework_max).max_concurrency
    best = max(1, min(conservative_default, hard))
    for e in measured_envelopes or []:
        if not e.get("safe", True):
            continue
        evidence=str(e.get("evidence") or "MEASURED").upper()
        if evidence not in {"MEASURED","PROVEN"}:
            continue
        try:
            c = int(e.get("concurrency") or 1)
            ctx = int(e.get("max_context") or e.get("context") or 0)
        except Exception:
            continue
        if ctx < requested_context:
            continue
        if c > 1:
            client_tg=envelope_client_tg(e)
            if client_tg is None or client_tg < float(min_client_tg):
                continue
        best = max(best, min(c, hard))
    return max(1, min(best, hard))


def admission_decision(*, active: int, capacity: int, queue_depth: int, queue_limit: int = 64) -> str:
    if active < max(1, capacity):
        return "ADMIT"
    if queue_depth < max(0, queue_limit):
        return "QUEUE"
    return "REJECT_BUSY"
