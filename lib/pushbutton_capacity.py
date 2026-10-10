"""Validated per-instance capacity settings and conservative planning budgets."""
from __future__ import annotations

import functools
import math
import re

OPTION_KEYS = {
    "slots", "context", "output", "client_context", "compact", "quant", "bits",
    "kv_k", "kv_v", "min_tps", "admission", "admission_limit", "safety",
}
KV_TYPES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625,
            "q4_0": 0.5625, "q4_1": 0.625, "q5_0": 0.6875, "q5_1": 0.75}


def quant_bits(quant):
    match = re.search(r"(?:^|[-_])(?:IQ|Q|MXFP|NVFP|FP|BF)([0-9]+)(?:$|[_-])",
                      str(quant).upper())
    return int(match[1]) if match else None


def matches_quant(quant, settings):
    return (not settings.get("quant") or quant.upper() == settings["quant"]) and (
        not settings.get("bits") or quant_bits(quant) == settings["bits"])


def model_bits_specs(models):
    """Apply a standalone bits=N fallback without overriding per-model bits."""
    bits = None
    specs = []
    for text in models:
        if text.startswith("bits="):
            if bits is not None:
                raise ValueError("duplicate bits specifier")
            bits = parse_options({"bits": text.split("=", 1)[1]})["bits"]
        else:
            specs.append(text)
    if bits is None:
        return specs
    return [text if any(field.strip().split("=", 1)[0].lower() == "bits"
                        for field in text.partition("@")[2].split(","))
            else text + ("," if "@" in text else "@") + f"bits={bits}"
            for text in specs]


def positive_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    text = str(value).strip()
    match = re.fullmatch(r"([0-9]+)([kKmM]?)", text)
    if not match:
        raise ValueError(f"{name} must be a positive integer")
    result = int(match[1]) * {"": 1, "k": 1024, "m": 1048576}[match[2].lower()]
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _compact(value):
    if isinstance(value, bool):
        raise ValueError("compact must be a token count or a fraction between 0 and 1")
    text = str(value).strip()
    if text.endswith("%"):
        try:
            result = float(text[:-1]) / 100
        except ValueError as exc:
            raise ValueError("invalid compact percentage") from exc
        if not 0 < result < 1:
            raise ValueError("compact percentage must be between 0 and 100")
        return result
    if "." in text:
        try:
            result = float(text)
        except ValueError as exc:
            raise ValueError("invalid compact fraction") from exc
        if not math.isfinite(result) or not 0 < result < 1:
            raise ValueError("compact fraction must be between 0 and 1")
        return result
    return positive_int(value, "compact")


def parse_options(options: dict) -> dict:
    """Normalize supplied settings; omitted values remain omitted for fallback."""
    if not isinstance(options, dict):
        raise ValueError("capacity settings must be an object")
    result = {}
    for key, value in options.items():
        if key not in OPTION_KEYS:
            raise ValueError(f"unknown capacity key '{key}'")
        name = "admission_limit" if key == "admission" else key
        if name in result:
            raise ValueError(f"duplicate capacity key '{name}'")
        if name in {"slots", "context", "output", "client_context", "admission_limit", "safety"}:
            result[name] = positive_int(value, name)
            if name == "slots" and result[name] > 128:
                raise ValueError("slots must be between 1 and 128")
        elif name == "bits":
            if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value).strip()) or int(value) < 1:
                raise ValueError("bits must be a positive integer")
            result[name] = int(value)
        elif name == "compact":
            result[name] = _compact(value)
        elif name == "min_tps":
            if isinstance(value, bool):
                raise ValueError("min_tps must be positive and finite")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("min_tps must be positive and finite") from exc
            if not math.isfinite(number) or number <= 0:
                raise ValueError("min_tps must be positive and finite")
            result[name] = number
        elif name in {"kv_k", "kv_v"}:
            kind = str(value).strip().lower()
            if kind not in KV_TYPES:
                raise ValueError(f"unsupported {name} precision '{value}'")
            result[name] = kind
        else:
            quant = value.strip() if isinstance(value, str) else ""
            if not quant or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", quant):
                raise ValueError("quant must be a supported profile quant name")
            result[name] = quant.upper()
    return result


def resolve_budget(context, output=None, client_context=None, compact=None, safety=None) -> dict:
    context = positive_int(context, "context")
    client = context if client_context is None else positive_int(client_context, "client_context")
    if client > context:
        raise ValueError("client_context cannot exceed server context")
    output = max(1, min(32768, context // 8)) if output is None else positive_int(output, "output")
    safety = max(1, min(1024, context // 32)) if safety is None else positive_int(safety, "safety")
    input_tokens = client - output - safety
    if input_tokens < 1:
        raise ValueError("output and safety must leave room for input within client_context")
    threshold = 0.8 if compact is None else _compact(compact)
    trigger = max(1, int(input_tokens * threshold)) if isinstance(threshold, float) else threshold
    if trigger >= input_tokens:
        raise ValueError("compact trigger must be strictly below the input budget")
    result = {"context": context, "output_tokens": output, "client_context": client,
              "compact_trigger": trigger, "input_tokens": input_tokens, "safety_tokens": safety}
    if not 0 < result["compact_trigger"] < result["input_tokens"] <= result["client_context"] <= result["context"]:
        raise ValueError("compaction trigger must be below the input, client, and hard context limits")
    return result


def resolve_options(options: dict, context: int, slots: int = 1) -> dict:
    settings = parse_options(options)
    budget = resolve_budget(settings.get("context", context), settings.get("output"),
                            settings.get("client_context"), settings.get("compact"),
                            settings.get("safety"))
    fallback_slots = positive_int(slots, "slots")
    if fallback_slots > 128:
        raise ValueError("slots must be between 1 and 128")
    count = settings.get("slots", fallback_slots)
    admission = settings.get("admission_limit", 1)
    if admission > count:
        raise ValueError("admission limit cannot exceed slots")
    return {**settings, **budget, "slots": count, "admission_limit": admission,
            "admission_explicit": "admission_limit" in settings,
            "min_tps": settings.get("min_tps")}


# llama.cpp CUDA context, compute/graph buffers and recurrent state that do not
# scale with KV context, plus a small allocator margin on catalog weight sizes.
GPU_RUNTIME_OVERHEAD_MIB = 1536
WEIGHT_MARGIN = 1.03


@functools.lru_cache(maxsize=None)
def catalog_weight_mib(model, quant):
    """Real GGUF weight size from configs/model-catalog.json, if catalogued."""
    import cpu_platform
    for q in cpu_platform._catalog().get(model, {}).get("quants", []) or []:
        if str(q.get("name", "")).upper() == str(quant).upper():
            return float(q["weight_gib"]) * 1024
    return None


def kv_bytes_per_token(model, kv_k, kv_v):
    """K+V bytes per context token from the model's attention architecture."""
    import cpu_platform
    arch = cpu_platform.MODEL_ARCH.get(model)
    if not arch:
        return None
    return arch["attn_layers"] * arch["kv_dim"] * (KV_TYPES[kv_k] + KV_TYPES[kv_v])


def memory_estimate(profile, capacity: dict) -> dict:
    """Weights + architecture-derived KV for every slot + runtime overhead.

    Weights use the catalogued GGUF size when known. Otherwise the profile's
    calibrated full-native-context envelope minus its own native KV share is
    kept as the weights+runtime base, so only the KV term is re-derived.
    """
    context = capacity["context"]
    if context > profile.native_context:
        raise ValueError(f"context {context} exceeds supported native context {profile.native_context}")
    k = capacity.get("kv_k", profile.kv_k)
    v = capacity.get("kv_v", profile.kv_v)
    tokens = context * capacity["slots"]
    per_token = kv_bytes_per_token(profile.model, k, v)
    if per_token is None:
        ratio = max(1.0, KV_TYPES[k] / KV_TYPES[profile.kv_k], KV_TYPES[v] / KV_TYPES[profile.kv_v])
        scale = tokens / profile.native_context * ratio
        required = math.ceil(profile.required_mib * (0.85 + 0.15 * scale))
        return {"status": "ESTIMATED", "method": "15% context-dependent envelope heuristic",
                "architecture_exact": False, "required_mib": required,
                "context_tokens_total": tokens, "kv_precision_multiplier": ratio,
                "note": "No architecture data; validate actual allocation at startup."}
    kv_mib = tokens * per_token / 2**20
    weights = catalog_weight_mib(profile.model, profile.quant)
    if weights is not None:
        weights_mib, source = weights, "catalog"
        overhead_mib = GPU_RUNTIME_OVERHEAD_MIB + weights * (WEIGHT_MARGIN - 1)
        base_mib = weights_mib + overhead_mib
    else:
        native_kv = profile.native_context * kv_bytes_per_token(profile.model, profile.kv_k, profile.kv_v) / 2**20
        base_mib = max(profile.required_mib - native_kv, GPU_RUNTIME_OVERHEAD_MIB)
        overhead_mib = GPU_RUNTIME_OVERHEAD_MIB
        weights_mib, source = base_mib - overhead_mib, "profile-envelope"
    required = math.ceil(base_mib + kv_mib)
    return {"status": "ESTIMATED", "method": "weights + architecture KV x context x slots + runtime overhead",
            "architecture_exact": False, "required_mib": required,
            "weights_mib": math.ceil(weights_mib), "weights_source": source,
            "kv_mib": math.ceil(kv_mib), "kv_bytes_per_token": per_token,
            "overhead_mib": math.ceil(overhead_mib), "context_tokens_total": tokens,
            "kv_k": k, "kv_v": v,
            "kv_precision_multiplier": round(per_token / kv_bytes_per_token(profile.model, profile.kv_k, profile.kv_v), 4),
            "note": "Planning architecture assumptions; validate actual allocation at startup."}
