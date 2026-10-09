"""Host-aware startup planning. Fallback capacities require calibrated metadata.

The older planner API remains available to the other frontends; this module is
the strict, joint RAM/VRAM startup path used by claude-local.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Host:
    available_mib: int
    cgroup_remaining_mib: int | None
    usable_mib: int
    cpu_ids: tuple[int, ...]
    cpu_capacity: float
    numa_nodes: tuple[str, ...]
    reserve_mib: int = 1024


def host_inventory(proc=Path("/proc"), sys=Path("/sys")) -> Host:
    mem = dict(line.split(":", 1) for line in (proc / "meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) // 1024
    cpus = tuple(sorted(os.sched_getaffinity(0)))
    capacity = float(len(cpus))
    remaining = []
    mounts = []
    mountinfo = proc / "self/mountinfo"
    if mountinfo.is_file():
        for line in mountinfo.read_text().splitlines():
            before, after = line.split(" - ", 1)
            fields, fs = before.split(), after.split()
            if fs[0] in ("cgroup", "cgroup2"):
                unescape = lambda s: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), s)
                mounts.append((fs[0], unescape(fields[3]), Path(unescape(fields[4])), fs[2].split(",")))
    # Account for all visible ancestor limits, not just the leaf container.
    for line in (proc / "self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if not controllers:
            root = sys / "fs/cgroup"
            memory, usage, quota = "memory.max", "memory.current", "cpu.max"
        elif "memory" in controllers.split(","):
            root = sys / "fs/cgroup/memory"
            memory, usage = "memory.limit_in_bytes", "memory.usage_in_bytes"
            quota = "cpu.cfs_quota_us" if "cpu" in controllers.split(",") else ""
        elif "cpu" in controllers.split(","):
            root = sys / "fs/cgroup/cpu"
            memory, usage, quota = "", "", "cpu.cfs_quota_us"
        else:
            continue
        mount_root = "/"
        for fs, mounted_root, mounted_path, options in mounts:
            if (not controllers and fs == "cgroup2" or
                    controllers and fs == "cgroup" and
                    ("memory" if memory else "cpu") in options):
                root, mount_root = mounted_path, mounted_root
                break
        if controllers and "cpu" in controllers.split(",") and not mounts and not root.exists():
            root = sys / "fs/cgroup/cpu,cpuacct"
        rel = Path(relative)
        try:
            relative_to_mount = rel.relative_to(mount_root)
        except ValueError:
            # Ancestors hidden by a cgroup namespace cannot be inspected.
            relative_to_mount = Path(".")
        leaf = root / relative_to_mount
        # A cgroup namespace may mount the current subtree as its root.
        if not leaf.is_dir():
            leaf = root
        for directory in (leaf, *leaf.parents):
            if directory != root and root not in directory.parents:
                break
            if memory and (directory / memory).is_file():
                limit = (directory / memory).read_text().strip()
                if limit != "max":
                    current = int((directory / usage).read_text().strip())
                    remaining.append(max(0, int(limit) - current) // 1048576)
            if quota and (directory / quota).is_file():
                values = (directory / quota).read_text().split()
                if quota == "cpu.max":
                    q, period = values
                else:
                    q = values[0]
                    period = (directory / "cpu.cfs_period_us").read_text().strip()
                if q not in ("max", "-1"):
                    capacity = min(capacity, int(q) / int(period))
    if not cpus or capacity <= 0:
        raise ValueError("no usable CPU capacity")
    cgroup = min(remaining) if remaining else None
    usable = min(available, cgroup) if cgroup is not None else available
    nodes = tuple(
        f"{p.parent.name}: {p.read_text().strip()}"
        for p in sorted((sys / "devices/system/node").glob("node*/cpulist"))
    )
    return Host(available, cgroup, usable, cpus, capacity, nodes)


MEMORY_FIELDS = (
    "weights_host_mib", "weights_gpu_mib", "cache_host_mib_per_token",
    "cache_gpu_mib_per_token", "buffer_host_mib", "buffer_gpu_mib",
)


def validate_metadata(data):
    """Explicit measured placements avoid uniform-layer and KV-ratio guesses."""
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("memory metadata must have version 1")
    entries = data.get("placements")
    if not isinstance(entries, list) or not entries:
        raise ValueError("memory metadata needs a nonempty placements array")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("memory placement must be an object")
        for key in ("hf_spec", "source"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"memory placement requires {key}")
        for key in MEMORY_FIELDS:
            value = entry.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid calibrated {key}")
        for key in ("ngl", "layer_count", "max_context", "max_slots", "batch", "ubatch", "max_threads"):
            value = entry.get(key)
            if type(value) is not int or value < (0 if key == "ngl" else 1):
                raise ValueError(f"invalid calibrated {key}")
        if entry.get("mode") not in ("gpu", "cpu"):
            raise ValueError("placement mode must be gpu or cpu; hybrid weights are unsupported")
        mode = entry["mode"]
        if (mode == "cpu") != (entry["ngl"] == 0):
            raise ValueError("CPU placement requires ngl=0; GPU requires ngl>0")
        if mode == "gpu" and entry["ngl"] <= entry["layer_count"]:
            raise ValueError("GPU placement must offload all layers, including output")
        if entry.get("kv_k") not in ("f16", "q4_0", "q8_0") or entry.get("kv_v") not in ("f16", "q4_0", "q8_0"):
            raise ValueError("unsupported calibrated cache type")
        if mode != "gpu" and (entry["kv_k"], entry["kv_v"]) != ("f16", "f16"):
            raise ValueError("CPU cache optimization is deferred; fallback requires f16 caches")
        if entry.get("flash_attn") not in ("on", "off"):
            raise ValueError("calibration must specify flash_attn on/off")
        if mode == "cpu" and any(entry[k] for k in ("weights_gpu_mib", "cache_gpu_mib_per_token", "buffer_gpu_mib")):
            raise ValueError("CPU placement cannot reserve GPU memory")
        if entry["weights_host_mib"] + entry["weights_gpu_mib"] <= 0:
            raise ValueError("calibration must include model weights")
        if mode != "cpu" and entry["weights_gpu_mib"] <= 0:
            raise ValueError("GPU placement requires GPU weights")
        if mode == "gpu" and entry["cache_host_mib_per_token"]:
            raise ValueError("cache-only CPU placement is deferred")
    return entries


def placement_targets(config, roles, gpus, classifier_model, classifier_gpu):
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("role placement must have version 1")
    routes = config.get("roles")
    if not isinstance(routes, dict) or set(routes) != set(roles):
        raise ValueError("role placement must specify haiku, sonnet, opus and fable")
    targets, occupied = {}, {}
    requests = [(roles[role], False, routes[role]) for role in roles]
    if classifier_model:
        if "classifier" not in config:
            raise ValueError("role placement requires a dedicated classifier entry")
        requests.append((classifier_model, True, config["classifier"]))
    elif "classifier" in config:
        raise ValueError("classifier placement requires a classifier model")
    visible = {g.index for g in gpus}
    for model, classifier, route in requests:
        if not isinstance(route, dict) or route.get("backend") != "llama.cpp":
            raise ValueError("claude-local role placement supports only llama.cpp Anthropic serving; "
                             "OpenAI-only specialized engines must use pushbutton-bench")
        indices = route.get("gpus")
        if (not isinstance(indices, list) or
                any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices)):
            raise ValueError("placement gpus must be a list of distinct physical indices (empty for CPU)")
        target = frozenset(indices)
        if not target <= visible:
            raise ValueError("role placement GPU is unavailable")
        if classifier and (classifier_gpu is None or target != {classifier_gpu}):
            raise ValueError("classifier placement must match its dedicated physical GPU")
        key = (model, classifier)
        if key in targets and targets[key] != target:
            raise ValueError("roles sharing a model must use the same placement")
        if model == "qwen3.8-flash-next" and not classifier and len(target) != 4:
            raise ValueError("dedicated Flash-Next placement requires exactly four GPUs; two-card recipes are unverified")
        for index in target:
            if index in occupied and occupied[index] != key:
                raise ValueError("role/classifier GPU reservations must be disjoint")
            occupied[index] = key
        targets[key] = target
    return targets


def startup_plan(base, models, gpus, context, *, slots=2, startup_policy="gpu-only",
                 metadata=None, host=None, min_quality=0, classifier_model=None,
                 classifier_gpu=None, classifier_context=32768, max_layouts=3,
                 role_placement=None):
    if startup_policy not in ("gpu-only", "allow-cpu-only"):
        raise ValueError("unknown startup policy")
    if slots < 1 or min(context, classifier_context) < 1 or not 0 <= min_quality <= 100:
        raise ValueError("invalid context, slots or minimum quality")
    if not 1 <= max_layouts <= 8:
        raise ValueError("max layouts must be between 1 and 8")
    if classifier_gpu is not None and classifier_model is None:
        raise ValueError("classifier GPU requires a classifier model")
    if classifier_model and classifier_gpu is None and startup_policy == "gpu-only":
        raise ValueError("gpu-only classifier requires a physical GPU reservation")
    if classifier_gpu is not None and not any(g.index == classifier_gpu for g in gpus):
        raise ValueError("classifier GPU is unavailable")
    host = host or host_inventory()
    entries = validate_metadata(metadata) if metadata is not None else []
    roles = base.role_map(models)
    classifier_model = base.canonical_model(classifier_model) if classifier_model else None
    targets = (placement_targets(role_placement, roles, gpus, classifier_model, classifier_gpu)
               if role_placement is not None else None)
    unique = list(dict.fromkeys(roles.values()))
    requests = [(m, context, False) for m in unique]
    if classifier_model:
        requests.append((base.canonical_model(classifier_model), classifier_context, True))
    if len(requests) * slots > 64:
        raise ValueError("bounded startup supports at most 64 aggregate slots")
    if len(gpus) > 16:
        raise ValueError("strict startup search supports at most 16 GPUs")
    choices = []
    candidate_work = 0
    for model, ctx, classifier in requests:
        candidates = []
        target = targets[(model, classifier)] if targets is not None else None
        cards = [g for g in gpus if (g.index == classifier_gpu if classifier and classifier_gpu is not None
                                     else classifier or g.index != classifier_gpu)]
        if target is not None:
            cards = [g for g in cards if g.index in target]
        for profile in base.PROFILES[model]:
            if profile.quality < min_quality or ctx > profile.native_context:
                continue
            calibrated = [e for e in entries if e["hf_spec"] == profile.hf_spec]
            for e in calibrated:
                candidate_work += 1
                if candidate_work > 200000:
                    raise ValueError("startup candidate enumeration exceeds bounded complexity")
                mode = e["mode"]
                if mode == "cpu" and startup_policy == "gpu-only" and target != frozenset():
                    continue
                if target is not None and (mode == "cpu") != (not target):
                    continue
                if ctx > e["max_context"] or slots > e["max_slots"]:
                    continue
                ram = math.ceil(e["weights_host_mib"] + e["buffer_host_mib"] +
                                ctx * slots * e["cache_host_mib_per_token"])
                vram = math.ceil(e["weights_gpu_mib"] + e["buffer_gpu_mib"] +
                                 ctx * slots * e["cache_gpu_mib_per_token"])
                # Calibrated fallback placements currently target one GPU.
                groups = [[]] if mode == "cpu" else [[g] for g in cards if g.free_mib >= vram + 512]
                for group in groups:
                    if target is not None and {g.index for g in group} != target:
                        continue
                    candidates.append(_server(base, profile, group, ctx, slots, classifier,
                                              mode, ram, vram, e))
                    if len(candidates) > 4096:
                        raise ValueError("too many startup placements; reduce GPUs/models/calibrations")
            if profile.required_mib is not None and not any(e["mode"] == "gpu" for e in calibrated):
                # Legacy catalogue envelopes are indivisible, not 85/15
                # weight/cache estimates. Do not extrapolate fallback from them.
                envelope = profile.required_mib * max(1, math.ceil(ctx * slots / profile.native_context))
                for positions in base.gpu_subsets(cards):
                    candidate_work += 1
                    if candidate_work > 200000:
                        raise ValueError("startup candidate enumeration exceeds bounded complexity")
                    group = [cards[p] for p in positions]
                    if target is not None and {g.index for g in group} != target:
                        continue
                    if sum(g.free_mib - 512 for g in group) >= envelope:
                        candidates.append(_server(base, profile, group, ctx, slots, classifier,
                                                  "gpu", envelope, envelope, None))
                    if len(candidates) > 4096:
                        raise ValueError("too many startup placements; reduce GPUs/models/calibrations")
        if not candidates:
            raise ValueError(f"no safe {startup_policy} placement for {model}; provide calibrated "
                             "memory metadata or sufficient GPU capacity; context/slots/quality are never reduced")
        options = []
        for candidate in candidates:
            if startup_policy == "allow-cpu-only" and candidate["mode"] == "gpu" and targets is None:
                # Overflow must already be resident and budgeted, using exactly
                # the requested weights, context and classifier identity.
                replicas = [s for s in candidates if s["mode"] == "cpu" and
                            s["profile"]["hf_spec"] == candidate["profile"]["hf_spec"]]
                for replica in replicas:
                    replica = {**replica, "id": candidate["id"] + "-cpu", "fallback": True}
                    options.append([{**candidate, "fallback_id": replica["id"]}, replica])
                    if len(options) > 4096:
                        raise ValueError("too many startup placements; reduce GPUs/models/calibrations")
            else:
                if classifier and classifier_gpu is not None and candidate["mode"] == "cpu":
                    continue
                options.append([candidate])
        if not options:
            raise ValueError(f"no calibrated CPU-only replica for {model}; overflow requires matching weights")
        choices.append(options)
    # Reserve CPU threads jointly, including CPU helper threads for GPU servers.
    if host.cpu_capacity < len(requests):
        raise ValueError("CPU quota cannot provide one concurrent thread per server")
    best = {0: [], 1: [], 2: []}
    explored = 0

    def search(i, selected, used, ram):
        nonlocal explored
        explored += 1
        if explored > 200000:
            raise ValueError("startup search exceeds bounded complexity; reduce models/GPUs/calibrations")
        if i == len(choices):
            if len(selected) > host.cpu_capacity or sum(s["slots"] for s in selected) > 64:
                return
            primaries = [s for s in selected if not s.get("fallback")]
            primary = next(s for s in primaries if not s["classifier"] and s["model"] == roles["sonnet"])
            classifier = next((s for s in primaries if s["classifier"]), None)
            rank = {"gpu": 2, "cpu": 0}
            score = (rank[primary["mode"]] + (rank[classifier["mode"]] if classifier else 0),
                     rank[primary["mode"]], sum(rank[s["mode"]] for s in primaries),
                     -sum(len(s["gpus"]) for s in selected),
                     sum(s["profile"]["quality"] for s in selected), -ram)
            stage = (0 if all(s["mode"] == "gpu" for s in primaries)
                     else 2 if all(s["mode"] == "cpu" for s in primaries) else 1)
            best[stage].append((score, selected))
            best[stage].sort(key=lambda x: x[0], reverse=True)
            del best[stage][max_layouts:]
            return
        for option in choices[i]:
            indices = {g["index"] for s in option for g in s["gpus"]}
            required = sum(s["ram_required_mib"] for s in option)
            if used & indices or ram + required > host.usable_mib - host.reserve_mib:
                continue
            search(i + 1, selected + option, used | indices, ram + required)

    search(0, [], set(), 0)
    stages = [stage for stage in best.values() if stage]
    if not stages:
        raise ValueError("no complete layout fits joint RAM/VRAM/CPU budgets (swap excluded)")
    # Reserve one attempt for each permitted fallback stage before using extra
    # attempts on another quant/device arrangement in the same stage.
    ranked = [stage[0] for stage in stages]
    ranked.extend(proposal for stage in stages for proposal in stage[1:])
    layouts = []
    for _, selected in ranked[:max_layouts]:
        threads = max(1, math.floor(host.cpu_capacity / len(selected)))
        servers = [{**s, "threads": min(threads, s["max_threads"]),
                    "threads_batch": min(threads, s["max_threads"])} for s in selected]
        role_ids = {r: next(s["id"] for s in servers if not s["classifier"] and s["model"] == m)
                    for r, m in roles.items()}
        used = {g["index"] for s in servers for g in s["gpus"]}
        layouts.append({"context": context, "roles": roles, "role_ids": role_ids,
                        "classifier_id": "local-classifier" if classifier_model else None,
                        "servers": servers, "unused_gpus": [asdict(g) for g in gpus if g.index not in used],
                        "placement_policy": "joint-host-vram", "startup_policy": startup_policy,
                        "role_placement": role_placement,
                        "host": asdict(host), "ram_required_mib": sum(s["ram_required_mib"] for s in servers),
                        "min_quality": min_quality})
    return {**layouts[0], "alternatives": layouts[1:]}


def _server(base, profile, group, context, slots, classifier, mode, ram, vram, entry):
    if len(group) > 1:
        slowest = min(group, key=lambda g: (g.link_score, g.speed_score))
        group = [slowest] + sorted((g for g in group if g != slowest),
                                   key=lambda g: (-g.link_score, -g.speed_score))
    p = {**asdict(profile), "extra_env": dict(profile.extra_env), "hf_spec": profile.hf_spec}
    if entry:
        for key in ("kv_k", "kv_v", "flash_attn", "batch", "ubatch"):
            p[key] = entry[key]
    return {
        "id": "local-classifier" if classifier else f"local-{profile.model.replace(':', '-').replace('.', '')}",
        "model": profile.model, "classifier": classifier, "context": context, "slots": slots,
        "backend": "llama.cpp",
        "profile": p, "gpus": [asdict(g) for g in group],
        "cuda_visible_devices": ",".join(str(g.index) for g in group), "multi_gpu": len(group) > 1,
        "mode": mode, "ngl": entry["ngl"] if entry else 999, "cache_on_cpu": mode == "cpu",
        "ram_required_mib": ram, "required_mib": vram, "allocated_mib": sum(g.free_mib for g in group),
        "headroom_mib": sum(g.free_mib for g in group) - vram,
        "memory_estimate": {k: entry[k] for k in MEMORY_FIELDS} if entry else {"legacy_envelope_mib": ram},
        "estimate_source": entry["source"] if entry else "uncalibrated conservative catalogue envelope; RAM includes staging",
        "max_threads": entry["max_threads"] if entry else 2**31,
        "placement_policy": "joint-host-vram",
    }
