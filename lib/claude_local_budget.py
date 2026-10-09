#!/usr/bin/env python3
"""Context policy shared by the launcher and the Anthropic gateway."""
from __future__ import annotations

import argparse
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.request import ProxyHandler, Request, build_opener

MIN_COMPACT_WINDOW = 100000
MAX_COMPACT_WINDOW = 1000000
PROMPT_RESERVE = 8192
COMPACT_RESERVE = 8192
DEFAULT_OUTPUT = 8192


class BudgetError(ValueError):
    pass


def positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not str(value).isascii() or not str(value).isdigit() or int(value) < 1:
        raise BudgetError(f"{name} must be a positive plain integer")
    return int(value)


def context_policy(capacity: int, client: str = "", env: dict | None = None) -> dict:
    env = os.environ if env is None else env
    capacity = positive_integer(capacity, "backend context")
    for name in ("DISABLE_AUTO_COMPACT", "DISABLE_COMPACT"):
        if env.get(name, "").lower() not in ("", "0", "false"):
            raise BudgetError(f"{name} disables compaction; unset it for claude-local")
    if env.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"):
        raise BudgetError("Unset CLAUDE_AUTOCOMPACT_PCT_OVERRIDE; use the supported CLAUDE_CODE_AUTO_COMPACT_WINDOW integer instead")
    output = positive_integer(env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", str(DEFAULT_OUTPUT)), "CLAUDE_CODE_MAX_OUTPUT_TOKENS")
    safe_client = capacity - output - PROMPT_RESERVE
    safe_compact = safe_client - COMPACT_RESERVE
    if safe_compact < MIN_COMPACT_WINDOW:
        raise BudgetError(
            f"Effective backend context {capacity} cannot fit the documented minimum auto-compaction window "
            f"{MIN_COMPACT_WINDOW} plus output {output}, prompt reserve {PROMPT_RESERVE} and compaction reserve "
            f"{COMPACT_RESERVE}. Smaller windows are unsupported; use another backend/configuration that "
            "fits your hardware, not silent clamping or disabling compaction."
        )
    assumed = min(200000, safe_client)
    for value, name in ((client, "--local-client-context / CLAUDE_LOCAL_CLIENT_CONTEXT"),
                        (env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS", ""), "CLAUDE_CODE_MAX_CONTEXT_TOKENS")):
        if value:
            n = positive_integer(value, name)
            if n > safe_client:
                raise BudgetError(f"{name}={n} is unsafe: maximum {safe_client} for effective context {capacity}, output {output} and prompt reserve {PROMPT_RESERVE}")
            assumed = n if name.startswith("--local") else min(assumed, n)
    if assumed < MIN_COMPACT_WINDOW:
        raise BudgetError(f"Client window {assumed} is below the supported {MIN_COMPACT_WINDOW} minimum; cannot safely clamp compaction upward")
    compact_limit = min(assumed, safe_compact, MAX_COMPACT_WINDOW)
    compact = positive_integer(env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW", str(compact_limit)), "CLAUDE_CODE_AUTO_COMPACT_WINDOW")
    if not MIN_COMPACT_WINDOW <= compact <= compact_limit:
        raise BudgetError(f"CLAUDE_CODE_AUTO_COMPACT_WINDOW={compact} is unsafe/unsupported; use {MIN_COMPACT_WINDOW}..{compact_limit} for effective context {capacity}")
    return {"capacity": capacity, "client_context": assumed, "compact_window": compact,
            "max_output_tokens": output, "prompt_reserve": PROMPT_RESERVE,
            "compact_reserve": COMPACT_RESERVE}


def request_json(url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = Request(url, data=data, headers={"content-type": "application/json"})
    with build_opener(ProxyHandler({})).open(req, timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise BudgetError("backend returned a non-object response")
    return result


def effective_context(props: dict, expected_slots: int | None = None) -> int:
    # Require runtime per-slot context, never aggregate KV or GGUF native context.
    requested_slots = positive_integer(1 if expected_slots is None else expected_slots, "requested slots")
    slots = props.get("total_slots", props.get("n_parallel"))
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < requested_slots:
        raise BudgetError(f"/props must prove at least {requested_slots} requested slots")
    if expected_slots is None and slots != 1:
        raise BudgetError("Expected total_slots=1 for the launcher's legacy -np 1 backend")
    settings = props.get("default_generation_settings") or {}
    context = props.get("n_ctx_slot", props.get("n_ctx_per_slot", settings.get("n_ctx")))
    if isinstance(context, bool) or not isinstance(context, int):
        raise BudgetError("/props must prove integer per-slot n_ctx, not aggregate context")
    return positive_integer(context, "/props per-slot n_ctx")


def input_tokens(result: dict) -> int:
    if not isinstance(result, dict):
        raise BudgetError("Backend count_tokens returned a non-object response")
    n = result.get("input_tokens")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise BudgetError("Backend /v1/messages/count_tokens did not return a non-negative integer input_tokens")
    return n


def request_budget_error(body: dict, prompt_tokens: int, route: dict) -> str | None:
    output = positive_integer(body.get("max_tokens"), "max_tokens")
    capacity = positive_integer(route["context_capacity"], "route context")
    reserve = positive_integer(route["prompt_reserve"], "prompt reserve")
    if prompt_tokens + output + reserve <= capacity:
        return None
    return (
        f"prompt is too long: request ({prompt_tokens} input tokens + {output} requested output tokens "
        f"+ {reserve} reserve) exceeds the available context size ({capacity} tokens). "
        "Compact earlier or start a new session with a saved handoff summary. An already oversized "
        "resumed transcript may also be too large for /compact. No history or max_tokens was changed."
    )


def validate_claude_args(args: list[str], env: dict) -> None:
    for name in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"):
        if env.get(name, "").lower() not in ("", "0", "false"):
            raise BudgetError(f"{name} bypasses the local gateway; unset it")
    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in ("--settings", "--setting-sources", "--no-autocompact"):
            raise BudgetError(f"{flag} can bypass isolated compaction policy; remove it. Use CLAUDE_CODE_AUTO_COMPACT_WINDOW for a safe lower window")
        if "[1m]" in arg.lower():
            raise BudgetError("[1m] assumes an unverified million-token window; remove it for local backends")


def frontend_policy_env(args: list[str], env: dict) -> dict:
    result = dict(env)
    for i, arg in enumerate(args):
        if arg.split("=", 1)[0] != "--autocompact":
            continue
        value = arg.split("=", 1)[1] if "=" in arg else (args[i + 1] if i + 1 < len(args) else "")
        if value == "auto":
            continue
        window = positive_integer(value, "--autocompact (use auto or a plain token integer, not off)")
        if not MIN_COMPACT_WINDOW <= window <= MAX_COMPACT_WINDOW:
            raise BudgetError(f"--autocompact must be {MIN_COMPACT_WINDOW}..{MAX_COMPACT_WINDOW}")
        existing = result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "")
        if existing:
            window = min(window, positive_integer(existing, "CLAUDE_CODE_AUTO_COMPACT_WINDOW"))
        result["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(window)
    return result


def validate_managed_settings(path: Path = Path("/etc/claude-code/managed-settings.json")) -> None:
    try:
        files = [path] if path.exists() else []
        directory = path.with_suffix(".d")
        if directory.exists():
            files += sorted(p for p in directory.iterdir() if not p.name.startswith(".") and p.suffix == ".json")
        settings = {}
        managed_env = {}
        for file in files:
            data = json.loads(file.read_text())
            if not isinstance(data, dict) or not isinstance(data.get("env", {}), dict):
                raise BudgetError("invalid managed settings object/env")
            settings.update(data)
            managed_env.update(data.get("env", {}))
        if settings.get("autoCompactEnabled") is False:
            raise BudgetError("managed autoCompactEnabled=false disables compaction")
        if settings.get("policyHelper"):
            raise BudgetError("dynamic managed policyHelper cannot be verified before launch")
        protected = {"DISABLE_COMPACT", "DISABLE_AUTO_COMPACT", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
                     "CLAUDE_CONFIG_DIR", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
                     "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
                     "CLAUDE_CODE_MAX_CONTEXT_TOKENS", "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
                     "CLAUDE_CODE_MAX_OUTPUT_TOKENS"}
        if any(name in managed_env for name in protected):
            raise BudgetError("managed env overrides local context/gateway enforcement")
    except (ValueError, AttributeError, OSError) as exc:
        raise BudgetError(f"Cannot safely apply local policy with {path}: {exc}; ask your administrator to resolve the conflict") from exc


def claude_capabilities() -> str:
    executable = shutil.which("claude")
    if not executable:
        raise BudgetError("Claude Code is not installed")
    version = subprocess.check_output([executable, "--version"], text=True, timeout=30).strip()
    help_text = subprocess.check_output([executable, "--help"], text=True, timeout=30)
    parsed = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", version)
    if not parsed or tuple(map(int, parsed.groups())) < (2, 1, 221):
        raise BudgetError(f"Claude Code {version}: require 2.1.221+ with documented --autocompact support; update Claude Code")
    # Both official native installs and npm's cli.js symlink carry these
    # feature names. Do not guess a first supporting release from its number.
    try:
        with Path(executable).resolve().open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as artifact:
            supported = all(artifact.find(name.encode()) >= 0 for name in (
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS"))
    except (OSError, ValueError):
        supported = False
    if not supported or not all(flag in help_text for flag in ("--autocompact", "--setting-sources", "--settings")):
        raise BudgetError(f"Claude Code {version}: cannot verify explicit auto-compaction-window support in the installed artifact. Update the official native/npm install; opaque wrappers are unsupported")
    return version


def shared_policy(routes: dict, client: str, env: dict) -> dict:
    capacities = [route["capacity"] for route in routes.values() if "capacity" in route]
    policy_env = dict(env)
    if capacities:
        output_limit = min(positive_integer(c["output_tokens"], "model output_tokens") for c in capacities)
        output = positive_integer(env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", min(DEFAULT_OUTPUT, output_limit)),
                                  "CLAUDE_CODE_MAX_OUTPUT_TOKENS")
        if output > output_limit:
            raise BudgetError(f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={output} exceeds the smallest model output budget {output_limit}; lower it")
        policy_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(output)
    policy = context_policy(min(route["context_capacity"] for route in routes.values()), client, policy_env)
    if not capacities:
        return policy
    client_limit = min(positive_integer(c["client_context"], "model client_context") for c in capacities)
    for value, name in ((client, "--local-client-context / CLAUDE_LOCAL_CLIENT_CONTEXT"),
                        (env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS", ""), "CLAUDE_CODE_MAX_CONTEXT_TOKENS")):
        if value and positive_integer(value, name) > client_limit:
            raise BudgetError(f"{name}={value} exceeds the smallest model client_context {client_limit}; lower it")
    policy["client_context"] = min(policy["client_context"], client_limit)
    compact_limit = min(policy["client_context"],
                        policy["capacity"] - policy["max_output_tokens"] - PROMPT_RESERVE - COMPACT_RESERVE,
                        MAX_COMPACT_WINDOW,
                        *(positive_integer(c["compact_trigger"], "model compact_trigger") for c in capacities))
    if compact_limit < MIN_COMPACT_WINDOW:
        raise BudgetError(f"Per-model client/compact budget {compact_limit} is below the documented minimum auto-compaction window {MIN_COMPACT_WINDOW}; increase the model's context/client_context/compact capacity or choose another backend. Smaller windows and disabling compaction are unsupported")
    if env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW"):
        compact = positive_integer(env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "CLAUDE_CODE_AUTO_COMPACT_WINDOW")
        if compact > compact_limit:
            raise BudgetError(f"CLAUDE_CODE_AUTO_COMPACT_WINDOW={compact} exceeds the smallest model compact limit {compact_limit}; lower it")
    else:
        compact = compact_limit
    policy["compact_window"] = compact
    return policy


def planned_routes(plan: dict) -> dict:
    capacities = {server["id"]: server["capacity"] for server in plan.get("servers", [])}
    return {role: {"context_capacity": positive_integer(capacities[mid]["context"], "model context"),
                   "capacity": capacities[mid]}
            for role, mid in plan["role_ids"].items()}


def read_backends(path: str) -> dict:
    backends = {}
    with open(path) as f:
        for line in f:
            row = line.rstrip("\n").split("\t")
            if len(row) not in (3, 4) or not row[0] or row[0] in backends:
                raise BudgetError("Backend registry requires unique model IDs and 3 or 4 tab-separated columns")
            entry = {"port": positive_integer(row[1], "backend port")}
            if len(row) == 4:
                try:
                    capacity = json.loads(row[3])
                except ValueError as exc:
                    raise BudgetError("Backend registry capacity must be valid JSON") from exc
                if not isinstance(capacity, dict):
                    raise BudgetError("Backend registry capacity must be an object")
                entry["capacity"] = capacity
            backends[row[0]] = entry
    return backends


def prepare_config(plan: dict, backends: dict, requested: int, client: str, env: dict) -> dict:
    routes = {}
    planned = {server["id"]: server["capacity"] for server in plan.get("servers", [])}
    for mid, backend in backends.items():
        port = backend["port"] if isinstance(backend, dict) else backend
        model_capacity = backend.get("capacity", planned.get(mid)) if isinstance(backend, dict) else planned.get(mid)
        requested_context = positive_integer(model_capacity["context"], "model context") if model_capacity else requested
        requested_slots = positive_integer(model_capacity.get("slots", 1), "model slots") if model_capacity else 1
        url = f"http://127.0.0.1:{port}"
        try:
            capacity = effective_context(request_json(url + "/props"), requested_slots)
            if model_capacity and capacity < requested_context:
                raise BudgetError(f"backend proves per-slot context={capacity}; model requires {requested_context}")
            # Exercise the native Anthropic conversion + chat template counter,
            # not /tokenize on a concatenation or a character-count estimate.
            probe = {"model": mid, "max_tokens": 1, "system": "Local context probe.",
                     "tools": [{"name": "probe", "description": "Probe", "input_schema": {"type": "object"}}],
                     "messages": [{"role": "user", "content": "Hello"}]}
            input_tokens(request_json(url + "/v1/messages/count_tokens", probe))
        except Exception as exc:
            raise BudgetError(f"{mid}: cannot verify runtime slot capacity/native Anthropic tokenizer ({type(exc).__name__}: {exc}). Check /props and /v1/messages/count_tokens; rebuild/update llama.cpp with CLAUDE_LOCAL_REBUILD=1. Claude was not launched") from exc
        routes[mid] = {"model_id": mid, "backend_alias": mid, "url": url,
                       "context_capacity": min(capacity, requested_context) if model_capacity else capacity,
                       "prompt_reserve": PROMPT_RESERVE}
        if model_capacity:
            routes[mid]["capacity"] = {**model_capacity, "no_context_shift": True}
        if capacity != requested_context:
            print(f"[claude-local] Context mismatch: {mid}: requested={requested_context}, effective per-slot={capacity}; using verified capacity bounded by model limits, not aggregate -c", file=sys.stderr)
    roles = {role: routes[mid] for role, mid in plan["role_ids"].items()}
    policy = shared_policy(roles, client, env)
    return {"roles": roles, "budget": policy}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requested", required=True)
    ap.add_argument("--client", default="")
    ap.add_argument("--plan")
    ap.add_argument("--backends")
    ap.add_argument("--config")
    ap.add_argument("--model-spec", action="append", default=[])
    ap.add_argument("--slots", default="1")
    ap.add_argument("--check-claude", action="store_true")
    ap.add_argument("claude_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    try:
        requested = positive_integer(args.requested, "--local-context")
        validate_claude_args(args.claude_args, dict(os.environ))
        env = frontend_policy_env(args.claude_args, dict(os.environ))
        validate_managed_settings()
        plan = None
        if args.plan:
            with open(args.plan) as f:
                plan = json.load(f)
        elif args.model_spec:
            from coder_local_plan import parse_model_spec
            from pushbutton_capacity import resolve_options
            capacities = [resolve_options(parse_model_spec(spec).capacity, requested,
                                          positive_integer(args.slots, "--slots"))
                          for spec in args.model_spec]
            plan = {"servers":[{"id":str(i), "capacity":capacity}
                               for i, capacity in enumerate(capacities)],
                    "role_ids":{str(i):str(i) for i in range(len(capacities))}}
        if not args.backends:
            if plan:
                shared_policy(planned_routes(plan), args.client, env)
            else:
                context_policy(requested, args.client, env)
        version = claude_capabilities() if args.check_claude else None
        if args.backends:
            if plan is None or not args.config:
                raise BudgetError("--backends requires --plan and --config")
            backends = read_backends(args.backends)
            config = prepare_config(plan, backends, requested, args.client, env)
            config["claude_version"] = version
            with open(args.config, "w") as f:
                json.dump(config, f, indent=2)
            p = config["budget"]
            print(f"[claude-local] Claude Code {version}; requested context={requested}; smallest effective per-slot={p['capacity']}; "
                  f"client assumed window={p['client_context']}; auto-compaction window={p['compact_window']}; "
                  f"max output={p['max_output_tokens']}; prompt reserve={p['prompt_reserve']}; compaction headroom={p['compact_reserve']}")
            print("[claude-local] Compaction enabled with isolated Claude settings; native tokenizer preflight enforced on every role. Auto-compaction alone is not an overflow guarantee.")
            print("[claude-local] Installed artifact/help compatibility checked; local managed policy conflicts are rejected. Cached/remote policy cannot be independently verified; resolve conflicts with your administrator.")
            if any(a.split("=", 1)[0] == "--model" for a in args.claude_args):
                print("[claude-local] Frontend --model override: recognized Claude IDs may ignore the configured custom-model assumed window; explicit compaction window and gateway route budgets remain enforced.")
            if any(a.split("=", 1)[0] in ("--resume", "--continue", "-r", "-c") for a in args.claude_args):
                print("[claude-local] Resume warning: lowering the window does not compact an already oversized transcript; /compact may also fail. Use a new session with a saved handoff summary.")
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"[claude-local] Context policy error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
