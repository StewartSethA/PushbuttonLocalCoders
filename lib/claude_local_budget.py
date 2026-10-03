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
from urllib.request import Request, urlopen

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
    with urlopen(req, timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise BudgetError("backend returned a non-object response")
    return result


def effective_context(props: dict) -> int:
    # Official /props reports meta.slot_n_ctx here, not total KV or GGUF native
    # context. This includes runtime fitting and current per-slot caps.
    if props.get("total_slots") != 1:
        raise BudgetError("Expected total_slots=1 for the launcher's -np 1 backend")
    return positive_integer(props.get("default_generation_settings", {}).get("n_ctx"), "/props per-slot n_ctx")


def input_tokens(result: dict) -> int:
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
    if not path.exists():
        return
    try:
        settings = json.loads(path.read_text())
        managed_env = settings.get("env", {})
        if settings.get("autoCompactEnabled") is False:
            raise BudgetError("managed autoCompactEnabled=false disables compaction")
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


def prepare_config(plan: dict, backends: dict, requested: int, client: str, env: dict) -> dict:
    routes = {}
    for mid, port in backends.items():
        url = f"http://127.0.0.1:{port}"
        try:
            capacity = effective_context(request_json(url + "/props"))
            # Exercise the native Anthropic conversion + chat template counter,
            # not /tokenize on a concatenation or a character-count estimate.
            probe = {"model": mid, "max_tokens": 1, "system": "Local context probe.",
                     "tools": [{"name": "probe", "description": "Probe", "input_schema": {"type": "object"}}],
                     "messages": [{"role": "user", "content": "Hello"}]}
            input_tokens(request_json(url + "/v1/messages/count_tokens", probe))
        except Exception as exc:
            raise BudgetError(f"{mid}: cannot verify runtime slot capacity/native Anthropic tokenizer ({type(exc).__name__}). Check /props and /v1/messages/count_tokens; rebuild/update llama.cpp with CLAUDE_LOCAL_REBUILD=1. Claude was not launched") from exc
        routes[mid] = {"model_id": mid, "backend_alias": mid, "url": url,
                       "context_capacity": capacity, "prompt_reserve": PROMPT_RESERVE}
        if capacity != requested:
            print(f"[claude-local] Context mismatch: {mid}: requested={requested}, effective per-slot={capacity}; using verified capacity, not requested -c", file=sys.stderr)
    roles = {role: routes[mid] for role, mid in plan["role_ids"].items()}
    policy = context_policy(min(route["context_capacity"] for route in roles.values()), client, env)
    return {"roles": roles, "budget": policy}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requested", required=True)
    ap.add_argument("--client", default="")
    ap.add_argument("--plan")
    ap.add_argument("--backends")
    ap.add_argument("--config")
    ap.add_argument("--check-claude", action="store_true")
    ap.add_argument("claude_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    try:
        requested = positive_integer(args.requested, "--local-context")
        validate_claude_args(args.claude_args, dict(os.environ))
        env = frontend_policy_env(args.claude_args, dict(os.environ))
        validate_managed_settings()
        context_policy(requested, args.client, env)
        version = claude_capabilities() if args.check_claude else None
        if args.plan:
            with open(args.plan) as f:
                plan = json.load(f)
            with open(args.backends) as f:
                backends = {row[0]: int(row[1]) for row in (line.rstrip("\n").split("\t") for line in f)}
            config = prepare_config(plan, backends, requested, args.client, env)
            config["claude_version"] = version
            with open(args.config, "w") as f:
                json.dump(config, f, indent=2)
            p = config["budget"]
            print(f"[claude-local] Claude Code {version}; requested context={requested}; smallest effective per-slot={p['capacity']}; "
                  f"client assumed window={p['client_context']}; auto-compaction window={p['compact_window']}; "
                  f"max output={p['max_output_tokens']}; prompt reserve={p['prompt_reserve']}; compaction headroom={p['compact_reserve']}")
            print("[claude-local] Compaction enabled with isolated Claude settings; native tokenizer preflight enforced on every role. Auto-compaction alone is not an overflow guarantee.")
            print("[claude-local] Installed artifact/help compatibility checked; remotely managed policy cannot be inspected here. A conflicting managed policy must be resolved, not bypassed.")
            if any(a.split("=", 1)[0] == "--model" for a in args.claude_args):
                print("[claude-local] Frontend --model override: explicit compaction window and gateway route budgets remain enforced.")
            if any(a.split("=", 1)[0] in ("--resume", "--continue", "-r", "-c") for a in args.claude_args):
                print("[claude-local] Resume warning: lowering the window does not compact an already oversized transcript; /compact may also fail. Use a new session with a saved handoff summary.")
    except (BudgetError, OSError, subprocess.SubprocessError) as exc:
        print(f"[claude-local] Context policy error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
