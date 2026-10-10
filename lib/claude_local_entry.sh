#!/usr/bin/env bash
# Runtime policy wrapper for claude-local installed commands.
set -euo pipefail
SELF="${BASH_SOURCE[0]}"
command -v readlink >/dev/null 2>&1 && SELF="$(readlink -f "$SELF" 2>/dev/null || printf '%s' "$SELF")"
ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"

STATE_DIR="${CLAUDE_LOCAL_STATE:-$HOME/.local/share/pushbutton/claude-local}"
WEB_MCP="${CLAUDE_LOCAL_WEB_MCP:-1}"
WEB_CONFIG="${CLAUDE_LOCAL_WEB_CONFIG:-}"
ARGS=()
INFO_ONLY=0
case "${1:-}" in
  help|-h|--help|models|doctor) exec "$ROOT/claude-local" "$@";;
esac
while (($#)); do
  case "$1" in
    --)
      ARGS+=("$@"); break;;
    --local-state)
      [[ $# -ge 2 ]] || { echo "--local-state needs a directory" >&2; exit 2; }
      STATE_DIR="$2"; export CLAUDE_LOCAL_STATE="$2"; shift 2;;
    --local-cache)
      [[ $# -ge 2 ]] || { echo "--local-cache needs a directory" >&2; exit 2; }
      export CLAUDE_LOCAL_CACHE="$2"; ARGS+=("$1" "$2"); shift 2;;
    --local-no-web)
      WEB_MCP=0; shift;;
    --local-web-config)
      [[ $# -ge 2 ]] || { echo "--local-web-config needs a JSON file" >&2; exit 2; }
      WEB_CONFIG="$2"; shift 2;;
    --folders|--system-info)
      INFO_ONLY=1; ARGS+=("$1"); shift;;
    --local-context|--local-client-context|--local-port-base|--slots)
      [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; exit 2; }
      ARGS+=("$1" "$2"); shift 2;;
    --slots=*|--agents=[0-9]*)
      ARGS+=("$1"); shift;;
    --agents)
      # Numeric values are harness replica counts; JSON belongs to Claude Code.
      if [[ "${2:-}" =~ ^[0-9]+$ ]]; then ARGS+=("$1" "$2"); shift 2; else ARGS+=("$@"); break; fi;;
    --select|--quiet|--no-parallel|--local-no-teams|--local-keep-servers|--local-allow-offload|--local-dry-run|--local-verbose)
      ARGS+=("$1"); shift;;
    -*)
      ARGS+=("$@"); break;;
    *) ARGS+=("$1"); shift;;
  esac
done

((INFO_ONLY == 0)) || exec "$ROOT/claude-local" "${ARGS[@]}"
export CLAUDE_LOCAL_WEB_CONFIG="$WEB_CONFIG"
if CLAUDE_LOCAL_STARTUP_ONLY=1 CLAUDE_LOCAL_WEB_MCP="$WEB_MCP" "$ROOT/claude-local" "${ARGS[@]}"; then
  exit 0
else
  startup_rc=$?
  [[ $startup_rc -eq 125 ]] || exit "$startup_rc"
fi
source "$ROOT/lib/pushbutton_folders.sh"
initialize_folders 1

# Local inference can have long first-token and tool/subagent queues. Claude's
# hosted defaults are too aggressive for a single 3090/V100 under a large
# prompt. Keep retries low so a failed 10-minute local generation is not
# repeated ten times, but allow a healthy local stream enough time to finish.
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-1800000}"
export CLAUDE_STREAM_IDLE_TIMEOUT_MS="${CLAUDE_STREAM_IDLE_TIMEOUT_MS:-900000}"
export CLAUDE_ENABLE_BYTE_WATCHDOG="${CLAUDE_ENABLE_BYTE_WATCHDOG:-1}"
export CLAUDE_ENABLE_STREAM_WATCHDOG="${CLAUDE_ENABLE_STREAM_WATCHDOG:-1}"
export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-2}"
export CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS="${CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS:-1800000}"
export MCP_TIMEOUT="${MCP_TIMEOUT:-60000}"

# Do not let ten concurrent Claude requests pile up behind a local -np 1
# backend. Default to one concurrent request per visible NVIDIA GPU, capped at
# four; users can override CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY explicitly.
if [[ -z "${CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  n="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
  [[ "$n" =~ ^[0-9]+$ ]] || n=1
  (( n < 1 )) && n=1
  (( n > 4 )) && n=4
  export CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY="$n"
fi

# Provider-neutral web search/fetch for local models. Exa's hosted MCP supports
# search + page fetch without requiring a local Node process or an API key.
# Keep this config isolated under Pushbutton state rather than changing the
# user's ~/.claude.json. Preserve edits so the user's backend choice is sticky.
if (( WEB_MCP )); then
  mkdir -p "$STATE_DIR"
  MCP_CONFIG="${WEB_CONFIG:-$STATE_DIR/web-mcp.json}"
  if [[ -z "$WEB_CONFIG" && ! -e "$MCP_CONFIG" ]]; then
    (umask 077; set -o noclobber; cat >"$MCP_CONFIG" <<'JSON'
{
  "mcpServers": {
    "pushbutton-web": {
      "type": "http",
      "url": "https://mcp.exa.ai/mcp"
    }
  }
}
JSON
    )
  fi
  printf '[claude-local] Sticky web backend config: %s\n' "$MCP_CONFIG"
  [[ -r "$MCP_CONFIG" ]] || { echo "Web MCP config is not readable: $MCP_CONFIG" >&2; exit 2; }
  python3 - "$MCP_CONFIG" <<'PY'
import json, sys
from urllib.parse import urlsplit
try:
    with open(sys.argv[1]) as f:
        config = json.load(f)
    servers = config["mcpServers"]
    if not isinstance(servers, dict) or not servers:
        raise ValueError("mcpServers must be a non-empty object")
    if any(not isinstance(server, dict) for server in servers.values()):
        raise ValueError("each MCP server must be an object")
except (OSError, ValueError, KeyError, TypeError) as exc:
    sys.exit(f"Invalid web MCP config: {exc}. Edit the disclosed config or use --local-no-web.")
print("[claude-local] Web MCP servers: " + ", ".join(servers) + " (verify connection/tools with /mcp)")
for name, server in servers.items():
    # Disclose provider hosts, never URL credentials, query strings or env values.
    host = urlsplit(server.get("url", "")).hostname if isinstance(server.get("url", ""), str) else None
    print(f"[claude-local] Web backend {name}: {host or 'local MCP process (see config)'}")
PY
  printf '[claude-local] Change backend: edit that JSON and restart; or use --local-web-config FILE. Disable automatic MCP with --local-no-web.\n'
  ARGS+=(--mcp-config "$MCP_CONFIG")
else
  printf '[claude-local] Automatic web MCP disabled; built-in web tools remain unavailable. Supply --mcp-config FILE for custom web tools.\n'
fi

if [[ -x "$ROOT/claude-local-safe" ]]; then
  exec "$ROOT/claude-local-safe" "${ARGS[@]}"
fi
exec "$ROOT/claude-local" "${ARGS[@]}"
