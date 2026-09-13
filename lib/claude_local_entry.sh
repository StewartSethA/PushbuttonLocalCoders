#!/usr/bin/env bash
# Runtime policy wrapper for claude-local installed commands.
set -euo pipefail
SELF="${BASH_SOURCE[0]}"
command -v readlink >/dev/null 2>&1 && SELF="$(readlink -f "$SELF" 2>/dev/null || printf '%s' "$SELF")"
ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"

STATE_DIR="${CLAUDE_LOCAL_STATE:-$HOME/.local/share/pushbutton/claude-local}"
WEB_MCP=1
ARGS=()
while (($#)); do
  case "$1" in
    --local-state)
      [[ $# -ge 2 ]] || { echo "--local-state needs a directory" >&2; exit 2; }
      STATE_DIR="$2"; export CLAUDE_LOCAL_STATE="$2"; shift 2;;
    --local-no-web)
      WEB_MCP=0; shift;;
    *) ARGS+=("$1"); shift;;
  esac
done

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
# user's ~/.claude.json. Firecrawl/Tavily/etc. can still be supplied separately.
if (( WEB_MCP )); then
  mkdir -p "$STATE_DIR"
  MCP_CONFIG="$STATE_DIR/web-mcp.json"
  cat >"$MCP_CONFIG" <<'JSON'
{
  "mcpServers": {
    "pushbutton-web": {
      "type": "http",
      "url": "https://mcp.exa.ai/mcp"
    }
  }
}
JSON
  ARGS+=(--mcp-config "$MCP_CONFIG")
fi

exec "$ROOT/claude-local" "${ARGS[@]}"
