#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/lib/claude_local_policy.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
export HOME="$tmp/home"
export CLAUDE_LOCAL_STATE="$tmp/state"
unset CLAUDE_CONFIG_DIR
mkdir -p "$HOME/.claude" "$tmp/project/.claude" "$CLAUDE_LOCAL_STATE/claude-config"
cd "$tmp/project"

cat >"$CLAUDE_LOCAL_STATE/claude-config/settings.json" <<'JSON'
{"permissions":{"defaultMode":"auto"}}
JSON
claude_local_auto_mode_requested

claude_local_auto_mode_requested --permission-mode auto
! claude_local_auto_mode_requested --permission-mode default

cat >"$CLAUDE_LOCAL_STATE/claude-config/settings.json" <<'JSON'
{"permissions":{"defaultMode":"default"}}
JSON
! claude_local_auto_mode_requested

cat >"$tmp/managed-settings.json" <<'JSON'
{"permissions":{"defaultMode":"default"}}
JSON
mkdir -p "$tmp/managed-settings.d"
cat >"$tmp/managed-settings.d/10-auto.json" <<'JSON'
{"permissions":{"defaultMode":"auto"}}
JSON
export CLAUDE_CODE_MANAGED_SETTINGS_PATH="$tmp/managed-settings.json"
claude_local_auto_mode_requested
unset CLAUDE_CODE_MANAGED_SETTINGS_PATH

cat >"$tmp/plan.json" <<'JSON'
{"servers":[{"id":"m","capacity":{"slots":1}},{"id":"w","capacity":{"slots":3}}],
 "role_ids":{"haiku":"m","sonnet":"w","opus":"w","fable":"w"}}
JSON
claude_local_plan_has_single_slot "$tmp/plan.json"
cat >"$tmp/plan.json" <<'JSON'
{"servers":[{"id":"m","capacity":{"slots":2}}],
 "role_ids":{"haiku":"m","sonnet":"m","opus":"m","fable":"m"}}
JSON
! claude_local_plan_has_single_slot "$tmp/plan.json"

CLAUDE_ARGS=(--permission-mode auto --resume session)
claude_local_set_permission_mode default
[[ "${CLAUDE_ARGS[*]}" == "--resume session --permission-mode default" ]]

CLAUDE_ARGS=(--permission-mode=auto --continue)
claude_local_set_permission_mode acceptEdits
[[ "${CLAUDE_ARGS[*]}" == "--continue --permission-mode acceptEdits" ]]

printf 'claude-local policy tests passed\n'
