#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/lib/claude_local_policy.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
export HOME="$tmp/home"
mkdir -p "$HOME/.claude" "$tmp/project/.claude"
cd "$tmp/project"

cat >"$HOME/.claude/settings.json" <<'JSON'
{"permissions":{"defaultMode":"auto"}}
JSON
claude_local_auto_mode_requested

claude_local_auto_mode_requested --permission-mode auto
! claude_local_auto_mode_requested --permission-mode default

cat >"$HOME/.claude/settings.json" <<'JSON'
{"permissions":{"defaultMode":"default"}}
JSON
! claude_local_auto_mode_requested

cat >"$tmp/custom-settings.json" <<'JSON'
{"defaultMode":"auto"}
JSON
claude_local_auto_mode_requested --settings "$tmp/custom-settings.json"

CLAUDE_ARGS=(--permission-mode auto --resume session)
claude_local_set_permission_mode default
[[ "${CLAUDE_ARGS[*]}" == "--resume session --permission-mode default" ]]

CLAUDE_ARGS=(--permission-mode=auto --continue)
claude_local_set_permission_mode acceptEdits
[[ "${CLAUDE_ARGS[*]}" == "--continue --permission-mode acceptEdits" ]]

printf 'claude-local policy tests passed\n'
