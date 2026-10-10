#!/usr/bin/env bash

claude_local_auto_mode_requested() {
    local explicit_mode="" arg next
    local -a setting_files=()

    while (($#)); do
        arg="$1"
        case "$arg" in
            --permission-mode)
                shift
                explicit_mode="${1:-}"
                ;;
            --permission-mode=*)
                explicit_mode="${arg#*=}"
                ;;
            --settings)
                shift
                [[ -f "${1:-}" ]] && setting_files+=("$1")
                ;;
            --settings=*)
                next="${arg#*=}"
                [[ -f "$next" ]] && setting_files+=("$next")
                ;;
        esac
        shift
    done

    if [[ -n "$explicit_mode" ]]; then
        [[ "$explicit_mode" == auto ]]
        return
    fi

    local config_dir="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
    local managed="${CLAUDE_CODE_MANAGED_SETTINGS_PATH:-/etc/claude-code/managed-settings.json}"
    local project="${PWD}/.claude"
    local -a candidates=("$managed" "${setting_files[@]}" "$project/settings.local.json" "$project/settings.json" "$config_dir/settings.json")
    python3 - "${candidates[@]}" <<'PY'
import json
import pathlib
import sys

for raw_path in sys.argv[1:]:
    path = pathlib.Path(raw_path)
    if not path.is_file():
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        continue
    mode = data.get("defaultMode")
    permissions = data.get("permissions")
    if mode is None and isinstance(permissions, dict):
        mode = permissions.get("defaultMode")
    if isinstance(mode, str):
        raise SystemExit(0 if mode == "auto" else 1)
raise SystemExit(1)
PY
}

claude_local_set_permission_mode() {
    local new_mode="$1" arg skip=0
    local -a updated=()
    while (("${#CLAUDE_ARGS[@]}")); do
        arg="${CLAUDE_ARGS[0]}"
        CLAUDE_ARGS=("${CLAUDE_ARGS[@]:1}")
        if (( skip )); then
            skip=0
            continue
        fi
        case "$arg" in
            --permission-mode)
                skip=1
                ;;
            --permission-mode=*)
                ;;
            *)
                updated+=("$arg")
                ;;
        esac
    done
    CLAUDE_ARGS=("${updated[@]}" --permission-mode "$new_mode")
}
