#!/usr/bin/env bash

claude_local_auto_mode_requested() {
    local explicit_mode="" arg

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
        esac
        shift
    done

    if [[ -n "$explicit_mode" ]]; then
        [[ "$explicit_mode" == auto ]]
        return
    fi

    local managed="${CLAUDE_CODE_MANAGED_SETTINGS_PATH:-/etc/claude-code/managed-settings.json}"
    local config_dir="${CLAUDE_LOCAL_STATE:-$HOME/.local/share/pushbutton/claude-local}/claude-config"
    local -a candidates=("$managed" "$config_dir/settings.json")
    python3 - "${candidates[@]}" <<'PY'
import json
import pathlib
import sys

def load(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None

def permission_mode(data):
    if data is None:
        return None
    mode = data.get("defaultMode")
    permissions = data.get("permissions")
    if mode is None and isinstance(permissions, dict):
        mode = permissions.get("defaultMode")
    return mode if isinstance(mode, str) else None

managed_path = pathlib.Path(sys.argv[1])
managed_files = [managed_path]
managed_dir = managed_path.with_suffix(".d")
if managed_dir.is_dir():
    managed_files.extend(sorted(p for p in managed_dir.iterdir()
                                if not p.name.startswith(".") and p.suffix == ".json"))
managed_settings = {}
for file in managed_files:
    data = load(file)
    if data is not None:
        managed_settings.update(data)
mode = permission_mode(managed_settings)
if mode is not None:
    raise SystemExit(0 if mode == "auto" else 1)

for raw_path in sys.argv[2:]:
    data = load(pathlib.Path(raw_path))
    mode = permission_mode(data)
    if mode is not None:
        raise SystemExit(0 if mode == "auto" else 1)
raise SystemExit(1)
PY
}

claude_local_plan_has_single_slot() {
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    plan = json.load(f)
servers = {server["id"]: server for server in plan.get("servers", [])}
route_ids = set(plan.get("role_ids", {}).values())
slots = [
    int(servers[mid].get("capacity", {}).get("slots", 1))
    for mid in route_ids
    if mid in servers
]
raise SystemExit(0 if not slots or min(slots) <= 1 else 1)
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
