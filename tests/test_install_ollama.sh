#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/lib/install_ollama.sh"

REMOTE_PAYLOAD='{"models":[{"name":"qwen3.8:27b"}]}'
LAST_CURL_URL=""
ERRORS=""
LOCAL_CALLS=()

curl() {
    LAST_CURL_URL="${!#}"
    [[ "$LAST_CURL_URL" == *"/api/tags" ]] || return 2
    printf '%s' "$REMOTE_PAYLOAD"
}
tui_error() { ERRORS+="$*\n"; }
tui_success() { :; }
ensure_ollama_installed() { LOCAL_CALLS+=(install); }
start_ollama_service() { LOCAL_CALLS+=(start); }
pull_model() { LOCAL_CALLS+=("pull:$1"); }

OLLAMA_HOST="https://192.168.1.20:11434"
setup_ollama qwen3.8:27b
[[ "$LAST_CURL_URL" == "https://192.168.1.20:11434/api/tags" ]]
[[ ${#LOCAL_CALLS[@]} -eq 0 ]]

ERRORS=""
if setup_ollama missing-model; then
    echo "expected missing remote model to fail" >&2
    exit 1
fi
[[ "$ERRORS" == *"not found on remote Ollama"* ]]
[[ ${#LOCAL_CALLS[@]} -eq 0 ]]

REMOTE_PAYLOAD='invalid-json'
ERRORS=""
if setup_ollama; then
    echo "expected invalid remote response to fail" >&2
    exit 1
fi
[[ "$ERRORS" == *"invalid model list"* ]]

unset OLLAMA_HOST
LOCAL_CALLS=()
setup_ollama qwen3.8:27b
[[ "${LOCAL_CALLS[*]}" == "install start pull:qwen3.8:27b" ]]
