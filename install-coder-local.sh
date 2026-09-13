#!/usr/bin/env bash
set -euo pipefail
REPO_URL="${PUSHBUTTON_REPO_URL:-https://github.com/StewartSethA/PushbuttonLocalCoders.git}"
REF="${PUSHBUTTON_REF:-main}"
ROOT="${PUSHBUTTON_DIR:-$HOME/.local/share/pushbutton}"
DEST="$ROOT/PushbuttonLocalCoders"
SYSTEM=0
[[ "${1:-}" == --system ]] && { SYSTEM=1; shift; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
mkdir -p "$ROOT"
if [[ -d "$DEST/.git" ]]; then
  git -C "$DEST" fetch --depth=1 origin "$REF"
  git -C "$DEST" checkout -q --detach FETCH_HEAD
else
  git clone --depth=1 --branch "$REF" "$REPO_URL" "$DEST"
fi
chmod +x "$DEST/coder-local" "$DEST/opencode-local" "$DEST/deepseek-local" "$DEST/mini-swe-local"

install_wrapper(){
  local target="$1" body="$2" tmp="$ROOT/.coder-shim.$$"
  printf '#!/usr/bin/env bash\nexec %s "$@"\n' "$body" >"$tmp"
  chmod +x "$tmp"
  if [[ "$target" == /usr/local/bin/* ]]; then
    sudo install -m755 "$tmp" "$target"
  else
    mkdir -p "$(dirname "$target")"
    mv "$tmp" "$target"
  fi
  rm -f "$tmp" 2>/dev/null || true
}

qwen_defaults="$(printf '%q' "$DEST/configs/qwen-local-defaults.json")"
qwen_body="env QWEN_CODE_SYSTEM_DEFAULTS_PATH=$qwen_defaults $(printf '%q' "$DEST/coder-local") --frontend qwen"
opencode_body="$(printf '%q' "$DEST/opencode-local")"
deepseek_body="$(printf '%q' "$DEST/deepseek-local")"
mini_body="$(printf '%q' "$DEST/mini-swe-local")"

install_wrapper "$HOME/.local/bin/qwen-local" "$qwen_body"
install_wrapper "$HOME/.local/bin/opencode-local" "$opencode_body"
install_wrapper "$HOME/.local/bin/deepseek-local" "$deepseek_body"
install_wrapper "$HOME/.local/bin/mini-swe-local" "$mini_body"

if ((SYSTEM)); then
  install_wrapper /usr/local/bin/qwen-local "$qwen_body"
  install_wrapper /usr/local/bin/opencode-local "$opencode_body"
  install_wrapper /usr/local/bin/deepseek-local "$deepseek_body"
  install_wrapper /usr/local/bin/mini-swe-local "$mini_body"
fi

printf '[pushbutton] Installed replica-aware coder frontends: qwen-local, opencode-local, deepseek-local, mini-swe-local\n'
printf '[pushbutton] Qwen Code web MCP: Exa search + fetch enabled by default\n'
if (($#)); then
  export QWEN_CODE_SYSTEM_DEFAULTS_PATH="$DEST/configs/qwen-local-defaults.json"
  exec "$DEST/coder-local" --frontend qwen "$@"
fi
export QWEN_CODE_SYSTEM_DEFAULTS_PATH="$DEST/configs/qwen-local-defaults.json"
exec "$DEST/coder-local" --frontend qwen --help
