#!/usr/bin/env bash
set -euo pipefail
REPO_URL="${PUSHBUTTON_REPO_URL:-https://github.com/StewartSethA/PushbuttonLocalCoders.git}"
REF="${PUSHBUTTON_REF:-main}"
ROOT="${PUSHBUTTON_DIR:-$HOME/.local/share/pushbutton}"
DEST="$ROOT/PushbuttonLocalCoders"
SYSTEM=0
[[ "${1:-}" == --system ]] && { SYSTEM=1; shift; }
SELECT=0
[[ "${1:-}" == --select ]] && { SELECT=1; shift; }
INSTALL_ONLY="${PUSHBUTTON_INSTALL_ONLY:-0}"
[[ "${1:-}" == --install-only ]] && { INSTALL_ONLY=1; shift; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
mkdir -p "$ROOT"
if [[ -d "$DEST/.git" ]]; then
  git -C "$DEST" fetch --depth=1 origin "$REF"
  git -C "$DEST" checkout -q --detach FETCH_HEAD
else
  git clone --depth=1 --branch "$REF" "$REPO_URL" "$DEST"
fi
chmod +x \
  "$DEST/pushbutton" "$DEST/pushbutton-instance" "$DEST/pushbutton-broker" "$DEST/pushbutton-proxy" \
  "$DEST/coder-local" "$DEST/qwen-local" "$DEST/opencode-local" "$DEST/deepseek-local" "$DEST/mini-swe-local" \
  "$DEST/pushbutton-backend" "$DEST/pushbutton-bench" "$DEST/pushbutton-select" "$DEST/pushbutton-observe"

install_wrapper(){
  local target="$1" body="$2" tmp="$ROOT/.coder-shim.$$"
  printf '#!/usr/bin/env bash\nexec %s "$@"\n' "$body" >"$tmp"
  chmod +x "$tmp"
  if [[ "$target" == /usr/local/bin/* ]]; then sudo install -m755 "$tmp" "$target"; else mkdir -p "$(dirname "$target")"; mv "$tmp" "$target"; fi
  rm -f "$tmp" 2>/dev/null || true
}

qwen_defaults="$(printf '%q' "$DEST/configs/qwen-local-defaults.json")"
qwen_body="env QWEN_CODE_SYSTEM_DEFAULTS_PATH=$qwen_defaults $(printf '%q' "$DEST/qwen-local")"
opencode_body="$(printf '%q' "$DEST/opencode-local")"
deepseek_body="$(printf '%q' "$DEST/deepseek-local")"
mini_body="$(printf '%q' "$DEST/mini-swe-local")"
pushbutton_body="$(printf '%q' "$DEST/pushbutton")"
instance_body="$(printf '%q' "$DEST/pushbutton-instance")"
broker_body="$(printf '%q' "$DEST/pushbutton-broker")"
proxy_body="$(printf '%q' "$DEST/pushbutton-proxy")"
backend_body="$(printf '%q' "$DEST/pushbutton-backend")"
bench_body="$(printf '%q' "$DEST/pushbutton-bench")"
select_body="$(printf '%q' "$DEST/pushbutton-select")"
observe_body="$(printf '%q' "$DEST/pushbutton-observe")"

for spec in \
 "$HOME/.local/bin/pushbutton|$pushbutton_body" \
 "$HOME/.local/bin/pushbutton-instance|$instance_body" \
 "$HOME/.local/bin/pushbutton-broker|$broker_body" \
 "$HOME/.local/bin/pushbutton-proxy|$proxy_body" \
 "$HOME/.local/bin/qwen-local|$qwen_body" \
 "$HOME/.local/bin/opencode-local|$opencode_body" \
 "$HOME/.local/bin/deepseek-local|$deepseek_body" \
 "$HOME/.local/bin/mini-swe-local|$mini_body" \
 "$HOME/.local/bin/pushbutton-backend|$backend_body" \
 "$HOME/.local/bin/pushbutton-bench|$bench_body" \
 "$HOME/.local/bin/pushbutton-select|$select_body" \
 "$HOME/.local/bin/pushbutton-observe|$observe_body"; do
  install_wrapper "${spec%%|*}" "${spec#*|}"
done

if ((SYSTEM)); then
  install_wrapper /usr/local/bin/pushbutton "$pushbutton_body"
  install_wrapper /usr/local/bin/pushbutton-instance "$instance_body"
  install_wrapper /usr/local/bin/pushbutton-broker "$broker_body"
  install_wrapper /usr/local/bin/pushbutton-proxy "$proxy_body"
  install_wrapper /usr/local/bin/qwen-local "$qwen_body"
  install_wrapper /usr/local/bin/opencode-local "$opencode_body"
  install_wrapper /usr/local/bin/deepseek-local "$deepseek_body"
  install_wrapper /usr/local/bin/mini-swe-local "$mini_body"
  install_wrapper /usr/local/bin/pushbutton-backend "$backend_body"
  install_wrapper /usr/local/bin/pushbutton-bench "$bench_body"
  install_wrapper /usr/local/bin/pushbutton-select "$select_body"
  install_wrapper /usr/local/bin/pushbutton-observe "$observe_body"
fi

printf '[pushbutton] Installed unified runtime: pushbutton\n'
printf '[pushbutton] Expert frontends/tools remain available: qwen-local, opencode-local, deepseek-local, mini-swe-local, pushbutton-select, pushbutton-bench, pushbutton-observe\n'

export QWEN_CODE_SYSTEM_DEFAULTS_PATH="$DEST/configs/qwen-local-defaults.json"
if ((INSTALL_ONLY)); then exit 0; fi
if ((SELECT)); then exec "$DEST/pushbutton-select" "$@"; fi
exec "$DEST/pushbutton" "$@"
