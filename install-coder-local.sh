#!/usr/bin/env bash
set -euo pipefail
REPO_URL="${PUSHBUTTON_REPO_URL:-https://github.com/StewartSethA/PushbuttonLocalCoders.git}"
REF="${PUSHBUTTON_REF:-main}"
ROOT="${PUSHBUTTON_DIR:-$HOME/.local/share/pushbutton}"
DEST="$ROOT/PushbuttonLocalCoders"
SYSTEM=0
[[ "${1:-}" == --system ]]&&{ SYSTEM=1;shift; }
command -v git >/dev/null||{ echo "git is required" >&2;exit 1; }
mkdir -p "$ROOT"
if [[ -d "$DEST/.git" ]];then git -C "$DEST" fetch --depth=1 origin "$REF";git -C "$DEST" checkout -q --detach FETCH_HEAD;else git clone --depth=1 --branch "$REF" "$REPO_URL" "$DEST";fi
chmod +x "$DEST/coder-local"
write_one(){ local p="$1" f="$2"; local tmp="$ROOT/.coder-shim.$$";cat >"$tmp" <<EOF
#!/usr/bin/env bash
exec $(printf '%q' "$DEST/coder-local") --frontend $(printf '%q' "$f") "\$@"
EOF
chmod +x "$tmp";if [[ "$p" == /usr/local/bin/* ]];then sudo install -m755 "$tmp" "$p";else mkdir -p "$(dirname "$p")";mv "$tmp" "$p";fi;rm -f "$tmp" 2>/dev/null||true; }
write_one "$HOME/.local/bin/qwen-local" qwen
((SYSTEM))&&write_one /usr/local/bin/qwen-local qwen
if (($#));then exec "$DEST/coder-local" --frontend qwen "$@";fi
exec "$DEST/coder-local" --frontend qwen --help
