#!/usr/bin/env bash
# Curl-safe bootstrap for the claude-local harness.
set -euo pipefail

REPO_URL="${PUSHBUTTON_REPO_URL:-https://github.com/StewartSethA/PushbuttonLocalCoders.git}"
REF="${PUSHBUTTON_REF:-main}"
INSTALL_ROOT="${PUSHBUTTON_DIR:-$HOME/.local/share/pushbutton}"
DEST="$INSTALL_ROOT/PushbuttonLocalCoders"

say() { printf '\033[1;34m[pushbutton]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[pushbutton]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[pushbutton]\033[0m %s\n' "$*" >&2; exit 1; }

ensure_git() {
    command -v git >/dev/null 2>&1 && return 0
    say "git not found; installing it..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y || warn "apt-get update reported errors; continuing with usable/cached indexes"
        sudo apt-get install -y git ca-certificates || true
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y git ca-certificates || true
    fi
    command -v git >/dev/null 2>&1 || die "git is required and could not be installed automatically"
}

install_user_shim() {
    local bindir="$HOME/.local/bin" shim="$HOME/.local/bin/claude-local"
    mkdir -p "$bindir"
    cat > "$shim" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(printf '%q' "$DEST")
need_hostcc=1
for arg in "\$@"; do
    case "\$arg" in --local-dry-run|doctor|models|help|-h|--help) need_hostcc=0 ;; esac
done
if (( need_hostcc )); then
    source "\$ROOT/lib/claude_local_hostcc.sh"
    claude_local_prepare_hostcc
fi
exec "\$ROOT/claude-local" "\$@"
EOF
    chmod +x "$shim"
}

prepare_hostcc_if_needed() {
    local need=1 arg
    for arg in "$@"; do
        case "$arg" in --local-dry-run|doctor|models|help|-h|--help) need=0 ;; esac
    done
    (( need )) || return 0
    [[ -f "$DEST/lib/claude_local_hostcc.sh" ]] || die "missing CUDA host-compiler bootstrap helper"
    # shellcheck source=/dev/null
    source "$DEST/lib/claude_local_hostcc.sh"
    claude_local_prepare_hostcc
}

ensure_git
mkdir -p "$INSTALL_ROOT"

if [[ -d "$DEST/.git" ]]; then
    say "Updating PushbuttonLocalCoders ($REF)..."
    git -C "$DEST" remote set-url origin "$REPO_URL"
    git -C "$DEST" fetch --depth=1 origin "$REF"
    git -C "$DEST" checkout -q --detach FETCH_HEAD
else
    say "Installing PushbuttonLocalCoders ($REF)..."
    rm -rf "$DEST.tmp"
    git clone --depth=1 --branch "$REF" "$REPO_URL" "$DEST.tmp"
    rm -rf "$DEST"
    mv "$DEST.tmp" "$DEST"
fi

[[ -x "$DEST/claude-local" ]] || chmod +x "$DEST/claude-local"
install_user_shim

# With no arguments, install/update the user command and print help. With
# arguments, prepare any host-compiler compatibility needed by CUDA and run the
# harness immediately, so the curl one-liner doubles as a fresh-machine test.
if [[ $# -eq 0 ]]; then
    say "Installed command: $HOME/.local/bin/claude-local"
    exec "$DEST/claude-local" --help
fi

prepare_hostcc_if_needed "$@"
exec "$DEST/claude-local" "$@"
