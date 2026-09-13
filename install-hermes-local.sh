#!/usr/bin/env bash
# Curl-safe bootstrap for the hermes-local frontend.
set -euo pipefail

REPO_URL="${PUSHBUTTON_REPO_URL:-https://github.com/StewartSethA/PushbuttonLocalCoders.git}"
REF="${PUSHBUTTON_REF:-main}"
INSTALL_ROOT="${PUSHBUTTON_DIR:-$HOME/.local/share/pushbutton}"
DEST="$INSTALL_ROOT/PushbuttonLocalCoders"
SYSTEM_INSTALL=0

say() { printf '\033[1;35m[pushbutton]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[pushbutton]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[pushbutton]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ "${1:-}" == "--system" ]]; then
    SYSTEM_INSTALL=1
    shift
fi

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

install_commands() {
    mkdir -p "$HOME/.local/bin"
    ln -sfn "$DEST/hermes-local" "$HOME/.local/bin/hermes-local"
    if (( SYSTEM_INSTALL )); then
        sudo ln -sfn "$DEST/hermes-local" /usr/local/bin/hermes-local
        say "Installed system command: /usr/local/bin/hermes-local"
    else
        say "Installed user command: $HOME/.local/bin/hermes-local"
    fi
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

chmod +x "$DEST/hermes-local"
install_commands

if [[ $# -eq 0 ]]; then
    exec "$DEST/hermes-local" --help
fi

exec "$DEST/hermes-local" "$@"
