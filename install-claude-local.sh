#!/usr/bin/env bash
# Curl-safe bootstrap for the claude-local harness.
set -euo pipefail

REPO_URL="${PUSHBUTTON_REPO_URL:-https://github.com/StewartSethA/PushbuttonLocalCoders.git}"
REF="${PUSHBUTTON_REF:-main}"
INSTALL_ROOT="${PUSHBUTTON_DIR:-$HOME/.local/share/pushbutton}"
DEST="$INSTALL_ROOT/PushbuttonLocalCoders"

say() { printf '\033[1;34m[pushbutton]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[pushbutton]\033[0m %s\n' "$*" >&2; exit 1; }

ensure_git() {
    command -v git >/dev/null 2>&1 && return 0
    say "git not found; installing it..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y
        sudo apt-get install -y git ca-certificates
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y git ca-certificates
    else
        die "git is required and could not be installed automatically"
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

[[ -x "$DEST/claude-local" ]] || chmod +x "$DEST/claude-local"

# With no arguments, install the command and print help. With arguments, run
# the harness immediately so the curl one-liner doubles as a fresh-machine test.
if [[ $# -eq 0 ]]; then
    "$DEST/claude-local" install
    exec "$DEST/claude-local" --help
fi

exec "$DEST/claude-local" "$@"
