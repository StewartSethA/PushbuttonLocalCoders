#!/usr/bin/env bash
# Curl-safe bootstrap for the hermes-local frontend.
set -euo pipefail

REPO_URL="${PUSHBUTTON_REPO_URL:-https://github.com/StewartSethA/PushbuttonLocalCoders.git}"
REF="${PUSHBUTTON_REF:-main}"
INSTALL_ROOT="${PUSHBUTTON_DIR:-$HOME/.local/share/pushbutton}"
DEST="$INSTALL_ROOT/PushbuttonLocalCoders"
SYSTEM_INSTALL=0
SELECT=0

say() { printf '\033[1;35m[pushbutton]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[pushbutton]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[pushbutton]\033[0m %s\n' "$*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        --system) SYSTEM_INSTALL=1; shift;;
        --select) SELECT=1; shift;;
        -h|--help) echo "Usage: install-hermes-local.sh [--system] [--select] [MODEL ...] [options]"; exit 0;;
        *) break;;
    esac
done
if ((SELECT)) && [[ ! -t 0 || ! -t 1 ]]; then
    die "run --select in an interactive terminal, or supply a MODEL without --select. Piped installs without arguments only install and print help."
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
    ln -sfn "$DEST/pushbutton-select" "$HOME/.local/bin/pushbutton-select"
    if (( SYSTEM_INSTALL )); then
        sudo ln -sfn "$DEST/hermes-local" /usr/local/bin/hermes-local
        sudo ln -sfn "$DEST/pushbutton-select" /usr/local/bin/pushbutton-select
        say "Installed system command: /usr/local/bin/hermes-local"
    else
        say "Installed user command: $HOME/.local/bin/hermes-local"
    fi
}

if [[ $# -eq 0 && ( ! -t 0 || ! -t 1 ) ]] && ! command -v git >/dev/null 2>&1; then
    die "git is required to install launcher assets; install git and rerun. No tools or models were installed."
fi
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

chmod +x "$DEST/hermes-local" "$DEST/pushbutton-select"
for asset in lib/pushbutton_metrics.py lib/coder_local_plan.py lib/claude_local_plan.py configs/backend-registry.json; do
    [[ -f "$DEST/$asset" ]] || die "missing selector asset: $asset"
done
install_commands

if ((SELECT)); then exec "$DEST/hermes-local" --select "$@"; fi
if [[ $# -eq 0 ]]; then
    if [[ -t 0 && -t 1 ]]; then exec "$DEST/hermes-local"; fi
    exec "$DEST/hermes-local" --help
fi

exec "$DEST/hermes-local" "$@"
