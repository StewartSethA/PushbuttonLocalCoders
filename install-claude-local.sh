#!/usr/bin/env bash
# Curl-safe bootstrap for the claude-local harness.
set -euo pipefail

REPO_URL="${PUSHBUTTON_REPO_URL:-https://github.com/StewartSethA/PushbuttonLocalCoders.git}"
REF="${PUSHBUTTON_REF:-main}"
INSTALL_ROOT="${PUSHBUTTON_DIR:-$HOME/.local/share/pushbutton}"
DEST="$INSTALL_ROOT/PushbuttonLocalCoders"
SYSTEM_INSTALL=0

say() { printf '\033[1;34m[pushbutton]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[pushbutton]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[pushbutton]\033[0m %s\n' "$*" >&2; exit 1; }

# Bootstrap-only option. This is consumed here rather than passed through to
# Claude Code. It may be used by itself (install only) or before model args.
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

write_shim() {
    local target="$1" tmp="$INSTALL_ROOT/.claude-local-shim.$$"
    mkdir -p "$INSTALL_ROOT"
    cat > "$tmp" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(printf '%q' "$DEST")
exec "\$ROOT/lib/claude_local_entry.sh" "\$@"
EOF
    chmod +x "$tmp"
    if [[ "$target" == /usr/local/bin/* ]]; then
        sudo install -m 0755 "$tmp" "$target"
        rm -f "$tmp"
    else
        mkdir -p "$(dirname "$target")"
        mv "$tmp" "$target"
    fi
}

install_command_shims() {
    # Always keep the per-user command available. --system additionally places
    # the same wrapper in /usr/local/bin.
    write_shim "$HOME/.local/bin/claude-local"
    if (( SYSTEM_INSTALL )); then
        write_shim /usr/local/bin/claude-local
        say "Installed system command: /usr/local/bin/claude-local"
    else
        say "Installed user command: $HOME/.local/bin/claude-local"
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
[[ -x "$DEST/lib/claude_local_entry.sh" ]] || chmod +x "$DEST/lib/claude_local_entry.sh"
install_command_shims

# With no remaining arguments, install/update the command and print help. With
# arguments, run the hardened entry wrapper immediately so the curl one-liner
# doubles as a fresh-machine test.
if [[ $# -eq 0 ]]; then
    exec "$DEST/lib/claude_local_entry.sh" --help
fi

exec "$DEST/lib/claude_local_entry.sh" "$@"
