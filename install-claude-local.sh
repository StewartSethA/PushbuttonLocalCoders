#!/usr/bin/env bash
# Curl-safe bootstrap for the claude-local harness.
set -euo pipefail

REPO_URL="${PUSHBUTTON_REPO_URL:-https://github.com/StewartSethA/PushbuttonLocalCoders.git}"
REF="${PUSHBUTTON_REF:-main}"
INSTALL_ROOT="${PUSHBUTTON_DIR:-$HOME/.local/share/pushbutton}"
DEST="$INSTALL_ROOT/PushbuttonLocalCoders"
SYSTEM_INSTALL=0
SELECT=0

say() { printf '\033[1;34m[pushbutton]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[pushbutton]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[pushbutton]\033[0m %s\n' "$*" >&2; exit 1; }

# Bootstrap-only option. This is consumed here rather than passed through to
# Claude Code. It may be used by itself (install only) or before model args.
while (($#)); do
    case "$1" in
        --system) SYSTEM_INSTALL=1; shift;;
        --select) SELECT=1; shift;;
        -h|--help) echo "Usage: install-claude-local.sh [--system] [--select] [MODEL ...] [Claude Code flags...]"; exit 0;;
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
    ln -sfn "$DEST/pushbutton-select" "$HOME/.local/bin/pushbutton-select"
    if (( SYSTEM_INSTALL )); then
        write_shim /usr/local/bin/claude-local
        sudo ln -sfn "$DEST/pushbutton-select" /usr/local/bin/pushbutton-select
        say "Installed system command: /usr/local/bin/claude-local"
    else
        say "Installed user command: $HOME/.local/bin/claude-local"
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

[[ -x "$DEST/claude-local" ]] || chmod +x "$DEST/claude-local"
[[ -x "$DEST/lib/claude_local_entry.sh" ]] || chmod +x "$DEST/lib/claude_local_entry.sh"
chmod +x "$DEST/pushbutton-select"
for asset in lib/pushbutton_metrics.py lib/coder_local_plan.py lib/claude_local_plan.py lib/pushbutton_capacity.py lib/pushbutton_request_budget.py lib/pushbutton_capacity_proxy.py configs/backend-registry.json; do
    [[ -f "$DEST/$asset" ]] || die "missing launcher asset: $asset"
done
install_command_shims

if ((SELECT)); then exec "$DEST/lib/claude_local_entry.sh" --select "$@"; fi
# A piped no-argument install must never read from an implicit controlling TTY.
if [[ $# -eq 0 ]]; then
    if [[ -t 0 && -t 1 ]]; then exec "$DEST/lib/claude_local_entry.sh"; fi
    exec "$DEST/lib/claude_local_entry.sh" --help
fi

exec "$DEST/lib/claude_local_entry.sh" "$@"
