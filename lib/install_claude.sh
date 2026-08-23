#!/usr/bin/env bash
# install_claude.sh — Install the Anthropic Claude CLI (claude) and configure it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_hardware.sh"
source "$SCRIPT_DIR/tui.sh"

# ── Installation ───────────────────────────────────────────────────────────────
install_claude_cli() {
    local os
    os=$(detect_os)

    tui_step "Installing Claude CLI…"

    # The official Anthropic CLI is distributed via npm
    if ! command -v npm &>/dev/null; then
        tui_info "npm not found — installing Node.js first…"
        install_nodejs "$os"
    fi

    npm install -g @anthropic-ai/claude-cli 2>/dev/null || \
        npm install -g claude 2>/dev/null || \
        pip3 install claude-cli 2>/dev/null || true

    if command -v claude &>/dev/null; then
        tui_success "Claude CLI installed ✓"
        return 0
    fi

    # Fallback: pip-based claude-code or anthropic SDK wrapper
    tui_info "Trying pip fallback for claude…"
    pip3 install --quiet anthropic 2>/dev/null || true
    tui_warn "Claude CLI not found in PATH. Ensure ANTHROPIC_API_KEY is set and 'claude' is in PATH."
}

install_nodejs() {
    local os="$1"
    case "$os" in
        linux)
            if command -v apt-get &>/dev/null; then
                curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - 2>/dev/null
                sudo apt-get install -y nodejs 2>/dev/null
            elif command -v dnf &>/dev/null; then
                sudo dnf module install nodejs:lts -y 2>/dev/null
            elif command -v brew &>/dev/null; then
                brew install node
            else
                tui_warn "Cannot auto-install Node.js. Visit https://nodejs.org"
            fi
            ;;
        mac)
            if command -v brew &>/dev/null; then
                brew install node
            else
                tui_warn "Homebrew not found. Visit https://nodejs.org to install Node.js"
            fi
            ;;
    esac
}

# ── API key configuration ──────────────────────────────────────────────────────
configure_claude_api_key() {
    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        tui_success "ANTHROPIC_API_KEY already set ✓"
        return 0
    fi

    local config_file="$HOME/.config/pushbutton/claude.env"
    mkdir -p "$(dirname "$config_file")"

    if [[ -f "$config_file" ]]; then
        # shellcheck source=/dev/null
        source "$config_file"
        if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
            export ANTHROPIC_API_KEY
            tui_success "ANTHROPIC_API_KEY loaded from $config_file ✓"
            return 0
        fi
    fi

    if [[ -t 0 ]]; then
        echo ""
        tui_info "Enter your Anthropic API key (leave blank to skip):"
        read -r -s ANTHROPIC_API_KEY
        echo ""
        if [[ -n "$ANTHROPIC_API_KEY" ]]; then
            echo "ANTHROPIC_API_KEY=\"$ANTHROPIC_API_KEY\"" > "$config_file"
            chmod 600 "$config_file"
            export ANTHROPIC_API_KEY
            tui_success "API key saved to $config_file"
        else
            tui_warn "No API key provided — Claude cloud features will be unavailable."
        fi
    else
        tui_warn "Non-interactive session and no ANTHROPIC_API_KEY set. Skipping claude config."
    fi
}

# ── Cloud model aliases ────────────────────────────────────────────────────────
# Returns the best available Claude model for the given role.
# Roles: orchestrator | developer | reviewer
claude_model_for_role() {
    local role="${1:-orchestrator}"
    # Prefer the latest available model; fall back gracefully
    local default_model="claude-opus-4-5"
    case "$role" in
        orchestrator) echo "${CLAUDE_ORCHESTRATOR_MODEL:-claude-opus-4-5}" ;;
        developer)    echo "${CLAUDE_DEVELOPER_MODEL:-claude-sonnet-4-5}"  ;;
        reviewer)     echo "${CLAUDE_REVIEWER_MODEL:-claude-haiku-4-5}"    ;;
        *)            echo "$default_model" ;;
    esac
}

setup_claude() {
    install_claude_cli
    configure_claude_api_key
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_claude
fi
