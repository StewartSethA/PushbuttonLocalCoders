#!/usr/bin/env bash
# install_ollama.sh — Install Ollama on Linux or Mac, then pull the chosen model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_hardware.sh"
source "$SCRIPT_DIR/tui.sh"

OLLAMA_MIN_VERSION="0.3.0"

# ── Version helpers ────────────────────────────────────────────────────────────
version_gte() {
    # Returns 0 (true) if $1 >= $2 in semver
    local a b
    IFS='.' read -ra a <<< "$1"
    IFS='.' read -ra b <<< "$2"
    local i
    for (( i=0; i<3; i++ )); do
        local av=${a[$i]:-0}
        local bv=${b[$i]:-0}
        (( av > bv )) && return 0
        (( av < bv )) && return 1
    done
    return 0
}

# ── Installation ───────────────────────────────────────────────────────────────
install_ollama_linux() {
    tui_step "Installing Ollama for Linux…"
    if command -v curl &>/dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh
    elif command -v wget &>/dev/null; then
        wget -qO- https://ollama.com/install.sh | sh
    else
        tui_error "Neither curl nor wget found. Please install one and retry."
        return 1
    fi
}

install_ollama_mac() {
    tui_step "Installing Ollama for macOS…"
    if command -v brew &>/dev/null; then
        brew install ollama
    else
        tui_info "Homebrew not found — downloading Ollama.app directly…"
        local tmp_dmg
        tmp_dmg=$(mktemp /tmp/ollama-XXXXXX.dmg)
        curl -fsSL "https://ollama.com/download/Ollama-darwin.dmg" -o "$tmp_dmg"
        hdiutil attach "$tmp_dmg" -quiet
        cp -R /Volumes/Ollama/Ollama.app /Applications/
        hdiutil detach /Volumes/Ollama -quiet
        rm -f "$tmp_dmg"
        # Add CLI to PATH
        if [[ ! -f /usr/local/bin/ollama ]]; then
            ln -sf /Applications/Ollama.app/Contents/Resources/ollama /usr/local/bin/ollama || true
        fi
    fi
}

ensure_ollama_installed() {
    local os
    os=$(detect_os)

    if command -v ollama &>/dev/null; then
        local current_ver
        current_ver=$(ollama --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1 || echo "0.0.0")
        if version_gte "$current_ver" "$OLLAMA_MIN_VERSION"; then
            tui_success "Ollama $current_ver already installed ✓"
            return 0
        else
            tui_warn "Ollama $current_ver is older than required $OLLAMA_MIN_VERSION — upgrading…"
        fi
    fi

    case "$os" in
        linux)   install_ollama_linux ;;
        mac)     install_ollama_mac   ;;
        windows) tui_error "Windows automatic install not yet supported. Download from https://ollama.com/download" ; return 1 ;;
        *)       tui_error "Unsupported OS: $os" ; return 1 ;;
    esac

    tui_success "Ollama installed ✓"
}

# ── Service management ─────────────────────────────────────────────────────────
start_ollama_service() {
    local os
    os=$(detect_os)

    # Check if already running
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
        tui_success "Ollama service already running ✓"
        return 0
    fi

    case "$os" in
        linux)
            if systemctl is-enabled --quiet ollama 2>/dev/null; then
                sudo systemctl start ollama
            else
                nohup ollama serve &>/tmp/ollama.log &
                disown
            fi
            ;;
        mac)
            # Ollama.app auto-starts; if using brew service:
            if command -v brew &>/dev/null; then
                brew services start ollama 2>/dev/null || nohup ollama serve &>/tmp/ollama.log & disown
            else
                open -a Ollama 2>/dev/null || nohup ollama serve &>/tmp/ollama.log & disown
            fi
            ;;
        *)
            nohup ollama serve &>/tmp/ollama.log &
            disown
            ;;
    esac

    # Wait for API to become available (up to 30 s)
    tui_step "Waiting for Ollama service…"
    local attempts=0
    until curl -sf http://localhost:11434/api/tags &>/dev/null; do
        sleep 1
        (( attempts++ ))
        if (( attempts > 30 )); then
            tui_error "Ollama did not start within 30 seconds. Check /tmp/ollama.log"
            return 1
        fi
    done
    tui_success "Ollama service running ✓"
}

# ── Model pull ─────────────────────────────────────────────────────────────────
pull_model() {
    local model_tag="$1"
    if ollama list 2>/dev/null | grep -q "^${model_tag//:/\:}"; then
        tui_success "Model $model_tag already present ✓"
        return 0
    fi

    tui_step "Pulling model: $model_tag (this may take a while)…"
    ollama pull "$model_tag"
    tui_success "Model $model_tag pulled ✓"
}

# ── Entrypoint ─────────────────────────────────────────────────────────────────
setup_ollama() {
    local model_tag="${1:-}"
    ensure_ollama_installed
    start_ollama_service
    if [[ -n "$model_tag" ]]; then
        pull_model "$model_tag"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_ollama "${1:-}"
fi
