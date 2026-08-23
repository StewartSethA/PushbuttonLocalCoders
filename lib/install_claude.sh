#!/usr/bin/env bash
# install_claude.sh — Install Claude Code and bridge it to a local Ollama model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_hardware.sh"
source "$SCRIPT_DIR/tui.sh"

PUSHBUTTON_CONFIG_DIR="${PUSHBUTTON_CONFIG_DIR:-$HOME/.config/pushbutton}"
CLAUDE_GATEWAY_PORT="${PUSHBUTTON_CLAUDE_GATEWAY_PORT:-4000}"
CLAUDE_GATEWAY_HOST="${PUSHBUTTON_CLAUDE_GATEWAY_HOST:-127.0.0.1}"
CLAUDE_GATEWAY_MODEL="${PUSHBUTTON_CLAUDE_GATEWAY_MODEL:-pushbutton-local}"
LITELLM_PYPI_SPEC="${LITELLM_PYPI_SPEC:-litellm[proxy]==1.98.0}"

# ── Installation ───────────────────────────────────────────────────────────────
install_claude_cli() {
    local os
    os=$(detect_os)

    tui_step "Installing Claude Code…"

    # The official Anthropic CLI is distributed via npm
    if ! command -v npm &>/dev/null; then
        tui_info "npm not found — installing Node.js first…"
        install_nodejs "$os"
    fi

    npm install -g @anthropic-ai/claude-code 2>/dev/null || true
    export PATH="$(npm prefix -g 2>/dev/null)/bin:$PATH"

    if command -v claude &>/dev/null; then
        tui_success "Claude Code installed ✓"
        return 0
    fi

    tui_warn "Claude Code not found in PATH after install attempt."
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

    local config_file="$PUSHBUTTON_CONFIG_DIR/claude.env"
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

    if [[ "${PUSHBUTTON_PROMPT_FOR_ANTHROPIC_KEY:-0}" == "1" ]] && [[ -r /dev/tty ]]; then
        printf "\n" > /dev/tty
        printf "Enter your Anthropic API key (leave blank to skip):\n" > /dev/tty
        read -r -s ANTHROPIC_API_KEY < /dev/tty
        printf "\n" > /dev/tty
        if [[ -n "$ANTHROPIC_API_KEY" ]]; then
            echo "ANTHROPIC_API_KEY=\"$ANTHROPIC_API_KEY\"" > "$config_file"
            chmod 600 "$config_file"
            export ANTHROPIC_API_KEY
            tui_success "API key saved to $config_file"
            return 0
        fi
    fi

    tui_info "No Anthropic API key configured; local Claude Code sessions will use the Ollama bridge."
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

ensure_litellm_proxy() {
    if command -v python3 &>/dev/null; then
        export PATH="$(python3 -m site --user-base 2>/dev/null)/bin:$PATH"
    fi

    if command -v litellm &>/dev/null; then
        tui_success "LiteLLM proxy available ✓"
        return 0
    fi

    if ! command -v python3 &>/dev/null; then
        tui_warn "python3 not found; cannot install LiteLLM bridge for local Claude Code."
        return 1
    fi

    if ! command -v pip3 &>/dev/null; then
        tui_warn "pip3 not found; cannot install LiteLLM bridge for local Claude Code."
        return 1
    fi

    tui_step "Installing LiteLLM proxy…"
    pip3 install --user "$LITELLM_PYPI_SPEC" 2>/dev/null || pip3 install "$LITELLM_PYPI_SPEC" 2>/dev/null || {
        tui_warn "LiteLLM install failed; falling back to direct Ollama chat."
        return 1
    }

    export PATH="$(python3 -m site --user-base 2>/dev/null)/bin:$PATH"

    if command -v litellm &>/dev/null; then
        tui_success "LiteLLM proxy installed ✓"
        return 0
    fi

    tui_warn "LiteLLM installed but not found in PATH."
    return 1
}

write_local_claude_gateway_config() {
    local model_tag="${1:?model_tag required}"
    local gateway_config="$PUSHBUTTON_CONFIG_DIR/litellm.local.yaml"
    local ollama_host="${OLLAMA_HOST:-http://127.0.0.1:11434}"

    mkdir -p "$PUSHBUTTON_CONFIG_DIR"
    cat > "$gateway_config" <<EOF
model_list:
  - model_name: $CLAUDE_GATEWAY_MODEL
    litellm_params:
      model: ollama_chat/$model_tag
      api_base: ${ollama_host%/}
EOF

    echo "$gateway_config"
}

start_local_claude_gateway() {
    local model_tag="${1:?model_tag required}"
    local pid_file="$PUSHBUTTON_CONFIG_DIR/litellm.pid"
    local log_file="$PUSHBUTTON_CONFIG_DIR/litellm.log"
    local base_url="http://$CLAUDE_GATEWAY_HOST:$CLAUDE_GATEWAY_PORT"
    local gateway_config
    gateway_config="$(write_local_claude_gateway_config "$model_tag")"

    ensure_litellm_proxy || return 1

    if [[ -f "$pid_file" ]]; then
        local existing_pid
        existing_pid="$(<"$pid_file")"
        if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
            kill "$existing_pid" 2>/dev/null || true
            local shutdown_attempts=0
            while kill -0 "$existing_pid" 2>/dev/null; do
                sleep 1
                (( shutdown_attempts++ ))
                if (( shutdown_attempts > 5 )); then
                    tui_warn "Timed out waiting for the previous LiteLLM bridge to stop."
                    break
                fi
            done
        fi
        rm -f "$pid_file"
    fi

    tui_step "Starting local Claude Code gateway on $base_url …"
    nohup litellm --config "$gateway_config" --host "$CLAUDE_GATEWAY_HOST" --port "$CLAUDE_GATEWAY_PORT" \
        >"$log_file" 2>&1 &
    echo "$!" > "$pid_file"

    local attempts=0
    until curl -sf "$base_url/health/liveliness" &>/dev/null || curl -sf "$base_url/health" &>/dev/null; do
        sleep 1
        (( attempts++ ))
        if (( attempts > 30 )); then
            tui_warn "Local Claude Code gateway did not start. Check $log_file"
            return 1
        fi
    done

    tui_success "Local Claude Code gateway running ✓"
}

launch_interactive_claude_session() {
    local model_tag="${1:?model_tag required}"
    local session_dir="${2:-$PWD}"
    local base_url="http://$CLAUDE_GATEWAY_HOST:$CLAUDE_GATEWAY_PORT"

    if ! command -v claude &>/dev/null; then
        tui_warn "Claude Code is unavailable; falling back to an interactive Ollama session."
        if [[ -r /dev/tty ]]; then
            (cd "$session_dir" && exec ollama run "$model_tag" < /dev/tty > /dev/tty 2> /dev/tty)
            return 0
        fi
        return 1
    fi

    if ! [[ -r /dev/tty ]]; then
        tui_warn "No controlling TTY found; skipping interactive Claude Code launch."
        return 1
    fi

    if ! start_local_claude_gateway "$model_tag"; then
        tui_warn "Falling back to a direct interactive Ollama session."
        (cd "$session_dir" && exec ollama run "$model_tag" < /dev/tty > /dev/tty 2> /dev/tty)
        return 0
    fi

    tui_header "Starting Claude Code"
    tui_info "Workspace: $session_dir"
    tui_info "Model: $model_tag (via LiteLLM → Ollama bridge)"

    (
        cd "$session_dir"
        export ANTHROPIC_BASE_URL="$base_url"
        export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-pushbutton-local}"
        export ANTHROPIC_AUTH_TOKEN=""
        exec claude --model "$CLAUDE_GATEWAY_MODEL" < /dev/tty > /dev/tty 2> /dev/tty
    )
}

setup_claude() {
    install_claude_cli
    configure_claude_api_key
    ensure_litellm_proxy || true
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_claude
fi
