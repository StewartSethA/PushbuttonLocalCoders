#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# PushbuttonLocalCoders — One-shot bootstrap installer
#
# Quick start (curl install):
#   curl -fsSL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install.sh | bash
#
# Or clone and run:
#   git clone https://github.com/StewartSethA/PushbuttonLocalCoders.git
#   cd PushbuttonLocalCoders && bash install.sh [options]
#
# Actions:
#   --quick          Just get me running (default): capability-discover, install
#                    local runtime, recommend models, then open a prompt shell.
#   --explore        Explore model/quant tradeoffs for this machine.
#   --agent          Wrap a project directory in a sandboxed Docker agent.
#   --monitor        Launch live GPU/CPU/node monitor TUI.
#   --build-llamacpp Build llama.cpp with platform-appropriate optimisations.
#   --submit-benchmarks  Prepare a benchmark contribution file for a PR.
#   --help           Show this help.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_URL="https://github.com/StewartSethA/PushbuttonLocalCoders.git"
INSTALL_DIR="${PUSHBUTTON_DIR:-$HOME/.local/share/pushbutton}"
CONFIG_DIR="$HOME/.config/pushbutton"
SCRIPT_DIR=""

if [[ ${BASH_SOURCE[0]+set} ]]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [[ -n "${0:-}" ]] && [[ -f "$0" ]]; then
    SCRIPT_PATH="$0"
fi

if [[ -n "${SCRIPT_PATH:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
fi

# ── Determine lib directory ────────────────────────────────────────────────────
# When run via curl pipe the script is downloaded to a tmp file without the lib/
# directory next to it, so we clone the repo first.
bootstrap_repo() {
    if [[ -n "$SCRIPT_DIR" ]] && [[ -d "$SCRIPT_DIR/lib" ]]; then
        LIB_DIR="$SCRIPT_DIR/lib"
        AGENTS_DIR="$SCRIPT_DIR/agents"
        ENTRY_SCRIPT="$SCRIPT_DIR/install.sh"
        return 0
    fi

    echo "Cloning PushbuttonLocalCoders to $INSTALL_DIR …"
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        git -C "$INSTALL_DIR" pull --ff-only 2>/dev/null || true
    else
        git clone --depth=1 "$REPO_URL" "$INSTALL_DIR" 2>/dev/null || {
            echo "ERROR: git clone failed. Ensure git is installed and you have internet access."
            exit 1
        }
    fi
    LIB_DIR="$INSTALL_DIR/lib"
    AGENTS_DIR="$INSTALL_DIR/agents"
    ENTRY_SCRIPT="$INSTALL_DIR/install.sh"
}

bootstrap_repo

# ── Load libraries ─────────────────────────────────────────────────────────────
source "$LIB_DIR/tui.sh"
source "$LIB_DIR/detect_hardware.sh"
source "$LIB_DIR/select_model.sh"
source "$LIB_DIR/install_ollama.sh"
source "$LIB_DIR/install_claude.sh"
source "$LIB_DIR/install_llamacpp.sh"
source "$LIB_DIR/benchmark.sh"
source "$LIB_DIR/docker_agent.sh"
source "$LIB_DIR/ablation.sh"
source "$LIB_DIR/network_nodes.sh"
source "$LIB_DIR/orchestrator.sh"
source "$LIB_DIR/apple_ablate.sh"

mkdir -p "$CONFIG_DIR"

# ── Argument parsing ───────────────────────────────────────────────────────────
MODE="quick"
PROJECT_DIR=""
TASK=""
NUM_DEVS=1
MONITOR_INTERVAL=2
NODES_SUBCMD="live"
NODES_ARG=""
RUN_BENCHMARK="ask"
BENCHMARK_CONTEXT_SWEEP="ask"
BENCHMARK_FRAMEWORK="ollama"
MODE_EXPLICIT=false
PROMPT_SHELL="${PROMPT_SHELL:-claude}"
SESSION_NETWORK_MODE="${SESSION_NETWORK_MODE:-host}"
TARGET_AGENT_REQUEST="${TARGET_AGENT_REQUEST:-}"
ENABLE_WEB_SEARCH="${ENABLE_WEB_SEARCH:-1}"

collect_requested_models() {
    local models=()
    local seen="|"
    local candidate

    for candidate in "${PRIMARY_CODER_MODEL:-}" "${ORCHESTRATOR_MODEL:-}"; do
        [[ -z "$candidate" ]] && continue
        if [[ "$seen" != *"|$candidate|"* ]]; then
            models+=("$candidate")
            seen="${seen}${candidate}|"
        fi
    done

    if [[ -n "${ADDITIONAL_CODER_MODELS:-}" ]]; then
        local extras=()
        IFS=',' read -r -a extras <<< "$ADDITIONAL_CODER_MODELS"
        for candidate in "${extras[@]}"; do
            [[ -z "$candidate" ]] && continue
            if [[ "$seen" != *"|$candidate|"* ]]; then
                models+=("$candidate")
                seen="${seen}${candidate}|"
            fi
        done
    fi

    if [[ "${ENABLE_CPU_MODELS:-0}" == "1" ]] && [[ -n "${CPU_FALLBACK_MODELS:-}" ]]; then
        local cpu_models=()
        IFS=',' read -r -a cpu_models <<< "$CPU_FALLBACK_MODELS"
        for candidate in "${cpu_models[@]}"; do
            [[ -z "$candidate" ]] && continue
            if [[ "$seen" != *"|$candidate|"* ]]; then
                models+=("$candidate")
                seen="${seen}${candidate}|"
            fi
        done
    fi

    printf '%s\n' "${models[@]}"
}

print_help() {
    cat <<HELP
${BOLD}PushbuttonLocalCoders${RESET} — Local AI coding assistant bootstrap

Usage: install.sh [ACTION] [OPTIONS]

Actions:
  --quick              (default) Install Ollama + Claude Code, bridge it to the local model, and launch it
  --explore            Run hardware ablation to find optimal model/quant
  --agent              Wrap project in Docker agent sandbox
  --team               Launch multi-agent team via Docker Compose
  --monitor            Live GPU/CPU/node monitor
  --build-llamacpp     Build llama.cpp with GPU/CPU optimisations
  --benchmark          Run an Ollama speed benchmark and save the runtime profile
  --nodes              Network node monitor (add/list/scan/live)
  --orchestrator       Start local orchestrator + developer agents
  --apple-ablate      Run the Apple Silicon multi-backend ablation lab
  --cpu-lab           Launch the extracted cpu-llama-lab toolkit
  --submit-benchmarks  Prepare a benchmark report file for PR submission
  --help               Show this help

Options:
  --project  <dir>     Project directory for agent/team mode
  --task     <text>    Task description for agent/team mode
  --devs     <n>       Number of developer agents (default: 1)
  --interval <s>       Monitor refresh interval in seconds (default: 2)
  --model    <tag>     Override model tag (Ollama format)
  --prompt-shell <id>  Prompt shell: claude (default), opencode, hermes
  --network-mode <id>  Session/agent sandbox network mode: host, bridge, none
  --request  <text>    Target a specific hardware/model request for agent startup
  --framework <name>   Benchmark framework: ollama or ollama-cpu
  --run-benchmark      Force the post-setup benchmark in quick mode
  --skip-benchmark     Skip the post-setup benchmark prompt
  --context-sweep      Run the longer context sweep after the quick benchmark

Environment:
  ANTHROPIC_API_KEY    Anthropic API key for Claude cloud features
  OLLAMA_HOST          Ollama API host (default: http://localhost:11434)
  DEVELOPER_MODEL      Override primary developer model
  ORCHESTRATOR_MODEL   Override orchestrator model
  PUSHBUTTON_CLAUDE_GATEWAY_PORT  LiteLLM bridge port (default: 4000)
  PUSHBUTTON_ACCEPT_MODEL_PLAN=1   Accept the shown plan non-interactively
  PUSHBUTTON_CLOUD_PROVIDERS       Cloud provider entries (name|url|auth|token;...)
  PUSHBUTTON_CLOUD_MODELS          Comma-separated cloud model IDs for agent inventory

HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)           MODE="quick"          ; MODE_EXPLICIT=true ; shift ;;
        --explore)         MODE="explore"        ; MODE_EXPLICIT=true ; shift ;;
        --agent)           MODE="agent"          ; MODE_EXPLICIT=true ; shift ;;
        --team)            MODE="team"           ; MODE_EXPLICIT=true ; shift ;;
        --monitor)         MODE="monitor"        ; MODE_EXPLICIT=true ; shift ;;
        --build-llamacpp)  MODE="llamacpp"       ; MODE_EXPLICIT=true ; shift ;;
        --benchmark)       MODE="benchmark"       ; MODE_EXPLICIT=true ; shift ;;
        --nodes)           MODE="nodes"          ; shift
                           MODE_EXPLICIT=true
                           # Capture optional subcommand (add/rm/list/live)
                           if [[ $# -gt 0 ]] && [[ "$1" != --* ]]; then
                               NODES_SUBCMD="$1"; shift
                               # Some subcommands take an extra argument (IP)
                               if [[ "$NODES_SUBCMD" =~ ^(add|rm)$ ]] && [[ $# -gt 0 ]] && [[ "$1" != --* ]]; then
                                   NODES_ARG="$1"; shift
                               fi
                           fi
                           ;;
        --orchestrator)    MODE="orchestrator"   ; MODE_EXPLICIT=true ; shift ;;
        --apple-ablate)    MODE="apple-ablate"   ; MODE_EXPLICIT=true ; shift ;;
        --cpu-lab)         MODE="cpu-lab"        ; MODE_EXPLICIT=true ; shift ;;
        --submit-benchmarks) MODE="submit-benchmarks" ; MODE_EXPLICIT=true ; shift ;;
        --help|-h)         print_help ; exit 0   ;;
        --project)         PROJECT_DIR="$2"      ; shift 2 ;;
        --task)            TASK="$2"             ; shift 2 ;;
        --devs)            NUM_DEVS="$2"         ; shift 2 ;;
        --interval)        MONITOR_INTERVAL="$2" ; shift 2 ;;
        --model)           SELECTED_MODEL="$2"   ; shift 2 ;;
        --prompt-shell)    PROMPT_SHELL="$2"     ; shift 2 ;;
        --network-mode)    SESSION_NETWORK_MODE="$2" ; shift 2 ;;
        --request)         TARGET_AGENT_REQUEST="$2" ; shift 2 ;;
        --framework)       BENCHMARK_FRAMEWORK="$2" ; shift 2 ;;
        --run-benchmark)   RUN_BENCHMARK="yes"      ; shift ;;
        --skip-benchmark)  RUN_BENCHMARK="no"       ; shift ;;
        --context-sweep)   BENCHMARK_CONTEXT_SWEEP="yes" ; shift ;;
        *)                 tui_warn "Unknown option: $1" ; shift ;;
    esac
done

# ── Mode implementations ───────────────────────────────────────────────────────

validate_network_mode() {
    case "${1:-host}" in
        host|bridge|none) return 0 ;;
        *) return 1 ;;
    esac
}

initialize_capability_profile() {
    eval "$(detect_hardware_profile)"
    tui_header "Hardware capability discovery"
    tui_info "OS=$HW_PROFILE_OS GPU=${HW_PROFILE_GPU_MODEL} (${HW_PROFILE_GPU_VENDOR}) VRAM=${HW_PROFILE_VRAM_GB}GB RAM=${HW_PROFILE_RAM_GB}GB"
    tui_info "CUDA available=${HW_PROFILE_CUDA_AVAILABLE} version=${HW_PROFILE_CUDA_VERSION} provider=${HW_PROFILE_CUDA_PROVIDER}"
    tui_info "Apple silicon=${HW_PROFILE_APPLE_SILICON} CPU caps=${HW_PROFILE_CPU_CAPABILITIES}"

    if [[ "$HW_PROFILE_GPU_VENDOR" == "nvidia" ]]; then
        tui_step "Ensuring usable CUDA toolchain (auto-provisioning local nvcc if needed)…"
        local nvcc_path=""
        nvcc_path="$(ensure_nvcc 2>/dev/null || true)"
        if [[ -n "$nvcc_path" ]]; then
            tui_success "CUDA ready: $nvcc_path"
        else
            tui_warn "CUDA toolchain unavailable; GPU-specific builds may fall back to CPU."
        fi
    fi

    if [[ "$HW_PROFILE_APPLE_SILICON" == "1" ]]; then
        export PUSHBUTTON_APPLE_OPTIMIZED=1
        tui_success "Apple Silicon optimizations enabled automatically."
    fi
}

launch_selected_prompt_session() {
    local model_tag="${1:?model required}"
    local session_dir="${2:-$PWD}"
    local shell_id="${PROMPT_SHELL:-claude}"

    export PUSHBUTTON_WEB_SEARCH="${ENABLE_WEB_SEARCH:-1}"

    case "$shell_id" in
        claude)
            launch_interactive_claude_session "$model_tag" "$session_dir"
            ;;
        opencode)
            if command -v opencode >/dev/null 2>&1; then
                (cd "$session_dir" && opencode < /dev/tty > /dev/tty 2> /dev/tty)
            else
                tui_warn "OpenCode not found; falling back to Claude Code."
                launch_interactive_claude_session "$model_tag" "$session_dir"
            fi
            ;;
        hermes)
            if command -v hermes >/dev/null 2>&1; then
                (cd "$session_dir" && hermes < /dev/tty > /dev/tty 2> /dev/tty)
            else
                tui_warn "Hermes CLI not found; falling back to Claude Code."
                launch_interactive_claude_session "$model_tag" "$session_dir"
            fi
            ;;
        *)
            tui_warn "Unknown prompt shell '$shell_id'; using claude."
            launch_interactive_claude_session "$model_tag" "$session_dir"
            ;;
    esac
}

if ! validate_network_mode "$SESSION_NETWORK_MODE"; then
    tui_error "Invalid --network-mode '$SESSION_NETWORK_MODE' (expected host|bridge|none)."
    exit 1
fi

mode_quick() {
    tui_header "PushbuttonLocalCoders — Quick Setup"
    initialize_capability_profile

    # 1. Detect hardware and confirm a model plan
    print_model_recommendation
    configure_model_plan || {
        tui_warn "Cancelled before installing models."
        return 0
    }

    tui_info "Primary coder: $PRIMARY_CODER_MODEL"

    # 2. Install Ollama and pull selected models
    local requested_models=()
    while IFS= read -r model_tag; do
        [[ -n "$model_tag" ]] && requested_models+=("$model_tag")
    done < <(collect_requested_models)
    setup_ollama "${requested_models[@]}"
    if [[ "${ENABLE_CPU_MODELS:-0}" == "1" ]]; then
        ensure_cpu_optimizations
    fi

    # 3. Record estimated vs actual PP/TG
    benchmark_selected_models "${requested_models[@]}"

    # 4. Install Claude CLI
    setup_claude

    # 5. Offer a quick speed benchmark and persist the runtime profile
    maybe_run_post_setup_benchmark "$SELECTED_MODEL" "$BENCHMARK_FRAMEWORK" "$RUN_BENCHMARK" "$BENCHMARK_CONTEXT_SWEEP"

    tui_header "Setup Complete"
    echo ""
    echo "  Claude Code :  ANTHROPIC_BASE_URL=http://${PUSHBUTTON_CLAUDE_GATEWAY_HOST:-127.0.0.1}:${PUSHBUTTON_CLAUDE_GATEWAY_PORT:-4000} claude --model ${PUSHBUTTON_CLAUDE_GATEWAY_MODEL:-pushbutton-local}"
    echo "  Prompt shell:  ${PROMPT_SHELL:-claude}"
    echo "  Run a query :  ollama run $SELECTED_MODEL \"Write a hello world in Python\""
    echo "  Monitor     :  bash $ENTRY_SCRIPT --monitor"
    echo "  Agent mode  :  bash $ENTRY_SCRIPT --agent --project /your/project --task 'Improve this code'"
    echo "  Explore     :  bash $ENTRY_SCRIPT --explore"
    echo "  Runtime env :  source $RUNTIME_ENV_FILE"
    echo ""

    launch_selected_prompt_session "$SELECTED_MODEL" "${PROJECT_DIR:-$PWD}" || true
}

mode_explore() {
    tui_header "PushbuttonLocalCoders — Explore Mode"
    initialize_capability_profile
    tui_info "Running modern PP/TG benchmarks and recording estimate accuracy…"

    setup_ollama  # ensure Ollama is running
    run_hardware_ablation
}

mode_agent() {
    tui_header "PushbuttonLocalCoders — Agent Mode"
    initialize_capability_profile
    [[ -z "$PROJECT_DIR" ]] && PROJECT_DIR="$(pwd)"
    [[ -z "$TASK"        ]] && TASK="Improve code quality and fix any issues"

    configure_model_plan || {
        tui_warn "Cancelled before installing models."
        return 0
    }

    local requested_models=()
    while IFS= read -r model_tag; do
        [[ -n "$model_tag" ]] && requested_models+=("$model_tag")
    done < <(collect_requested_models)
    setup_ollama "${requested_models[@]}"
    benchmark_selected_models "$PRIMARY_CODER_MODEL"
    if [[ "${ENABLE_CPU_MODELS:-0}" == "1" ]]; then
        ensure_cpu_optimizations
    fi

    local target_model="$PRIMARY_CODER_MODEL"
    if [[ -n "$TARGET_AGENT_REQUEST" ]]; then
        eval "$(resolve_model_target_request "$TARGET_AGENT_REQUEST" "$PRIMARY_CODER_MODEL")"
        target_model="$RESOLVED_AGENT_MODEL"
        tui_info "Resolved request '$TARGET_AGENT_REQUEST' → $target_model ($RESOLVED_AGENT_SOURCE)"
    fi
    run_agent_sandbox "$PROJECT_DIR" "$TASK" "$target_model" "$SESSION_NETWORK_MODE"
}

mode_team() {
    tui_header "PushbuttonLocalCoders — Agent Team Mode"
    initialize_capability_profile
    [[ -z "$PROJECT_DIR" ]] && PROJECT_DIR="$(pwd)"
    [[ -z "$TASK"        ]] && TASK="Develop and iterate on this codebase"

    configure_model_plan || {
        tui_warn "Cancelled before installing models."
        return 0
    }

    local requested_models=()
    while IFS= read -r model_tag; do
        [[ -n "$model_tag" ]] && requested_models+=("$model_tag")
    done < <(collect_requested_models)
    setup_ollama "${requested_models[@]}"
    benchmark_selected_models "${requested_models[@]}"
    if [[ "${ENABLE_CPU_MODELS:-0}" == "1" ]]; then
        ensure_cpu_optimizations
    fi

    start_agent_team "$PROJECT_DIR" "$TASK"
}

mode_monitor() {
    monitor_live "$MONITOR_INTERVAL"
}

mode_llamacpp() {
    tui_header "PushbuttonLocalCoders — Build llama.cpp"
    initialize_capability_profile
    build_llamacpp
}

mode_nodes() {
    case "$NODES_SUBCMD" in
        add)  add_node "${NODES_ARG:?IP required}"    ;;
        rm)   remove_node "${NODES_ARG:?IP required}" ;;
        list) read_nodes                               ;;
        scan) scan_nodes_models                        ;;
        *)    monitor_nodes_live "$MONITOR_INTERVAL"   ;;
    esac
}

mode_orchestrator() {
    tui_header "PushbuttonLocalCoders — Local Orchestrator"
    initialize_capability_profile
    [[ -z "$TASK" ]] && TASK="Improve this codebase"
    configure_model_plan || {
        tui_warn "Cancelled before installing models."
        return 0
    }
    local requested_models=()
    while IFS= read -r model_tag; do
        [[ -n "$model_tag" ]] && requested_models+=("$model_tag")
    done < <(collect_requested_models)
    setup_ollama "${requested_models[@]}"
    benchmark_selected_models "${requested_models[@]}"
    if [[ "${ENABLE_CPU_MODELS:-0}" == "1" ]]; then
        ensure_cpu_optimizations
    fi
    if [[ -n "$TARGET_AGENT_REQUEST" ]]; then
        eval "$(resolve_model_target_request "$TARGET_AGENT_REQUEST" "$DEVELOPER_MODEL")"
        DEVELOPER_MODEL="$RESOLVED_AGENT_MODEL"
        DEVELOPER_MODELS="$RESOLVED_AGENT_MODEL"
        tui_info "Resolved request '$TARGET_AGENT_REQUEST' → $DEVELOPER_MODEL ($RESOLVED_AGENT_SOURCE)"
    fi
    run_orchestrator "$TASK" "$NUM_DEVS"
}

mode_benchmark() {
    tui_header "PushbuttonLocalCoders — Benchmark"
    initialize_capability_profile
    if [[ -z "${SELECTED_MODEL:-}" ]]; then
        auto_select_model
    fi
    setup_ollama "$SELECTED_MODEL"
    maybe_run_post_setup_benchmark "$SELECTED_MODEL" "$BENCHMARK_FRAMEWORK" "yes" "$BENCHMARK_CONTEXT_SWEEP"
}

mode_submit_benchmarks() {
    tui_header "PushbuttonLocalCoders — Submit Benchmarks"
    prepare_system_benchmark_submission
}

mode_apple_ablate() {
    tui_header "PushbuttonLocalCoders — Apple Silicon Ablation"
    setup_apple_ablate "${APPLE_ABLATE_LAB_DIR:-$PWD/.apple-llm-lab}"
}

mode_cpu_lab() {
    tui_header "PushbuttonLocalCoders — CPU llama lab"
    bash "$SCRIPT_DIR/cpu-llama-lab/cpu-llama-lab.sh"
}

# ── Dispatch ───────────────────────────────────────────────────────────────────
case "$MODE" in
    quick)        mode_quick       ;;
    explore)      mode_explore     ;;
    agent)        mode_agent       ;;
    team)         mode_team        ;;
    monitor)      mode_monitor     ;;
    llamacpp)     mode_llamacpp    ;;
    benchmark)    mode_benchmark   ;;
    nodes)        mode_nodes        ;;
    orchestrator) mode_orchestrator ;;
    apple-ablate) mode_apple_ablate ;;
    cpu-lab)      mode_cpu_lab     ;;
    submit-benchmarks) mode_submit_benchmarks ;;
    *)
        tui_error "Unknown action: $MODE"
        print_help
        exit 1
        ;;
esac
