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
# Modes:
#   --quick          Just get me running (default): install Ollama + Claude CLI,
#                    pull best-fit coder model.
#   --explore        Explore better/faster models: run hardware ablation to find
#                    the optimal model and quantisation for this machine.
#   --agent          Wrap a project directory in a sandboxed Docker agent team.
#   --monitor        Launch live GPU/CPU/node monitor TUI.
#   --build-llamacpp Build llama.cpp with GPU optimisations.
#   --help           Show this help.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_URL="https://github.com/StewartSethA/PushbuttonLocalCoders.git"
INSTALL_DIR="${PUSHBUTTON_DIR:-$HOME/.local/share/pushbutton}"
CONFIG_DIR="$HOME/.config/pushbutton"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Determine lib directory ────────────────────────────────────────────────────
# When run via curl pipe the script is downloaded to a tmp file without the lib/
# directory next to it, so we clone the repo first.
bootstrap_repo() {
    if [[ -d "$SCRIPT_DIR/lib" ]]; then
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

print_help() {
    cat <<HELP
${BOLD}PushbuttonLocalCoders${RESET} — Local AI coding assistant bootstrap

Usage: install.sh [MODE] [OPTIONS]

Modes:
  --quick              (default) Install Ollama + Claude CLI, pull best model
  --explore            Run hardware ablation to find optimal model/quant
  --agent              Wrap project in Docker agent sandbox
  --team               Launch multi-agent team via Docker Compose
  --monitor            Live GPU/CPU/node monitor
  --build-llamacpp     Build llama.cpp with GPU/CPU optimisations
  --benchmark          Run an Ollama speed benchmark and save the runtime profile
  --nodes              Network node monitor (add/list/live)
  --orchestrator       Start local orchestrator + developer agents
  --help               Show this help

Options:
  --project  <dir>     Project directory for agent/team mode
  --task     <text>    Task description for agent/team mode
  --devs     <n>       Number of developer agents (default: 1)
  --interval <s>       Monitor refresh interval in seconds (default: 2)
  --model    <tag>     Override model tag (Ollama format)
  --framework <name>   Benchmark framework: ollama or ollama-cpu
  --skip-benchmark     Skip the post-setup benchmark prompt
  --context-sweep      Run the longer context sweep after the quick benchmark

Environment:
  ANTHROPIC_API_KEY    Anthropic API key for Claude cloud features
  OLLAMA_HOST          Ollama API host (default: http://localhost:11434)
  DEVELOPER_MODEL      Override developer model
  ORCHESTRATOR_MODEL   Override orchestrator model

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
        --benchmark)       if [[ "$MODE_EXPLICIT" == false ]]; then
                               MODE="benchmark"
                               MODE_EXPLICIT=true
                           else
                               RUN_BENCHMARK="yes"
                           fi
                           shift ;;
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
        --help|-h)         print_help ; exit 0   ;;
        --project)         PROJECT_DIR="$2"      ; shift 2 ;;
        --task)            TASK="$2"             ; shift 2 ;;
        --devs)            NUM_DEVS="$2"         ; shift 2 ;;
        --interval)        MONITOR_INTERVAL="$2" ; shift 2 ;;
        --model)           SELECTED_MODEL="$2"   ; shift 2 ;;
        --framework)       BENCHMARK_FRAMEWORK="$2" ; shift 2 ;;
        --skip-benchmark)  RUN_BENCHMARK="no"       ; shift ;;
        --context-sweep)   BENCHMARK_CONTEXT_SWEEP="yes" ; shift ;;
        *)                 tui_warn "Unknown option: $1" ; shift ;;
    esac
done

# ── Mode implementations ───────────────────────────────────────────────────────

mode_quick() {
    tui_header "PushbuttonLocalCoders — Quick Setup"

    # 1. Detect hardware and pick best model
    print_model_recommendation
    eval "$(detect_inference_memory)"

    if [[ -z "${SELECTED_MODEL:-}" ]]; then
        auto_select_model
    fi

    tui_info "Selected model: $SELECTED_MODEL"

    # 2. Install Ollama and pull model
    setup_ollama "$SELECTED_MODEL"

    # 3. Install Claude CLI
    setup_claude

    # 4. Offer a quick speed benchmark and persist the runtime profile
    maybe_run_post_setup_benchmark "$SELECTED_MODEL" "$BENCHMARK_FRAMEWORK" "$RUN_BENCHMARK" "$BENCHMARK_CONTEXT_SWEEP"

    tui_header "Setup Complete"
    echo ""
    echo "  Run a query :  ollama run $SELECTED_MODEL \"Write a hello world in Python\""
    echo "  Monitor     :  bash $ENTRY_SCRIPT --monitor"
    echo "  Agent mode  :  bash $ENTRY_SCRIPT --agent --project /your/project --task 'Improve this code'"
    echo "  Explore     :  bash $ENTRY_SCRIPT --explore"
    echo "  Runtime env :  source $RUNTIME_ENV_FILE"
    echo ""
}

mode_explore() {
    tui_header "PushbuttonLocalCoders — Explore Mode"
    tui_info "Running hardware ablation to find optimal model/quant…"

    setup_ollama  # ensure Ollama is running
    run_hardware_ablation
}

mode_agent() {
    tui_header "PushbuttonLocalCoders — Agent Mode"
    [[ -z "$PROJECT_DIR" ]] && PROJECT_DIR="$(pwd)"
    [[ -z "$TASK"        ]] && TASK="Improve code quality and fix any issues"

    eval "$(detect_inference_memory)"
    if [[ -z "${SELECTED_MODEL:-}" ]]; then
        auto_select_model
    fi

    run_agent_sandbox "$PROJECT_DIR" "$TASK" "$SELECTED_MODEL"
}

mode_team() {
    tui_header "PushbuttonLocalCoders — Agent Team Mode"
    [[ -z "$PROJECT_DIR" ]] && PROJECT_DIR="$(pwd)"
    [[ -z "$TASK"        ]] && TASK="Develop and iterate on this codebase"

    start_agent_team "$PROJECT_DIR" "$TASK"
}

mode_monitor() {
    monitor_live "$MONITOR_INTERVAL"
}

mode_llamacpp() {
    tui_header "PushbuttonLocalCoders — Build llama.cpp"
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
    [[ -z "$TASK" ]] && TASK="Improve this codebase"
    run_orchestrator "$TASK" "$NUM_DEVS"
}

mode_benchmark() {
    tui_header "PushbuttonLocalCoders — Benchmark"
    if [[ -z "${SELECTED_MODEL:-}" ]]; then
        auto_select_model
    fi
    setup_ollama "$SELECTED_MODEL"
    maybe_run_post_setup_benchmark "$SELECTED_MODEL" "$BENCHMARK_FRAMEWORK" "yes" "$BENCHMARK_CONTEXT_SWEEP"
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
    *)
        tui_error "Unknown mode: $MODE"
        print_help
        exit 1
        ;;
esac
