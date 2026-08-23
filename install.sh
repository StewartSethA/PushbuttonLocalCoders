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
}

bootstrap_repo

# ── Load libraries ─────────────────────────────────────────────────────────────
source "$LIB_DIR/tui.sh"
source "$LIB_DIR/detect_hardware.sh"
source "$LIB_DIR/select_model.sh"
source "$LIB_DIR/install_ollama.sh"
source "$LIB_DIR/install_claude.sh"
source "$LIB_DIR/install_llamacpp.sh"
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
  --nodes              Network node monitor (add/list/live)
  --orchestrator       Start local orchestrator + developer agents
  --help               Show this help

Options:
  --project  <dir>     Project directory for agent/team mode
  --task     <text>    Task description for agent/team mode
  --devs     <n>       Number of developer agents (default: 1)
  --interval <s>       Monitor refresh interval in seconds (default: 2)
  --model    <tag>     Override model tag (Ollama format)

Environment:
  ANTHROPIC_API_KEY    Anthropic API key for Claude cloud features
  OLLAMA_HOST          Ollama API host (default: http://localhost:11434)
  DEVELOPER_MODEL      Override developer model
  ORCHESTRATOR_MODEL   Override orchestrator model

HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)           MODE="quick"         ; shift ;;
        --explore)         MODE="explore"        ; shift ;;
        --agent)           MODE="agent"          ; shift ;;
        --team)            MODE="team"           ; shift ;;
        --monitor)         MODE="monitor"        ; shift ;;
        --build-llamacpp)  MODE="llamacpp"       ; shift ;;
        --nodes)           MODE="nodes"          ; shift ;;
        --orchestrator)    MODE="orchestrator"   ; shift ;;
        --help|-h)         print_help ; exit 0   ;;
        --project)         PROJECT_DIR="$2"      ; shift 2 ;;
        --task)            TASK="$2"             ; shift 2 ;;
        --devs)            NUM_DEVS="$2"         ; shift 2 ;;
        --interval)        MONITOR_INTERVAL="$2" ; shift 2 ;;
        --model)           SELECTED_MODEL="$2"   ; shift 2 ;;
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

    tui_header "Setup Complete"
    echo ""
    echo "  Run a query :  ollama run $SELECTED_MODEL \"Write a hello world in Python\""
    echo "  Monitor     :  bash $0 --monitor"
    echo "  Agent mode  :  bash $0 --agent --project /your/project --task 'Improve this code'"
    echo "  Explore     :  bash $0 --explore"
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
    local sub="${1:-live}"
    case "$sub" in
        add)  add_node "${2:?IP required}"    ;;
        rm)   remove_node "${2:?IP required}" ;;
        list) read_nodes                       ;;
        *)    monitor_nodes_live "$MONITOR_INTERVAL" ;;
    esac
}

mode_orchestrator() {
    tui_header "PushbuttonLocalCoders — Local Orchestrator"
    [[ -z "$TASK" ]] && TASK="Improve this codebase"
    run_orchestrator "$TASK" "$NUM_DEVS"
}

# ── Dispatch ───────────────────────────────────────────────────────────────────
case "$MODE" in
    quick)        mode_quick       ;;
    explore)      mode_explore     ;;
    agent)        mode_agent       ;;
    team)         mode_team        ;;
    monitor)      mode_monitor     ;;
    llamacpp)     mode_llamacpp    ;;
    nodes)        mode_nodes "${2:-}" ;;
    orchestrator) mode_orchestrator ;;
    *)
        tui_error "Unknown mode: $MODE"
        print_help
        exit 1
        ;;
esac
