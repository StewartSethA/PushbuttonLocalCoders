#!/usr/bin/env bash
# orchestrator.sh — Multi-agent orchestrator setup.
# Spins up an orchestrator process and one or more developer agents,
# each backed by an appropriate local model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_hardware.sh"
source "$SCRIPT_DIR/select_model.sh"
source "$SCRIPT_DIR/tui.sh"

AGENT_PIDS=()
LOG_DIR="${LOG_DIR:-$HOME/.config/pushbutton/logs}"
mkdir -p "$LOG_DIR"
FIFO_DIR=""

cleanup() {
    tui_step "Shutting down agents…"
    for pid in "${AGENT_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    if [[ -n "$FIFO_DIR" ]] && [[ -d "$FIFO_DIR" ]]; then
        rm -rf "$FIFO_DIR"
    fi
    tui_success "All agents stopped."
}

register_orchestrator_cleanup() {
    trap cleanup EXIT INT TERM
}

# ── Determine models ───────────────────────────────────────────────────────────
resolve_agent_models() {
    eval "$(detect_inference_memory)"

    # Orchestrator gets a lighter, faster model
    ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-}"
    if [[ -z "$ORCHESTRATOR_MODEL" ]]; then
        auto_select_orchestrator
        ORCHESTRATOR_MODEL="$SELECTED_MODEL"
    fi

    # Developer agent(s) get the best-fit coder model
    DEVELOPER_MODEL="${DEVELOPER_MODEL:-}"
    if [[ -z "$DEVELOPER_MODEL" ]]; then
        auto_select_model
        DEVELOPER_MODEL="$SELECTED_MODEL"
    fi

    DEVELOPER_MODELS="${DEVELOPER_MODELS:-$DEVELOPER_MODEL}"

    export ORCHESTRATOR_MODEL DEVELOPER_MODEL DEVELOPER_MODELS
}

# ── Launch a single agent process ─────────────────────────────────────────────
# Each agent is a looping Ollama chat that receives tasks from stdin / a FIFO.
launch_agent() {
    local role="$1"          # orchestrator | developer | reviewer
    local model="$2"
    local agent_id="$3"
    local task_fifo="$4"
    local log_file="$LOG_DIR/${role}_${agent_id}.log"

    tui_step "Starting $role agent ($model) — log: $log_file"

    (
        echo "[$role-$agent_id] Ready. Model=$model" | tee -a "$log_file"
        while IFS= read -r task_line; do
            [[ -z "$task_line" ]] && continue
            echo "[$role-$agent_id] Task: $task_line" | tee -a "$log_file"
            local response
            response=$(ollama run "$model" "$task_line" 2>>"$log_file" || echo "ERROR running model")
            echo "[$role-$agent_id] Response: $response" | tee -a "$log_file"
        done < "$task_fifo"
    ) &

    AGENT_PIDS+=($!)
}

# ── Setup FIFOs ────────────────────────────────────────────────────────────────
create_agent_fifo() {
    local name="$1"
    local fifo="$FIFO_DIR/$name"
    mkfifo "$fifo"
    echo "$fifo"
}

# ── Main orchestrator loop ─────────────────────────────────────────────────────
run_orchestrator() {
    local objective="${1:-Improve code quality and fix any issues}"
    local num_developers="${2:-1}"

    register_orchestrator_cleanup
    resolve_agent_models

    tui_header "Launching Agent Team"
    tui_info "Objective   : $objective"
    tui_info "Orchestrator: $ORCHESTRATOR_MODEL"
    tui_info "Developer(s): $DEVELOPER_MODELS"
    echo ""

    # Create FIFOs
    FIFO_DIR=$(mktemp -d /tmp/pushbutton-fifos-XXXXX)
    local orch_fifo
    orch_fifo=$(create_agent_fifo "orchestrator")
    local dev_fifos=()
    for (( d=0; d<num_developers; d++ )); do
        dev_fifos+=("$(create_agent_fifo "developer_$d")")
    done

    # Launch orchestrator
    launch_agent "orchestrator" "$ORCHESTRATOR_MODEL" "0" "$orch_fifo"

    # Launch developer agents
    local developer_models=()
    IFS=',' read -r -a developer_models <<< "$DEVELOPER_MODELS"
    for (( d=0; d<num_developers; d++ )); do
        local dev_model="${developer_models[$(( d % ${#developer_models[@]} ))]}"
        launch_agent "developer" "$dev_model" "$d" "${dev_fifos[$d]}"
    done

    tui_success "Agent team running. PIDs: ${AGENT_PIDS[*]}"
    tui_info "Send tasks by writing to: $orch_fifo"
    tui_info "Logs in: $LOG_DIR"

    # Seed the orchestrator with the main objective
    echo "$objective" > "$orch_fifo" &

    # Keep process alive until Ctrl-C
    wait
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    run_orchestrator "${1:-Improve this codebase}" "${2:-1}"
fi
