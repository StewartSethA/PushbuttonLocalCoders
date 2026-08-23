#!/usr/bin/env bash
# docker_agent.sh — Wrap a project in a Docker agent sandbox with
#                   "dangerously skip permissions" for unattended agentic runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tui.sh"

DOCKER_IMAGE="${DOCKER_IMAGE:-pushbutton-agent:latest}"
AGENT_DOCKERFILE="$SCRIPT_DIR/../agents/Dockerfile.agent"
COMPOSE_FILE="$SCRIPT_DIR/../agents/docker-compose.yml"

# ── Prerequisite checks ────────────────────────────────────────────────────────
ensure_docker() {
    if ! command -v docker &>/dev/null; then
        tui_error "Docker not installed. Visit https://docs.docker.com/get-docker/"
        return 1
    fi
    if ! docker info &>/dev/null 2>&1; then
        tui_error "Docker daemon not running. Start Docker and retry."
        return 1
    fi
    tui_success "Docker available ✓"
}

# ── Build agent image ──────────────────────────────────────────────────────────
build_agent_image() {
    local context_dir
    context_dir="$(dirname "$AGENT_DOCKERFILE")"

    tui_step "Building agent Docker image: $DOCKER_IMAGE …"
    docker build -f "$AGENT_DOCKERFILE" -t "$DOCKER_IMAGE" "$context_dir"
    tui_success "Docker image $DOCKER_IMAGE built ✓"
}

# ── Run project in sandbox ─────────────────────────────────────────────────────
# Arguments:
#   $1 — project directory (mounted at /workspace inside container)
#   $2 — task/objective (passed to the agent as TASK env var)
#   $3 — model tag (optional, defaults to env SELECTED_MODEL)
run_agent_sandbox() {
    local project_dir="${1:?project_dir required}"
    local task="${2:-Improve this codebase}"
    local model_tag="${3:-${SELECTED_MODEL:-qwen2.5-coder:7b-instruct-q4_K_M}}"

    project_dir="$(realpath "$project_dir")"

    ensure_docker

    if ! docker image inspect "$DOCKER_IMAGE" &>/dev/null 2>&1; then
        build_agent_image
    fi

    tui_step "Launching agent sandbox for: $project_dir"
    tui_info "Task: $task"
    tui_info "Model: $model_tag"

    local container_name
    container_name="pushbutton-agent-$(date +%s)"

    docker run --rm \
        --name "$container_name" \
        --network host \
        -v "$project_dir":/workspace \
        -v "$HOME/.config/pushbutton":/root/.config/pushbutton:ro \
        -e TASK="$task" \
        -e OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}" \
        -e MODEL_TAG="$model_tag" \
        -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
        -e DANGEROUSLY_SKIP_PERMISSIONS=true \
        "$DOCKER_IMAGE"
}

# ── Docker Compose multi-agent setup ──────────────────────────────────────────
start_agent_team() {
    local project_dir="${1:?project_dir required}"
    local task="${2:-Develop and iterate on this codebase}"

    ensure_docker

    tui_step "Starting multi-agent team via Docker Compose…"
    WORKSPACE="$(realpath "$project_dir")" \
    TASK="$task" \
    ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-qwen2.5:7b-instruct-q4_K_M}" \
    DEVELOPER_MODEL="${DEVELOPER_MODEL:-qwen2.5-coder:7b-instruct-q4_K_M}" \
    ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    docker compose -f "$COMPOSE_FILE" up --build -d

    tui_success "Agent team started. Logs: docker compose -f $COMPOSE_FILE logs -f"
}

stop_agent_team() {
    docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
    tui_success "Agent team stopped."
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        build)   build_agent_image ;;
        run)     run_agent_sandbox "${2:?}" "${3:-}" "${4:-}" ;;
        team)    start_agent_team  "${2:?}" "${3:-}" ;;
        stop)    stop_agent_team ;;
        *)
            echo "Usage: $0 {build|run <project_dir> [task] [model]|team <project_dir> [task]|stop}"
            exit 1
            ;;
    esac
fi
