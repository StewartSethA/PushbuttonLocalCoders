#!/usr/bin/env bash
# agent_entrypoint.sh — Container entrypoint for the coding agent.
# Environment variables expected:
#   TASK                     — the objective/task description
#   OLLAMA_HOST              — Ollama API endpoint (default: http://localhost:11434)
#   MODEL_TAG                — Ollama model tag to use
#   ANTHROPIC_API_KEY        — optional Anthropic API key for cloud fallback
#   DANGEROUSLY_SKIP_PERMISSIONS — set to true to allow unrestricted file writes

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODEL_TAG="${MODEL_TAG:-qwen3.8:27b-q4_K_M}"
TASK="${TASK:-Describe yourself and list files in /workspace}"

echo "═══════════════════════════════════════════════════"
echo "  PushbuttonLocalCoders — Coding Agent"
echo "  Model : $MODEL_TAG"
echo "  Host  : $OLLAMA_HOST"
echo "  Task  : $TASK"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Wait for Ollama ────────────────────────────────────────────────────────────
wait_for_ollama() {
    local attempts=0
    until curl -sf "$OLLAMA_HOST/api/tags" &>/dev/null; do
        sleep 2
        (( attempts++ ))
        if (( attempts > 15 )); then
            echo "ERROR: Ollama not reachable at $OLLAMA_HOST after 30s"
            exit 1
        fi
    done
    echo "Ollama ready ✓"
}

# ── Pull model if needed ───────────────────────────────────────────────────────
pull_model_if_needed() {
    local tag="$1"
    if ! curl -sf "$OLLAMA_HOST/api/tags" | grep -q "\"$tag\"" 2>/dev/null; then
        echo "Pulling model $tag …"
        curl -s -X POST "$OLLAMA_HOST/api/pull" \
             -H "Content-Type: application/json" \
             -d "{\"name\":\"$tag\"}" | tail -1
    else
        echo "Model $tag already present ✓"
    fi
}

# ── Submit task to Ollama via REST ─────────────────────────────────────────────
run_task() {
    local model="$1"
    local task="$2"

    echo ""
    echo "Running task…"
    echo "─────────────────────────────────────────────────"

    curl -s -X POST "$OLLAMA_HOST/api/generate" \
         -H "Content-Type: application/json" \
         -d "$(jq -n \
                --arg model "$model" \
                --arg prompt "$task" \
                '{model: $model, prompt: $prompt, stream: false}')" \
    | jq -r '.response // "No response received"'
}

wait_for_ollama
pull_model_if_needed "$MODEL_TAG"
run_task "$MODEL_TAG" "$TASK"
