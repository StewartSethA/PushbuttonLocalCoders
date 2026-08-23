#!/usr/bin/env bash
# select_model.sh — Choose optimal model + quant given available inference memory.
# Sources detect_hardware.sh for hardware info.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_hardware.sh"

# ── Model catalogue ────────────────────────────────────────────────────────────
# Format: "ollama_tag|display_name|min_vram_gb|param_billions|quality_score"
# quality_score is a rough 0-100 ranking (higher = better coder)
# Models are listed in decreasing quality order so we pick the best that fits.
declare -a MODEL_CATALOGUE=(
    # Flagship coders
    "qwen2.5-coder:32b-instruct-q8_0|Qwen2.5-Coder 32B Q8|34|32|98"
    "qwen2.5-coder:32b-instruct-q6_K|Qwen2.5-Coder 32B Q6|26|32|96"
    "qwen2.5-coder:32b-instruct-q4_K_M|Qwen2.5-Coder 32B Q4|20|32|94"
    "qwen2.5-coder:14b-instruct-q8_0|Qwen2.5-Coder 14B Q8|16|14|90"
    "qwen2.5-coder:14b-instruct-q4_K_M|Qwen2.5-Coder 14B Q4|9|14|87"
    "nemotron-mini:4b-instruct-q8_0|Nemotron-Mini 4B Q8|6|4|72"
    # Mid-range coders
    "qwen2.5-coder:7b-instruct-q8_0|Qwen2.5-Coder 7B Q8|9|7|82"
    "qwen2.5-coder:7b-instruct-q4_K_M|Qwen2.5-Coder 7B Q4|5|7|78"
    "codellama:13b-instruct-q4_K_M|CodeLlama 13B Q4|9|13|74"
    "codellama:7b-instruct-q4_K_M|CodeLlama 7B Q4|5|7|68"
    # Lightweight fallbacks
    "qwen2.5-coder:3b-instruct-q4_K_M|Qwen2.5-Coder 3B Q4|3|3|62"
    "qwen2.5-coder:1.5b-instruct-q4_K_M|Qwen2.5-Coder 1.5B Q4|2|1.5|50"
    "tinyllama:1.1b-chat-q4_K_M|TinyLlama 1.1B Q4|1|1.1|30"
)

# Orchestrator models (faster, smaller — for routing/planning)
declare -a ORCHESTRATOR_CATALOGUE=(
    "qwen2.5:7b-instruct-q4_K_M|Qwen2.5 7B Q4|5|7|80"
    "qwen2.5:3b-instruct-q4_K_M|Qwen2.5 3B Q4|3|3|65"
    "qwen2.5:1.5b-instruct-q4_K_M|Qwen2.5 1.5B Q4|2|1.5|50"
    "tinyllama:1.1b-chat-q4_K_M|TinyLlama 1.1B Q4|1|1.1|30"
)

# ── Model selection ────────────────────────────────────────────────────────────
# Returns the best-fit model from the catalogue given available GB of inference
# memory.  Writes: SELECTED_MODEL, SELECTED_MODEL_NAME, SELECTED_QUALITY
select_best_model() {
    local inference_gb="${1:-0}"
    local catalogue_var="${2:-MODEL_CATALOGUE[@]}"

    SELECTED_MODEL=""
    SELECTED_MODEL_NAME=""
    SELECTED_QUALITY=0

    local best_quality=0
    local best_tag=""
    local best_name=""

    for entry in "${!catalogue_var}"; do
        local tag name min_gb quality
        IFS='|' read -r tag name min_gb _params quality <<< "$entry"
        if (( inference_gb >= min_gb )) && (( quality > best_quality )); then
            best_quality=$quality
            best_tag=$tag
            best_name=$name
        fi
    done

    SELECTED_MODEL="$best_tag"
    SELECTED_MODEL_NAME="$best_name"
    SELECTED_QUALITY=$best_quality

    echo "SELECTED_MODEL=\"$SELECTED_MODEL\""
    echo "SELECTED_MODEL_NAME=\"$SELECTED_MODEL_NAME\""
    echo "SELECTED_QUALITY=$SELECTED_QUALITY"
}

# Convenience wrapper that auto-detects hardware then selects a model
auto_select_model() {
    eval "$(detect_inference_memory)"
    select_best_model "$INFERENCE_GB" "MODEL_CATALOGUE[@]"
}

auto_select_orchestrator() {
    eval "$(detect_inference_memory)"
    # Reserve half the inference budget for developer model; use rest for orchestrator
    local orch_gb=$(( INFERENCE_GB / 2 ))
    [[ $orch_gb -lt 1 ]] && orch_gb=1
    select_best_model "$orch_gb" "ORCHESTRATOR_CATALOGUE[@]"
}

# Print a human-readable summary
print_model_recommendation() {
    eval "$(detect_all)"

    echo ""
    echo "════════════════════════════════════════════════"
    echo "  Hardware Summary"
    echo "════════════════════════════════════════════════"
    echo "  OS        : $OS"
    echo "  CPU       : $CPU_MODEL ($CPU_CORES cores)"
    echo "  RAM       : ${RAM_GB} GB"
    echo "  GPU       : $GPU_MODEL ($GPU_VENDOR)"
    echo "  VRAM      : ${VRAM_GB} GB"
    echo "  Memory for inference: ${INFERENCE_GB} GB ($MEMORY_TYPE)"
    echo "════════════════════════════════════════════════"

    auto_select_model
    echo ""
    echo "  Recommended coder model : $SELECTED_MODEL_NAME"
    echo "    Ollama tag             : $SELECTED_MODEL"
    echo "    Quality score          : $SELECTED_QUALITY/100"

    auto_select_orchestrator
    echo ""
    echo "  Recommended orchestrator: $SELECTED_MODEL_NAME"
    echo "    Ollama tag             : $SELECTED_MODEL"
    echo "════════════════════════════════════════════════"
    echo ""
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    print_model_recommendation
fi
