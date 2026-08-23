#!/usr/bin/env bash
# ablation.sh — Rapid model/quant ablation runner.
# Benchmarks a set of models/quants using llama.cpp and ranks them by
# tokens/second on the local hardware.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_hardware.sh"
source "$SCRIPT_DIR/tui.sh"

RESULTS_DIR="${RESULTS_DIR:-$HOME/.config/pushbutton/ablation_results}"
mkdir -p "$RESULTS_DIR"

# ── Benchmark a single model ───────────────────────────────────────────────────
# Returns tok/s written to stdout and a result file.
benchmark_model() {
    local tag="$1"          # ollama model tag
    local n_prompt="${2:-64}"
    local n_gen="${3:-64}"

    tui_step "Benchmarking $tag …"

    # Pull if needed (silently)
    ollama pull "$tag" &>/dev/null 2>&1 || {
        tui_warn "Could not pull $tag — skipping."
        echo "0"
        return 0
    }

    # Run a quick generation and measure wall time
    local start_ts end_ts elapsed_s tokens_per_sec
    local prompt="def fibonacci(n):"
    start_ts=$(date +%s%3N)
    local output
    output=$(ollama run "$tag" "$prompt" 2>/dev/null | head -c 500 || true)
    end_ts=$(date +%s%3N)

    local token_count
    token_count=$(echo "$output" | wc -w)
    elapsed_s=$(echo "scale=3; ($end_ts - $start_ts) / 1000" | bc 2>/dev/null || echo "1")
    tokens_per_sec=$(echo "scale=1; $token_count / $elapsed_s" | bc 2>/dev/null || echo "0")

    echo "$tokens_per_sec"
}

# ── Run ablation over a list of models ────────────────────────────────────────
run_ablation() {
    local models=("$@")
    local total=${#models[@]}
    local result_file="$RESULTS_DIR/ablation_$(date +%Y%m%d_%H%M%S).csv"

    tui_header "Model Ablation Run"
    echo "tag,tok_per_sec,quality_score" > "$result_file"

    local i=0
    declare -A tps_map

    for entry in "${models[@]}"; do
        (( i++ ))
        local tag name quality
        IFS='|' read -r tag name _min_gb _params quality <<< "$entry"

        tui_progress_bar "$i" "$total" "$name"

        local tps
        tps=$(benchmark_model "$tag")
        tps_map["$tag"]=$tps
        echo "$tag,$tps,$quality" >> "$result_file"
    done

    echo ""
    tui_success "Ablation complete. Results: $result_file"

    # Print sorted summary
    echo ""
    printf "%-50s  %10s  %7s\n" "Model" "Tok/s" "Quality"
    printf '%0.s─' {1..72}; echo ""
    sort -t',' -k2 -rn "$result_file" | tail -n +2 | while IFS=',' read -r tag tps qual; do
        printf "%-50s  %10s  %7s\n" "$tag" "$tps" "$qual"
    done
    echo ""

    echo "$result_file"
}

# ── Shortcut: ablate best candidates for detected hardware ────────────────────
run_hardware_ablation() {
    eval "$(detect_inference_memory)"

    source "$SCRIPT_DIR/select_model.sh"

    # Build candidate list: top models that fit in INFERENCE_GB
    local candidates=()
    for entry in "${MODEL_CATALOGUE[@]}"; do
        local _tag _name min_gb _params _q
        IFS='|' read -r _tag _name min_gb _params _q <<< "$entry"
        if (( INFERENCE_GB >= min_gb )); then
            candidates+=("$entry")
        fi
    done

    if (( ${#candidates[@]} == 0 )); then
        tui_warn "No models fit in ${INFERENCE_GB} GB. Using smallest available."
        candidates=("${MODEL_CATALOGUE[-1]}")
    fi

    run_ablation "${candidates[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-hardware}" in
        hardware) run_hardware_ablation ;;
        custom)   shift; run_ablation "$@" ;;
        *)        run_hardware_ablation ;;
    esac
fi
