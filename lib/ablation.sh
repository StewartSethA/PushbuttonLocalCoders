#!/usr/bin/env bash
# ablation.sh — Modern model benchmarking, estimate tracking, and submission prep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_hardware.sh"
source "$SCRIPT_DIR/tui.sh"
source "$SCRIPT_DIR/select_model.sh"

RESULTS_DIR="${RESULTS_DIR:-$HOME/.config/pushbutton/benchmarks}"
HISTORY_FILE="$RESULTS_DIR/benchmark_history.csv"
SYSTEM_BENCH_DIR="$SCRIPT_DIR/../benchmarks/system"
mkdir -p "$RESULTS_DIR" "$SYSTEM_BENCH_DIR"

ensure_history_file() {
    if [[ ! -f "$HISTORY_FILE" ]]; then
        cat > "$HISTORY_FILE" <<EOF
timestamp,hostname,os,gpu_vendor,gpu_model,cpu_model,memory_type,inference_gb,model_tag,model_quant,kv_quant,target_context,estimated_pp_tps,actual_pp_tps,pp_delta_pct,estimated_tg_tps,actual_tg_tps,tg_delta_pct,result_file
EOF
    fi
}

json_escape() {
    local value="${1:-}"
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//$'\n'/\\n}
    printf '%s' "$value"
}

bench_prompt() {
    cat <<'EOF'
You are benchmarking local coding throughput.
Summarize the following repository concerns in bullet points:
- model selection
- benchmark collection
- install UX
- hardware fit
Then produce a short refactor checklist.
EOF
}

collect_generate_metrics() {
    local model_tag="$1"
    local target_context="${2:-16384}"
    local prompt payload response
    prompt=$(bench_prompt)
    payload=$(cat <<EOF
{"model":"$model_tag","prompt":"$(json_escape "$prompt")","stream":false,"options":{"temperature":0,"num_predict":96,"num_ctx":$target_context}}
EOF
)
    curl -fsS "${OLLAMA_HOST:-http://localhost:11434}/api/generate" \
        -H 'Content-Type: application/json' \
        -d "$payload"
}

parse_json_number() {
    local json="$1"
    local key="$2"
    sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p" <<< "$json" | head -1
}

ns_to_tps() {
    local count="${1:-0}"
    local duration_ns="${2:-0}"
    if [[ -z "$count" || -z "$duration_ns" || "$duration_ns" == "0" ]]; then
        echo "0.0"
        return 0
    fi
    awk "BEGIN { printf \"%.1f\", ($count * 1000000000) / $duration_ns }"
}

pct_delta() {
    local estimated="${1:-0}"
    local actual="${2:-0}"
    if awk "BEGIN { exit !($estimated == 0) }"; then
        echo "0.0"
        return 0
    fi
    awk "BEGIN { printf \"%.1f\", (($actual - $estimated) / $estimated) * 100 }"
}

lookup_model_entry_by_tag() {
    local full_tag="$1"
    local base_tag
    base_tag="$(strip_model_quant_suffix "$full_tag")"
    local entry
    for entry in "${MODEL_CATALOGUE[@]}" "${ORCHESTRATOR_CATALOGUE[@]}"; do
        if [[ "$(entry_field "$entry" tag)" == "$base_tag" ]]; then
            echo "$entry"
            return 0
        fi
    done
    return 1
}

benchmark_model_detailed() {
    local model_tag="$1"
    local quant="${2:-${MODEL_QUANT:-q4_K_M}}"
    local kv_quant="${3:-${KV_CACHE_QUANT:-q6_K}}"
    local target_context="${4:-${TARGET_CONTEXT_LENGTH:-16384}}"

    local entry
    entry="$(lookup_model_entry_by_tag "$model_tag" || true)"
    if [[ -z "$entry" ]]; then
        tui_warn "Unknown model metadata for $model_tag — skipping benchmark."
        echo "0.0,0.0,0.0,0.0,"
        return 0
    fi

    local base_tag name params native_ctx estimated_pp estimated_tg response
    base_tag="$(entry_field "$entry" tag)"
    name="$(entry_field "$entry" name)"
    params="$(entry_field "$entry" params)"
    native_ctx="$(entry_field "$entry" ctx)"
    estimated_pp="$(estimate_prompt_tps "$params" "$quant")"
    estimated_tg="$(estimate_generation_tps "$params" "$quant")"

    tui_step "Benchmarking $model_tag …"
    ollama pull "$model_tag" >/dev/null 2>&1 || {
        tui_warn "Could not pull $model_tag — skipping benchmark."
        echo "${estimated_pp},0.0,${estimated_tg},0.0,"
        return 0
    }

    response="$(collect_generate_metrics "$model_tag" "$target_context" 2>/dev/null || true)"
    if [[ -z "$response" ]]; then
        tui_warn "No benchmark response for $model_tag."
        echo "${estimated_pp},0.0,${estimated_tg},0.0,"
        return 0
    fi

    local prompt_eval_count prompt_eval_duration eval_count eval_duration actual_pp actual_tg
    prompt_eval_count="$(parse_json_number "$response" "prompt_eval_count")"
    prompt_eval_duration="$(parse_json_number "$response" "prompt_eval_duration")"
    eval_count="$(parse_json_number "$response" "eval_count")"
    eval_duration="$(parse_json_number "$response" "eval_duration")"

    actual_pp="$(ns_to_tps "${prompt_eval_count:-0}" "${prompt_eval_duration:-0}")"
    actual_tg="$(ns_to_tps "${eval_count:-0}" "${eval_duration:-0}")"

    local result_file="$RESULTS_DIR/${base_tag//[:\/]/_}_$(date +%Y%m%d_%H%M%S).json"
    cat > "$result_file" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host": "$(hostname)",
  "model_name": "$(json_escape "$name")",
  "model_tag": "$(json_escape "$model_tag")",
  "base_model": "$(json_escape "$base_tag")",
  "model_quant": "$quant",
  "kv_quant": "$kv_quant",
  "target_context": $target_context,
  "native_context": $native_ctx,
  "estimated_pp_tps": $estimated_pp,
  "actual_pp_tps": $actual_pp,
  "estimated_tg_tps": $estimated_tg,
  "actual_tg_tps": $actual_tg
}
EOF

    echo "${estimated_pp},${actual_pp},${estimated_tg},${actual_tg},${result_file}"
}

record_benchmark_result() {
    local model_tag="$1"
    local quant="$2"
    local kv_quant="$3"
    local target_context="$4"
    local estimated_pp="$5"
    local actual_pp="$6"
    local estimated_tg="$7"
    local actual_tg="$8"
    local result_file="$9"

    ensure_history_file
    eval "$(detect_all)"
    local pp_delta tg_delta
    pp_delta="$(pct_delta "$estimated_pp" "$actual_pp")"
    tg_delta="$(pct_delta "$estimated_tg" "$actual_tg")"
    local safe_gpu_model safe_cpu_model safe_model_tag safe_result_file
    safe_gpu_model="${GPU_MODEL//,/;}"
    safe_cpu_model="${CPU_MODEL//,/;}"
    safe_model_tag="${model_tag//,/;}"
    safe_result_file="${result_file//,/;}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "$OS" "$GPU_VENDOR" \
        "$safe_gpu_model" "$safe_cpu_model" "$MEMORY_TYPE" "$INFERENCE_GB" "$safe_model_tag" "$quant" "$kv_quant" "$target_context" \
        "$estimated_pp" "$actual_pp" "$pp_delta" "$estimated_tg" "$actual_tg" "$tg_delta" "$safe_result_file" >> "$HISTORY_FILE"

    tui_info "Estimated vs actual PP tok/s: ${estimated_pp} → ${actual_pp} (${pp_delta}%)"
    tui_info "Estimated vs actual TG tok/s: ${estimated_tg} → ${actual_tg} (${tg_delta}%)"
}

run_ablation() {
    local models=("$@")
    local total=${#models[@]}
    eval "$(detect_inference_memory)"
    local quant="${MODEL_QUANT:-$(recommend_model_quant "$INFERENCE_GB")}"
    local kv_quant="${KV_CACHE_QUANT:-q6_K}"
    local target_context="${TARGET_CONTEXT_LENGTH:-16384}"
    local result_file="$RESULTS_DIR/ablation_$(date +%Y%m%d_%H%M%S).csv"

    tui_header "Model Ablation Run"
    echo "model_tag,estimated_pp_tps,actual_pp_tps,estimated_tg_tps,actual_tg_tps" > "$result_file"

    local i=0
    for model_tag in "${models[@]}"; do
        (( i++ ))
        tui_progress_bar "$i" "$total" "$model_tag"
        local measured estimated_pp actual_pp estimated_tg actual_tg detail_file
        measured="$(benchmark_model_detailed "$model_tag" "$quant" "$kv_quant" "$target_context")"
        IFS=',' read -r estimated_pp actual_pp estimated_tg actual_tg detail_file <<< "$measured"
        echo "$model_tag,$estimated_pp,$actual_pp,$estimated_tg,$actual_tg" >> "$result_file"
        record_benchmark_result "$model_tag" "$quant" "$kv_quant" "$target_context" "$estimated_pp" "$actual_pp" "$estimated_tg" "$actual_tg" "$detail_file"
    done

    echo ""
    tui_success "Ablation complete. Results: $result_file"
    printf "%-34s %-12s %-12s %-12s %-12s\n" "Model" "Est PP" "Act PP" "Est TG" "Act TG"
    printf '%0.s─' {1..90}; echo ""
    tail -n +2 "$result_file" | while IFS=',' read -r tag est_pp act_pp est_tg act_tg; do
        printf "%-34s %-12s %-12s %-12s %-12s\n" "$tag" "$est_pp" "$act_pp" "$est_tg" "$act_tg"
    done
    echo "$result_file"
}

run_hardware_ablation() {
    eval "$(detect_inference_memory)"
    local quant="${MODEL_QUANT:-$(recommend_model_quant "$INFERENCE_GB")}"
    local kv_quant="${KV_CACHE_QUANT:-$(recommend_kv_quant "$INFERENCE_GB")}"
    local candidates=()
    local entry

    for entry in "${MODEL_CATALOGUE[@]}"; do
        if model_fits_hardware "$entry" "$quant" "$INFERENCE_GB"; then
            candidates+=("$(build_model_tag "$(entry_field "$entry" tag)" "$quant")")
        fi
    done

    if (( ${#candidates[@]} == 0 )); then
        tui_error "No modern models fit this hardware with quant $quant."
        return 1
    fi

    echo ""
    tui_info "This will pull and benchmark:"
    local tag
    for tag in "${candidates[@]}"; do
        echo "  - $tag (KV ${kv_quant}, target ctx ${TARGET_CONTEXT_LENGTH:-16384})"
    done
    echo ""

    if [[ -t 0 ]]; then
        read -r -p "Proceed with benchmarking these models? [y/N] " proceed
        [[ "$proceed" =~ ^[Yy]$ ]] || return 1
    fi

    run_ablation "${candidates[@]}"
}

benchmark_selected_models() {
    local tags=("$@")
    local tag
    for tag in "${tags[@]}"; do
        [[ -z "$tag" ]] && continue
        local measured estimated_pp actual_pp estimated_tg actual_tg detail_file
        measured="$(benchmark_model_detailed "$tag" "${MODEL_QUANT:-q4_K_M}" "${KV_CACHE_QUANT:-q6_K}" "${TARGET_CONTEXT_LENGTH:-16384}")"
        IFS=',' read -r estimated_pp actual_pp estimated_tg actual_tg detail_file <<< "$measured"
        record_benchmark_result "$tag" "${MODEL_QUANT:-q4_K_M}" "${KV_CACHE_QUANT:-q6_K}" "${TARGET_CONTEXT_LENGTH:-16384}" "$estimated_pp" "$actual_pp" "$estimated_tg" "$actual_tg" "$detail_file"
    done
}

prepare_system_benchmark_submission() {
    ensure_history_file
    if [[ ! -s "$HISTORY_FILE" ]] || [[ $(wc -l < "$HISTORY_FILE") -le 1 ]]; then
        tui_error "No benchmark history found yet."
        return 1
    fi

    local stamp host output_file
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    host="$(hostname | tr '[:space:]/' '__')"
    output_file="$SYSTEM_BENCH_DIR/${stamp}_${host}.md"

    {
        echo "# System benchmark submission"
        echo ""
        echo "- Host: \`$host\`"
        echo "- Generated: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
        echo ""
        echo "## Recent measured PP/TG"
        echo ""
        echo "| Timestamp | Model | Quant | KV | Target ctx | Est PP | Act PP | Est TG | Act TG |"
        echo "|---|---|---|---|---:|---:|---:|---:|---:|"
        tail -n 10 "$HISTORY_FILE" | while IFS=',' read -r ts host_name os gpu_vendor gpu_model cpu_model memory_type inference_gb model_tag model_quant kv_quant target_context est_pp act_pp pp_delta est_tg act_tg tg_delta result_file; do
            echo "| $ts | $model_tag | $model_quant | $kv_quant | $target_context | $est_pp | $act_pp | $est_tg | $act_tg |"
        done
        echo ""
        echo "## Raw history source"
        echo ""
        echo "- CSV: \`$HISTORY_FILE\`"
    } > "$output_file"

    tui_success "Prepared benchmark submission: $output_file"
    tui_info "Open a PR with this file to contribute your real PP/TG data."

    local repo_root
    repo_root="$(cd "$SCRIPT_DIR/.." && pwd)"
    if [[ -t 0 ]] && command -v gh >/dev/null 2>&1 && git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        read -r -p "Create a benchmark contribution branch and PR now? [y/N] " create_pr
        if [[ "$create_pr" =~ ^[Yy]$ ]]; then
            local branch_name rel_output
            branch_name="benchmarks/$(date -u +%Y%m%dT%H%M%SZ)-${host}"
            rel_output="${output_file#$repo_root/}"
            git -C "$repo_root" checkout -b "$branch_name"
            git -C "$repo_root" add "$rel_output"
            git -C "$repo_root" commit -m "Add system benchmark for ${host}"
            (
                cd "$repo_root"
                gh pr create \
                    --title "Add system benchmark for ${host}" \
                    --body "Adds measured prefill (PP) and text generation (TG) throughput captured by PushbuttonLocalCoders." \
                    || tui_warn "Branch committed locally; open the PR manually if gh could not create it."
            )
        fi
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-hardware}" in
        hardware) run_hardware_ablation ;;
        custom)   shift; run_ablation "$@" ;;
        submit)   prepare_system_benchmark_submission ;;
        *)        run_hardware_ablation ;;
    esac
fi
