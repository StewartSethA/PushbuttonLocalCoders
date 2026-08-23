#!/usr/bin/env bash
# benchmark.sh — Quick Ollama benchmark helpers and runtime persistence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_hardware.sh"
source "$SCRIPT_DIR/tui.sh"

PUSHBUTTON_CONFIG_DIR="${PUSHBUTTON_CONFIG_DIR:-$HOME/.config/pushbutton}"
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-$PUSHBUTTON_CONFIG_DIR/runtime.env}"
RUNTIME_HISTORY_FILE="${RUNTIME_HISTORY_FILE:-$PUSHBUTTON_CONFIG_DIR/runtime_history.tsv}"

ensure_runtime_store() {
    mkdir -p "$PUSHBUTTON_CONFIG_DIR"
    if [[ ! -f "$RUNTIME_HISTORY_FILE" ]]; then
        printf "timestamp\tframework\tendpoint\tmodel\tquant\tprompt_tps\tgen_tps\ttotal_tps\tprompt_tokens\tgen_tokens\n" > "$RUNTIME_HISTORY_FILE"
    fi
}

extract_model_size_billions() {
    local model="${1:-}"
    local size
    size=$(grep -oE '[0-9]+([.][0-9]+)?b' <<< "$model" | head -1 | tr -d 'b' || true)
    [[ -n "$size" ]] || size="7"
    echo "$size"
}

extract_model_quant() {
    local model="${1:-}"
    local quant
    quant=$(grep -oE 'q[0-9]+(_[A-Za-z0-9]+)?' <<< "$model" | tail -1 || true)
    [[ -n "$quant" ]] || quant="unknown"
    echo "$quant"
}

benchmark_prompt() {
    local prompt_tokens="${1:-128}"
    python3 - "$prompt_tokens" <<'PY'
import sys
n = int(sys.argv[1])
print(" ".join(f"pbtok{i:03d}" for i in range(n)))
PY
}

estimate_ollama_speeds() {
    local model="${1:?model required}"
    local framework="${2:-ollama}"
    local size_b quant family base_pp base_tg quant_factor family_factor hw_factor

    size_b=$(extract_model_size_billions "$model")
    quant=$(extract_model_quant "$model")
    family="general"
    [[ "$model" == *coder* ]] && family="coder"

    case "$framework" in
        ollama-cpu)
            base_pp=260
            base_tg=14
            eval "$(detect_cpu)"
            hw_factor=$(awk -v t="${CPU_THREADS:-16}" 'BEGIN { v=t/32.0; if (v<0.55) v=0.55; if (v>2.0) v=2.0; printf "%.3f", v }')
            ;;
        *)
            base_pp=2400
            base_tg=120
            eval "$(detect_gpu)"
            case "${GPU_MODEL:-}" in
                *H100*|*A100*) hw_factor="1.45" ;;
                *V100*|*A40*|*RTX\ 6000*|*L40*) hw_factor="1.20" ;;
                *4090*|*3090*) hw_factor="1.10" ;;
                *T4*|*A10*) hw_factor="0.85" ;;
                *) hw_factor="1.00" ;;
            esac
            ;;
    esac

    case "$quant" in
        q8*) quant_factor="0.58" ;;
        q6*) quant_factor="0.74" ;;
        q5*) quant_factor="0.88" ;;
        q4*) quant_factor="1.00" ;;
        *)   quant_factor="0.92" ;;
    esac

    case "$family" in
        coder) family_factor="0.92" ;;
        *)     family_factor="1.00" ;;
    esac

    awk -v base_pp="$base_pp" -v base_tg="$base_tg" -v size="$size_b" \
        -v quant="$quant_factor" -v family="$family_factor" -v hw="$hw_factor" '
        BEGIN {
            scale = 7.0 / size
            pp = base_pp * scale * quant * family * hw
            tg = base_tg * scale * quant * family * hw
            if (pp < 20) pp = 20
            if (tg < 1) tg = 1
            printf "%.1f\t%.1f\n", pp, tg
        }'
}

calc_quick_gen_tokens() {
    local prompt_tokens="${1:-128}"
    local guessed_total="${2:-32}"
    awk -v prompt="$prompt_tokens" -v total="$guessed_total" '
        BEGIN {
            budget = int(total * 8.0)
            gen = budget - prompt
            if (gen < 32) gen = 32
            if (gen > 128) gen = 128
            print gen
        }'
}

bar_fill() {
    local filled="${1:-0}"
    local empty="${2:-0}"
    local out=""
    local i
    for (( i=0; i<filled; i++ )); do out+="█"; done
    for (( i=0; i<empty; i++ )); do out+="░"; done
    printf "%s" "$out"
}

print_speed_bar() {
    local label="${1:-metric}"
    local value="${2:-0}"
    local scale="${3:-1}"
    local unit="${4:-tok/s}"
    local width=24
    local filled
    filled=$(awk -v value="$value" -v scale="$scale" -v width="$width" 'BEGIN { if (scale <= 0) scale = 1; v = int((value / scale) * width); if (v < 0) v = 0; if (v > width) v = width; print v }')
    local empty=$(( width - filled ))
    printf "  %-12s [%s] %7.1f %s\n" "$label" "$(bar_fill "$filled" "$empty")" "$value" "$unit"
}

print_speed_comparison() {
    local label="${1:-metric}"
    local actual="${2:-0}"
    local guessed="${3:-0}"
    local unit="${4:-tok/s}"
    local scale
    scale=$(awk -v a="$actual" -v g="$guessed" 'BEGIN { m = (a > g ? a : g); if (m < 1) m = 1; printf "%.1f", m * 1.15 }')
    echo ""
    printf "  %s\n" "$label"
    print_speed_bar "actual" "$actual" "$scale" "$unit"
    print_speed_bar "guessed" "$guessed" "$scale" "$unit"
}

persist_runtime_profile() {
    local framework="${1:?framework required}"
    local endpoint="${2:?endpoint required}"
    local model="${3:?model required}"
    local prompt_tps="${4:-0}"
    local gen_tps="${5:-0}"
    local total_tps="${6:-0}"
    local prompt_tokens="${7:-128}"
    local gen_tokens="${8:-64}"
    local quant
    local timestamp

    ensure_runtime_store
    quant=$(extract_model_quant "$model")
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    cat > "$RUNTIME_ENV_FILE" <<EOF
PUSHBUTTON_DEFAULT_FRAMEWORK="$framework"
PUSHBUTTON_DEFAULT_ENDPOINT="$endpoint"
PUSHBUTTON_DEFAULT_MODEL="$model"
PUSHBUTTON_DEFAULT_QUANT="$quant"
PUSHBUTTON_LAST_PROMPT_TPS="$prompt_tps"
PUSHBUTTON_LAST_GENERATION_TPS="$gen_tps"
PUSHBUTTON_LAST_TOTAL_TPS="$total_tps"
PUSHBUTTON_LAST_BENCHMARK_AT="$timestamp"
EOF

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$timestamp" "$framework" "$endpoint" "$model" "$quant" \
        "$prompt_tps" "$gen_tps" "$total_tps" "$prompt_tokens" "$gen_tokens" >> "$RUNTIME_HISTORY_FILE"
}

detect_model_max_context() {
    local model="${1:?model required}"
    local max_ctx=""
    max_ctx=$(ollama show "$model" --modelfile 2>/dev/null | awk '/^PARAMETER num_ctx /{print $3; exit}' || true)
    [[ -n "$max_ctx" ]] || max_ctx="8192"
    echo "$max_ctx"
}

run_ollama_benchmark_once() {
    local model="${1:?model required}"
    local framework="${2:-ollama}"
    local prompt_tokens="${3:-128}"
    local gen_tokens="${4:-64}"
    local render="${5:-full}"

    local host="${OLLAMA_HOST:-http://localhost:11434}"
    local prompt quant guessed_pp guessed_tg guessed_total
    local payload_file result_file
    local curl_pid start_ms now_ms elapsed_ms est_pp_ms est_tg_ms phase_elapsed
    local num_gpu="-1"

    IFS=$'\t' read -r GUESSED_PP GUESSED_TG < <(estimate_ollama_speeds "$model" "$framework")
    guessed_total=$(awk -v pp="$GUESSED_PP" -v tg="$GUESSED_TG" 'BEGIN { printf "%.1f", (pp + tg) / 2.0 }')

    prompt=$(benchmark_prompt "$prompt_tokens")
    payload_file=$(mktemp /tmp/pushbutton-benchmark-payload-XXXXXX.json)
    result_file=$(mktemp /tmp/pushbutton-benchmark-result-XXXXXX.json)

    [[ "$framework" == "ollama-cpu" ]] && num_gpu="0"

    python3 - "$payload_file" "$model" "$prompt" "$gen_tokens" "$prompt_tokens" "$num_gpu" <<'PY'
import json
import sys

payload_path, model, prompt, gen_tokens, prompt_tokens, num_gpu = sys.argv[1:]
options = {
    "num_predict": int(gen_tokens),
    "num_ctx": max(512, int(prompt_tokens) + int(gen_tokens) + 64),
    "temperature": 0,
    "top_k": 1,
}
if int(num_gpu) >= 0:
    options["num_gpu"] = int(num_gpu)

payload = {
    "model": model,
    "prompt": prompt,
    "stream": False,
    "options": options,
}
with open(payload_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
PY

    if [[ "$render" == "full" ]]; then
        tui_header "Quick Benchmark"
        tui_info "Framework : $framework"
        tui_info "Endpoint  : $host"
        tui_info "Model     : $model"
        tui_info "Quant     : $(extract_model_quant "$model")"
        tui_info "Prompt    : ${prompt_tokens} toks"
        tui_info "Generate  : ${gen_tokens} toks"
        tui_info "Guess     : PP ${GUESSED_PP} tok/s · TG ${GUESSED_TG} tok/s"
    fi

    curl -fsS "$host/api/generate" \
        -H 'Content-Type: application/json' \
        --data @"$payload_file" > "$result_file" &
    curl_pid=$!

    start_ms=$(date +%s%3N)
    est_pp_ms=$(awk -v prompt="$prompt_tokens" -v rate="$GUESSED_PP" 'BEGIN { if (rate <= 0) rate = 1; printf "%d", (prompt / rate) * 1000 }')
    est_tg_ms=$(awk -v gen="$gen_tokens" -v rate="$GUESSED_TG" 'BEGIN { if (rate <= 0) rate = 1; printf "%d", (gen / rate) * 1000 }')
    [[ "$est_pp_ms" -lt 200 ]] && est_pp_ms=200
    [[ "$est_tg_ms" -lt 300 ]] && est_tg_ms=300

    if [[ "$render" == "full" ]]; then
        while kill -0 "$curl_pid" 2>/dev/null; do
            now_ms=$(date +%s%3N)
            elapsed_ms=$(( now_ms - start_ms ))
            if (( elapsed_ms <= est_pp_ms )); then
                tui_progress_bar "$elapsed_ms" "$est_pp_ms" "PP (estimated)"
            else
                phase_elapsed=$(( elapsed_ms - est_pp_ms ))
                tui_progress_bar "$phase_elapsed" "$est_tg_ms" "TG (estimated)"
            fi
            sleep 0.1
        done
        wait "$curl_pid"
        tui_progress_bar 1 1 "Done"
        printf "\r\033[K"
    else
        wait "$curl_pid"
    fi

    IFS=$'\t' read -r ACTUAL_PP ACTUAL_TG ACTUAL_TOTAL PROMPT_EVAL_COUNT EVAL_COUNT < <(python3 - "$result_file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)

def safe_rate(count_key, dur_key):
    count = payload.get(count_key, 0) or 0
    duration = payload.get(dur_key, 0) or 0
    if duration <= 0:
        return 0.0
    return count / (duration / 1_000_000_000)

prompt_tps = safe_rate("prompt_eval_count", "prompt_eval_duration")
gen_tps = safe_rate("eval_count", "eval_duration")
prompt_count = payload.get("prompt_eval_count", 0) or 0
eval_count = payload.get("eval_count", 0) or 0
total_duration = (payload.get("prompt_eval_duration", 0) or 0) + (payload.get("eval_duration", 0) or 0)
total_tps = 0.0 if total_duration <= 0 else (prompt_count + eval_count) / (total_duration / 1_000_000_000)

print(f"{prompt_tps:.1f}\t{gen_tps:.1f}\t{total_tps:.1f}\t{prompt_count}\t{eval_count}")
PY
)

    if [[ "$render" == "full" ]]; then
        print_speed_comparison "PP tok/s" "${ACTUAL_PP:-0}" "${GUESSED_PP:-0}" "tok/s"
        print_speed_comparison "TG tok/s" "${ACTUAL_TG:-0}" "${GUESSED_TG:-0}" "tok/s"
        echo ""
        tui_info "Actual total  : ${ACTUAL_TOTAL:-0} tok/s"
        tui_info "Guessed total : $guessed_total tok/s"
        persist_runtime_profile "$framework" "$host" "$model" "${ACTUAL_PP:-0}" "${ACTUAL_TG:-0}" "${ACTUAL_TOTAL:-0}" "$prompt_tokens" "$gen_tokens"
        tui_success "Runtime profile saved to $RUNTIME_ENV_FILE"
    fi

    rm -f "$payload_file" "$result_file"
}

run_ollama_context_sweep() {
    local model="${1:?model required}"
    local framework="${2:-ollama}"
    local initial_total="${3:-0}"

    local max_ctx
    local -a contexts=()
    local total_budget_s=0
    local ctx=256

    max_ctx=$(detect_model_max_context "$model")
    while (( ctx <= max_ctx )); do
        local projected
        projected=$(awk -v ctx="$ctx" -v total="${initial_total:-1}" 'BEGIN { if (total <= 0) total = 1; printf "%.2f", (ctx + 64) / total }')
        total_budget_s=$(awk -v a="$total_budget_s" -v b="$projected" 'BEGIN { printf "%.2f", a + b }')
        if awk -v total="$total_budget_s" 'BEGIN { exit !(total <= 1800) }'; then
            contexts+=("$ctx")
            ctx=$(( ctx * 2 ))
        else
            break
        fi
    done

    if (( ${#contexts[@]} == 0 )); then
        tui_warn "Skipping context sweep — projected runtime exceeds 30 minutes."
        return 0
    fi

    local accepted_budget_s=0
    for ctx_val in "${contexts[@]}"; do
        accepted_budget_s=$(awk -v a="$accepted_budget_s" -v ctx="$ctx_val" -v total="${initial_total:-1}" \
            'BEGIN { if (total <= 0) total = 1; printf "%.2f", a + ((ctx + 64) / total) }')
    done

    tui_header "Context Sweep"
    tui_info "Projected runtime from initial speed: ${accepted_budget_s}s"

    local best_pp=1
    local best_tg=1
    local ctx_val
    declare -a rows=()

    for ctx_val in "${contexts[@]}"; do
        run_ollama_benchmark_once "$model" "$framework" "$ctx_val" 64 "quiet"
        rows+=("${ctx_val}|${ACTUAL_PP:-0}|${ACTUAL_TG:-0}")
        if awk -v a="${ACTUAL_PP:-0}" -v b="$best_pp" 'BEGIN { exit !(a > b) }'; then
            best_pp="${ACTUAL_PP:-0}"
        fi
        if awk -v a="${ACTUAL_TG:-0}" -v b="$best_tg" 'BEGIN { exit !(a > b) }'; then
            best_tg="${ACTUAL_TG:-0}"
        fi
    done

    printf "  %-8s %-12s %-30s %-12s %-30s\n" "Context" "PP tok/s" "PP curve" "TG tok/s" "TG curve"
    local row sweep_ctx sweep_pp sweep_tg pp_bar tg_bar pp_fill tg_fill width=28
    for row in "${rows[@]}"; do
        IFS='|' read -r sweep_ctx sweep_pp sweep_tg <<< "$row"
        pp_fill=$(awk -v v="$sweep_pp" -v m="$best_pp" -v w="$width" 'BEGIN { if (m <= 0) m = 1; x = int((v / m) * w); if (x < 0) x = 0; if (x > w) x = w; print x }')
        tg_fill=$(awk -v v="$sweep_tg" -v m="$best_tg" -v w="$width" 'BEGIN { if (m <= 0) m = 1; x = int((v / m) * w); if (x < 0) x = 0; if (x > w) x = w; print x }')
        pp_bar=$(bar_fill "$pp_fill" $(( width - pp_fill )))
        tg_bar=$(bar_fill "$tg_fill" $(( width - tg_fill )))
        printf "  %-8s %-12.1f %-30s %-12.1f %-30s\n" "$sweep_ctx" "$sweep_pp" "$pp_bar" "$sweep_tg" "$tg_bar"
    done
}

maybe_run_post_setup_benchmark() {
    local model="${1:?model required}"
    local framework="${2:-ollama}"
    local mode="${3:-ask}"
    local sweep_mode="${4:-ask}"
    local prompt_tokens=128
    local gen_tokens guessed_total

    IFS=$'\t' read -r GUESSED_PP GUESSED_TG < <(estimate_ollama_speeds "$model" "$framework")
    guessed_total=$(awk -v pp="$GUESSED_PP" -v tg="$GUESSED_TG" 'BEGIN { printf "%.1f", (pp + tg) / 2.0 }')
    gen_tokens=$(calc_quick_gen_tokens "$prompt_tokens" "$guessed_total")

    case "$mode" in
        yes) ;;
        no) return 0 ;;
        *)
            if [[ ! -t 0 ]]; then
                tui_info "Skipping benchmark prompt in non-interactive mode."
                return 0
            fi
            echo ""
            read -r -p "Run a quick ${framework} speed benchmark now? [Y/n] " reply
            case "${reply:-y}" in
                n|N|no|NO) return 0 ;;
            esac
            ;;
    esac

    run_ollama_benchmark_once "$model" "$framework" "$prompt_tokens" "$gen_tokens" "full"

    case "$sweep_mode" in
        no) return 0 ;;
        yes)
            run_ollama_context_sweep "$model" "$framework" "${ACTUAL_TOTAL:-0}"
            ;;
        *)
            if [[ -t 0 ]]; then
                echo ""
                read -r -p "Run a longer context sweep (up to the model limit, capped to <30m)? [y/N] " sweep_reply
                case "${sweep_reply:-n}" in
                    y|Y|yes|YES) run_ollama_context_sweep "$model" "$framework" "${ACTUAL_TOTAL:-0}" ;;
                esac
            fi
            ;;
    esac
}
