#!/usr/bin/env bash
# select_model.sh — Choose modern model/quant combinations for local coding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_hardware.sh"
source "$SCRIPT_DIR/tui.sh"

MODEL_PLAN_DIR="${MODEL_PLAN_DIR:-$HOME/.config/pushbutton}"
MODEL_PLAN_FILE="$MODEL_PLAN_DIR/model_plan.env"
mkdir -p "$MODEL_PLAN_DIR"

# Format: "base_tag|display_name|param_billions|quality_score|native_context|role"
declare -a MODEL_CATALOGUE=(
    "qwen3.6:35b|Qwen 3.6 35B-A3B|35|99|262144|coder"
    "qwen3.8:27b|Qwen 3.8 27B|27|97|262144|coder"
    "nemotron-3.5-lightning|Nemotron 3.5 Lightning 30B-A3B|30|95|262144|coder"
    "qwen3.8-27b:q4_K_M|Qwen3.8-27B Q4_K_M|27|90|262144|coder"
    "qwen3.6:35b-a3b-q4_K_M|Qwen3.6-35B-A3B Q4_K_M|35|88|131072|coder"
    "nemotron-3.5-lightning:30b-a3b-q4_K_M|Nemotron-Lightning-30B-A3B Q4_K_M|30|86|32768|coder"
)

declare -a ORCHESTRATOR_CATALOGUE=(
    "nemotron-3.5-lightning|Nemotron 3.5 Lightning 30B-A3B|30|97|262144|orchestrator"
    "qwen3.8:27b|Qwen 3.8 27B|27|93|262144|orchestrator"
    "qwen3.6:35b|Qwen 3.6 35B-A3B|35|90|262144|orchestrator"
)

declare -a MODEL_QUANT_OPTIONS=("q3_K_M" "q4_K_M" "q6_K" "q8_0")
declare -a KV_QUANT_OPTIONS=("q4_0" "q6_K" "q8_0")
declare -a CONTEXT_PRESETS=(16384 32768 65536 131072 262144)

float_eval() {
    local expr="$1"
    awk "BEGIN { printf \"%.1f\", ($expr) }"
}

int_eval() {
    local expr="$1"
    awk "BEGIN { printf \"%d\", ($expr) }"
}

entry_field() {
    local entry="$1"
    local field="$2"
    IFS='|' read -r f1 f2 f3 f4 f5 f6 <<< "$entry"
    case "$field" in
        tag)     echo "$f1" ;;
        name)    echo "$f2" ;;
        params)  echo "$f3" ;;
        quality) echo "$f4" ;;
        ctx)     echo "$f5" ;;
        role)    echo "$f6" ;;
        *)       echo "" ;;
    esac
}

quant_bits() {
    case "${1:-q6_K}" in
        q3_K_M)             echo "3.5" ;;
        q4_K_M|q4_0|nvfp4) echo "4" ;;
        q6_K)              echo "6" ;;
        q8_0|fp8)          echo "8" ;;
        *)                 echo "6" ;;
    esac
}

quant_speed_factor() {
    case "${1:-q6_K}" in
        q3_K_M)             echo "1.35" ;;
        q4_K_M|q4_0|nvfp4) echo "1.20" ;;
        q6_K)              echo "1.00" ;;
        q8_0|fp8)          echo "0.78" ;;
        *)                 echo "1.00" ;;
    esac
}

kv_quant_factor() {
    case "${1:-q6_K}" in
        q4_0)       echo "0.50" ;;
        q6_K)       echo "0.75" ;;
        q8_0|fp8)   echo "1.00" ;;
        *)          echo "0.75" ;;
    esac
}

context_gb_per_1k_tokens() {
    case "$1" in
        qwen3.6:35b)             echo "0.14" ;;
        qwen3.8:27b)             echo "0.11" ;;
        nemotron-3.5-lightning)  echo "0.12" ;;
        *)                       echo "0.12" ;;
    esac
}

build_model_tag() {
    local base_tag="$1"
    local quant="${2:-}"
    if [[ -z "$quant" ]]; then
        echo "$base_tag"
    else
        echo "${base_tag}-${quant}"
    fi
}

strip_model_quant_suffix() {
    local model_tag="$1"
    for quant in "${MODEL_QUANT_OPTIONS[@]}"; do
        if [[ "$model_tag" == *"-$quant" ]]; then
            echo "${model_tag%-${quant}}"
            return 0
        fi
    done
    echo "$model_tag"
}

estimate_model_disk_gb() {
    local params_b="$1"
    local quant="$2"
    local bits
    bits=$(quant_bits "$quant")
    float_eval "($params_b * $bits / 8) * 1.05"
}

estimate_model_runtime_gb() {
    local params_b="$1"
    local quant="$2"
    local disk_gb
    disk_gb=$(estimate_model_disk_gb "$params_b" "$quant")
    float_eval "($disk_gb * 1.08) + 1.5"
}

estimate_effective_context() {
    local base_tag="$1"
    local params_b="$2"
    local native_ctx="$3"
    local quant="$4"
    local kv_quant="$5"
    local inference_gb="$6"
    local runtime_gb kv_per_1k kv_factor remaining_gb max_ctx_k

    runtime_gb=$(estimate_model_runtime_gb "$params_b" "$quant")
    kv_per_1k=$(context_gb_per_1k_tokens "$base_tag")
    kv_factor=$(kv_quant_factor "$kv_quant")
    remaining_gb=$(float_eval "$inference_gb - $runtime_gb - 2.0")

    if awk "BEGIN { exit !($remaining_gb <= 0) }"; then
        echo "4096"
        return 0
    fi

    max_ctx_k=$(int_eval "$remaining_gb / ($kv_per_1k * $kv_factor)")
    if (( max_ctx_k < 4 )); then
        echo "4096"
        return 0
    fi

    local effective_ctx=$(( max_ctx_k * 1024 ))
    if (( effective_ctx > native_ctx )); then
        effective_ctx="$native_ctx"
    fi
    echo "$effective_ctx"
}

estimate_hardware_speed_multiplier() {
    if [[ -z "${GPU_VENDOR:-}" || -z "${GPU_COUNT:-}" ]]; then
        eval "$(detect_gpu)"
    fi
    if [[ -z "${CPU_THREADS:-}" ]]; then
        eval "$(detect_cpu)"
    fi

    case "$GPU_VENDOR" in
        nvidia) echo "$(float_eval "1.35 * ($GPU_COUNT > 0 ? $GPU_COUNT : 1)")" ;;
        amd)    echo "0.95" ;;
        apple)  echo "0.75" ;;
        *)      echo "$(float_eval "0.22 * sqrt(($CPU_THREADS > 0 ? $CPU_THREADS : 1) / 8)")" ;;
    esac
}

estimate_generation_tps() {
    local params_b="$1"
    local quant="$2"
    local hw_mult quant_mult
    hw_mult=$(estimate_hardware_speed_multiplier)
    quant_mult=$(quant_speed_factor "$quant")
    float_eval "($hw_mult * $quant_mult * 980) / $params_b"
}

estimate_prompt_tps() {
    local params_b="$1"
    local quant="$2"
    local gen_tps
    gen_tps=$(estimate_generation_tps "$params_b" "$quant")
    float_eval "$gen_tps * 1.9"
}

recommend_model_quant() {
    local inference_gb="${1:-0}"
    if (( inference_gb >= 42 )); then
        echo "q8_0"
    elif (( inference_gb >= 28 )); then
        echo "q6_K"
    elif (( inference_gb >= 17 )); then
        echo "q4_K_M"
    else
        echo "q3_K_M"
    fi
}

recommend_kv_quant() {
    local inference_gb="${1:-0}"
    if (( inference_gb >= 56 )); then
        echo "q8_0"
    elif (( inference_gb >= 28 )); then
        echo "q6_K"
    else
        echo "q4_0"
    fi
}

model_fits_hardware() {
    local entry="$1"
    local quant="$2"
    local inference_gb="$3"
    local params runtime_gb
    params=$(entry_field "$entry" params)
    runtime_gb=$(estimate_model_runtime_gb "$params" "$quant")
    awk "BEGIN { exit !($runtime_gb <= $inference_gb) }"
}

select_best_model() {
    local inference_gb="${1:-0}"
    local catalogue_var="${2:-MODEL_CATALOGUE[@]}"
    local quant="${3:-$(recommend_model_quant "$inference_gb")}"
    local kv_quant="${4:-$(recommend_kv_quant "$inference_gb")}"

    SELECTED_MODEL=""
    SELECTED_MODEL_ID=""
    SELECTED_MODEL_NAME=""
    SELECTED_QUALITY=0
    SELECTED_MODEL_DISK_GB="0.0"
    SELECTED_MODEL_RUNTIME_GB="0.0"
    SELECTED_CONTEXT_LENGTH=4096
    SELECTED_MODEL_QUANT="$quant"
    SELECTED_KV_QUANT="$kv_quant"
    SELECTED_EST_PP_TPS="0.0"
    SELECTED_EST_TG_TPS="0.0"

    local best_quality=0
    for entry in "${!catalogue_var}"; do
        local tag name params quality native_ctx
        tag=$(entry_field "$entry" tag)
        name=$(entry_field "$entry" name)
        params=$(entry_field "$entry" params)
        quality=$(entry_field "$entry" quality)
        native_ctx=$(entry_field "$entry" ctx)

        if ! model_fits_hardware "$entry" "$quant" "$inference_gb"; then
            continue
        fi

        if (( quality > best_quality )); then
            best_quality=$quality
            SELECTED_MODEL_ID="$tag"
            SELECTED_MODEL_NAME="$name"
            SELECTED_MODEL="$(build_model_tag "$tag" "$quant")"
            SELECTED_QUALITY="$quality"
            SELECTED_MODEL_DISK_GB="$(estimate_model_disk_gb "$params" "$quant")"
            SELECTED_MODEL_RUNTIME_GB="$(estimate_model_runtime_gb "$params" "$quant")"
            SELECTED_CONTEXT_LENGTH="$(estimate_effective_context "$tag" "$params" "$native_ctx" "$quant" "$kv_quant" "$inference_gb")"
            SELECTED_EST_PP_TPS="$(estimate_prompt_tps "$params" "$quant")"
            SELECTED_EST_TG_TPS="$(estimate_generation_tps "$params" "$quant")"
        fi
    done

    echo "SELECTED_MODEL=\"$SELECTED_MODEL\""
    echo "SELECTED_MODEL_ID=\"$SELECTED_MODEL_ID\""
    echo "SELECTED_MODEL_NAME=\"$SELECTED_MODEL_NAME\""
    echo "SELECTED_QUALITY=$SELECTED_QUALITY"
    echo "SELECTED_MODEL_DISK_GB=\"$SELECTED_MODEL_DISK_GB\""
    echo "SELECTED_MODEL_RUNTIME_GB=\"$SELECTED_MODEL_RUNTIME_GB\""
    echo "SELECTED_CONTEXT_LENGTH=$SELECTED_CONTEXT_LENGTH"
    echo "SELECTED_MODEL_QUANT=\"$SELECTED_MODEL_QUANT\""
    echo "SELECTED_KV_QUANT=\"$SELECTED_KV_QUANT\""
    echo "SELECTED_EST_PP_TPS=\"$SELECTED_EST_PP_TPS\""
    echo "SELECTED_EST_TG_TPS=\"$SELECTED_EST_TG_TPS\""
}

auto_select_model() {
    eval "$(detect_inference_memory)"
    select_best_model "$INFERENCE_GB" "MODEL_CATALOGUE[@]"
}

auto_select_orchestrator() {
    eval "$(detect_inference_memory)"
    local quant kv_quant
    quant=$(recommend_model_quant "$INFERENCE_GB")
    kv_quant=$(recommend_kv_quant "$INFERENCE_GB")
    select_best_model "$INFERENCE_GB" "ORCHESTRATOR_CATALOGUE[@]" "$quant" "$kv_quant"
}

menu_choose_one() {
    local prompt="$1"
    local out_var="$2"
    shift 2
    local options=("$@")
    local choice
    PS3="$prompt "
    select choice in "${options[@]}"; do
        if [[ -n "${choice:-}" ]]; then
            printf -v "$out_var" '%s' "$choice"
            break
        fi
        tui_warn "Invalid selection."
    done
}

catalogue_labels_for_quant() {
    local catalogue_var="${1:-MODEL_CATALOGUE[@]}"
    local quant="$2"
    local kv_quant="$3"
    local inference_gb="$4"
    local labels=()

    for entry in "${!catalogue_var}"; do
        if ! model_fits_hardware "$entry" "$quant" "$inference_gb"; then
            continue
        fi
        local tag name params native_ctx runtime_gb disk_gb eff_ctx pp tg
        tag=$(entry_field "$entry" tag)
        name=$(entry_field "$entry" name)
        params=$(entry_field "$entry" params)
        native_ctx=$(entry_field "$entry" ctx)
        runtime_gb=$(estimate_model_runtime_gb "$params" "$quant")
        disk_gb=$(estimate_model_disk_gb "$params" "$quant")
        eff_ctx=$(estimate_effective_context "$tag" "$params" "$native_ctx" "$quant" "$kv_quant" "$inference_gb")
        pp=$(estimate_prompt_tps "$params" "$quant")
        tg=$(estimate_generation_tps "$params" "$quant")
        labels+=("$tag|$name | ~${disk_gb} GB disk | ~${runtime_gb} GB runtime | ctx ${eff_ctx} | est ${pp}/${tg} PP/TG tok/s")
    done

    printf '%s\n' "${labels[@]}"
}

menu_choose_additional_models() {
    local primary_tag="$1"
    local catalogue_var="$2"
    local quant="$3"
    local kv_quant="$4"
    local inference_gb="$5"

    local selected=()
    while true; do
        local menu_labels=()
        local menu_tags=()
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            local tag label
            tag="${line%%|*}"
            label="${line#*|}"
            [[ "$tag" == "$primary_tag" ]] && continue
            local already_selected="0"
            for picked in "${selected[@]}"; do
                [[ "$picked" == "$tag" ]] && already_selected="1"
            done
            [[ "$already_selected" == "1" ]] && continue
            menu_tags+=("$tag")
            menu_labels+=("$label")
        done < <(catalogue_labels_for_quant "$catalogue_var" "$quant" "$kv_quant" "$inference_gb")

        menu_labels+=("Done")
        local choice
        PS3="Select an additional coder (or Done): "
        select choice in "${menu_labels[@]}"; do
            if [[ "$choice" == "Done" ]]; then
                echo "$(IFS=,; echo "${selected[*]:-}")"
                return 0
            fi
            if [[ -n "${choice:-}" ]]; then
                local selection_index selected_tag
                selection_index=$((REPLY-1))
                if (( selection_index < 0 || selection_index >= ${#menu_tags[@]} )); then
                    tui_warn "Invalid selection."
                    continue
                fi
                selected_tag="${menu_tags[$selection_index]}"
                [[ -n "$selected_tag" ]] || {
                    tui_warn "Invalid selection."
                    continue
                }
                selected+=("$selected_tag")
                break
            fi
            tui_warn "Invalid selection."
        done
    done
}

best_context_preset() {
    local effective_ctx="$1"
    local best=4096
    for preset in "${CONTEXT_PRESETS[@]}"; do
        if (( preset <= effective_ctx )); then
            best="$preset"
        fi
    done
    echo "$best"
}

print_model_recommendation() {
    eval "$(detect_all)"
    local quant kv_quant
    quant=$(recommend_model_quant "$INFERENCE_GB")
    kv_quant=$(recommend_kv_quant "$INFERENCE_GB")

    eval "$(select_best_model "$INFERENCE_GB" "MODEL_CATALOGUE[@]" "$quant" "$kv_quant")"
    local coder_model="$SELECTED_MODEL_NAME"
    local coder_tag="$SELECTED_MODEL"
    local coder_ctx="$SELECTED_CONTEXT_LENGTH"
    local coder_pp="$SELECTED_EST_PP_TPS"
    local coder_tg="$SELECTED_EST_TG_TPS"

    eval "$(select_best_model "$INFERENCE_GB" "ORCHESTRATOR_CATALOGUE[@]" "$quant" "$kv_quant")"
    local orch_model="$SELECTED_MODEL_NAME"
    local orch_tag="$SELECTED_MODEL"

    echo ""
    echo "════════════════════════════════════════════════"
    echo "  Hardware Summary"
    echo "════════════════════════════════════════════════"
    echo "  OS        : $OS"
    echo "  CPU       : $CPU_MODEL ($CPU_CORES cores)"
    echo "  RAM       : ${RAM_GB} GB"
    echo "  GPU       : $GPU_MODEL ($GPU_VENDOR)"
    echo "  VRAM      : ${VRAM_GB} GB"
    echo "  Inference : ${INFERENCE_GB} GB ($MEMORY_TYPE)"
    echo "════════════════════════════════════════════════"
    echo ""
    echo "  Recommended quant       : $quant"
    echo "  Recommended KV quant    : $kv_quant"
    echo "  Recommended coder       : $coder_model"
    echo "    Ollama tag            : $coder_tag"
    echo "    Effective max context : $coder_ctx"
    echo "    Estimated PP/TG tok/s : ${coder_pp}/${coder_tg}"
    echo ""
    echo "  Recommended orchestrator: $orch_model"
    echo "    Ollama tag            : $orch_tag"
    echo "════════════════════════════════════════════════"
    echo ""
}

persist_model_plan() {
    cat > "$MODEL_PLAN_FILE" <<EOF
PRIMARY_CODER_MODEL="${PRIMARY_CODER_MODEL:-}"
PRIMARY_CODER_NAME="${PRIMARY_CODER_NAME:-}"
ADDITIONAL_CODER_MODELS="${ADDITIONAL_CODER_MODELS:-}"
MODEL_QUANT="${MODEL_QUANT:-}"
KV_CACHE_QUANT="${KV_CACHE_QUANT:-}"
TARGET_CONTEXT_LENGTH="${TARGET_CONTEXT_LENGTH:-4096}"
DEVELOPER_MODEL="${DEVELOPER_MODEL:-}"
DEVELOPER_MODELS="${DEVELOPER_MODELS:-}"
ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-}"
EOF
}

describe_selected_models() {
    local inference_gb="$1"
    local quant="$2"
    local kv_quant="$3"
    local target_ctx="$4"
    shift 4
    local models=("$@")
    local total_disk="0.0"

    echo ""
    printf "%-12s %-34s %-12s %-12s %-12s %-12s %-14s\n" "Role" "Model" "Disk GB" "Runtime GB" "Max ctx" "Target ctx" "Est PP/TG"
    printf '%0.s─' {1..124}; echo ""

    for spec in "${models[@]}"; do
        local role="${spec%%:*}"
        local tag="${spec#*:}"
        local base
        base="$(strip_model_quant_suffix "$tag")"
        local matched=""
        for entry in "${MODEL_CATALOGUE[@]}" "${ORCHESTRATOR_CATALOGUE[@]}"; do
            if [[ "$(entry_field "$entry" tag)" == "$base" ]]; then
                matched="$entry"
                break
            fi
        done
        [[ -z "$matched" ]] && continue
        local name params native_ctx disk_gb runtime_gb eff_ctx pp tg
        name=$(entry_field "$matched" name)
        params=$(entry_field "$matched" params)
        native_ctx=$(entry_field "$matched" ctx)
        disk_gb=$(estimate_model_disk_gb "$params" "$quant")
        runtime_gb=$(estimate_model_runtime_gb "$params" "$quant")
        eff_ctx=$(estimate_effective_context "$base" "$params" "$native_ctx" "$quant" "$kv_quant" "$inference_gb")
        pp=$(estimate_prompt_tps "$params" "$quant")
        tg=$(estimate_generation_tps "$params" "$quant")
        total_disk=$(float_eval "$total_disk + $disk_gb")
        printf "%-12s %-34s %-12s %-12s %-12s %-12s %-14s\n" \
            "$role" "$name" "$disk_gb" "$runtime_gb" "$eff_ctx" "$target_ctx" "${pp}/${tg}"
    done

    echo ""
    echo "  Quant            : $quant"
    echo "  KV quant         : $kv_quant"
    echo "  Target context   : $target_ctx"
    echo "  Total disk pull  : ~${total_disk} GB"
    echo "  Active memory    : peak runtime of one model at a time on ${inference_gb} GB ${MEMORY_TYPE}"
    echo ""
}

configure_model_plan() {
    eval "$(detect_all)"

    local quant kv_quant
    quant=$(recommend_model_quant "$INFERENCE_GB")
    kv_quant=$(recommend_kv_quant "$INFERENCE_GB")

    if [[ ! -t 0 ]]; then
        tui_warn "Non-interactive shell detected."
        tui_warn "Re-run in an interactive terminal or set PUSHBUTTON_ACCEPT_MODEL_PLAN=1 after reviewing the recommendation."
        print_model_recommendation
        [[ "${PUSHBUTTON_ACCEPT_MODEL_PLAN:-0}" == "1" ]] || return 1
    fi

    if [[ -z "${SELECTED_MODEL:-}" ]]; then
        tui_header "Model install plan"
        tui_info "Nothing older than Qwen 3.x or Nemotron 3.5 is offered."

        menu_choose_one "Choose model quant:" quant "${MODEL_QUANT_OPTIONS[@]}"
        menu_choose_one "Choose KV quant:" kv_quant "${KV_QUANT_OPTIONS[@]}"

        local label_lines=()
        while IFS= read -r line; do
            [[ -n "$line" ]] && label_lines+=("$line")
        done < <(catalogue_labels_for_quant "MODEL_CATALOGUE[@]" "$quant" "$kv_quant" "$INFERENCE_GB")

        if (( ${#label_lines[@]} == 0 )); then
            tui_error "None of the modern models fit this machine with quant $quant."
            return 1
        fi

        local primary_options=()
        local primary_tags=()
        for line in "${label_lines[@]}"; do
            primary_tags+=("${line%%|*}")
            primary_options+=("${line#*|}")
        done

        local primary_choice
        PS3="Choose primary coder: "
        select primary_choice in "${primary_options[@]}"; do
            if [[ -n "${primary_choice:-}" ]]; then
                PRIMARY_CODER_MODEL="$(build_model_tag "${primary_tags[$((REPLY-1))]}" "$quant")"
                PRIMARY_CODER_ID="${primary_tags[$((REPLY-1))]}"
                break
            fi
            tui_warn "Invalid selection."
        done

        ADDITIONAL_CODER_MODELS="$(menu_choose_additional_models "$PRIMARY_CODER_ID" "MODEL_CATALOGUE[@]" "$quant" "$kv_quant" "$INFERENCE_GB")"
        if [[ -n "$ADDITIONAL_CODER_MODELS" ]]; then
            local extras=()
            IFS=',' read -r -a extras <<< "$ADDITIONAL_CODER_MODELS"
            for i in "${!extras[@]}"; do
                extras[$i]="$(build_model_tag "${extras[$i]}" "$quant")"
            done
            ADDITIONAL_CODER_MODELS="$(IFS=,; echo "${extras[*]}")"
        fi
    else
        quant="${MODEL_QUANT:-$quant}"
        kv_quant="${KV_CACHE_QUANT:-$kv_quant}"
        PRIMARY_CODER_MODEL="$SELECTED_MODEL"
        PRIMARY_CODER_ID="$(strip_model_quant_suffix "$SELECTED_MODEL")"
        ADDITIONAL_CODER_MODELS="${ADDITIONAL_CODER_MODELS:-}"
    fi

    local primary_entry=""
    for entry in "${MODEL_CATALOGUE[@]}"; do
        if [[ "$(entry_field "$entry" tag)" == "$PRIMARY_CODER_ID" ]]; then
            primary_entry="$entry"
            break
        fi
    done
    [[ -z "$primary_entry" ]] && primary_entry="${MODEL_CATALOGUE[0]}"

    PRIMARY_CODER_NAME="$(entry_field "$primary_entry" name)"
    local primary_params primary_native_ctx effective_ctx default_ctx
    primary_params=$(entry_field "$primary_entry" params)
    primary_native_ctx=$(entry_field "$primary_entry" ctx)
    effective_ctx=$(estimate_effective_context "$PRIMARY_CODER_ID" "$primary_params" "$primary_native_ctx" "$quant" "$kv_quant" "$INFERENCE_GB")
    default_ctx=$(best_context_preset "$effective_ctx")

    local ctx_options=()
    for preset in "${CONTEXT_PRESETS[@]}"; do
        if (( preset <= effective_ctx )); then
            ctx_options+=("$preset")
        fi
    done
    (( ${#ctx_options[@]} == 0 )) && ctx_options=("4096")

    local chosen_ctx="${TARGET_CONTEXT_LENGTH:-$default_ctx}"
    if [[ -z "${SELECTED_MODEL:-}" ]] && [[ -t 0 ]]; then
        menu_choose_one "Choose target context:" chosen_ctx "${ctx_options[@]}"
    fi

    MODEL_QUANT="$quant"
    KV_CACHE_QUANT="$kv_quant"
    TARGET_CONTEXT_LENGTH="$chosen_ctx"
    DEVELOPER_MODEL="$PRIMARY_CODER_MODEL"
    DEVELOPER_MODELS="$PRIMARY_CODER_MODEL"
    if [[ -n "$ADDITIONAL_CODER_MODELS" ]]; then
        DEVELOPER_MODELS="$DEVELOPER_MODELS,$ADDITIONAL_CODER_MODELS"
    fi

    eval "$(select_best_model "$INFERENCE_GB" "ORCHESTRATOR_CATALOGUE[@]" "$quant" "$kv_quant")"
    ORCHESTRATOR_MODEL="$SELECTED_MODEL"

    local described_models=("primary:$PRIMARY_CODER_MODEL" "orchestrator:$ORCHESTRATOR_MODEL")
    if [[ -n "$ADDITIONAL_CODER_MODELS" ]]; then
        local extra_tag
        local -a __extra_models=()
        IFS=',' read -r -a __extra_models <<< "$ADDITIONAL_CODER_MODELS"
        for extra_tag in "${__extra_models[@]}"; do
            described_models+=("additional:$extra_tag")
        done
    fi
    describe_selected_models "$INFERENCE_GB" "$MODEL_QUANT" "$KV_CACHE_QUANT" "$TARGET_CONTEXT_LENGTH" "${described_models[@]}"

    if [[ -t 0 ]]; then
        read -r -p "Proceed with these model installs and benchmarks? [y/N] " proceed
        [[ "$proceed" =~ ^[Yy]$ ]] || return 1
    fi

    persist_model_plan
    export PRIMARY_CODER_MODEL PRIMARY_CODER_NAME ADDITIONAL_CODER_MODELS
    export MODEL_QUANT KV_CACHE_QUANT TARGET_CONTEXT_LENGTH
    export DEVELOPER_MODEL DEVELOPER_MODELS ORCHESTRATOR_MODEL
    SELECTED_MODEL="$PRIMARY_CODER_MODEL"
    SELECTED_MODEL_NAME="$PRIMARY_CODER_NAME"
    export SELECTED_MODEL SELECTED_MODEL_NAME
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    print_model_recommendation
fi
