#!/usr/bin/env bash
# install_gguf.sh — GGUF + llama.cpp deployment helpers for large local coder models.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tui.sh"
source "$SCRIPT_DIR/detect_hardware.sh"
source "$SCRIPT_DIR/install_llamacpp.sh"
source "$SCRIPT_DIR/install_claude.sh"

GGUF_DEFAULT_STATE_ROOT="${GGUF_DEFAULT_STATE_ROOT:-$HOME/.local/share/pushbutton/gguf}"
GGUF_DEFAULT_MODEL_ALIAS="${GGUF_DEFAULT_MODEL_ALIAS:-qwen3.8-27b-local}"
GGUF_DEFAULT_LOCAL_KEY="${GGUF_DEFAULT_LOCAL_KEY:-local-key}"
GGUF_DEFAULT_REASONING_EFFORT="${GGUF_DEFAULT_REASONING_EFFORT:-medium}"
GGUF_SERVER_TIMEOUT="${GGUF_SERVER_TIMEOUT:-180}"
GGUF_BENCH_REPS_DEFAULT="${GGUF_BENCH_REPS_DEFAULT:-3}"
GGUF_CHAT_TEMPLATE_REF="${GGUF_CHAT_TEMPLATE_REF:-main}"
GGUF_QWEN_CHAT_TEMPLATE_URL="${GGUF_QWEN_CHAT_TEMPLATE_URL:-https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/${GGUF_CHAT_TEMPLATE_REF}/chat_template.jinja?download=true}"

GGUF_MODEL_REPO="${GGUF_MODEL_REPO:-}"
GGUF_STATE_DIR="${GGUF_STATE_DIR:-}"
GGUF_MODEL_DIR="${GGUF_MODEL_DIR:-}"
GGUF_LOG_DIR="${GGUF_LOG_DIR:-}"
GGUF_RUN_DIR="${GGUF_RUN_DIR:-}"
GGUF_BENCH_DIR="${GGUF_BENCH_DIR:-}"
GGUF_TEMPLATE_DIR="${GGUF_TEMPLATE_DIR:-}"
GGUF_HF_HOME="${GGUF_HF_HOME:-}"
GGUF_INSTANCE="${GGUF_INSTANCE:-main}"
GGUF_INSTANCE_SAFE="${GGUF_INSTANCE_SAFE:-main}"
GGUF_PROFILE_FILE="${GGUF_PROFILE_FILE:-}"
GGUF_PID_FILE="${GGUF_PID_FILE:-}"
GGUF_SERVER_LOG="${GGUF_SERVER_LOG:-}"
GGUF_PORT="${GGUF_PORT:-8080}"
GGUF_HOST="${GGUF_HOST:-127.0.0.1}"
GGUF_MODEL_ALIAS="${GGUF_MODEL_ALIAS:-$GGUF_DEFAULT_MODEL_ALIAS}"
GGUF_LOCAL_KEY="${GGUF_LOCAL_KEY:-$GGUF_DEFAULT_LOCAL_KEY}"
GGUF_REASONING_EFFORT="${GGUF_REASONING_EFFORT:-$GGUF_DEFAULT_REASONING_EFFORT}"
GGUF_COMPACT_WINDOW="${GGUF_COMPACT_WINDOW:-180000}"
GGUF_COMPACT_PCT="${GGUF_COMPACT_PCT:-85}"
GGUF_MODEL_FAMILY="${GGUF_MODEL_FAMILY:-qwen}"
GGUF_GPU_SELECTOR="${GGUF_GPU_SELECTOR:-0}"
GGUF_GPU_NAME="${GGUF_GPU_NAME:-}"
GGUF_GPU_ARCH="${GGUF_GPU_ARCH:-0}"
GGUF_GPU_FREE_MIB="${GGUF_GPU_FREE_MIB:-0}"
GGUF_GPU_TOTAL_MIB="${GGUF_GPU_TOTAL_MIB:-0}"
GGUF_CTX="${GGUF_CTX:-}"
GGUF_KV="${GGUF_KV:-}"
GGUF_BATCH="${GGUF_BATCH:-}"
GGUF_UBATCH="${GGUF_UBATCH:-}"
GGUF_QUANT="${GGUF_QUANT:-}"
GGUF_MODEL_PATH="${GGUF_MODEL_PATH:-}"
GGUF_MODEL_FILE="${GGUF_MODEL_FILE:-}"
GGUF_CHAT_TEMPLATE_PATH="${GGUF_CHAT_TEMPLATE_PATH:-}"
GGUF_BENCH_BIN="${GGUF_BENCH_BIN:-$LLAMACPP_BIN/llama-bench}"
GGUF_SERVER_BIN="${GGUF_SERVER_BIN:-$LLAMACPP_BIN/llama-server}"

_gguf_need() {
    command -v "$1" >/dev/null 2>&1
}

_gguf_first_visible_gpu() {
    if [[ -n "${GPU:-}" ]]; then
        printf '%s\n' "$GPU"
        return 0
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        printf '%s\n' "${CUDA_VISIBLE_DEVICES%%,*}"
        return 0
    fi
    printf '0\n'
}

_gguf_map_arch_by_name() {
    case "$1" in
        *"P40"*) printf '61\n' ;;
        *"P100"*|*"Tesla P"*) printf '60\n' ;;
        *"V100"*|*"Tesla V"*) printf '70\n' ;;
        *"A100"*) printf '80\n' ;;
        *"3090"*) printf '86\n' ;;
        *"4060 Ti"*|*"4060Ti"*) printf '89\n' ;;
        *) printf '\n' ;;
    esac
}

_gguf_detect_nvidia_gpu() {
    _gguf_need nvidia-smi || {
        tui_error "nvidia-smi not found; NVIDIA driver required"
        return 1
    }

    GGUF_GPU_SELECTOR="$(_gguf_first_visible_gpu)"
    local line
    line="$(nvidia-smi -i "$GGUF_GPU_SELECTOR" --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1)"
    if [[ -z "$line" ]]; then
        line="$(nvidia-smi -i "$GGUF_GPU_SELECTOR" --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)"
    fi
    [[ -n "$line" ]] || {
        tui_error "cannot query GPU '$GGUF_GPU_SELECTOR' with nvidia-smi"
        return 1
    }

    local cc_raw=""
    IFS=',' read -r GGUF_GPU_NAME GGUF_GPU_TOTAL_MIB GGUF_GPU_FREE_MIB cc_raw <<< "$line"
    GGUF_GPU_NAME="$(printf '%s' "$GGUF_GPU_NAME" | xargs)"
    GGUF_GPU_TOTAL_MIB="$(printf '%s' "$GGUF_GPU_TOTAL_MIB" | tr -dc '0-9')"
    GGUF_GPU_FREE_MIB="$(printf '%s' "$GGUF_GPU_FREE_MIB" | tr -dc '0-9')"
    cc_raw="$(printf '%s' "$cc_raw" | xargs || true)"

    if [[ -n "$cc_raw" ]]; then
        GGUF_GPU_ARCH="$(printf '%s' "$cc_raw" | tr -d '.')"
    else
        GGUF_GPU_ARCH="$(_gguf_map_arch_by_name "$GGUF_GPU_NAME")"
    fi
    [[ -n "$GGUF_GPU_ARCH" ]] || {
        tui_error "could not determine CUDA architecture for '$GGUF_GPU_NAME'"
        return 1
    }

    tui_step "Detected GGUF target GPU"
    tui_info "selector=$GGUF_GPU_SELECTOR name=$GGUF_GPU_NAME compute=$GGUF_GPU_ARCH total=${GGUF_GPU_TOTAL_MIB}MiB free=${GGUF_GPU_FREE_MIB}MiB"
}

_gguf_model_size_from_repo() {
    local repo="${1:-}"
    if [[ "$repo" =~ ([0-9]+)(\.[0-9]+)?[[:space:]]*B ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    elif [[ "$repo" =~ ([0-9]+)(\.[0-9]+)?-?B ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    else
        printf '27\n'
    fi
}

_gguf_quant_bits() {
    case "$1" in
        Q8_0|UD-Q8_K_L) printf '8.2\n' ;;
        UD-Q6_K_XL|UD-Q6_K_L|UD-Q6_K_M|UD-Q6_K) printf '6.6\n' ;;
        UD-Q5_K_XL|UD-Q5_K_M|UD-Q5_K_S) printf '5.6\n' ;;
        UD-Q4_K_XL|UD-Q4_K_M|UD-Q4_K_S|Q4_K_M|Q4_K_S|UD-IQ4_XS) printf '4.6\n' ;;
        UD-Q3_K_XL|UD-IQ3_S|UD-IQ3_XXS) printf '3.6\n' ;;
        UD-Q2_K_XL|UD-IQ2_S|UD-IQ2_XXS) printf '2.6\n' ;;
        *) printf '4.6\n' ;;
    esac
}

_gguf_estimated_weight_mib() {
    local model_size_b="${1:-27}"
    local quant="${2:-Q4_K_M}"
    local bits
    bits="$(_gguf_quant_bits "$quant")"
    awk -v params="$model_size_b" -v bits="$bits" 'BEGIN { printf "%d\n", ((params * 1000000000 * bits / 8) / 1048576) * 1.03 }'
}

_gguf_repo_manifest() {
    local repo="${1:?repo required}"
    python3 - "$repo" <<'PY'
import json
import sys
import urllib.request

repo = sys.argv[1]
url = f"https://huggingface.co/api/models/{repo}"
with urllib.request.urlopen(url, timeout=60) as response:
    data = json.load(response)
for item in data.get("siblings", []):
    name = item.get("rfilename") or ""
    if not name.lower().endswith(".gguf"):
        continue
    size = item.get("size")
    if size is None:
        size = ((item.get("lfs") or {}).get("size"))
    size_mib = int(size / 1024 / 1024) if isinstance(size, (int, float)) else 0
    print(f"{name}|{size_mib}")
PY
}

_gguf_download_hf_file() {
    local repo="${1:?repo required}"
    local file="${2:?file required}"
    local dest="${3:?dest required}"

    if python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
        python3 - "$repo" "$file" "$dest" <<'PY'
import os
import shutil
import sys
from huggingface_hub import hf_hub_download

repo, filename, dest = sys.argv[1:4]
path = hf_hub_download(repo_id=repo, filename=filename, local_dir=os.path.dirname(dest), local_dir_use_symlinks=False)
if os.path.realpath(path) != os.path.realpath(dest):
    shutil.copy2(path, dest)
PY
        return 0
    fi

    gguf_download_file "https://huggingface.co/${repo}/resolve/main/${file}?download=true" "$dest"
}

_gguf_prepare_state() {
    local state_dir="${1:?state dir required}"
    local instance="${2:-main}"
    local model_repo="${3:-$GGUF_MODEL_REPO}"

    GGUF_STATE_DIR="$state_dir"
    GGUF_MODEL_REPO="$model_repo"
    GGUF_INSTANCE="$instance"
    GGUF_INSTANCE_SAFE="$(printf '%s' "$instance" | tr -c 'A-Za-z0-9_.-' '_')"
    GGUF_MODEL_DIR="$GGUF_STATE_DIR/models"
    GGUF_LOG_DIR="$GGUF_STATE_DIR/logs"
    GGUF_RUN_DIR="$GGUF_STATE_DIR/run"
    GGUF_BENCH_DIR="$GGUF_STATE_DIR/bench"
    GGUF_TEMPLATE_DIR="$GGUF_STATE_DIR/templates"
    GGUF_HF_HOME="$GGUF_STATE_DIR/hf-cache"
    GGUF_PROFILE_FILE="$GGUF_RUN_DIR/profile-${GGUF_INSTANCE_SAFE}.env"
    GGUF_PID_FILE="$GGUF_RUN_DIR/server-${GGUF_INSTANCE_SAFE}.pid"
    GGUF_SERVER_LOG="$GGUF_LOG_DIR/server-${GGUF_INSTANCE_SAFE}.log"
    mkdir -p "$GGUF_MODEL_DIR" "$GGUF_LOG_DIR" "$GGUF_RUN_DIR" "$GGUF_BENCH_DIR" "$GGUF_TEMPLATE_DIR" "$GGUF_HF_HOME"
}

gguf_quant_candidates() {
    local _model_size_b="${1:-27}"
    local _vram_mib="${2:-0}"
    cat <<'EOFQ'
Q8_0
UD-Q8_K_L
UD-Q6_K_XL
UD-Q6_K_L
UD-Q6_K_M
UD-Q6_K
UD-Q5_K_XL
UD-Q5_K_M
UD-Q5_K_S
UD-Q4_K_XL
UD-Q4_K_M
UD-Q4_K_S
UD-IQ4_XS
UD-Q3_K_XL
UD-IQ3_S
UD-IQ3_XXS
UD-Q2_K_XL
UD-IQ2_S
UD-IQ2_XXS
EOFQ
}

gguf_estimated_nonweight_mib() {
    local native_ctx="${1:-262144}"
    local kv_type="${2:-${GGUF_KV:-q4_0}}"
    local base
    if [[ "$kv_type" == "q8_0" ]]; then
        base=11200
    else
        base=7000
    fi
    awk -v base="$base" -v ctx="$native_ctx" 'BEGIN {
        scaled = int((base * ctx) / 262144)
        if (scaled < 1500) scaled = 1500
        printf "%d\n", scaled
    }'
}

gguf_choose_kv_type() {
    local vram_mib="${1:-0}"
    if [[ -n "${KV_TYPE:-}" ]]; then
        case "$KV_TYPE" in
            q4_0|q8_0) printf '%s\n' "$KV_TYPE"; return 0 ;;
            *) tui_error "KV_TYPE must be q4_0 or q8_0"; return 1 ;;
        esac
    fi
    if (( vram_mib >= 60000 )); then
        printf 'q8_0\n'
    else
        printf 'q4_0\n'
    fi
}

gguf_choose_batch_profile() {
    local vram_mib="${1:-0}"
    local arch="${GGUF_GPU_ARCH:-0}"
    if (( arch >= 80 && vram_mib >= 30000 )); then
        printf '2048 512\n'
    elif (( arch >= 80 || vram_mib >= 24000 )); then
        printf '1024 256\n'
    else
        printf '512 256\n'
    fi
}

gguf_prefilter_quant_list() {
    local vram_mib="${1:?vram required}"
    local native_ctx="${2:-262144}"
    local model_size_b="${3:-27}"

    if [[ -n "${QUANT:-}" ]]; then
        printf '%s\n' "$QUANT"
        return 0
    fi

    local kv budget nonweight quant weight
    kv="$(gguf_choose_kv_type "$vram_mib")"
    nonweight="$(gguf_estimated_nonweight_mib "$native_ctx" "$kv")"
    budget=$(( vram_mib - nonweight ))
    (( budget > 0 )) || return 1

    while IFS= read -r quant; do
        [[ -n "$quant" ]] || continue
        weight="$(_gguf_estimated_weight_mib "$model_size_b" "$quant")"
        if (( weight <= budget )); then
            printf '%s\n' "$quant"
        fi
    done < <(gguf_quant_candidates "$model_size_b" "$vram_mib")
}

gguf_ensure_chat_template() {
    local dest_dir="${1:?destination directory required}"
    local model_family="${2:-qwen}"
    mkdir -p "$dest_dir"

    if [[ "$model_family" != "qwen" ]]; then
        printf '\n'
        return 0
    fi

    local template_path="$dest_dir/qwen-fixed-chat-template.jinja"
    if [[ -s "$template_path" && "${REFRESH_TEMPLATE:-0}" != "1" ]]; then
        tui_info "chat template cached: $template_path"
        printf '%s\n' "$template_path"
        return 0
    fi

    tui_step "Downloading fixed Qwen chat template"
    gguf_download_file "$GGUF_QWEN_CHAT_TEMPLATE_URL" "$template_path"
    grep -q 'template_version' "$template_path" || {
        tui_error "downloaded chat template does not look valid"
        return 1
    }
    printf '%s\n' "$template_path"
}

gguf_download_file() {
    local url="${1:?url required}"
    local dest="${2:?destination required}"
    local sha256="${3:-}"
    local part="${dest}.partial"

    mkdir -p "$(dirname "$dest")"
    if [[ -s "$dest" ]]; then
        if [[ -n "$sha256" ]]; then
            local actual
            actual="$(sha256sum "$dest" | awk '{print $1}')"
            if [[ "$actual" == "$sha256" ]]; then
                tui_info "cached: $(basename "$dest")"
                return 0
            fi
            rm -f "$dest"
        else
            tui_info "cached: $(basename "$dest")"
            return 0
        fi
    fi

    tui_step "Downloading $(basename "$dest")"
    if _gguf_need curl; then
        curl -L --fail --retry 8 --retry-delay 3 --connect-timeout 30 -C - -o "$part" "$url"
    elif _gguf_need wget; then
        wget -O "$part" "$url"
    else
        tui_error "Neither curl nor wget is available"
        return 1
    fi
    mv -f "$part" "$dest"

    if [[ -n "$sha256" ]]; then
        local actual
        actual="$(sha256sum "$dest" | awk '{print $1}')"
        [[ "$actual" == "$sha256" ]] || {
            tui_error "SHA256 mismatch for $dest"
            return 1
        }
    fi
}

gguf_server_args() {
    local model_path="${1:?model path required}"
    local ctx="${2:?context required}"
    local kv_type="${3:?kv type required}"
    local batch="${4:?batch required}"
    local ubatch="${5:?ubatch required}"
    local chat_template="${6:-}"
    local port="${7:?port required}"
    local host="${8:-127.0.0.1}"

    cat <<EOFARGS
-m
$model_path
--alias
${GGUF_MODEL_ALIAS}
--host
$host
--port
$port
-c
$ctx
-ngl
999
-sm
none
-mg
0
--fit
off
--kv-offload
-fa
on
-ctk
$kv_type
-ctv
$kv_type
-b
$batch
-ub
$ubatch
-np
1
EOFARGS
    if [[ -n "$chat_template" ]]; then
        cat <<EOFARGS
--jinja
--chat-template-file
$chat_template
EOFARGS
    fi
    cat <<EOFARGS
--reasoning-format
deepseek
--reasoning-preserve
--reasoning-effort
${GGUF_REASONING_EFFORT}
--cache-prompt
--cache-reuse
256
--metrics
--api-key
${GGUF_LOCAL_KEY}
--no-mmproj
EOFARGS
}

gguf_wait_health() {
    local port="${1:?port required}"
    local timeout="${2:-$GGUF_SERVER_TIMEOUT}"
    local host="${GGUF_HOST:-127.0.0.1}"
    local elapsed=0

    while (( elapsed < timeout )); do
        if [[ -n "${GGUF_WAIT_PID:-}" ]] && ! kill -0 "$GGUF_WAIT_PID" 2>/dev/null; then
            return 1
        fi
        if curl -fsS --max-time 2 "http://${host}:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

gguf_port_in_use() {
    local port="${1:?port required}"
    local host="${GGUF_HOST:-127.0.0.1}"
    if curl -fsS --max-time 1 "http://${host}:${port}/health" >/dev/null 2>&1; then
        return 0
    fi
    if _gguf_need ss && ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
        return 0
    fi
    return 1
}

gguf_kill_pid_clean() {
    local pid_file="${1:?pid file required}"
    [[ -f "$pid_file" ]] || return 0
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        local i=0
        while kill -0 "$pid" 2>/dev/null && (( i < 20 )); do
            sleep 0.25
            i=$((i + 1))
        done
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
}

gguf_try_full_context_load() {
    local model_path="${1:?model path required}"
    local ctx="${2:?context required}"
    local kv_type="${3:?kv type required}"
    local batch="${4:?batch required}"
    local ubatch="${5:?ubatch required}"
    local template="${6:-}"
    local port="${7:?port required}"
    local host="${8:-127.0.0.1}"
    local log_file="${9:?log file required}"
    local server_bin="${GGUF_SERVER_BIN:-$LLAMACPP_BIN/llama-server}"
    local pid_file="${GGUF_PID_FILE:-$GGUF_RUN_DIR/fit-${GGUF_INSTANCE_SAFE}.pid}"

    [[ -x "$server_bin" ]] || {
        tui_error "llama-server not found at $server_bin"
        return 1
    }
    gguf_port_in_use "$port" && {
        tui_warn "port $port already in use; cannot run fit check"
        return 1
    }

    local args=()
    mapfile -t args < <(gguf_server_args "$model_path" "$ctx" "$kv_type" "$batch" "$ubatch" "$template" "$port" "$host")
    : > "$log_file"

    CUDA_VISIBLE_DEVICES="$GGUF_GPU_SELECTOR" \
    HF_HOME="$GGUF_HF_HOME" LLAMA_CACHE="$GGUF_HF_HOME" \
    "$server_bin" "${args[@]}" >"$log_file" 2>&1 &
    local pid=$!
    printf '%s\n' "$pid" > "$pid_file"
    GGUF_WAIT_PID="$pid"

    if gguf_wait_health "$port" "$GGUF_SERVER_TIMEOUT"; then
        if grep -Eqi 'out of memory|failed to allocate|cuda error' "$log_file"; then
            gguf_kill_pid_clean "$pid_file"
            return 1
        fi
        if ! grep -Eq "n_ctx(_per_seq)?[[:space:]]*=[[:space:]]*$ctx([[:space:]]|$)" "$log_file"; then
            gguf_kill_pid_clean "$pid_file"
            return 1
        fi
        local offload_line offloaded total pair
        offload_line="$(grep -E 'offloaded [0-9]+/[0-9]+ layers to GPU' "$log_file" | tail -n1 || true)"
        if [[ -z "$offload_line" ]]; then
            gguf_kill_pid_clean "$pid_file"
            return 1
        fi
        pair="$(printf '%s\n' "$offload_line" | sed -nE 's/.*offloaded ([0-9]+)\/([0-9]+) layers to GPU.*/\1 \2/p')"
        offloaded="$(printf '%s\n' "$pair" | awk '{print $1}')"
        total="$(printf '%s\n' "$pair" | awk '{print $2}')"
        if [[ -z "$offloaded" || -z "$total" || "$offloaded" != "$total" ]]; then
            gguf_kill_pid_clean "$pid_file"
            return 1
        fi
        if ! grep -Eqi '(llama_kv_cache|KV buffer).*CUDA0|CUDA0.*(llama_kv_cache|KV buffer)' "$log_file"; then
            gguf_kill_pid_clean "$pid_file"
            return 1
        fi
        export GGUF_CTX="$ctx" GGUF_KV="$kv_type" GGUF_BATCH="$batch" GGUF_UBATCH="$ubatch"
        gguf_kill_pid_clean "$pid_file"
        return 0
    fi

    gguf_kill_pid_clean "$pid_file"
    return 1
}

gguf_select_download_and_validate() {
    local model_repo="${1:?model repo required}"
    local model_dir="${2:?model dir required}"
    local vram_mib="${3:?vram required}"
    local native_ctx="${4:?context required}"
    local model_size_b
    model_size_b="$(_gguf_model_size_from_repo "$model_repo")"

    GGUF_MODEL_REPO="$model_repo"
    mkdir -p "$model_dir"

    GGUF_CHAT_TEMPLATE_PATH="$(gguf_ensure_chat_template "$GGUF_TEMPLATE_DIR" "$GGUF_MODEL_FAMILY")"
    GGUF_KV="$(gguf_choose_kv_type "$vram_mib")"
    read -r GGUF_BATCH GGUF_UBATCH < <(gguf_choose_batch_profile "$vram_mib")

    tui_step "Quant / full-context plan"
    tui_info "repo=$model_repo KV=$GGUF_KV context=$native_ctx batch=$GGUF_BATCH ubatch=$GGUF_UBATCH"

    local candidates
    candidates="$(gguf_prefilter_quant_list "$vram_mib" "$native_ctx" "$model_size_b" || true)"
    if [[ -z "$candidates" ]]; then
        candidates='UD-IQ2_XXS'
    fi

    local manifest
    manifest="$(_gguf_repo_manifest "$model_repo")"
    [[ -n "$manifest" ]] || {
        tui_error "could not list GGUF files for $model_repo"
        return 1
    }

    local quant file size attempt=0 fitlog
    while IFS= read -r quant; do
        [[ -n "$quant" ]] || continue
        file="$(printf '%s\n' "$manifest" | awk -F'|' -v q="$quant" 'toupper($1) ~ toupper(q) { print $1; exit }')"
        size="$(printf '%s\n' "$manifest" | awk -F'|' -v q="$quant" 'toupper($1) ~ toupper(q) { print $2; exit }')"
        [[ -n "$file" ]] || continue

        attempt=$((attempt + 1))
        GGUF_QUANT="$quant"
        GGUF_MODEL_FILE="$file"
        GGUF_MODEL_PATH="$model_dir/$file"
        tui_info "candidate #$attempt: $quant (~${size:-0} MiB)"
        _gguf_download_hf_file "$model_repo" "$file" "$GGUF_MODEL_PATH" || continue

        fitlog="$GGUF_LOG_DIR/fit-${GGUF_INSTANCE_SAFE}-${quant}.log"
        tui_step "Proving VRAM fit: $quant + $native_ctx context + $GGUF_KV KV"
        if gguf_try_full_context_load "$GGUF_MODEL_PATH" "$native_ctx" "$GGUF_KV" "$GGUF_BATCH" "$GGUF_UBATCH" "$GGUF_CHAT_TEMPLATE_PATH" "$GGUF_PORT" "$GGUF_HOST" "$fitlog"; then
            gguf_save_profile "$GGUF_PROFILE_FILE"
            return 0
        fi

        tui_warn "$quant failed full-context load; see $fitlog"
        if [[ -n "${QUANT:-}" ]]; then
            tui_error "forced QUANT=$QUANT failed the full-context load test"
            return 1
        fi
        _gguf_detect_nvidia_gpu >/dev/null
    done < <(printf '%s\n' "$candidates")

    tui_error "no candidate quant could pass full-context GPU validation for $model_repo"
    return 1
}

gguf_save_profile() {
    local profile_file="${1:?profile file required}"
    cat > "$profile_file" <<EOFPROFILE
GGUF_MODEL_REPO=$(printf '%q' "$GGUF_MODEL_REPO")
GGUF_MODEL_ALIAS=$(printf '%q' "$GGUF_MODEL_ALIAS")
GGUF_LOCAL_KEY=$(printf '%q' "$GGUF_LOCAL_KEY")
GGUF_MODEL_FAMILY=$(printf '%q' "$GGUF_MODEL_FAMILY")
GGUF_MODEL_DIR=$(printf '%q' "$GGUF_MODEL_DIR")
GGUF_MODEL_FILE=$(printf '%q' "$GGUF_MODEL_FILE")
GGUF_MODEL_PATH=$(printf '%q' "$GGUF_MODEL_PATH")
GGUF_QUANT=$(printf '%q' "$GGUF_QUANT")
GGUF_CTX=$(printf '%q' "$GGUF_CTX")
GGUF_KV=$(printf '%q' "$GGUF_KV")
GGUF_BATCH=$(printf '%q' "$GGUF_BATCH")
GGUF_UBATCH=$(printf '%q' "$GGUF_UBATCH")
GGUF_HOST=$(printf '%q' "$GGUF_HOST")
GGUF_PORT=$(printf '%q' "$GGUF_PORT")
GGUF_GPU_SELECTOR=$(printf '%q' "$GGUF_GPU_SELECTOR")
GGUF_GPU_NAME=$(printf '%q' "$GGUF_GPU_NAME")
GGUF_GPU_ARCH=$(printf '%q' "$GGUF_GPU_ARCH")
GGUF_GPU_FREE_MIB=$(printf '%q' "$GGUF_GPU_FREE_MIB")
GGUF_GPU_TOTAL_MIB=$(printf '%q' "$GGUF_GPU_TOTAL_MIB")
GGUF_CHAT_TEMPLATE_PATH=$(printf '%q' "$GGUF_CHAT_TEMPLATE_PATH")
GGUF_SERVER_BIN=$(printf '%q' "${GGUF_SERVER_BIN:-$LLAMACPP_BIN/llama-server}")
GGUF_BENCH_BIN=$(printf '%q' "${GGUF_BENCH_BIN:-$LLAMACPP_BIN/llama-bench}")
GGUF_PID_FILE=$(printf '%q' "$GGUF_PID_FILE")
GGUF_SERVER_LOG=$(printf '%q' "$GGUF_SERVER_LOG")
GGUF_COMPACT_WINDOW=$(printf '%q' "$GGUF_COMPACT_WINDOW")
GGUF_COMPACT_PCT=$(printf '%q' "$GGUF_COMPACT_PCT")
GGUF_REASONING_EFFORT=$(printf '%q' "$GGUF_REASONING_EFFORT")
EOFPROFILE
}

gguf_load_profile() {
    local profile_file="${1:?profile file required}"
    [[ -f "$profile_file" ]] || {
        tui_error "GGUF profile not found: $profile_file"
        return 1
    }
    # shellcheck disable=SC1090
    source "$profile_file"
}

gguf_bench() {
    local model_path="${1:?model path required}"
    local _port="${2:-$GGUF_PORT}"
    local bench_reps="${3:-$GGUF_BENCH_REPS_DEFAULT}"
    local bench_bin="${GGUF_BENCH_BIN:-$LLAMACPP_BIN/llama-bench}"
    [[ -x "$bench_bin" ]] || {
        tui_warn "llama-bench not found at $bench_bin"
        return 1
    }

    local j128="$GGUF_BENCH_DIR/${GGUF_INSTANCE_SAFE}-d128.json"
    local j4096="$GGUF_BENCH_DIR/${GGUF_INSTANCE_SAFE}-d4096.json"

    tui_step "Running GGUF smoke benchmark"
    CUDA_VISIBLE_DEVICES="$GGUF_GPU_SELECTOR" "$bench_bin" -m "$model_path" -ngl 999 -sm none -mg 0 \
        -fa on -ctk "$GGUF_KV" -ctv "$GGUF_KV" -b "$GGUF_BATCH" -ub "$GGUF_UBATCH" \
        -p 128 -n 128 -d '0,128' -r "$bench_reps" -o json > "$j128"
    CUDA_VISIBLE_DEVICES="$GGUF_GPU_SELECTOR" "$bench_bin" -m "$model_path" -ngl 999 -sm none -mg 0 \
        -fa on -ctk "$GGUF_KV" -ctv "$GGUF_KV" -b "$GGUF_BATCH" -ub "$GGUF_UBATCH" \
        -p 4096 -n 128 -d '0,4096' -r "$bench_reps" -o json > "$j4096"

    local pp128 tg128 pp4096 tg4096
    pp128="$(jq -r '[.[] | select(.n_prompt==128 and .n_depth==0) | .avg_ts][0] // empty' "$j128" 2>/dev/null || true)"
    tg128="$(jq -r '[.[] | select(.n_gen==128 and .n_depth==128) | .avg_ts][0] // empty' "$j128" 2>/dev/null || true)"
    pp4096="$(jq -r '[.[] | select(.n_prompt==4096 and .n_depth==0) | .avg_ts][0] // empty' "$j4096" 2>/dev/null || true)"
    tg4096="$(jq -r '[.[] | select(.n_gen==128 and .n_depth==4096) | .avg_ts][0] // empty' "$j4096" 2>/dev/null || true)"
    if [[ -n "$pp128" && -n "$tg128" && -n "$pp4096" && -n "$tg4096" ]]; then
        printf '\n%-14s %16s %18s\n' 'INPUT/DEPTH' 'PP tok/s' 'TG128 tok/s'
        printf '%-14s %16.2f %18.2f\n' '128' "$pp128" "$tg128"
        printf '%-14s %16.2f %18.2f\n\n' '4096' "$pp4096" "$tg4096"
    fi
}

gguf_is_running() {
    local pid_file="${1:?pid file required}"
    local port="${2:?port required}"
    [[ -f "$pid_file" ]] || return 1
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && gguf_wait_health "$port" 2
}

gguf_start_server() {
    local model_path="${1:?model path required}"
    local ctx="${2:?ctx required}"
    local kv_type="${3:?kv type required}"
    local batch="${4:?batch required}"
    local ubatch="${5:?ubatch required}"
    local template="${6:-}"
    local port="${7:?port required}"
    local host="${8:-127.0.0.1}"
    local pid_file="${9:?pid file required}"
    local log_file="${10:?log file required}"
    local server_bin="${GGUF_SERVER_BIN:-$LLAMACPP_BIN/llama-server}"

    [[ -x "$server_bin" ]] || {
        tui_error "llama-server not found at $server_bin"
        return 1
    }

    if gguf_is_running "$pid_file" "$port"; then
        tui_info "server already healthy: http://${host}:${port}"
        return 0
    fi

    gguf_port_in_use "$port" && {
        tui_error "port $port is already in use"
        return 1
    }
    local args=()
    mapfile -t args < <(gguf_server_args "$model_path" "$ctx" "$kv_type" "$batch" "$ubatch" "$template" "$port" "$host")
    : > "$log_file"
    CUDA_VISIBLE_DEVICES="$GGUF_GPU_SELECTOR" \
    HF_HOME="$GGUF_HF_HOME" LLAMA_CACHE="$GGUF_HF_HOME" \
    nohup "$server_bin" "${args[@]}" >"$log_file" 2>&1 &
    local pid=$!
    printf '%s\n' "$pid" > "$pid_file"
    GGUF_WAIT_PID="$pid"
    if ! gguf_wait_health "$port" "$GGUF_SERVER_TIMEOUT"; then
        gguf_kill_pid_clean "$pid_file"
        tail -n 120 "$log_file" >&2 || true
        tui_error "server failed to become healthy"
        return 1
    fi
    tui_success "GGUF server healthy at http://${host}:${port}"
}

gguf_stop_server() {
    local pid_file="${1:?pid file required}"
    gguf_kill_pid_clean "$pid_file"
}

gguf_api_smoke() {
    local port="${1:?port required}"
    local host="${GGUF_HOST:-127.0.0.1}"
    local models_out="$GGUF_LOG_DIR/api-models-${GGUF_INSTANCE_SAFE}.json"
    local chat_out="$GGUF_LOG_DIR/api-chat-${GGUF_INSTANCE_SAFE}.json"
    local request_file="$GGUF_RUN_DIR/api-chat-${GGUF_INSTANCE_SAFE}.json"

    curl -fsS "http://${host}:${port}/v1/models" > "$models_out"
    jq -e '.data[0].id' "$models_out" >/dev/null 2>&1 || {
        tui_error "/v1/models smoke failed"
        return 1
    }

    cat > "$request_file" <<EOFJSON
{"model":"${GGUF_MODEL_ALIAS}","messages":[{"role":"user","content":"Reply with GGUF_API_SMOKE_OK."}],"temperature":0,"max_tokens":64}
EOFJSON
    local code
    code="$(curl -sS --max-time 300 -o "$chat_out" -w '%{http_code}' -H 'Content-Type: application/json' --data-binary "@$request_file" "http://${host}:${port}/v1/chat/completions")"
    [[ "$code" == "200" ]] || {
        cat "$chat_out" >&2
        tui_error "/v1/chat/completions returned HTTP $code"
        return 1
    }
    grep -q 'GGUF_API_SMOKE_OK' "$chat_out" || {
        tui_warn "chat smoke did not contain expected marker"
        return 1
    }
    tui_success "GGUF OpenAI-compatible API smoke passed"
}

gguf_export_claude_env() {
    local port="${1:?port required}"
    local compact_window="${2:-$GGUF_COMPACT_WINDOW}"
    local compact_pct="${3:-$GGUF_COMPACT_PCT}"
    local model_alias="${4:-$GGUF_MODEL_ALIAS}"
    local local_key="${5:-$GGUF_LOCAL_KEY}"
    local host="${GGUF_HOST:-127.0.0.1}"

    export ANTHROPIC_BASE_URL="http://${host}:${port}"
    export ANTHROPIC_API_KEY="$local_key"
    unset ANTHROPIC_AUTH_TOKEN
    export ANTHROPIC_MODEL="$model_alias"
    export ANTHROPIC_DEFAULT_SONNET_MODEL="$model_alias"
    export ANTHROPIC_DEFAULT_OPUS_MODEL="$model_alias"
    export ANTHROPIC_DEFAULT_HAIKU_MODEL="$model_alias"
    export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$GGUF_CTX"
    export CLAUDE_CODE_AUTO_COMPACT_WINDOW="$compact_window"
    export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="$compact_pct"
    unset DISABLE_AUTO_COMPACT
    unset DISABLE_COMPACT
    export CLAUDE_CODE_ATTRIBUTION_HEADER=0
    export CLAUDE_CODE_ENABLE_TELEMETRY=0
    export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
    export API_TIMEOUT_MS="${API_TIMEOUT_MS:-1800000}"
}

gguf_claude_smoke() {
    if ! _gguf_need claude; then
        tui_warn "Claude Code executable not installed"
        return 0
    fi
    claude --version
}

gguf_claude_exec() {
    if ! _gguf_need claude; then
        tui_error "Claude Code not found"
        return 1
    fi
    if [[ ! -r /dev/tty ]]; then
        tui_warn "No controlling TTY; skipping interactive Claude session"
        claude --version
        return 0
    fi
    tui_step "Launching Claude Code against local GGUF endpoint"
    claude --model "$GGUF_MODEL_ALIAS" < /dev/tty > /dev/tty 2> /dev/tty
}

setup_qwen38_cuda() {
    local model_repo="${1:-unsloth/Qwen3.8-27B-GGUF}"
    local state_dir="${2:-$GGUF_DEFAULT_STATE_ROOT/qwen38}"
    local port="${3:-8080}"
    local instance="${4:-main}"

    GGUF_MODEL_ALIAS="${GGUF_MODEL_ALIAS:-qwen3.8-27b-local}"
    GGUF_MODEL_FAMILY="qwen"
    GGUF_PORT="$port"
    GGUF_HOST="${GGUF_HOST:-127.0.0.1}"
    _gguf_prepare_state "$state_dir" "$instance" "$model_repo"
    _gguf_detect_nvidia_gpu

    if [[ "${GGUF_SKIP_BUILD:-0}" != "1" ]]; then
        build_llamacpp
    fi
    GGUF_SERVER_BIN="${GGUF_SERVER_BIN:-$LLAMACPP_BIN/llama-server}"
    GGUF_BENCH_BIN="${GGUF_BENCH_BIN:-$LLAMACPP_BIN/llama-bench}"

    local native_ctx="${NATIVE_CTX:-262144}"
    gguf_select_download_and_validate "$model_repo" "$GGUF_MODEL_DIR" "$GGUF_GPU_FREE_MIB" "$native_ctx"
    gguf_bench "$GGUF_MODEL_PATH" "$GGUF_PORT" "${BENCH_REPS:-$GGUF_BENCH_REPS_DEFAULT}" || true
    gguf_start_server "$GGUF_MODEL_PATH" "$GGUF_CTX" "$GGUF_KV" "$GGUF_BATCH" "$GGUF_UBATCH" "$GGUF_CHAT_TEMPLATE_PATH" "$GGUF_PORT" "$GGUF_HOST" "$GGUF_PID_FILE" "$GGUF_SERVER_LOG"
    gguf_api_smoke "$GGUF_PORT"
    setup_claude || true
    gguf_export_claude_env "$GGUF_PORT" "$GGUF_COMPACT_WINDOW" "$GGUF_COMPACT_PCT" "$GGUF_MODEL_ALIAS" "$GGUF_LOCAL_KEY"
    gguf_claude_smoke || true
    gguf_claude_exec || true
}
