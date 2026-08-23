#!/usr/bin/env bash
# Qwen3.8-27B + llama.cpp + local Claude Code, single-NVIDIA-GPU pushbutton deployer.
#
# Goals:
#   * Everything owned by this script lives beside this script.
#   * Detect RTX 3090 / V100 / P40 / A100 / RTX 4060 Ti (and generally other NVIDIA GPUs).
#   * Build a GPU-specific llama.cpp CUDA binary from current upstream.
#   * Choose the highest-quality GGUF quant that can plausibly coexist with the model's
#     full native 262,144-token context in VRAM, then PROVE the choice by loading it with
#     --fit off, all model layers + KV on GPU. Fall back to smaller quants on OOM/load failure.
#   * Run PP/TG smoke benchmarks at 128 and 4096 input/context tokens.
#   * Expose llama.cpp's Anthropic-compatible /v1/messages endpoint to Claude Code.
#   * Keep Claude Code's effective auto-compaction window safely below server context.
#
# Quick start:
#   chmod +x ./qwen38_llamacpp_claude.sh
#   CUDA_VISIBLE_DEVICES=0 ./qwen38_llamacpp_claude.sh install
#   ./qwen38_llamacpp_claude.sh claude
#
# Second physical GPU (independent server/model):
#   CUDA_VISIBLE_DEVICES=1 INSTANCE=agent2 PORT=8081 ./qwen38_llamacpp_claude.sh install
#   INSTANCE=agent2 PORT=8081 ./qwen38_llamacpp_claude.sh claude
#
# Same GPU, second independent Claude Code agent: share the same model server (no second
# copy of 27B weights/KV in VRAM): launch ./qwen38_llamacpp_claude.sh claude in another shell.
#
# Useful overrides:
#   QUANT=UD-Q4_K_M        force a quant (fit is still verified)
#   KV_TYPE=q4_0           force q4_0 or q8_0 KV
#   LLAMA_REF=master       llama.cpp git ref (default master)
#   PORT=8080 INSTANCE=main
#   COMPACT_WINDOW=180000  Claude Code effective auto-compaction capacity
#   COMPACT_PCT=85         secondary early-trigger hint
#   BENCH_REPS=3
#   AUTO_DEPS=1            attempt distro package install for ordinary build tools
#   CUDA_HOME=/usr/local/cuda-12.9
#
# Deliberately NOT using `set -euo pipefail`: failures are checked at the operation where
# they matter so diagnostics remain readable and cleanup can run.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
STATE_DIR="$SCRIPT_DIR/.qwen38-llamacpp-claude"
SRC_DIR="$STATE_DIR/llama.cpp"
BUILD_ROOT="$STATE_DIR/builds"
MODEL_DIR="$STATE_DIR/models"
LOG_DIR="$STATE_DIR/logs"
RUN_DIR="$STATE_DIR/run"
BENCH_DIR="$STATE_DIR/bench"
HF_HOME="$STATE_DIR/hf-cache"
TMP_ROOT="$STATE_DIR/tmp"
CLAUDE_ROOT="$STATE_DIR/claude"
TEMPLATE_DIR="$STATE_DIR/templates"

MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.8-27B-GGUF}"
MODEL_BASE_URL="https://huggingface.co/${MODEL_REPO}/resolve/main"
NATIVE_CTX="${NATIVE_CTX:-262144}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.8-27b-local}"
LOCAL_KEY="${LOCAL_KEY:-local-key}"
LLAMA_REF="${LLAMA_REF:-master}"
CHAT_TEMPLATE_REF="${CHAT_TEMPLATE_REF:-main}"
CHAT_TEMPLATE_URL="${CHAT_TEMPLATE_URL:-https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/${CHAT_TEMPLATE_REF}/chat_template.jinja?download=true}"
CHAT_TEMPLATE_PATH="$TEMPLATE_DIR/qwen-fixed-chat-template.jinja"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
INSTANCE="${INSTANCE:-main}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
AUTO_DEPS="${AUTO_DEPS:-1}"
BENCH_REPS="${BENCH_REPS:-3}"
COMPACT_WINDOW="${COMPACT_WINDOW:-180000}"
COMPACT_PCT="${COMPACT_PCT:-85}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-180}"

INSTANCE_SAFE="$(printf '%s' "$INSTANCE" | tr -c 'A-Za-z0-9_.-' '_')"
PID_FILE="$RUN_DIR/server-${INSTANCE_SAFE}.pid"
CFG_FILE="$RUN_DIR/profile-${INSTANCE_SAFE}.env"
SERVER_LOG="$LOG_DIR/server-${INSTANCE_SAFE}.log"
CLAUDE_CONFIG_DIR_LOCAL="$CLAUDE_ROOT/${INSTANCE_SAFE}"
CLAUDE_TMP_DIR_LOCAL="$TMP_ROOT/claude-${INSTANCE_SAFE}"

mkdir -p "$STATE_DIR" "$BUILD_ROOT" "$MODEL_DIR" "$LOG_DIR" "$RUN_DIR" "$BENCH_DIR" "$HF_HOME" "$TMP_ROOT" "$CLAUDE_ROOT" "$TEMPLATE_DIR" "$CLAUDE_CONFIG_DIR_LOCAL" "$CLAUDE_TMP_DIR_LOCAL" || {
  echo "ERROR: cannot create state directories beside script: $STATE_DIR" >&2
  exit 2
}

say()  { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || return 1; }

print_header() {
  cat <<HDR
Qwen3.8-27B / llama.cpp / Claude Code local deployer
  script dir : $SCRIPT_DIR
  state dir  : $STATE_DIR
  instance   : $INSTANCE
  endpoint   : http://$HOST:$PORT
  native ctx : $NATIVE_CTX tokens
HDR
}

install_basic_deps() {
  local missing=()
  local c
  for c in git cmake curl jq gcc g++ make; do
    need "$c" || missing+=("$c")
  done
  if [ ${#missing[@]} -eq 0 ]; then
    return 0
  fi

  [ "$AUTO_DEPS" = "1" ] || die "missing tools: ${missing[*]}. Install them or rerun with AUTO_DEPS=1."
  say "Installing ordinary build dependencies (${missing[*]})"

  local SUDO=""
  if [ "$(id -u)" != "0" ]; then
    need sudo || die "need root/sudo to install dependencies: ${missing[*]}"
    SUDO="sudo"
  fi

  if need apt-get; then
    $SUDO apt-get update || die "apt-get update failed"
    $SUDO apt-get install -y build-essential cmake ninja-build git curl jq ca-certificates || die "apt dependency install failed"
  elif need dnf; then
    $SUDO dnf install -y gcc gcc-c++ cmake ninja-build make git curl jq ca-certificates || die "dnf dependency install failed"
  elif need pacman; then
    $SUDO pacman -Sy --needed --noconfirm base-devel cmake ninja git curl jq ca-certificates || die "pacman dependency install failed"
  else
    die "missing tools (${missing[*]}) and unsupported package manager; install them manually"
  fi
}

first_visible_gpu() {
  # CUDA_VISIBLE_DEVICES can contain UUIDs. nvidia-smi -i accepts both indices and UUIDs.
  if [ -n "${GPU:-}" ]; then
    printf '%s\n' "$GPU"
    return
  fi
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    printf '%s\n' "${CUDA_VISIBLE_DEVICES%%,*}"
    return
  fi
  printf '0\n'
}

GPU_SELECTOR=""
GPU_NAME=""
GPU_TOTAL_MIB=""
GPU_FREE_MIB=""
GPU_CC_RAW=""
GPU_ARCH=""

map_arch_by_name() {
  case "$1" in
    *"P40"*) printf '61\n' ;;
    *"V100"*) printf '70\n' ;;
    *"A100"*) printf '80\n' ;;
    *"3090"*) printf '86\n' ;;
    *"4060 Ti"*|*"4060Ti"*) printf '89\n' ;;
    *) printf '\n' ;;
  esac
}

detect_gpu() {
  need nvidia-smi || die "nvidia-smi not found; an NVIDIA driver is required"
  GPU_SELECTOR="$(first_visible_gpu)"

  local line
  line="$(nvidia-smi -i "$GPU_SELECTOR" --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1)"
  if [ -z "$line" ]; then
    # Older nvidia-smi may not expose compute_cap query.
    line="$(nvidia-smi -i "$GPU_SELECTOR" --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)"
    [ -n "$line" ] || die "cannot query GPU '$GPU_SELECTOR' with nvidia-smi"
    IFS=',' read -r GPU_NAME GPU_TOTAL_MIB GPU_FREE_MIB <<< "$line"
    GPU_CC_RAW=""
  else
    IFS=',' read -r GPU_NAME GPU_TOTAL_MIB GPU_FREE_MIB GPU_CC_RAW <<< "$line"
  fi

  GPU_NAME="$(printf '%s' "$GPU_NAME" | xargs)"
  GPU_TOTAL_MIB="$(printf '%s' "$GPU_TOTAL_MIB" | tr -dc '0-9')"
  GPU_FREE_MIB="$(printf '%s' "$GPU_FREE_MIB" | tr -dc '0-9')"
  GPU_CC_RAW="$(printf '%s' "$GPU_CC_RAW" | xargs)"

  [ -n "$GPU_TOTAL_MIB" ] && [ -n "$GPU_FREE_MIB" ] || die "could not parse GPU VRAM from nvidia-smi"

  if [ -n "$GPU_CC_RAW" ]; then
    GPU_ARCH="$(printf '%s' "$GPU_CC_RAW" | tr -d '.')"
  else
    GPU_ARCH="$(map_arch_by_name "$GPU_NAME")"
  fi
  [ -n "$GPU_ARCH" ] || die "could not determine CUDA compute capability for '$GPU_NAME'"

  case "$GPU_NAME" in
    *"RTX 3090"*|*"V100"*|*"P40"*|*"A100"*|*"RTX 4060 Ti"*) : ;;
    *) warn "GPU '$GPU_NAME' is not one of the five named targets; generic NVIDIA path will be used." ;;
  esac

  local used=$((GPU_TOTAL_MIB - GPU_FREE_MIB))
  say "Detected target GPU"
  info "selector=$GPU_SELECTOR  name=$GPU_NAME  compute=$GPU_ARCH  total=${GPU_TOTAL_MIB} MiB  free=${GPU_FREE_MIB} MiB  used=${used} MiB"
  if [ "$used" -gt 1024 ]; then
    warn "GPU already has >1 GiB allocated. Quant selection uses CURRENT FREE VRAM, so freeing other workloads may permit a better quant."
  fi
}

find_nvcc() {
  local candidates=()
  [ -n "${CUDA_HOME:-}" ] && candidates+=("$CUDA_HOME/bin/nvcc")
  [ -n "${CUDACXX:-}" ] && candidates+=("$CUDACXX")
  # Prefer CUDA 12.x first, especially because CUDA 13 dropped offline compilation for cc < 7.5.
  local p
  for p in /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda-12.8/bin/nvcc /usr/local/cuda-12.6/bin/nvcc /usr/local/cuda-12.5/bin/nvcc /usr/local/cuda-12.4/bin/nvcc /usr/local/cuda-12.3/bin/nvcc /usr/local/cuda-12.2/bin/nvcc /usr/local/cuda-12.1/bin/nvcc /usr/local/cuda-12.0/bin/nvcc /usr/local/cuda/bin/nvcc; do
    candidates+=("$p")
  done
  if need nvcc; then candidates+=("$(command -v nvcc)"); fi

  local nv=""
  for p in "${candidates[@]}"; do
    if [ -x "$p" ]; then nv="$p"; break; fi
  done
  [ -n "$nv" ] || die "nvcc not found. Install a CUDA toolkit. P40/V100 require CUDA 12.x (CUDA 13 cannot compile cc ${GPU_ARCH})."

  local release major
  release="$($nv --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | tail -n1)"
  major="${release:-0}"
  if [ "$GPU_ARCH" -lt 75 ] && [ "$major" -ge 13 ]; then
    die "selected nvcc is CUDA ${major}, which cannot offline-compile for cc ${GPU_ARCH}. Set CUDA_HOME to a CUDA 12.x toolkit."
  fi
  printf '%s\n' "$nv"
}

NVCC=""
BUILD_DIR=""
SERVER_BIN=""
BENCH_BIN=""
CLI_BIN=""
LOCAL_LD_PATH=""

build_llama() {
  install_basic_deps
  detect_gpu
  NVCC="$(find_nvcc)"
  BUILD_DIR="$BUILD_ROOT/cc${GPU_ARCH}"

  say "Fetching current llama.cpp ($LLAMA_REF)"
  if [ ! -d "$SRC_DIR/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$SRC_DIR" || die "git clone llama.cpp failed"
  fi
  (
    cd "$SRC_DIR" || exit 1
    git fetch --tags --prune origin || exit 1
    git fetch origin "$LLAMA_REF" || exit 1
    git checkout --detach FETCH_HEAD || exit 1
  ) || die "failed to update llama.cpp to '$LLAMA_REF'"

  local commit
  commit="$(git -C "$SRC_DIR" rev-parse --short=12 HEAD 2>/dev/null)"
  info "llama.cpp commit=$commit"
  info "nvcc=$NVCC"
  info "CUDA architecture=$GPU_ARCH"

  say "Configuring CUDA build"
  mkdir -p "$BUILD_DIR" || die "cannot create $BUILD_DIR"
  CUDACXX="$NVCC" cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DGGML_BUILD_TESTS=OFF \
    -DGGML_BUILD_EXAMPLES=OFF \
    -DGGML_CUDA_GRAPHS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -G Ninja >/dev/null 2>"$LOG_DIR/cmake-cc${GPU_ARCH}.log"
  if [ $? -ne 0 ]; then
    warn "Ninja configure failed; retrying with the default generator. See $LOG_DIR/cmake-cc${GPU_ARCH}.log"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    CUDACXX="$NVCC" cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=ON \
      -DGGML_BUILD_TESTS=OFF \
      -DGGML_BUILD_EXAMPLES=OFF \
      -DGGML_CUDA_GRAPHS=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >"$LOG_DIR/cmake-cc${GPU_ARCH}.log" 2>&1 || die "cmake configure failed; see $LOG_DIR/cmake-cc${GPU_ARCH}.log"
  fi

  say "Building llama-server, llama-cli, llama-bench"
  cmake --build "$BUILD_DIR" --config Release -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" --target llama-server llama-cli llama-bench || die "llama.cpp build failed"

  SERVER_BIN="$BUILD_DIR/bin/llama-server"
  BENCH_BIN="$BUILD_DIR/bin/llama-bench"
  CLI_BIN="$BUILD_DIR/bin/llama-cli"
  [ -x "$SERVER_BIN" ] && [ -x "$BENCH_BIN" ] && [ -x "$CLI_BIN" ] || die "expected llama.cpp binaries were not produced"

  LOCAL_LD_PATH="$BUILD_DIR/bin:$BUILD_DIR/src:$BUILD_DIR/ggml/src:$BUILD_DIR/ggml/src/ggml-cuda"
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then LOCAL_LD_PATH="$LOCAL_LD_PATH:$LD_LIBRARY_PATH"; fi

  say "ABI/library audit"
  LD_LIBRARY_PATH="$LOCAL_LD_PATH" ldd "$SERVER_BIN" >"$LOG_DIR/ldd-cc${GPU_ARCH}.txt" 2>&1 || die "ldd failed"
  if grep -q 'not found' "$LOG_DIR/ldd-cc${GPU_ARCH}.txt"; then
    cat "$LOG_DIR/ldd-cc${GPU_ARCH}.txt" >&2
    die "unresolved llama.cpp shared library; refusing to risk stale/mixed CUDA libraries"
  fi
  if ! LD_LIBRARY_PATH="$LOCAL_LD_PATH" "$SERVER_BIN" --version >"$LOG_DIR/version-cc${GPU_ARCH}.txt" 2>&1; then
    cat "$LOG_DIR/version-cc${GPU_ARCH}.txt" >&2
    die "fresh llama-server failed to run"
  fi
  info "build OK: $(head -n1 "$LOG_DIR/version-cc${GPU_ARCH}.txt")"
}

load_existing_build() {
  detect_gpu
  BUILD_DIR="$BUILD_ROOT/cc${GPU_ARCH}"
  SERVER_BIN="$BUILD_DIR/bin/llama-server"
  BENCH_BIN="$BUILD_DIR/bin/llama-bench"
  CLI_BIN="$BUILD_DIR/bin/llama-cli"
  LOCAL_LD_PATH="$BUILD_DIR/bin:$BUILD_DIR/src:$BUILD_DIR/ggml/src:$BUILD_DIR/ggml/src/ggml-cuda"
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then LOCAL_LD_PATH="$LOCAL_LD_PATH:$LD_LIBRARY_PATH"; fi
  [ -x "$SERVER_BIN" ] && [ -x "$BENCH_BIN" ] || die "no build for cc${GPU_ARCH}; run '$0 build' or '$0 install'"
}

# Candidate order: quality-first. Size is conservative approximate MiB based on repository file size.
# The authoritative gate is a real full-context llama-server load with --fit off.
quant_candidates() {
  cat <<'QEOF'
Q8_0|Qwen3.8-27B-Q8_0.gguf|28200
UD-Q8_K_L|Qwen3.8-27B-UD-Q8_K_L.gguf|27400
UD-Q6_K_XL|Qwen3.8-27B-UD-Q6_K_XL.gguf|24800
UD-Q6_K_L|Qwen3.8-27B-UD-Q6_K_L.gguf|23800
UD-Q6_K_M|Qwen3.8-27B-UD-Q6_K_M.gguf|22800
UD-Q6_K|Qwen3.8-27B-UD-Q6_K.gguf|21800
UD-Q5_K_XL|Qwen3.8-27B-UD-Q5_K_XL.gguf|20700
UD-Q5_K_M|Qwen3.8-27B-UD-Q5_K_M.gguf|19700
UD-Q5_K_S|Qwen3.8-27B-UD-Q5_K_S.gguf|18700
UD-Q4_K_XL|Qwen3.8-27B-UD-Q4_K_XL.gguf|17600
UD-Q4_K_M|Qwen3.8-27B-UD-Q4_K_M.gguf|16600
UD-Q4_K_S|Qwen3.8-27B-UD-Q4_K_S.gguf|15600
UD-IQ4_XS|Qwen3.8-27B-UD-IQ4_XS.gguf|14500
UD-Q3_K_XL|Qwen3.8-27B-UD-Q3_K_XL.gguf|13300
UD-IQ3_S|Qwen3.8-27B-UD-IQ3_S.gguf|12200
UD-IQ3_XXS|Qwen3.8-27B-UD-IQ3_XXS.gguf|11100
UD-Q2_K_XL|Qwen3.8-27B-UD-Q2_K_XL.gguf|10000
UD-IQ2_S|Qwen3.8-27B-UD-IQ2_S.gguf|8500
UD-IQ2_XXS|Qwen3.8-27B-UD-IQ2_XXS.gguf|7400
QEOF
}

quant_row() {
  local q="$1"
  quant_candidates | awk -F'|' -v q="$q" '$1==q {print; exit}'
}

# Qwen3.8-27B has only 16 full-attention layers, but at 262,144 tokens even a q4 KV cache
# is several GiB. These are intentionally conservative PRE-FILTER estimates; real loading proves fit.
estimated_nonweight_mib() {
  local kv="$1"
  if [ "$kv" = "q8_0" ]; then
    printf '11200\n'   # ~8.5 GiB attention KV + recurrent/scratch + safety
  else
    printf '7000\n'    # ~4.5 GiB attention KV + recurrent/scratch + safety
  fi
}

choose_kv_type() {
  if [ -n "${KV_TYPE:-}" ]; then
    case "$KV_TYPE" in q4_0|q8_0) printf '%s\n' "$KV_TYPE";; *) die "KV_TYPE must be q4_0 or q8_0";; esac
    return
  fi
  # q8 KV is worthwhile only with ample room; otherwise q4 buys a much better weight quant.
  if [ "$GPU_FREE_MIB" -ge 60000 ]; then printf 'q8_0\n'; else printf 'q4_0\n'; fi
}

KV_CHOSEN=""
QUANT_CHOSEN=""
MODEL_FILE=""
MODEL_PATH=""
BATCH_SIZE=""
UBATCH_SIZE=""

choose_batch_profile() {
  # Conservative scratch on old/small GPUs; larger batches on Ampere+ with room.
  if [ "$GPU_ARCH" -ge 80 ] && [ "$GPU_FREE_MIB" -ge 30000 ]; then
    BATCH_SIZE="${BATCH_SIZE:-2048}"
    UBATCH_SIZE="${UBATCH_SIZE:-512}"
  elif [ "$GPU_ARCH" -ge 80 ]; then
    BATCH_SIZE="${BATCH_SIZE:-1024}"
    UBATCH_SIZE="${UBATCH_SIZE:-256}"
  else
    BATCH_SIZE="${BATCH_SIZE:-512}"
    UBATCH_SIZE="${UBATCH_SIZE:-256}"
  fi
}

prefilter_quant_list() {
  local kv="$1" nonweight budget
  nonweight="$(estimated_nonweight_mib "$kv")"
  budget=$((GPU_FREE_MIB - nonweight))
  [ "$budget" -gt 0 ] || return 1

  if [ -n "${QUANT:-}" ]; then
    local row
    row="$(quant_row "$QUANT")"
    [ -n "$row" ] || die "unknown forced QUANT='$QUANT'"
    printf '%s\n' "$row"
    return 0
  fi

  quant_candidates | awk -F'|' -v b="$budget" '$3 <= b {print}'
}

ensure_chat_template() {
  if [ -s "$CHAT_TEMPLATE_PATH" ] && [ "${REFRESH_TEMPLATE:-0}" != "1" ]; then
    info "chat template cached: $CHAT_TEMPLATE_PATH"
    return 0
  fi
  say "Downloading Claude-Code-compatible Qwen3.8 Jinja template"
  info "source: froggeric/Qwen-Fixed-Chat-Templates ref=$CHAT_TEMPLATE_REF"
  local part="${CHAT_TEMPLATE_PATH}.partial"
  curl -L --fail --retry 6 --retry-delay 2 --connect-timeout 30 \
    -o "$part" "$CHAT_TEMPLATE_URL" || die "failed to download fixed Qwen chat template"
  grep -q 'template_version' "$part" || die "downloaded chat template does not look valid"
  mv -f "$part" "$CHAT_TEMPLATE_PATH" || die "cannot save chat template"
  local sha
  sha="$(sha256sum "$CHAT_TEMPLATE_PATH" 2>/dev/null | awk '{print $1}')"
  info "template SHA256=${sha:-unavailable}"
}

download_file() {
  local rel="$1" dest="$2"
  if [ -s "$dest" ]; then
    info "cached: $(basename "$dest") ($(du -h "$dest" | awk '{print $1}'))"
    return 0
  fi
  mkdir -p "$(dirname "$dest")" || return 1
  say "Downloading $(basename "$dest") into script-local cache"
  info "$MODEL_REPO/$rel"
  local part="${dest}.partial"
  curl -L --fail --retry 8 --retry-delay 3 --connect-timeout 30 -C - \
    -o "$part" "$MODEL_BASE_URL/$rel?download=true"
  if [ $? -ne 0 ]; then
    warn "download failed: $rel"
    return 1
  fi
  mv -f "$part" "$dest" || return 1
  [ -s "$dest" ] || return 1
}

server_args_common() {
  # Output one arg per line, consumed with mapfile. --fit off is NON-NEGOTIABLE: it prevents
  # llama.cpp from silently shrinking context/offload choices to fit memory.
  cat <<ARGS
-m
$MODEL_PATH
--alias
$MODEL_ALIAS
--host
$HOST
--port
$PORT
-c
$NATIVE_CTX
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
$KV_CHOSEN
-ctv
$KV_CHOSEN
-b
$BATCH_SIZE
-ub
$UBATCH_SIZE
-np
1
--jinja
--chat-template-file
$CHAT_TEMPLATE_PATH
--reasoning-format
deepseek
--reasoning-preserve
--reasoning-effort
$REASONING_EFFORT
--cache-prompt
--cache-reuse
256
--metrics
--api-key
$LOCAL_KEY
--no-mmproj
ARGS
}

wait_health_pid() {
  local pid="$1" seconds=0
  while [ "$seconds" -lt "$SERVER_TIMEOUT" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
    if curl -fsS --max-time 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then return 0; fi
    sleep 1
    seconds=$((seconds + 1))
  done
  return 1
}

port_in_use() {
  curl -fsS --max-time 1 "http://$HOST:$PORT/health" >/dev/null 2>&1
}

kill_pid_clean() {
  local pid="$1"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    local i=0
    while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 20 ]; do sleep 0.25; i=$((i+1)); done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

try_full_context_load() {
  local logfile="$1"
  if port_in_use; then
    warn "port $PORT is already serving /health; stop that service before fit validation"
    return 1
  fi
  local args=()
  mapfile -t args < <(server_args_common)
  : >"$logfile"

  # CUDA_VISIBLE_DEVICES presents exactly the chosen physical GPU as logical CUDA device 0.
  CUDA_VISIBLE_DEVICES="$GPU_SELECTOR" \
  HF_HOME="$HF_HOME" LLAMA_CACHE="$HF_HOME" \
  LD_LIBRARY_PATH="$LOCAL_LD_PATH" \
    "$SERVER_BIN" "${args[@]}" >"$logfile" 2>&1 &
  local pid=$!

  if wait_health_pid "$pid"; then
    # Context allocation happens at startup; health here is already a real allocation test.
    # Tighten that proof by requiring the requested context and complete layer offload in logs.
    if grep -Eqi 'out of memory|failed to allocate|cuda error' "$logfile"; then
      kill_pid_clean "$pid"
      return 1
    fi

    if ! grep -Eq "n_ctx(_per_seq)?[[:space:]]*=[[:space:]]*$NATIVE_CTX([[:space:]]|$)" "$logfile"; then
      warn "fit process became healthy but did not log n_ctx=$NATIVE_CTX"
      kill_pid_clean "$pid"
      return 1
    fi

    local offload_line offload_pair offloaded total
    offload_line="$(grep -E 'offloaded [0-9]+/[0-9]+ layers to GPU' "$logfile" | tail -n 1)"
    if [ -z "$offload_line" ]; then
      warn "fit process did not report complete GPU layer offload"
      kill_pid_clean "$pid"
      return 1
    fi
    offload_pair="$(printf '%s\n' "$offload_line" | sed -nE 's/.*offloaded ([0-9]+)\/([0-9]+) layers to GPU.*/\1 \2/p')"
    offloaded="$(printf '%s\n' "$offload_pair" | awk '{print $1}')"
    total="$(printf '%s\n' "$offload_pair" | awk '{print $2}')"
    if [ -z "$offloaded" ] || [ -z "$total" ] || [ "$offloaded" != "$total" ]; then
      warn "partial model offload detected: ${offloaded:-?}/${total:-?} layers"
      kill_pid_clean "$pid"
      return 1
    fi

    # --kv-offload + exactly one visible CUDA device requests GPU KV/state allocation. Confirm
    # current llama.cpp's startup log sees a CUDA KV/cache buffer as an additional sanity check.
    if ! grep -Eqi '(llama_kv_cache|KV buffer).*CUDA0|CUDA0.*(llama_kv_cache|KV buffer)' "$logfile"; then
      warn "no CUDA0 KV/cache-buffer marker found in startup log"
      kill_pid_clean "$pid"
      return 1
    fi

    info "fit proof: n_ctx=$NATIVE_CTX, layers=$offloaded/$total on GPU, KV/cache on CUDA0"
    kill_pid_clean "$pid"
    return 0
  fi
  kill_pid_clean "$pid"
  return 1
}

select_download_and_validate() {
  load_existing_build
  ensure_chat_template
  KV_CHOSEN="$(choose_kv_type)"
  choose_batch_profile

  say "Quant / full-native-context plan"
  info "KV=$KV_CHOSEN  context=$NATIVE_CTX  batch=$BATCH_SIZE  ubatch=$UBATCH_SIZE"
  info "Pre-filter is conservative; the decisive test is a real --fit off full-context GPU load."

  local rows row q file size attempt=0
  rows="$(prefilter_quant_list "$KV_CHOSEN")"
  if [ -z "$rows" ]; then
    # On very constrained/free-memory systems, attempt the smallest quant anyway; real load decides.
    rows="$(quant_row UD-IQ2_XXS)"
  fi

  while IFS='|' read -r q file size; do
    [ -n "$q" ] || continue
    attempt=$((attempt + 1))
    QUANT_CHOSEN="$q"
    MODEL_FILE="$file"
    MODEL_PATH="$MODEL_DIR/$file"
    info "candidate #$attempt: $q (~${size} MiB file)"
    download_file "$file" "$MODEL_PATH" || continue

    local fitlog="$LOG_DIR/fit-${INSTANCE_SAFE}-${q}.log"
    say "Proving VRAM fit: $q + $NATIVE_CTX context + $KV_CHOSEN KV"
    if try_full_context_load "$fitlog"; then
      info "PASS: full native context loaded with --fit off on GPU '$GPU_NAME'"
      save_profile
      return 0
    fi
    warn "$q failed full-context load; see $fitlog"
    if [ -n "${QUANT:-}" ]; then
      die "forced QUANT=$QUANT does not satisfy full native-context VRAM requirement"
    fi
    # refresh free VRAM in case driver retained/released allocations
    detect_gpu >/dev/null
  done <<< "$rows"

  die "no available quant could load Qwen3.8-27B with full $NATIVE_CTX context entirely on the selected GPU. Free VRAM or use a larger GPU."
}

save_profile() {
  cat >"$CFG_FILE" <<CFG
GPU_SELECTOR=$(printf '%q' "$GPU_SELECTOR")
GPU_NAME=$(printf '%q' "$GPU_NAME")
GPU_ARCH=$(printf '%q' "$GPU_ARCH")
GPU_TOTAL_MIB=$(printf '%q' "$GPU_TOTAL_MIB")
QUANT_CHOSEN=$(printf '%q' "$QUANT_CHOSEN")
MODEL_FILE=$(printf '%q' "$MODEL_FILE")
MODEL_PATH=$(printf '%q' "$MODEL_PATH")
KV_CHOSEN=$(printf '%q' "$KV_CHOSEN")
BATCH_SIZE=$(printf '%q' "$BATCH_SIZE")
UBATCH_SIZE=$(printf '%q' "$UBATCH_SIZE")
NATIVE_CTX=$(printf '%q' "$NATIVE_CTX")
MODEL_ALIAS=$(printf '%q' "$MODEL_ALIAS")
LOCAL_KEY=$(printf '%q' "$LOCAL_KEY")
BUILD_DIR=$(printf '%q' "$BUILD_DIR")
SERVER_BIN=$(printf '%q' "$SERVER_BIN")
BENCH_BIN=$(printf '%q' "$BENCH_BIN")
CLI_BIN=$(printf '%q' "$CLI_BIN")
LOCAL_LD_PATH=$(printf '%q' "$LOCAL_LD_PATH")
HOST=$(printf '%q' "$HOST")
PORT=$(printf '%q' "$PORT")
CHAT_TEMPLATE_PATH=$(printf '%q' "$CHAT_TEMPLATE_PATH")
REASONING_EFFORT=$(printf '%q' "$REASONING_EFFORT")
COMPACT_WINDOW=$(printf '%q' "$COMPACT_WINDOW")
COMPACT_PCT=$(printf '%q' "$COMPACT_PCT")
CFG
}

load_profile() {
  [ -f "$CFG_FILE" ] || die "no deployed profile for INSTANCE=$INSTANCE; run install first"
  # shellcheck disable=SC1090
  . "$CFG_FILE"
  [ -f "$MODEL_PATH" ] || die "profile model is missing: $MODEL_PATH"
  [ -x "$SERVER_BIN" ] || die "profile server binary is missing: $SERVER_BIN"
}

run_bench_one() {
  local depth="$1" out="$2"
  # Run both depth 0 and the requested depth. We retain PP(depth) at d=0 (true input/prefill
  # throughput from an empty context) and TG128 at d=depth (decode after that much context).
  # llama-bench excludes tokenization and sampling by design.
  CUDA_VISIBLE_DEVICES="$GPU_SELECTOR" \
  LD_LIBRARY_PATH="$LOCAL_LD_PATH" \
    "$BENCH_BIN" -m "$MODEL_PATH" -ngl 999 -sm none -mg 0 \
      -fa on -ctk "$KV_CHOSEN" -ctv "$KV_CHOSEN" \
      -b "$BATCH_SIZE" -ub "$UBATCH_SIZE" \
      -p "$depth" -n 128 -d "0,$depth" -r "$BENCH_REPS" -o json >"$out"
}

bench() {
  load_profile
  say "Smoke / mini benchmark (actual selected GPU + quant)"
  info "model=$QUANT_CHOSEN  KV=$KV_CHOSEN  GPU=$GPU_NAME"
  info "tests: PP128 from empty + TG128@d128; PP4096 from empty + TG128@d4096; reps=$BENCH_REPS"

  local j128="$BENCH_DIR/${INSTANCE_SAFE}-d128.json"
  local j4096="$BENCH_DIR/${INSTANCE_SAFE}-d4096.json"
  run_bench_one 128 "$j128" || die "128-token benchmark failed"
  run_bench_one 4096 "$j4096" || die "4096-token benchmark failed"

  local pp128 tg128 pp4096 tg4096
  pp128="$(jq -r '[.[] | select(.n_prompt==128 and .n_depth==0) | .avg_ts][0] // empty' "$j128")"
  tg128="$(jq -r '[.[] | select(.n_gen==128 and .n_depth==128) | .avg_ts][0] // empty' "$j128")"
  pp4096="$(jq -r '[.[] | select(.n_prompt==4096 and .n_depth==0) | .avg_ts][0] // empty' "$j4096")"
  tg4096="$(jq -r '[.[] | select(.n_gen==128 and .n_depth==4096) | .avg_ts][0] // empty' "$j4096")"

  [ -n "$pp128" ] && [ -n "$tg128" ] && [ -n "$pp4096" ] && [ -n "$tg4096" ] || {
    warn "could not parse expected benchmark rows; raw JSON retained in $BENCH_DIR"
    return 1
  }

  printf '\n%-14s %16s %18s\n' 'INPUT/DEPTH' 'PP tok/s' 'TG128 tok/s'
  printf '%-14s %16.2f %18.2f\n' '128' "$pp128" "$tg128"
  printf '%-14s %16.2f %18.2f\n' '4096' "$pp4096" "$tg4096"
  printf '\n'
  info "Raw benchmark JSON: $j128"
  info "                    $j4096"
}

is_running() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_server() {
  load_profile
  if is_running; then
    local pid="$(cat "$PID_FILE")"
    if curl -fsS --max-time 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      info "server already healthy: pid=$pid http://$HOST:$PORT"
      return 0
    fi
    warn "pidfile exists but health check failed; restarting"
    kill_pid_clean "$pid"
    rm -f "$PID_FILE"
  fi
  if port_in_use; then die "port $PORT is already in use by another HTTP health endpoint"; fi

  local args=()
  mapfile -t args < <(server_args_common)
  say "Starting deployed llama-server"
  : >"$SERVER_LOG"
  CUDA_VISIBLE_DEVICES="$GPU_SELECTOR" \
  HF_HOME="$HF_HOME" LLAMA_CACHE="$HF_HOME" \
  LD_LIBRARY_PATH="$LOCAL_LD_PATH" \
    nohup "$SERVER_BIN" "${args[@]}" >"$SERVER_LOG" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" >"$PID_FILE"
  if ! wait_health_pid "$pid"; then
    rm -f "$PID_FILE"
    tail -n 120 "$SERVER_LOG" >&2
    die "server failed to become healthy; see $SERVER_LOG"
  fi
  info "healthy: pid=$pid  http://$HOST:$PORT  metrics=http://$HOST:$PORT/metrics"
}

stop_server() {
  if ! is_running; then
    rm -f "$PID_FILE"
    info "server not running for INSTANCE=$INSTANCE"
    return 0
  fi
  local pid="$(cat "$PID_FILE")"
  say "Stopping server pid=$pid"
  kill_pid_clean "$pid"
  rm -f "$PID_FILE"
}

api_smoke() {
  load_profile
  start_server || return 1
  say "Anthropic /v1/messages semantic smoke test"
  local out="$LOG_DIR/api-smoke-${INSTANCE_SAFE}.json"
  local code
  code="$(curl -sS --max-time 300 -o "$out" -w '%{http_code}' \
    -H 'content-type: application/json' \
    -H "x-api-key: $LOCAL_KEY" \
    -H 'anthropic-version: 2023-06-01' \
    -d "{\"model\":\"$MODEL_ALIAS\",\"max_tokens\":192,\"system\":\"This is a deterministic health check.\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the exact marker QWEN38_SMOKE_OK.\"}]}" \
    "http://$HOST:$PORT/v1/messages")"
  if [ "$code" != "200" ]; then
    cat "$out" >&2
    die "Anthropic endpoint smoke failed with HTTP $code"
  fi
  if ! grep -q 'QWEN38_SMOKE_OK' "$out"; then
    warn "endpoint returned HTTP 200 but expected semantic marker was absent; inspecting output is recommended: $out"
    return 1
  fi
  info "PASS: /v1/messages produced the expected semantic marker"

  say "Anthropic forced tool-use smoke test"
  local tool_out="$LOG_DIR/api-tool-smoke-${INSTANCE_SAFE}.json"
  local tool_payload="$TMP_DIR/api-tool-smoke-${INSTANCE_SAFE}.json"
  cat >"$tool_payload" <<JSON
{"model":"$MODEL_ALIAS","max_tokens":512,"system":"This is an API/tool parser health check. Follow tool_choice exactly.","tools":[{"name":"local_echo","description":"Echo one string for a deployment health check.","input_schema":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":false}}],"tool_choice":{"type":"tool","name":"local_echo"},"messages":[{"role":"user","content":"Call local_echo exactly once with value TOOL_SMOKE_OK."}]}
JSON
  local tool_code
  tool_code="$(curl -sS --max-time 300 -o "$tool_out" -w '%{http_code}' \
    -H 'content-type: application/json' \
    -H "x-api-key: $LOCAL_KEY" \
    -H 'anthropic-version: 2023-06-01' \
    --data-binary "@$tool_payload" \
    "http://$HOST:$PORT/v1/messages")"
  if [ "$tool_code" != "200" ]; then
    cat "$tool_out" >&2
    warn "Anthropic forced-tool smoke failed with HTTP $tool_code"
    return 1
  fi
  if ! jq -e '.content[]? | select(.type == "tool_use" and .name == "local_echo" and .input.value == "TOOL_SMOKE_OK")' "$tool_out" >/dev/null 2>&1; then
    cat "$tool_out" >&2
    warn "Anthropic endpoint answered, but the forced local_echo tool call was not parsed correctly"
    return 1
  fi
  info "PASS: Anthropic tool_choice -> tool_use round trip parsed correctly"
}

export_claude_env() {
  # Claude Code itself is redirected into the script-local state tree as well.
  export ANTHROPIC_BASE_URL="http://$HOST:$PORT"
  export ANTHROPIC_API_KEY="$LOCAL_KEY"
  unset ANTHROPIC_AUTH_TOKEN
  export ANTHROPIC_MODEL="$MODEL_ALIAS"
  export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL_ALIAS"
  export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL_ALIAS"
  export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL_ALIAS"
  export CLAUDE_CONFIG_DIR="$CLAUDE_CONFIG_DIR_LOCAL"
  export CLAUDE_CODE_TMPDIR="$CLAUDE_TMP_DIR_LOCAL"
  # Claude Code otherwise has to guess the context of an unknown gateway model ID. Tell it
  # the real Qwen3.8 native limit, then compact comfortably inside that server boundary.
  export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$NATIVE_CTX"
  export CLAUDE_CODE_AUTO_COMPACT_WINDOW="$COMPACT_WINDOW"
  export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="$COMPACT_PCT"
  unset DISABLE_AUTO_COMPACT
  unset DISABLE_COMPACT
  export CLAUDE_CODE_ATTRIBUTION_HEADER=0
  export CLAUDE_CODE_ENABLE_TELEMETRY=0
  export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
  export API_TIMEOUT_MS="${API_TIMEOUT_MS:-1800000}"
}

claude_smoke() {
  if ! need claude; then
    warn "Claude Code executable 'claude' is not installed; skipping native Claude Code smoke."
    info "Install Claude Code with Anthropic's installer/npm, then run: $0 claude-smoke"
    return 0
  fi
  load_profile
  start_server || return 1
  export_claude_env
  say "Native Claude Code -> local llama.cpp compatibility smoke"
  local out="$LOG_DIR/claude-smoke-${INSTANCE_SAFE}.txt"
  local rc=0
  if need timeout; then
    timeout 900 claude -p --max-turns 1 --model "$MODEL_ALIAS" \
      "Reply exactly LOCAL_CLAUDE_SMOKE_OK. Do not use tools." >"$out" 2>&1 || rc=$?
  else
    claude -p --max-turns 1 --model "$MODEL_ALIAS" \
      "Reply exactly LOCAL_CLAUDE_SMOKE_OK. Do not use tools." >"$out" 2>&1 || rc=$?
  fi
  if [ "$rc" -ne 0 ] || ! grep -q 'LOCAL_CLAUDE_SMOKE_OK' "$out"; then
    warn "Claude Code compatibility smoke FAILED (rc=$rc)."
    warn "The fixed Qwen3.8 chat template is enabled, so this is a real harness/template/runtime incompatibility worth inspecting rather than silently accepting."
    warn "Claude output: $out"
    warn "Server log:    $SERVER_LOG"
    return 4
  fi
  info "PASS: Claude Code reached the local Qwen3.8 endpoint and returned the expected marker"
}

claude_exec() {
  need claude || die "Claude Code not found. Install it first (for example: npm install -g @anthropic-ai/claude-code)."
  load_profile
  start_server || return 1
  export_claude_env
  say "Launching Claude Code against local $MODEL_ALIAS"
  info "endpoint=$ANTHROPIC_BASE_URL  GPU=$GPU_SELECTOR/$GPU_NAME  compact-window=$COMPACT_WINDOW"
  exec claude --model "$MODEL_ALIAS" "$@"
}

print_usage_lines() {
  load_profile
  cat <<USAGE

DEPLOYED
  GPU       : $GPU_NAME (selector $GPU_SELECTOR, cc$GPU_ARCH)
  quant     : $QUANT_CHOSEN
  KV        : $KV_CHOSEN / $KV_CHOSEN
  context   : $NATIVE_CTX (full native context was load-tested with --fit off)
  template  : $CHAT_TEMPLATE_PATH (Claude-Code compatibility template)
  reasoning : $REASONING_EFFORT, deepseek parsing, preserve-history
  endpoint  : http://$HOST:$PORT
  model     : $MODEL_ALIAS
  logs      : $LOG_DIR

PRIMARY CLAUDE CODE (recommended wrapper; preserves safe local config + compaction env)
  $0 claude

LITERAL ONE-LINER
  env -u DISABLE_AUTO_COMPACT -u DISABLE_COMPACT -u ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL=http://$HOST:$PORT ANTHROPIC_API_KEY=$LOCAL_KEY ANTHROPIC_MODEL=$MODEL_ALIAS ANTHROPIC_DEFAULT_SONNET_MODEL=$MODEL_ALIAS ANTHROPIC_DEFAULT_OPUS_MODEL=$MODEL_ALIAS ANTHROPIC_DEFAULT_HAIKU_MODEL=$MODEL_ALIAS CLAUDE_CONFIG_DIR=$CLAUDE_CONFIG_DIR_LOCAL CLAUDE_CODE_TMPDIR=$CLAUDE_TMP_DIR_LOCAL CLAUDE_CODE_MAX_CONTEXT_TOKENS=$NATIVE_CTX CLAUDE_CODE_AUTO_COMPACT_WINDOW=$COMPACT_WINDOW CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=$COMPACT_PCT CLAUDE_CODE_ATTRIBUTION_HEADER=0 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude --model $MODEL_ALIAS

TARGET ONE PHYSICAL GPU DURING INSTALL/DEPLOY
  CUDA_VISIBLE_DEVICES=0 $0 install
  CUDA_VISIBLE_DEVICES=1 $0 install
  # The server records the chosen physical GPU in its instance profile, so later '$0 claude' uses it.

SECOND AGENT -- DIFFERENT GPU, FULLY INDEPENDENT SERVER/MODEL CONTEXT
  CUDA_VISIBLE_DEVICES=1 INSTANCE=agent2 PORT=8081 $0 install
  INSTANCE=agent2 PORT=8081 $0 claude

SECOND AGENT -- SAME GPU, SHARED SERVER (recommended on 16/24/32/40 GB cards)
  $0 claude
  # Run that in a second terminal. It is a separate Claude Code session; requests serialize through -np 1.

SECOND AGENT -- SAME GPU, FULLY INDEPENDENT SERVER (only when VRAM really permits it)
  CUDA_VISIBLE_DEVICES=$GPU_SELECTOR INSTANCE=agent2 PORT=8081 $0 install
  INSTANCE=agent2 $0 claude
  # This intentionally re-runs free-VRAM quant selection and the 262k fit proof. On smaller cards it
  # will refuse rather than duplicate weights/KV and silently spill/shrink. On a roomy A100 it can pass.

SERVER OPERATIONS
  $0 status
  $0 stop
  $0 start
  $0 bench
  $0 api-smoke
  $0 claude-smoke

AUTO-COMPACTION
  Claude Code is explicitly told the model context is $NATIVE_CTX tokens.
  Effective compaction capacity: $COMPACT_WINDOW tokens, comfortably inside that server boundary.
  The percentage hint is $COMPACT_PCT%; the lower 180k-style window is the primary safety guard.
  DISABLE_AUTO_COMPACT and DISABLE_COMPACT are explicitly unset by the wrapper.
USAGE
}

status() {
  if [ -f "$CFG_FILE" ]; then
    load_profile
    print_usage_lines
  else
    print_header
    info "no saved deployment profile for INSTANCE=$INSTANCE"
  fi
  if is_running; then
    local pid="$(cat "$PID_FILE")"
    info "server status: RUNNING pid=$pid"
    curl -fsS --max-time 2 "http://$HOST:$PORT/health" 2>/dev/null | jq . 2>/dev/null || true
  else
    info "server status: stopped"
  fi
}

plan() {
  print_header
  install_basic_deps
  detect_gpu
  local nv kv nonweight budget
  nv="$(find_nvcc)"
  kv="$(choose_kv_type)"
  nonweight="$(estimated_nonweight_mib "$kv")"
  budget=$((GPU_FREE_MIB - nonweight))
  choose_batch_profile
  say "Plan"
  info "nvcc=$nv"
  info "llama.cpp ref=$LLAMA_REF, CUDA arch=$GPU_ARCH"
  info "KV=$kv; conservative non-weight/context/scratch reservation=${nonweight} MiB"
  info "approx weight-file budget=${budget} MiB from CURRENT FREE VRAM"
  info "batch=$BATCH_SIZE ubatch=$UBATCH_SIZE"
  info "quant candidates that pass pre-filter:"
  prefilter_quant_list "$kv" | awk -F'|' '{printf "      %-14s %s (~%s MiB)\n",$1,$2,$3}'
  info "final choice is NOT trusted until a real $NATIVE_CTX-token --fit off GPU load succeeds"
}

install_all() {
  print_header
  build_llama
  select_download_and_validate
  bench || die "benchmark stage failed"
  start_server
  api_smoke || die "direct Anthropic API semantic/tool smoke failed"
  local claude_rc=0
  claude_smoke || claude_rc=$?
  print_usage_lines
  if [ "$claude_rc" -ne 0 ]; then
    printf '\nCLAUDE_COMPAT=FAIL (direct llama.cpp server is healthy; native Claude Code harness smoke failed)\n' >&2
    return "$claude_rc"
  fi
  printf '\nINSTALL=PASS\n'
}

usage() {
  cat <<HELP
Usage: $(basename "$0") COMMAND [claude args...]

Commands:
  install       deps + detect + build + quant select/download + full-ctx fit proof + bench + deploy + smokes
  plan          show detected GPU, build profile, VRAM budget, candidate quants; download nothing
  build         build fresh llama.cpp CUDA targets for selected GPU architecture
  model         choose/download quant and prove full native-context VRAM fit
  bench         PP/TG at input/context 128 and 4096
  start         start saved deployment
  stop          stop saved deployment
  restart       restart saved deployment
  status        show profile, endpoint, server state and exact Claude usage
  api-smoke     semantic + forced-tool tests of llama.cpp Anthropic /v1/messages
  claude-smoke  actual 'claude -p' test against the local endpoint
  claude        launch interactive Claude Code using local Qwen3.8 (remaining args passed through)
  help          this help

Examples:
  CUDA_VISIBLE_DEVICES=0 ./$(basename "$0") install
  ./$(basename "$0") claude
  ./$(basename "$0") claude --permission-mode plan
  CUDA_VISIBLE_DEVICES=1 INSTANCE=agent2 PORT=8081 ./$(basename "$0") install
  INSTANCE=agent2 PORT=8081 ./$(basename "$0") claude
HELP
}

cmd="${1:-install}"
if [ $# -gt 0 ]; then shift; fi
case "$cmd" in
  install) install_all "$@" ;;
  plan) plan "$@" ;;
  build) print_header; build_llama "$@" ;;
  model|download) print_header; select_download_and_validate "$@" ;;
  bench) bench "$@" ;;
  start) start_server "$@" ;;
  stop) stop_server "$@" ;;
  restart) stop_server; start_server ;;
  status) status "$@" ;;
  api-smoke|smoke) api_smoke "$@" ;;
  claude-smoke) claude_smoke "$@" ;;
  claude) claude_exec "$@" ;;
  help|-h|--help) usage ;;
  *) usage >&2; die "unknown command: $cmd" ;;
esac
