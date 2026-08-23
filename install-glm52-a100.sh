#!/usr/bin/env bash
# Push-button Docker installer/launcher/tester for GLM-5.2 AWQ INT4 on 8x A100 80GB.
#
# Payload: cyankiwi/GLM-5.2-AWQ-INT4, approximately 410 GB on disk.
# Model: approximately 743B total parameters / 39B active parameters (MoE).
# Hardware target: exactly 8 visible NVIDIA A100 80GB GPUs (SM80), TP=8.
# Served context default: 32,768 tokens (override with MAX_MODEL_LEN=...).
# Thinking mode: disabled server-wide by default (set ENABLE_THINKING=1 to enable).
# Performance: measured after startup by this script; no unverified tok/s is assumed.
#
# Default action:
#   ./install-glm52-a100.sh
#
# Other actions:
#   ./install-glm52-a100.sh build
#   ./install-glm52-a100.sh download
#   ./install-glm52-a100.sh run
#   ./install-glm52-a100.sh test
#   ./install-glm52-a100.sh benchmark
#   ./install-glm52-a100.sh logs
#   ./install-glm52-a100.sh status
#   ./install-glm52-a100.sh stop
#
# Common overrides:
#   MODEL_ROOT=/mnt/fast/models PORT=8000 BIND_ADDRESS=127.0.0.1 ./install-glm52-a100.sh
#   HF_TOKEN=hf_... ./install-glm52-a100.sh
#   MAX_MODEL_LEN=65536 GPU_MEMORY_UTILIZATION=0.92 ./install-glm52-a100.sh
#   SKIP_HARDWARE_CHECK=1 ./install-glm52-a100.sh build

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-all}"

# ----------------------------- Configuration -----------------------------
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:glm52-x86_64-cu129}"
IMAGE_NAME="${IMAGE_NAME:-vllm/vllm-openai:glm52-cu129-patched}"
CONTAINER_NAME="${CONTAINER_NAME:-glm52-awq-a100}"

MODEL_ID="${MODEL_ID:-cyankiwi/GLM-5.2-AWQ-INT4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ID}"
MODEL_ROOT="${MODEL_ROOT:-$SCRIPT_DIR/model}"
MODEL_SUBDIR="${MODEL_SUBDIR:-GLM-5.2-AWQ-INT4}"
MODEL_DIR="$MODEL_ROOT/$MODEL_SUBDIR"
CONTAINER_MODEL_DIR="/models/$MODEL_SUBDIR"
RUNTIME_CACHE_DIR="${RUNTIME_CACHE_DIR:-$SCRIPT_DIR/runtime-cache}"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/glm5-build}"

PATCH_REPO="${PATCH_REPO:-https://github.com/vllm-project/vllm.git}"
PATCH_PR="${PATCH_PR:-38476}"
PATCH_COMMIT="${PATCH_COMMIT:-}"

TP_SIZE="${TP_SIZE:-8}"
PORT="${PORT:-8000}"
BIND_ADDRESS="${BIND_ADDRESS:-127.0.0.1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
SHM_SIZE="${SHM_SIZE:-64g}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-7200}"
STARTUP_POLL_SECONDS="${STARTUP_POLL_SECONDS:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-440}"
HF_HUB_DOWNLOAD_WORKERS="${HF_HUB_DOWNLOAD_WORKERS:-8}"
BENCH_MAX_TOKENS="${BENCH_MAX_TOKENS:-256}"
BENCH_RUNS="${BENCH_RUNS:-2}"

ENABLE_THINKING="${ENABLE_THINKING:-0}"
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
SKIP_HARDWARE_CHECK="${SKIP_HARDWARE_CHECK:-0}"
SKIP_DISK_CHECK="${SKIP_DISK_CHECK:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
PULL_BASE_IMAGE="${PULL_BASE_IMAGE:-1}"

if [[ "$BIND_ADDRESS" == "0.0.0.0" || "$BIND_ADDRESS" == "127.0.0.1" || "$BIND_ADDRESS" == "localhost" ]]; then
    API_CONNECT_HOST="127.0.0.1"
else
    API_CONNECT_HOST="$BIND_ADDRESS"
fi
API_BASE="http://${API_CONNECT_HOST}:${PORT}"

# ------------------------------- Utilities -------------------------------
log()  { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*"; }
ok()   { printf '\033[1;32mOK:\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

on_error() {
    local rc=$?
    local line=${BASH_LINENO[0]:-unknown}
    printf '\n\033[1;31mFAILED\033[0m at line %s (exit %s).\n' "$line" "$rc" >&2
    if command -v docker >/dev/null 2>&1 && docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        printf '\nLast 120 container log lines:\n' >&2
        docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
    fi
    exit "$rc"
}
trap on_error ERR

usage() {
    cat <<USAGE
Usage: $(basename "$0") [action]

Actions:
  all         Check, build, verify, download, launch, test, benchmark (default)
  build       Create patch bundle and build the patched Docker image
  verify      Verify image patches and Docker GPU access
  download    Download/resume the model into MODEL_ROOT
  run         Start or replace the API container and wait until healthy
  test        Run model-list and arithmetic inference smoke tests
  benchmark   Run a short single-request decode benchmark
  logs        Follow server logs
  status      Show container, API, GPU, image, and model status
  stop        Stop and remove the API container
  help        Show this help

Important environment variables:
  BASE_IMAGE=$BASE_IMAGE
  IMAGE_NAME=$IMAGE_NAME
  MODEL_ROOT=$MODEL_ROOT
  MODEL_ID=$MODEL_ID
  PORT=$PORT
  MAX_MODEL_LEN=$MAX_MODEL_LEN
  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
  ENABLE_THINKING=$ENABLE_THINKING
  ENABLE_TOOL_CALLING=$ENABLE_TOOL_CALLING
USAGE
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

container_exists() {
    docker inspect "$CONTAINER_NAME" >/dev/null 2>&1
}

container_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == "true" ]]
}

image_exists() {
    docker image inspect "$IMAGE_NAME" >/dev/null 2>&1
}

human_banner() {
    cat <<BANNER

=======================================================================
 GLM-5.2 AWQ INT4 / vLLM / Docker / 8x A100 80GB (SM80)
=======================================================================
 Model                 : $MODEL_ID
 Approx. model payload : ~410 GB
 Model architecture    : ~743B total / ~39B active MoE
 GPUs / tensor parallel: 8 / TP=$TP_SIZE
 Served context        : $MAX_MODEL_LEN tokens
 Thinking default      : $([[ "$ENABLE_THINKING" == "1" ]] && echo enabled || echo disabled)
 Host model directory  : $MODEL_DIR
 Docker image          : $IMAGE_NAME
 API                    : http://$BIND_ADDRESS:$PORT/v1
=======================================================================
BANNER
}

check_host_prerequisites() {
    log "Checking host prerequisites"
    require_cmd docker
    require_cmd git
    require_cmd tar
    require_cmd curl
    require_cmd python3
    require_cmd df
    require_cmd awk
    require_cmd grep

    docker info >/dev/null 2>&1 || die "Docker daemon is not reachable by the current user."
    mkdir -p "$MODEL_ROOT" "$RUNTIME_CACHE_DIR" "$BUILD_DIR"
    ok "Docker, Git, curl, tar, and Python 3 are available"
}

check_disk_space() {
    [[ "$SKIP_DISK_CHECK" == "1" ]] && { warn "Skipping disk-space check."; return; }

    mkdir -p "$MODEL_ROOT"
    local available_kb available_gb existing_kb existing_gb effective_gb
    available_kb="$(df -Pk "$MODEL_ROOT" | awk 'NR==2 {print $4}')"
    [[ "$available_kb" =~ ^[0-9]+$ ]] || die "Could not determine free space for $MODEL_ROOT"
    available_gb=$((available_kb / 1024 / 1024))
    existing_kb=0
    if [[ -d "$MODEL_DIR" ]]; then
        existing_kb="$(du -sk "$MODEL_DIR" 2>/dev/null | awk '{print $1}')"
        [[ "$existing_kb" =~ ^[0-9]+$ ]] || existing_kb=0
    fi
    existing_gb=$((existing_kb / 1024 / 1024))
    effective_gb=$((available_gb + existing_gb))

    log "Checking model filesystem capacity"
    printf 'Free space at %s: %s GiB; existing model data: %s GiB; effective capacity: %s GiB\n' \
        "$MODEL_ROOT" "$available_gb" "$existing_gb" "$effective_gb"
    (( effective_gb >= MIN_FREE_GB )) || die \
        "The filesystem needs at least ${MIN_FREE_GB} GiB of free-plus-existing model capacity; only ${effective_gb} GiB is available. Set MODEL_ROOT to a larger filesystem."
    ok "Model filesystem has at least ${MIN_FREE_GB} GiB of effective capacity"
}

check_host_hardware() {
    [[ "$SKIP_HARDWARE_CHECK" == "1" ]] && { warn "Skipping host GPU checks."; return; }

    log "Checking the 8x A100 80GB host requirement"
    require_cmd nvidia-smi

    local -a gpu_lines
    local gpu_output
    if gpu_output="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null)"; then
        mapfile -t gpu_lines <<<"$gpu_output"
    else
        gpu_output="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
        mapfile -t gpu_lines <<<"$gpu_output"
    fi

    [[ "${#gpu_lines[@]}" -eq 8 ]] || die "Expected exactly 8 visible physical GPUs; found ${#gpu_lines[@]}."

    local i line name mem cap
    for i in "${!gpu_lines[@]}"; do
        line="${gpu_lines[$i]}"
        IFS=',' read -r name mem cap <<<"$line"
        name="${name#${name%%[![:space:]]*}}"; name="${name%${name##*[![:space:]]}}"
        mem="${mem//[[:space:]]/}"
        cap="${cap:-}"; cap="${cap//[[:space:]]/}"

        [[ "$name" == *A100* ]] || die "GPU $i is '$name', not an A100."
        [[ "$mem" =~ ^[0-9]+$ ]] || die "Could not parse memory for GPU $i: '$mem'"
        (( mem >= 79000 )) || die "GPU $i has ${mem} MiB, not an 80GB-class A100."
        if [[ -n "$cap" && "$cap" != "8.0" ]]; then
            die "GPU $i reports compute capability $cap; expected 8.0 (SM80)."
        fi
        printf 'GPU %d: %s, %s MiB%s\n' "$i" "$name" "$mem" "${cap:+, SM${cap/.}}"
    done
    ok "Eight A100 80GB GPUs are visible"
}

pull_base_image() {
    if [[ "$PULL_BASE_IMAGE" == "1" ]]; then
        log "Pulling the official GLM-5.2 CUDA 12.9 base image"
        docker pull "$BASE_IMAGE"
    elif ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
        die "Base image $BASE_IMAGE is absent and PULL_BASE_IMAGE=0."
    fi
}

prepare_build_context() {
    log "Preparing vLLM PR #$PATCH_PR Python overlay"
    rm -rf "$BUILD_DIR/vllm-pr"
    mkdir -p "$BUILD_DIR"

    git clone --filter=blob:none --no-checkout "$PATCH_REPO" "$BUILD_DIR/vllm-pr"
    git -C "$BUILD_DIR/vllm-pr" fetch --depth=1 origin \
        "pull/${PATCH_PR}/head:refs/remotes/origin/pr-${PATCH_PR}"

    local resolved_commit
    resolved_commit="$(git -C "$BUILD_DIR/vllm-pr" rev-parse "refs/remotes/origin/pr-${PATCH_PR}")"
    if [[ -n "$PATCH_COMMIT" && "$resolved_commit" != "$PATCH_COMMIT" ]]; then
        die "PR #$PATCH_PR currently resolves to $resolved_commit, not requested PATCH_COMMIT=$PATCH_COMMIT"
    fi
    git -C "$BUILD_DIR/vllm-pr" checkout --detach "$resolved_commit"

    local -a patch_files=(
        vllm/platforms/cuda.py
        vllm/v1/attention/backends/registry.py
        vllm/model_executor/layers/sparse_attn_indexer.py
        vllm/v1/attention/backends/mla/triton_mla_sparse.py
        vllm/v1/attention/ops/mqa_logits_triton.py
        vllm/v1/attention/ops/triton_mla_sparse_kernel.py
    )

    local f
    for f in "${patch_files[@]}"; do
        [[ -f "$BUILD_DIR/vllm-pr/$f" ]] || die "Expected PR patch file is missing: $f"
    done

    tar -C "$BUILD_DIR/vllm-pr" -czf "$BUILD_DIR/vllm_patch_files.tar.gz" "${patch_files[@]}"
    printf '%s\n' "$resolved_commit" > "$BUILD_DIR/PATCH_COMMIT"

    cat > "$BUILD_DIR/fix_ampere.py" <<'PY'
"""Apply idempotent Ampere (SM80) compatibility fixes to the installed vLLM."""
from pathlib import Path
import sys
import vllm

vllm_dir = Path(vllm.__file__).resolve().parent


def replace_or_verify(path: Path, old: str, new: str, label: str) -> None:
    src = path.read_text()
    if old in src:
        src = src.replace(old, new)
        path.write_text(src)
        print(f"{label}: applied")
    elif new in src:
        print(f"{label}: already applied")
    else:
        raise RuntimeError(
            f"{label}: found neither expected old text nor patched text in {path}"
        )


# On SM80, package presence is not equivalent to architectural support.
replace_or_verify(
    vllm_dir / "v1/attention/backends/mla/indexer.py",
    "has_deep_gemm",
    "is_deep_gemm_supported",
    "indexer.py has_deep_gemm -> is_deep_gemm_supported",
)

# Prevent DeepGEMM's CUDA path from being initialized on unsupported architectures.
replace_or_verify(
    vllm_dir / "utils/deep_gemm.py",
    "if not has_deep_gemm():",
    "if not is_deep_gemm_supported():",
    "deep_gemm.py _lazy_init architecture guard",
)

# PR #47629 later identified a separate GLM-5.2/A100 startup failure: the
# fused indexer-Q Triton kernel stores fp8e4nv, which requires SM89+.
deepseek_path = vllm_dir / "model_executor/models/deepseek_v2.py"
deepseek_src = deepseek_path.read_text()
marker = "self.use_fused_indexer_q = ("
if marker in deepseek_src:
    block_start = deepseek_src.index(marker)
    block_preview = deepseek_src[block_start : block_start + 700]
    capability_guard = "current_platform.has_device_capability(89)"
    if capability_guard in block_preview:
        print("deepseek_v2.py fused indexer-Q SM89 guard: already applied")
    else:
        old = "current_platform.is_cuda()\n            and self.quant_block_size"
        new = (
            "current_platform.is_cuda()\n"
            "            and current_platform.has_device_capability(89)\n"
            "            and self.quant_block_size"
        )
        if old not in deepseek_src:
            raise RuntimeError(
                "deepseek_v2.py: use_fused_indexer_q exists but its expected "
                "SM80-unsafe predicate was not found"
            )
        deepseek_path.write_text(deepseek_src.replace(old, new, 1))
        print("deepseek_v2.py fused indexer-Q SM89 guard: applied")
else:
    print("deepseek_v2.py: fused indexer-Q path absent; no SM89 guard needed")

sys.exit(0)
PY

    cat > "$BUILD_DIR/verify_patches.py" <<'PY'
"""Verify the PR overlay and the SM80 architecture guards."""
import importlib
import inspect
import sys
import vllm

print("vLLM version:", vllm.__version__)

modules = [
    "vllm.v1.attention.backends.mla.triton_mla_sparse",
    "vllm.v1.attention.ops.mqa_logits_triton",
    "vllm.v1.attention.ops.triton_mla_sparse_kernel",
]
for module_name in modules:
    importlib.import_module(module_name)
    print(module_name.rsplit(".", 1)[-1] + ": OK")

idx = importlib.import_module("vllm.v1.attention.backends.mla.indexer")
idx_src = inspect.getsource(idx)
assert "is_deep_gemm_supported" in idx_src, "indexer.py guard was not patched"
assert "has_deep_gemm" not in idx_src, "indexer.py still references has_deep_gemm"
print("indexer.py architecture guard: OK")

dg = importlib.import_module("vllm.utils.deep_gemm")
lazy_src = inspect.getsource(dg._lazy_init)
assert "if not is_deep_gemm_supported():" in lazy_src, (
    "deep_gemm.py _lazy_init architecture guard was not patched"
)
print("deep_gemm.py architecture guard: OK")

from pathlib import Path
vllm_dir = Path(vllm.__file__).resolve().parent
deepseek_src = (vllm_dir / "model_executor/models/deepseek_v2.py").read_text()
if "self.use_fused_indexer_q = (" in deepseek_src:
    start = deepseek_src.index("self.use_fused_indexer_q = (")
    block = deepseek_src[start : start + 700]
    assert "current_platform.has_device_capability(89)" in block, (
        "deepseek_v2.py fused indexer-Q path is not gated on SM89+"
    )
    print("deepseek_v2.py fused indexer-Q SM89 guard: OK")
else:
    print("deepseek_v2.py fused indexer-Q path absent: OK")

print("All GLM-5.2 SM80 patches verified successfully")
sys.exit(0)
PY

    cat > "$BUILD_DIR/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE=vllm/vllm-openai:glm52-x86_64-cu129
FROM ${BASE_IMAGE}

ARG PATCH_PR=38476
ARG PATCH_COMMIT=unknown
LABEL org.opencontainers.image.title="vLLM GLM-5.2 SM80 patched server" \
      org.opencontainers.image.description="TRITON_MLA_SPARSE overlay for GLM-5.2 on A100/SM80" \
      org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      io.vllm.patch.pr="${PATCH_PR}" \
      io.vllm.patch.commit="${PATCH_COMMIT}"

COPY vllm_patch_files.tar.gz /tmp/vllm_patch_files.tar.gz
RUN set -eux; \
    DEST="$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')"; \
    echo "vLLM site-packages: $DEST"; \
    mkdir -p /tmp/vllm-patch; \
    tar -xzf /tmp/vllm_patch_files.tar.gz -C /tmp/vllm-patch; \
    cp -rv /tmp/vllm-patch/vllm/. "$DEST/"; \
    rm -rf /tmp/vllm-patch /tmp/vllm_patch_files.tar.gz

COPY fix_ampere.py /tmp/fix_ampere.py
RUN python3 /tmp/fix_ampere.py && rm -f /tmp/fix_ampere.py

COPY verify_patches.py /tmp/verify_patches.py
RUN python3 /tmp/verify_patches.py && rm -f /tmp/verify_patches.py

ENV VLLM_ATTENTION_BACKEND=TRITON_MLA_SPARSE
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
DOCKERFILE

    rm -rf "$BUILD_DIR/vllm-pr"

    log "Patch bundle contents"
    tar -tzf "$BUILD_DIR/vllm_patch_files.tar.gz"
    printf 'Patch commit: %s\n' "$resolved_commit"
    ok "Build context created at $BUILD_DIR"
}

build_image() {
    pull_base_image
    prepare_build_context

    local patch_commit
    patch_commit="$(cat "$BUILD_DIR/PATCH_COMMIT")"
    log "Building $IMAGE_NAME"
    docker build \
        --build-arg "BASE_IMAGE=$BASE_IMAGE" \
        --build-arg "PATCH_PR=$PATCH_PR" \
        --build-arg "PATCH_COMMIT=$patch_commit" \
        --tag "$IMAGE_NAME" \
        --file "$BUILD_DIR/Dockerfile" \
        "$BUILD_DIR"
    ok "Built $IMAGE_NAME"
}

verify_image() {
    image_exists || die "Image $IMAGE_NAME does not exist. Run '$0 build' first."

    log "Verifying patch imports inside the image"
    # The explicit --entrypoint is required because the image normally launches the API server.
    docker run --rm --entrypoint python3 "$IMAGE_NAME" -c '
import importlib
import inspect
import vllm
print("vLLM version:", vllm.__version__)
for name in (
    "vllm.v1.attention.backends.mla.triton_mla_sparse",
    "vllm.v1.attention.ops.mqa_logits_triton",
    "vllm.v1.attention.ops.triton_mla_sparse_kernel",
):
    importlib.import_module(name)
    print(name, "OK")
from vllm.v1.attention.backends.mla import indexer
from vllm.utils import deep_gemm
assert "has_deep_gemm" not in inspect.getsource(indexer)
assert "if not is_deep_gemm_supported():" in inspect.getsource(deep_gemm._lazy_init)
from pathlib import Path
root = Path(vllm.__file__).resolve().parent
deepseek_src = (root / "model_executor/models/deepseek_v2.py").read_text()
if "self.use_fused_indexer_q = (" in deepseek_src:
    start = deepseek_src.index("self.use_fused_indexer_q = (")
    assert "current_platform.has_device_capability(89)" in deepseek_src[start:start+700]
print("Patch verification: OK")
'

    if [[ "$SKIP_HARDWARE_CHECK" != "1" ]]; then
        log "Verifying NVIDIA Container Toolkit and all eight SM80 GPUs"
        docker run --rm --gpus all --entrypoint python3 "$IMAGE_NAME" -c '
import torch
from vllm.utils.deep_gemm import is_deep_gemm_supported
n = torch.cuda.device_count()
print("CUDA device count:", n)
assert n == 8, f"expected exactly 8 GPUs, found {n}"
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    gib = p.total_memory / 1024**3
    print(f"GPU {i}: {p.name}; {gib:.1f} GiB; compute capability {p.major}.{p.minor}")
    assert "A100" in p.name, p.name
    assert gib >= 79.0, gib
    assert (p.major, p.minor) == (8, 0), (p.major, p.minor)
print("is_deep_gemm_supported:", is_deep_gemm_supported())
assert not is_deep_gemm_supported(), "DeepGEMM should be disabled on A100/SM80"
print("Docker GPU verification: OK")
'
    fi
    ok "Patched image verification passed"
}

download_model() {
    [[ "$SKIP_DOWNLOAD" == "1" ]] && { warn "SKIP_DOWNLOAD=1; not downloading model."; return; }
    image_exists || die "Image $IMAGE_NAME does not exist. Run '$0 build' first."
    check_disk_space
    mkdir -p "$MODEL_DIR" "$RUNTIME_CACHE_DIR"

    log "Downloading/resuming $MODEL_ID into $MODEL_DIR"
    local -a env_args=(-e "HF_HUB_DOWNLOAD_WORKERS=$HF_HUB_DOWNLOAD_WORKERS")
    [[ -n "${HF_TOKEN:-}" ]] && env_args+=(-e HF_TOKEN)

    docker run --rm -i \
        "${env_args[@]}" \
        -v "$MODEL_ROOT:/models" \
        -v "$RUNTIME_CACHE_DIR:/root/.cache" \
        --entrypoint python3 \
        "$IMAGE_NAME" - "$MODEL_ID" "$CONTAINER_MODEL_DIR" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1:3]
workers = int(os.environ.get("HF_HUB_DOWNLOAD_WORKERS", "8"))
print(f"Downloading {repo_id} -> {local_dir} with {workers} workers")
path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=workers,
)
print("Snapshot ready:", path)
PY

    [[ -s "$MODEL_DIR/config.json" ]] || die "Model download did not produce $MODEL_DIR/config.json"
    local shard_count
    shard_count="$(find "$MODEL_DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) | wc -l | tr -d ' ')"
    (( shard_count > 0 )) || die "No model weight shards were found in $MODEL_DIR"
    du -sh "$MODEL_DIR" || true
    ok "Model snapshot is present with $shard_count weight shard(s)"
}

stop_container() {
    if container_exists; then
        log "Stopping and removing $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME" >/dev/null
        ok "Removed $CONTAINER_NAME"
    else
        printf 'Container %s does not exist.\n' "$CONTAINER_NAME"
    fi
}

start_container() {
    image_exists || die "Image $IMAGE_NAME does not exist. Run '$0 build' first."
    [[ -s "$MODEL_DIR/config.json" ]] || die \
        "Model is not present at $MODEL_DIR. Run '$0 download' first."

    if container_exists; then
        log "Replacing existing container $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME" >/dev/null
    fi

    mkdir -p "$RUNTIME_CACHE_DIR"

    local -a env_args=(-e VLLM_ATTENTION_BACKEND=TRITON_MLA_SPARSE)
    [[ -n "${HF_TOKEN:-}" ]] && env_args+=(-e HF_TOKEN)

    local -a server_args=(
        --model "$CONTAINER_MODEL_DIR"
        --served-model-name "$SERVED_MODEL_NAME"
        --tensor-parallel-size "$TP_SIZE"
        --no-async-scheduling
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --max-model-len "$MAX_MODEL_LEN"
        --trust-remote-code
        --kv-cache-dtype "$KV_CACHE_DTYPE"
        --host 0.0.0.0
        --port 8000
    )

    if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
        server_args+=(
            --tool-call-parser glm47
            --reasoning-parser glm45
            --enable-auto-tool-choice
        )
    fi

    if [[ "$ENABLE_THINKING" != "1" ]]; then
        server_args+=(--default-chat-template-kwargs '{"enable_thinking":false}')
    fi

    log "Starting $CONTAINER_NAME"
    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        --gpus all \
        --shm-size "$SHM_SIZE" \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p "$BIND_ADDRESS:$PORT:8000" \
        "${env_args[@]}" \
        -v "$MODEL_ROOT:/models:ro" \
        -v "$RUNTIME_CACHE_DIR:/root/.cache" \
        "$IMAGE_NAME" \
        "${server_args[@]}" >/dev/null

    printf 'Container: %s\nLogs:      docker logs -f %q\n' "$CONTAINER_NAME" "$CONTAINER_NAME"
    wait_for_server
}

wait_for_server() {
    log "Waiting for the OpenAI-compatible API to become healthy"
    local start now elapsed next_log=0
    start="$(date +%s)"

    while true; do
        if curl -fsS --max-time 3 "$API_BASE/health" >/dev/null 2>&1; then
            ok "API health endpoint is ready"
            break
        fi

        container_running || {
            docker logs --tail 200 "$CONTAINER_NAME" >&2 || true
            die "Container exited before the API became healthy."
        }

        now="$(date +%s)"
        elapsed=$((now - start))
        (( elapsed < STARTUP_TIMEOUT )) || {
            docker logs --tail 200 "$CONTAINER_NAME" >&2 || true
            die "API did not become healthy within ${STARTUP_TIMEOUT}s."
        }

        if (( elapsed >= next_log )); then
            printf 'Still loading: %ss elapsed. Latest log line:\n' "$elapsed"
            docker logs --tail 1 "$CONTAINER_NAME" 2>&1 || true
            next_log=$((elapsed + 60))
        fi
        sleep "$STARTUP_POLL_SECONDS"
    done

    local logs
    logs="$(docker logs --tail 2000 "$CONTAINER_NAME" 2>&1 || true)"
    if grep -q 'TRITON_MLA_SPARSE' <<<"$logs"; then
        ok "Startup logs mention TRITON_MLA_SPARSE"
    else
        warn "The API is healthy, but TRITON_MLA_SPARSE was not found in captured startup logs. Inspect: docker logs $CONTAINER_NAME"
    fi
    if grep -Eq 'Triton fallback|DeepGEMM not supported|is_deep_gemm_supported' <<<"$logs"; then
        ok "Startup logs show the SM80/DeepGEMM fallback path"
    else
        warn "The expected DeepGEMM fallback wording was not found; log wording may have changed."
    fi
}

# Read JSON from a named file to avoid shell/heredoc stdin conflicts.
validate_models_response() {
    local response_file="$1"
    python3 - "$response_file" "$SERVED_MODEL_NAME" <<'PY'
import json
import sys
path, expected = sys.argv[1:3]
with open(path) as f:
    data = json.load(f)
ids = [item.get("id") for item in data.get("data", [])]
print("Served models:", ids)
assert expected in ids, f"expected {expected!r} in model list"
PY
}

run_inference_test() {
    log "Running non-thinking arithmetic inference test"
    local request_file response_file http_code
    request_file="$(mktemp)"
    response_file="$(mktemp)"
    python3 - "$request_file" "$SERVED_MODEL_NAME" <<'PY'
import json
import sys
path, model = sys.argv[1:3]
payload = {
    "model": model,
    "messages": [
        {"role": "user", "content": "What is 17 multiplied by 24? Explain briefly."}
    ],
    "temperature": 0,
    "max_tokens": 300,
    "chat_template_kwargs": {"enable_thinking": False},
}
with open(path, "w") as f:
    json.dump(payload, f)
PY

    http_code="$(curl -sS --max-time 600 \
        -o "$response_file" -w '%{http_code}' \
        "$API_BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@$request_file")"
    [[ "$http_code" == "200" ]] || {
        cat "$response_file" >&2
        rm -f "$request_file" "$response_file"
        die "Inference request returned HTTP $http_code"
    }

    python3 - "$response_file" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    data = json.load(f)
if "error" in data:
    raise SystemExit(f"API error: {data['error']}")
choices = data.get("choices") or []
assert choices, "response contains no choices"
message = choices[0].get("message") or {}
content = message.get("content") or ""
print("Assistant response:")
print(content)
print("Usage:", data.get("usage", {}))
assert "408" in content, "arithmetic smoke test did not contain the expected answer 408"
PY
    rm -f "$request_file" "$response_file"
    ok "Inference smoke test passed"
}

run_smoke_tests() {
    container_running || die "Container $CONTAINER_NAME is not running."
    wait_for_server

    local models_file
    models_file="$(mktemp)"
    curl -fsS "$API_BASE/v1/models" > "$models_file"
    validate_models_response "$models_file"
    rm -f "$models_file"
    run_inference_test
}

benchmark_server() {
    container_running || die "Container $CONTAINER_NAME is not running."
    wait_for_server

    log "Warming the model before the decode benchmark"
    local tmp_req tmp_resp
    tmp_req="$(mktemp)"
    tmp_resp="$(mktemp)"

    python3 - "$tmp_req" "$SERVED_MODEL_NAME" <<'PY'
import json
import sys
path, model = sys.argv[1:3]
with open(path, "w") as f:
    json.dump({
        "model": model,
        "messages": [{"role": "user", "content": "Count upward from 1, separated by spaces."}],
        "temperature": 0,
        "max_tokens": 32,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }, f)
PY
    curl -fsS --max-time 600 -o /dev/null \
        "$API_BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@$tmp_req"

    log "Running $BENCH_RUNS decode benchmark request(s), up to $BENCH_MAX_TOKENS completion tokens each"
    local run metrics elapsed completion rate
    local -a rates=()

    for ((run=1; run<=BENCH_RUNS; run++)); do
        python3 - "$tmp_req" "$SERVED_MODEL_NAME" "$BENCH_MAX_TOKENS" "$run" <<'PY'
import json
import sys
path, model, max_tokens, run = sys.argv[1:5]
payload = {
    "model": model,
    "messages": [{
        "role": "user",
        "content": f"Benchmark run {run}: count upward from 1 forever, separated only by spaces."
    }],
    "temperature": 0,
    "max_tokens": int(max_tokens),
    "ignore_eos": True,
    "chat_template_kwargs": {"enable_thinking": False},
}
with open(path, "w") as f:
    json.dump(payload, f)
PY

        metrics="$(curl -fsS --max-time 1200 \
            -o "$tmp_resp" -w '%{time_total}' \
            "$API_BASE/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            --data-binary "@$tmp_req")"
        elapsed="$metrics"
        completion="$(python3 - "$tmp_resp" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print((data.get("usage") or {}).get("completion_tokens", 0))
PY
)"
        [[ "$completion" =~ ^[0-9]+$ ]] || die "Could not parse completion token count."
        rate="$(python3 - "$completion" "$elapsed" <<'PY'
import sys
n = int(sys.argv[1]); seconds = float(sys.argv[2])
print(f"{(n / seconds if seconds else 0):.2f}")
PY
)"
        rates+=("$rate")
        printf 'Run %d: %s completion tokens in %ss = %s output tok/s (end-to-end)\n' \
            "$run" "$completion" "$elapsed" "$rate"
    done

    python3 - "${rates[@]}" <<'PY'
import statistics
import sys
rates = [float(x) for x in sys.argv[1:]]
print(f"Median end-to-end output rate: {statistics.median(rates):.2f} tok/s")
PY
    rm -f "$tmp_req" "$tmp_resp"
    ok "Benchmark completed"
}

show_status() {
    human_banner
    printf '\nDocker image:\n'
    docker image inspect "$IMAGE_NAME" --format \
        '  {{.RepoTags}}  created={{.Created}}  size={{.Size}} bytes' 2>/dev/null || echo '  not built'

    printf '\nContainer:\n'
    if container_exists; then
        docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format \
            '  {{.Names}}  {{.Status}}  {{.Ports}}  {{.Image}}'
    else
        echo '  not created'
    fi

    printf '\nModel directory:\n'
    if [[ -d "$MODEL_DIR" ]]; then
        du -sh "$MODEL_DIR" 2>/dev/null || true
        [[ -s "$MODEL_DIR/config.json" ]] && echo '  config.json present' || echo '  config.json missing'
    else
        echo '  not downloaded'
    fi

    printf '\nAPI:\n'
    if curl -fsS --max-time 3 "$API_BASE/health" >/dev/null 2>&1; then
        echo "  healthy at $API_BASE"
        curl -fsS "$API_BASE/v1/models" || true
        echo
    else
        echo "  unavailable at $API_BASE"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        printf '\nGPUs:\n'
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader || true
    fi
}

follow_logs() {
    container_exists || die "Container $CONTAINER_NAME does not exist."
    docker logs -f --tail 200 "$CONTAINER_NAME"
}

# --------------------------------- Main ----------------------------------
case "$ACTION" in
    all|install)
        human_banner
        check_host_prerequisites
        check_disk_space
        check_host_hardware
        build_image
        verify_image
        download_model
        start_container
        run_smoke_tests
        benchmark_server
        log "Installation complete"
        printf 'API endpoint: %s/v1\n' "$API_BASE"
        printf 'Container:    %s\n' "$CONTAINER_NAME"
        printf 'Logs:         docker logs -f %q\n' "$CONTAINER_NAME"
        ;;
    build)
        human_banner
        check_host_prerequisites
        build_image
        ;;
    verify)
        check_host_prerequisites
        check_host_hardware
        verify_image
        ;;
    download)
        human_banner
        check_host_prerequisites
        download_model
        ;;
    run|start)
        human_banner
        check_host_prerequisites
        check_host_hardware
        start_container
        ;;
    test)
        check_host_prerequisites
        run_smoke_tests
        ;;
    benchmark|bench)
        check_host_prerequisites
        benchmark_server
        ;;
    logs)
        check_host_prerequisites
        follow_logs
        ;;
    status)
        check_host_prerequisites
        show_status
        ;;
    stop)
        check_host_prerequisites
        stop_container
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        die "Unknown action: $ACTION"
        ;;
esac
