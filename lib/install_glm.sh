#!/usr/bin/env bash
# install_glm.sh — Dockerized GLM-5.2 AWQ INT4 deployment helpers for 8x A100 hosts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tui.sh"
source "$SCRIPT_DIR/install_claude.sh"

GLM_STATE_ROOT="${GLM_STATE_ROOT:-$HOME/.local/share/pushbutton/glm52}"
GLM_BASE_IMAGE="${GLM_BASE_IMAGE:-vllm/vllm-openai:glm52-x86_64-cu129}"
GLM_IMAGE_NAME="${GLM_IMAGE_NAME:-vllm/vllm-openai:glm52-cu129-patched}"
GLM_CONTAINER_NAME="${GLM_CONTAINER_NAME:-glm52-awq-a100}"
GLM_MODEL_REPO_DEFAULT="${GLM_MODEL_REPO_DEFAULT:-cyankiwi/GLM-5.2-AWQ-INT4}"
GLM_MODEL_SUBDIR_DEFAULT="${GLM_MODEL_SUBDIR_DEFAULT:-GLM-5.2-AWQ-INT4}"
GLM_PATCH_REPO="${GLM_PATCH_REPO:-https://github.com/vllm-project/vllm.git}"
GLM_PATCH_PR="${GLM_PATCH_PR:-38476}"
GLM_PATCH_COMMIT="${GLM_PATCH_COMMIT:-}"
GLM_GPU_MEMORY_UTILIZATION="${GLM_GPU_MEMORY_UTILIZATION:-0.90}"
GLM_KV_CACHE_DTYPE="${GLM_KV_CACHE_DTYPE:-auto}"
GLM_SHM_SIZE="${GLM_SHM_SIZE:-64g}"
GLM_STARTUP_TIMEOUT="${GLM_STARTUP_TIMEOUT:-7200}"
GLM_STARTUP_POLL_SECONDS="${GLM_STARTUP_POLL_SECONDS:-10}"
GLM_MIN_FREE_GB="${GLM_MIN_FREE_GB:-440}"
GLM_HF_HUB_DOWNLOAD_WORKERS="${GLM_HF_HUB_DOWNLOAD_WORKERS:-8}"
GLM_BENCH_MAX_TOKENS="${GLM_BENCH_MAX_TOKENS:-256}"
GLM_BENCH_RUNS="${GLM_BENCH_RUNS:-2}"
GLM_ENABLE_TOOL_CALLING="${GLM_ENABLE_TOOL_CALLING:-1}"
GLM_SKIP_HARDWARE_CHECK="${GLM_SKIP_HARDWARE_CHECK:-0}"
GLM_SKIP_DISK_CHECK="${GLM_SKIP_DISK_CHECK:-0}"
GLM_SKIP_DOWNLOAD="${GLM_SKIP_DOWNLOAD:-0}"
GLM_PULL_BASE_IMAGE="${GLM_PULL_BASE_IMAGE:-1}"
GLM_PORT="${GLM_PORT:-8000}"
GLM_BIND_ADDRESS="${GLM_BIND_ADDRESS:-127.0.0.1}"
GLM_TP="${GLM_TP:-8}"
GLM_MAX_MODEL_LEN="${GLM_MAX_MODEL_LEN:-32768}"
GLM_ENABLE_THINKING="${GLM_ENABLE_THINKING:-0}"

GLM_MODEL_ROOT="${GLM_MODEL_ROOT:-$GLM_STATE_ROOT/model}"
GLM_RUNTIME_CACHE_DIR="${GLM_RUNTIME_CACHE_DIR:-$GLM_STATE_ROOT/runtime-cache}"
GLM_BUILD_DIR="${GLM_BUILD_DIR:-$GLM_STATE_ROOT/build}"
GLM_RUNTIME_DIR="${GLM_RUNTIME_DIR:-$GLM_STATE_ROOT/runtime}"
GLM_LOG_DIR="${GLM_LOG_DIR:-$GLM_STATE_ROOT/logs}"

_glm_api_base() {
    local host
    if [[ "$GLM_BIND_ADDRESS" == "0.0.0.0" || "$GLM_BIND_ADDRESS" == "127.0.0.1" || "$GLM_BIND_ADDRESS" == "localhost" ]]; then
        host="127.0.0.1"
    else
        host="$GLM_BIND_ADDRESS"
    fi
    printf 'http://%s:%s\n' "$host" "$GLM_PORT"
}

_glm_model_subdir() {
    local repo="${1:-$GLM_MODEL_REPO_DEFAULT}"
    basename "$repo"
}

_glm_model_dir() {
    local repo="${1:-$GLM_MODEL_REPO_DEFAULT}"
    printf '%s/%s\n' "$GLM_MODEL_ROOT" "$(_glm_model_subdir "$repo")"
}

_glm_container_model_dir() {
    local repo="${1:-$GLM_MODEL_REPO_DEFAULT}"
    printf '/models/%s\n' "$(_glm_model_subdir "$repo")"
}

_glm_runtime_file() {
    mkdir -p "$GLM_RUNTIME_DIR"
    printf '%s/%s\n' "$GLM_RUNTIME_DIR" "$1"
}

glm_require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        tui_error "Required command not found: $1"
        return 1
    }
}

_glm_container_exists() {
    docker inspect "$GLM_CONTAINER_NAME" >/dev/null 2>&1
}

_glm_container_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "$GLM_CONTAINER_NAME" 2>/dev/null || true)" == "true" ]]
}

_glm_image_exists() {
    docker image inspect "$GLM_IMAGE_NAME" >/dev/null 2>&1
}

_glm_check_disk_space() {
    [[ "$GLM_SKIP_DISK_CHECK" == "1" ]] && return 0
    mkdir -p "$GLM_MODEL_ROOT"
    local model_dir available_kb available_gb existing_kb existing_gb effective_gb
    model_dir="$(_glm_model_dir "$GLM_MODEL_REPO_DEFAULT")"
    available_kb="$(df -Pk "$GLM_MODEL_ROOT" | awk 'NR==2 {print $4}')"
    [[ "$available_kb" =~ ^[0-9]+$ ]] || {
        tui_error "Could not determine free space for $GLM_MODEL_ROOT"
        return 1
    }
    available_gb=$((available_kb / 1024 / 1024))
    existing_kb=0
    if [[ -d "$model_dir" ]]; then
        existing_kb="$(du -sk "$model_dir" 2>/dev/null | awk '{print $1}')"
        [[ "$existing_kb" =~ ^[0-9]+$ ]] || existing_kb=0
    fi
    existing_gb=$((existing_kb / 1024 / 1024))
    effective_gb=$((available_gb + existing_gb))
    (( effective_gb >= GLM_MIN_FREE_GB )) || {
        tui_error "Need at least ${GLM_MIN_FREE_GB}GiB free+existing capacity for GLM-5.2"
        return 1
    }
}

glm_check_prereqs() {
    tui_header "GLM-5.2 A100 Prerequisites"
    local cmd
    for cmd in docker git tar curl python3 df awk grep; do
        glm_require_cmd "$cmd" || return 1
    done
    docker info >/dev/null 2>&1 || {
        tui_error "Docker daemon is not reachable"
        return 1
    }
    mkdir -p "$GLM_MODEL_ROOT" "$GLM_RUNTIME_CACHE_DIR" "$GLM_BUILD_DIR" "$GLM_RUNTIME_DIR" "$GLM_LOG_DIR"
    _glm_check_disk_space || return 1

    [[ "$GLM_SKIP_HARDWARE_CHECK" == "1" ]] && return 0

    glm_require_cmd nvidia-smi || return 1
    local gpu_output
    gpu_output="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -n "$gpu_output" ]] || gpu_output="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
    local -a gpu_lines=()
    mapfile -t gpu_lines <<< "$gpu_output"
    [[ "${#gpu_lines[@]}" -eq 8 ]] || {
        tui_error "Expected exactly 8 visible physical GPUs; found ${#gpu_lines[@]}"
        return 1
    }

    local i line name mem cap
    for i in "${!gpu_lines[@]}"; do
        line="${gpu_lines[$i]}"
        IFS=',' read -r name mem cap <<< "$line"
        name="${name#"${name%%[![:space:]]*}"}"
        name="${name%"${name##*[![:space:]]}"}"
        mem="${mem//[[:space:]]/}"
        cap="${cap//[[:space:]]/}"
        [[ "$name" == *A100* ]] || {
            tui_error "GPU $i is '$name', not an A100"
            return 1
        }
        (( mem >= 79000 )) || {
            tui_error "GPU $i has ${mem}MiB, not 80GB-class A100"
            return 1
        }
        if [[ -n "$cap" && "$cap" != "8.0" ]]; then
            tui_error "GPU $i compute capability is $cap, expected 8.0"
            return 1
        fi
    done

    tui_success "Docker and 8×A100 host checks passed"
}

_glm_pull_base_image() {
    if [[ "$GLM_PULL_BASE_IMAGE" == "1" ]]; then
        tui_step "Pulling GLM-5.2 base image"
        docker pull "$GLM_BASE_IMAGE"
    elif ! docker image inspect "$GLM_BASE_IMAGE" >/dev/null 2>&1; then
        tui_error "Base image $GLM_BASE_IMAGE is absent and GLM_PULL_BASE_IMAGE=0"
        return 1
    fi
}

_glm_prepare_build_context() {
    tui_step "Preparing vLLM PR #$GLM_PATCH_PR overlay"
    rm -rf "$GLM_BUILD_DIR/vllm-pr"
    mkdir -p "$GLM_BUILD_DIR"

    git clone --filter=blob:none --no-checkout "$GLM_PATCH_REPO" "$GLM_BUILD_DIR/vllm-pr"
    git -C "$GLM_BUILD_DIR/vllm-pr" fetch --depth=1 origin "pull/${GLM_PATCH_PR}/head:refs/remotes/origin/pr-${GLM_PATCH_PR}"

    local resolved_commit
    resolved_commit="$(git -C "$GLM_BUILD_DIR/vllm-pr" rev-parse "refs/remotes/origin/pr-${GLM_PATCH_PR}")"
    if [[ -n "$GLM_PATCH_COMMIT" && "$resolved_commit" != "$GLM_PATCH_COMMIT" ]]; then
        tui_error "PR #$GLM_PATCH_PR currently resolves to $resolved_commit, not requested GLM_PATCH_COMMIT=$GLM_PATCH_COMMIT"
        return 1
    fi
    git -C "$GLM_BUILD_DIR/vllm-pr" checkout --detach "$resolved_commit"

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
        [[ -f "$GLM_BUILD_DIR/vllm-pr/$f" ]] || {
            tui_error "Expected PR patch file is missing: $f"
            return 1
        }
    done

    tar -C "$GLM_BUILD_DIR/vllm-pr" -czf "$GLM_BUILD_DIR/vllm_patch_files.tar.gz" "${patch_files[@]}"
    printf '%s\n' "$resolved_commit" > "$GLM_BUILD_DIR/PATCH_COMMIT"

    cat > "$GLM_BUILD_DIR/fix_ampere.py" <<'PY'
from pathlib import Path
import vllm

vllm_dir = Path(vllm.__file__).resolve().parent

def replace_or_verify(path: Path, old: str, new: str) -> None:
    src = path.read_text()
    if old in src:
        path.write_text(src.replace(old, new))
    elif new in src:
        return
    else:
        raise RuntimeError(f"expected patch text missing in {path}")

replace_or_verify(vllm_dir / "v1/attention/backends/mla/indexer.py", "has_deep_gemm", "is_deep_gemm_supported")
replace_or_verify(vllm_dir / "utils/deep_gemm.py", "if not has_deep_gemm():", "if not is_deep_gemm_supported():")

deepseek_path = vllm_dir / "model_executor/models/deepseek_v2.py"
deepseek_src = deepseek_path.read_text()
marker = "self.use_fused_indexer_q = ("
if marker in deepseek_src and "current_platform.has_device_capability(89)" not in deepseek_src:
    old = "current_platform.is_cuda()\n            and self.quant_block_size"
    new = "current_platform.is_cuda()\n            and current_platform.has_device_capability(89)\n            and self.quant_block_size"
    if old not in deepseek_src:
        raise RuntimeError("deepseek_v2.py fused indexer-Q predicate changed")
    deepseek_path.write_text(deepseek_src.replace(old, new, 1))
PY

    cat > "$GLM_BUILD_DIR/verify_patches.py" <<'PY'
import importlib
import inspect
from pathlib import Path
import vllm

for name in (
    "vllm.v1.attention.backends.mla.triton_mla_sparse",
    "vllm.v1.attention.ops.mqa_logits_triton",
    "vllm.v1.attention.ops.triton_mla_sparse_kernel",
):
    importlib.import_module(name)

from vllm.v1.attention.backends.mla import indexer
from vllm.utils import deep_gemm

assert "has_deep_gemm" not in inspect.getsource(indexer)
assert "if not is_deep_gemm_supported():" in inspect.getsource(deep_gemm._lazy_init)
root = Path(vllm.__file__).resolve().parent
deepseek_src = (root / "model_executor/models/deepseek_v2.py").read_text()
if "self.use_fused_indexer_q = (" in deepseek_src:
    start = deepseek_src.index("self.use_fused_indexer_q = (")
    assert "current_platform.has_device_capability(89)" in deepseek_src[start:start+700]
PY

    cat > "$GLM_BUILD_DIR/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE=vllm/vllm-openai:glm52-x86_64-cu129
FROM ${BASE_IMAGE}

ARG PATCH_PR=38476
ARG PATCH_COMMIT=unknown
LABEL io.vllm.patch.pr="${PATCH_PR}" \
      io.vllm.patch.commit="${PATCH_COMMIT}"

COPY vllm_patch_files.tar.gz /tmp/vllm_patch_files.tar.gz
RUN set -eux; \
    DEST="$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')"; \
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

    rm -rf "$GLM_BUILD_DIR/vllm-pr"
}

_glm_build_image() {
    _glm_pull_base_image
    _glm_prepare_build_context
    local patch_commit
    patch_commit="$(cat "$GLM_BUILD_DIR/PATCH_COMMIT")"
    tui_step "Building $GLM_IMAGE_NAME"
    docker build \
        --build-arg "BASE_IMAGE=$GLM_BASE_IMAGE" \
        --build-arg "PATCH_PR=$GLM_PATCH_PR" \
        --build-arg "PATCH_COMMIT=$patch_commit" \
        --tag "$GLM_IMAGE_NAME" \
        --file "$GLM_BUILD_DIR/Dockerfile" \
        "$GLM_BUILD_DIR"
}

_glm_verify_image() {
    _glm_image_exists || {
        tui_error "Image $GLM_IMAGE_NAME does not exist"
        return 1
    }

    tui_step "Verifying patched image"
    docker run --rm --entrypoint python3 "$GLM_IMAGE_NAME" -c '
import importlib
import inspect
from pathlib import Path
import vllm
for name in (
    "vllm.v1.attention.backends.mla.triton_mla_sparse",
    "vllm.v1.attention.ops.mqa_logits_triton",
    "vllm.v1.attention.ops.triton_mla_sparse_kernel",
):
    importlib.import_module(name)
from vllm.v1.attention.backends.mla import indexer
from vllm.utils import deep_gemm
assert "has_deep_gemm" not in inspect.getsource(indexer)
assert "if not is_deep_gemm_supported():" in inspect.getsource(deep_gemm._lazy_init)
root = Path(vllm.__file__).resolve().parent
deepseek_src = (root / "model_executor/models/deepseek_v2.py").read_text()
if "self.use_fused_indexer_q = (" in deepseek_src:
    start = deepseek_src.index("self.use_fused_indexer_q = (")
    assert "current_platform.has_device_capability(89)" in deepseek_src[start:start+700]
'
}

_glm_download_model() {
    local model_repo="${1:-$GLM_MODEL_REPO_DEFAULT}"
    [[ "$GLM_SKIP_DOWNLOAD" == "1" ]] && return 0
    _glm_image_exists || _glm_build_image
    local model_dir container_dir
    model_dir="$(_glm_model_dir "$model_repo")"
    container_dir="$(_glm_container_model_dir "$model_repo")"
    mkdir -p "$GLM_MODEL_ROOT" "$GLM_RUNTIME_CACHE_DIR"

    tui_step "Downloading/resuming $model_repo"
    docker run --rm -i \
        -e "HF_HUB_DOWNLOAD_WORKERS=$GLM_HF_HUB_DOWNLOAD_WORKERS" \
        ${HF_TOKEN:+-e HF_TOKEN} \
        -v "$GLM_MODEL_ROOT:/models" \
        -v "$GLM_RUNTIME_CACHE_DIR:/root/.cache" \
        --entrypoint python3 \
        "$GLM_IMAGE_NAME" - "$model_repo" "$container_dir" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1:3]
workers = int(os.environ.get("HF_HUB_DOWNLOAD_WORKERS", "8"))
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=workers,
)
PY

    [[ -s "$model_dir/config.json" ]] || {
        tui_error "Model download did not produce $model_dir/config.json"
        return 1
    }
}

glm_stop_server() {
    local container_name="${1:-$GLM_CONTAINER_NAME}"
    if docker inspect "$container_name" >/dev/null 2>&1; then
        tui_step "Stopping $container_name"
        docker rm -f "$container_name" >/dev/null
    fi
}

_glm_wait_for_server() {
    local api_base
    api_base="$(_glm_api_base)"
    local start now elapsed next_log=0
    start="$(date +%s)"
    while true; do
        if curl -fsS --max-time 3 "$api_base/health" >/dev/null 2>&1; then
            return 0
        fi
        _glm_container_running || {
            docker logs --tail 200 "$GLM_CONTAINER_NAME" >&2 || true
            tui_error "Container exited before API became healthy"
            return 1
        }
        now="$(date +%s)"
        elapsed=$((now - start))
        (( elapsed < GLM_STARTUP_TIMEOUT )) || {
            docker logs --tail 200 "$GLM_CONTAINER_NAME" >&2 || true
            tui_error "API did not become healthy within ${GLM_STARTUP_TIMEOUT}s"
            return 1
        }
        if (( elapsed >= next_log )); then
            tui_info "Still loading: ${elapsed}s elapsed"
            docker logs --tail 1 "$GLM_CONTAINER_NAME" 2>&1 || true
            next_log=$((elapsed + 60))
        fi
        sleep "$GLM_STARTUP_POLL_SECONDS"
    done
}

glm_pull_and_run() {
    local model_repo="${1:-$GLM_MODEL_REPO_DEFAULT}"
    local max_model_len="${2:-$GLM_MAX_MODEL_LEN}"
    local enable_thinking="${3:-$GLM_ENABLE_THINKING}"
    local tp="${4:-$GLM_TP}"
    GLM_MODEL_REPO_DEFAULT="$model_repo"
    GLM_MAX_MODEL_LEN="$max_model_len"
    GLM_ENABLE_THINKING="$enable_thinking"
    GLM_TP="$tp"
    glm_check_prereqs
    _glm_build_image
    _glm_verify_image
    _glm_download_model "$model_repo"
    glm_start_server "$model_repo" "$max_model_len" "$tp" "$GLM_PORT" "$(_glm_runtime_file container.id)" "$GLM_LOG_DIR/server.log"
}

glm_start_server() {
    local model_repo="${1:-$GLM_MODEL_REPO_DEFAULT}"
    local max_model_len="${2:-$GLM_MAX_MODEL_LEN}"
    local tp="${3:-$GLM_TP}"
    local port="${4:-$GLM_PORT}"
    local pid_file="${5:-$(_glm_runtime_file container.id)}"
    local log_file="${6:-$GLM_LOG_DIR/server.log}"

    local model_dir container_model_dir
    model_dir="$(_glm_model_dir "$model_repo")"
    container_model_dir="$(_glm_container_model_dir "$model_repo")"
    [[ -s "$model_dir/config.json" ]] || {
        tui_error "Model is not present at $model_dir"
        return 1
    }
    GLM_PORT="$port"

    if _glm_container_exists; then
        docker rm -f "$GLM_CONTAINER_NAME" >/dev/null || true
    fi

    mkdir -p "$GLM_RUNTIME_CACHE_DIR" "$GLM_LOG_DIR"

    local -a env_args=(-e VLLM_ATTENTION_BACKEND=TRITON_MLA_SPARSE)
    [[ -n "${HF_TOKEN:-}" ]] && env_args+=(-e HF_TOKEN)
    local -a server_args=(
        --model "$container_model_dir"
        --served-model-name "$model_repo"
        --tensor-parallel-size "$tp"
        --no-async-scheduling
        --gpu-memory-utilization "$GLM_GPU_MEMORY_UTILIZATION"
        --max-model-len "$max_model_len"
        --trust-remote-code
        --kv-cache-dtype "$GLM_KV_CACHE_DTYPE"
        --host 0.0.0.0
        --port 8000
    )
    if [[ "$GLM_ENABLE_TOOL_CALLING" == "1" ]]; then
        server_args+=(--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice)
    fi
    if [[ "$GLM_ENABLE_THINKING" != "1" ]]; then
        server_args+=(--default-chat-template-kwargs '{"enable_thinking":false}')
    fi

    tui_step "Starting $GLM_CONTAINER_NAME"
    local container_id
    container_id="$(docker run -d \
        --name "$GLM_CONTAINER_NAME" \
        --restart unless-stopped \
        --gpus all \
        --shm-size "$GLM_SHM_SIZE" \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p "$GLM_BIND_ADDRESS:$GLM_PORT:8000" \
        "${env_args[@]}" \
        -v "$GLM_MODEL_ROOT:/models:ro" \
        -v "$GLM_RUNTIME_CACHE_DIR:/root/.cache" \
        "$GLM_IMAGE_NAME" \
        "${server_args[@]}")"
    printf '%s\n' "$container_id" > "$pid_file"
    _glm_wait_for_server
    docker logs --tail 200 "$GLM_CONTAINER_NAME" > "$log_file" 2>&1 || true
    tui_success "GLM API healthy at $(_glm_api_base)/v1"
}

_glm_validate_models_response() {
    local response_file="${1:?response file required}"
    local expected_model="${2:?model required}"
    python3 - "$response_file" "$expected_model" <<'PY'
import json
import sys
path, expected = sys.argv[1:3]
with open(path) as f:
    data = json.load(f)
ids = [item.get("id") for item in data.get("data", [])]
assert expected in ids, f"expected {expected!r} in model list"
PY
}

glm_api_smoke() {
    local port="${1:-$GLM_PORT}"
    GLM_PORT="$port"
    local api_base models_file request_file response_file http_code
    api_base="$(_glm_api_base)"
    models_file="$(_glm_runtime_file models.json)"
    request_file="$(_glm_runtime_file request.json)"
    response_file="$(_glm_runtime_file response.json)"

    curl -fsS "$api_base/v1/models" > "$models_file"
    _glm_validate_models_response "$models_file" "$GLM_MODEL_REPO_DEFAULT"

    python3 - "$request_file" "$GLM_MODEL_REPO_DEFAULT" <<'PY'
import json
import sys
path, model = sys.argv[1:3]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "What is 17 multiplied by 24? Explain briefly."}],
    "temperature": 0,
    "max_tokens": 300,
    "chat_template_kwargs": {"enable_thinking": False},
}
with open(path, "w") as f:
    json.dump(payload, f)
PY
    http_code="$(curl -sS --max-time 600 -o "$response_file" -w '%{http_code}' "$api_base/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$request_file")"
    [[ "$http_code" == "200" ]] || {
        cat "$response_file" >&2
        tui_error "Inference request returned HTTP $http_code"
        return 1
    }
    python3 - "$response_file" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    data = json.load(f)
choices = data.get("choices") or []
assert choices, "response contains no choices"
message = choices[0].get("message") or {}
content = message.get("content") or ""
assert "408" in content, "arithmetic smoke test did not contain the expected answer 408"
PY
    tui_success "GLM OpenAI smoke passed"
}

_glm_benchmark_server() {
    local api_base request_file response_file metrics elapsed completion rate run
    api_base="$(_glm_api_base)"
    request_file="$(_glm_runtime_file bench-request.json)"
    response_file="$(_glm_runtime_file bench-response.json)"
    local -a rates=()

    tui_step "Warming GLM server"
    python3 - "$request_file" "$GLM_MODEL_REPO_DEFAULT" <<'PY'
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
    curl -fsS --max-time 600 -o /dev/null "$api_base/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$request_file"

    for ((run=1; run<=GLM_BENCH_RUNS; run++)); do
        python3 - "$request_file" "$GLM_MODEL_REPO_DEFAULT" "$GLM_BENCH_MAX_TOKENS" "$run" <<'PY'
import json
import sys
path, model, max_tokens, run = sys.argv[1:5]
with open(path, "w") as f:
    json.dump({
        "model": model,
        "messages": [{"role": "user", "content": f"Benchmark run {run}: count upward from 1 forever, separated only by spaces."}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }, f)
PY
        metrics="$(curl -fsS --max-time 1200 -o "$response_file" -w '%{time_total}' "$api_base/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$request_file")"
        elapsed="$metrics"
        completion="$(python3 - "$response_file" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print((data.get("usage") or {}).get("completion_tokens", 0))
PY
)"
        rate="$(python3 - "$completion" "$elapsed" <<'PY'
import sys
n = int(sys.argv[1]); seconds = float(sys.argv[2])
print(f"{(n / seconds if seconds else 0):.2f}")
PY
)"
        rates+=("$rate")
        tui_info "Run $run: $completion completion tokens in ${elapsed}s = ${rate} tok/s"
    done
}

glm_export_claude_env() {
    local port="${1:-$GLM_PORT}"
    local api_base
    GLM_PORT="$port"
    api_base="$(_glm_api_base)"
    export ANTHROPIC_BASE_URL="$api_base"
    export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-local-key}"
    export ANTHROPIC_MODEL="$GLM_MODEL_REPO_DEFAULT"
    export ANTHROPIC_DEFAULT_SONNET_MODEL="$GLM_MODEL_REPO_DEFAULT"
    export ANTHROPIC_DEFAULT_OPUS_MODEL="$GLM_MODEL_REPO_DEFAULT"
    export ANTHROPIC_DEFAULT_HAIKU_MODEL="$GLM_MODEL_REPO_DEFAULT"
}

setup_glm52_a100() {
    local model_repo="${1:-$GLM_MODEL_REPO_DEFAULT}"
    local max_model_len="${2:-$GLM_MAX_MODEL_LEN}"
    glm_check_prereqs
    glm_pull_and_run "$model_repo" "$max_model_len" "$GLM_ENABLE_THINKING" "$GLM_TP"
    glm_api_smoke "$GLM_PORT"
    _glm_benchmark_server || true
    setup_claude || true
    glm_export_claude_env "$GLM_PORT"
}
