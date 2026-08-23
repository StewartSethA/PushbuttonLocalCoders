#!/usr/bin/env bash
# install_llamacpp.sh — Build llama.cpp with GPU acceleration (CUDA/Metal/ROCm)
#                       and a CPU-optimised fallback.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_hardware.sh"
source "$SCRIPT_DIR/tui.sh"

LLAMACPP_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/.local/share/llama.cpp}"
LLAMACPP_BIN="$LLAMACPP_DIR/build/bin"
LLAMA_INSTALL_PREFIX="${LLAMA_INSTALL_PREFIX:-$HOME/.local}"

# Private CUDA prefix installed by this script (never touches /usr/local).
CUDA_PRIVATE_PREFIX="${CUDA_PRIVATE_PREFIX:-$HOME/.local/share/cuda-toolkit}"
CUDA_RATTLER_CHANNEL="${CUDA_RATTLER_CHANNEL:-https://conda.anaconda.org/nvidia/label/cuda-12.9.1}"
CUDA_BOOTSTRAP_ENV="${CUDA_BOOTSTRAP_ENV:-$HOME/.local/share/pushbutton/cuda-bootstrap-py}"
CUDA_UV_TOOLS_DIR="${CUDA_UV_TOOLS_DIR:-$HOME/.local/share/pushbutton/tools}"
CUDA_UV_CACHE_DIR="${CUDA_UV_CACHE_DIR:-$HOME/.local/share/pushbutton/uv-cache}"
PY_RATTLER_VERSION="${PY_RATTLER_VERSION:-0.25.0}"

# CUDA 12.9.1 is the pinned version for legacy architectures (Volta sm_70,
# Pascal sm_61) because CUDA 13 removed their offline-compilation support.
CUDA12_VERSION="12.9.1"
CUDA12_CUBLAS_VERSION="12.9.1.4"

# ── CUDA architecture detection ────────────────────────────────────────────────
# Returns true (0) when the detected GPU requires CUDA 12.x because CUDA 13
# dropped offline-compilation support for it.
gpu_needs_cuda12() {
    local model="${GPU_MODEL:-}"
    # Volta (V100 = sm_70) and Pascal (P40, P100 = sm_61/sm_60)
    if echo "$model" | grep -qiE "V100|Tesla V|P40|P100|Tesla P"; then
        return 0
    fi
    return 1
}

# Map GPU model to CUDA architecture string for -DCMAKE_CUDA_ARCHITECTURES.
# Falls back to "native" (let CMake auto-detect) for modern cards.
cuda_arch_for_gpu() {
    local model="${GPU_MODEL:-}"
    if echo "$model" | grep -qiE "V100|Tesla V"; then
        echo "70"
    elif echo "$model" | grep -qiE "P40"; then
        echo "61"
    elif echo "$model" | grep -qiE "P100|Tesla P100"; then
        echo "60"
    else
        echo "native"
    fi
}

# ── CUDA toolkit installation ──────────────────────────────────────────────────
# Installs the CUDA toolkit into a private prefix using micromamba so that the
# system NVIDIA driver is left untouched and no files land in /usr/local.
#
# Sets and exports:
#   CUDA_ROOT          — prefix containing bin/, lib64/, include/
#   CUDA_HOME / CUDA_PATH
#   CUDACXX            — full path to nvcc
#   PATH, LD_LIBRARY_PATH, LIBRARY_PATH, CPATH
#
# Arguments:
#   $1  use_cuda12   — "true" to install the pinned CUDA 12.x toolkit,
#                      anything else installs the latest available toolkit.
install_cuda_toolkit() {
    local use_cuda12="${1:-false}"

    tui_step "Installing CUDA toolkit into private prefix $CUDA_PRIVATE_PREFIX…"

    # ── Bootstrap micromamba if needed ───────────────────────────────────────
    local mamba_bin="$HOME/.local/bin/micromamba"
    if ! command -v micromamba &>/dev/null && [[ ! -x "$mamba_bin" ]]; then
        tui_step "Downloading micromamba…"
        mkdir -p "$HOME/.local/bin"
        local mamba_url="https://micro.mamba.pm/api/micromamba/linux-64/latest"
        if command -v curl &>/dev/null; then
            curl -fsSL "$mamba_url" | tar -xj -C "$HOME/.local/bin" --strip-components=1 bin/micromamba
        elif command -v wget &>/dev/null; then
            wget -qO- "$mamba_url" | tar -xj -C "$HOME/.local/bin" --strip-components=1 bin/micromamba
        else
            tui_error "Neither curl nor wget found — cannot download micromamba."
            return 1
        fi
        chmod +x "$mamba_bin"
    fi
    local mamba_cmd
    mamba_cmd=$(command -v micromamba 2>/dev/null || echo "$mamba_bin")

    # ── Select package versions ───────────────────────────────────────────────
    local cuda_pkg libcublas_pkg
    if [[ "$use_cuda12" == "true" ]]; then
        tui_step "Pinning CUDA toolkit to $CUDA12_VERSION for legacy GPU architecture…"
        cuda_pkg="cuda-minimal-build=$CUDA12_VERSION"
        libcublas_pkg="libcublas-dev=$CUDA12_CUBLAS_VERSION"
    else
        cuda_pkg="cuda-minimal-build"
        libcublas_pkg="libcublas-dev"
    fi

    # ── Install into private prefix ───────────────────────────────────────────
    if ! "$mamba_cmd" install \
        --prefix "$CUDA_PRIVATE_PREFIX" \
        --channel nvidia \
        --channel conda-forge \
        --yes \
        "$cuda_pkg" \
        "$libcublas_pkg" 2>&1; then
        tui_error "micromamba install failed."
        return 1
    fi

    if [[ ! -x "$CUDA_PRIVATE_PREFIX/bin/nvcc" ]]; then
        tui_error "nvcc not found after CUDA toolkit installation."
        return 1
    fi

    # ── Export environment so CMake picks up the private prefix ──────────────
    export CUDA_ROOT="$CUDA_PRIVATE_PREFIX"
    export CUDA_HOME="$CUDA_PRIVATE_PREFIX"
    export CUDA_PATH="$CUDA_PRIVATE_PREFIX"
    export CUDACXX="$CUDA_PRIVATE_PREFIX/bin/nvcc"
    export PATH="$CUDA_PRIVATE_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PRIVATE_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$CUDA_PRIVATE_PREFIX/lib64:${LIBRARY_PATH:-}"
    export CPATH="$CUDA_PRIVATE_PREFIX/include:${CPATH:-}"

    tui_success "CUDA toolkit ready at $CUDA_PRIVATE_PREFIX ✓"
}

ensure_local_uv() {
    local uv_bin=""
    if command -v uv &>/dev/null; then
        uv_bin="$(command -v uv)"
    elif [[ -x "$CUDA_UV_TOOLS_DIR/bin/uv" ]]; then
        uv_bin="$CUDA_UV_TOOLS_DIR/bin/uv"
    else
        tui_step "Installing private uv bootstrap…"
        mkdir -p "$CUDA_UV_TOOLS_DIR/bin" "$CUDA_UV_CACHE_DIR"
        local installer="$CUDA_UV_TOOLS_DIR/uv-install.sh"
        curl -LsSf https://astral.sh/uv/install.sh -o "$installer"
        env UV_UNMANAGED_INSTALL="$CUDA_UV_TOOLS_DIR/bin" UV_NO_MODIFY_PATH=1 sh "$installer" >/dev/null
        uv_bin="$CUDA_UV_TOOLS_DIR/bin/uv"
    fi

    [[ -x "$uv_bin" ]] || {
        tui_error "uv bootstrap returned no executable"
        return 1
    }

    export UV_CACHE_DIR="$CUDA_UV_CACHE_DIR"
    export UV_PYTHON_INSTALL_DIR="$HOME/.local/share/pushbutton/uv-python"
    export UV_NO_MANAGED_PYTHON=1
    printf '%s\n' "$uv_bin"
}

bootstrap_private_cuda_rattler() {
    if [[ -x "$CUDA_PRIVATE_PREFIX/bin/nvcc" ]]; then
        printf '%s\n' "$CUDA_PRIVATE_PREFIX/bin/nvcc"
        return 0
    fi

    local uv_bin py
    uv_bin="$(ensure_local_uv)"
    py="$CUDA_BOOTSTRAP_ENV/bin/python"

    tui_step "Provisioning private CUDA $CUDA12_VERSION via uv + py-rattler…"
    mkdir -p "$CUDA_PRIVATE_PREFIX" "$CUDA_UV_CACHE_DIR"

    if [[ ! -x "$py" ]]; then
        "$uv_bin" venv "$CUDA_BOOTSTRAP_ENV" --python "$(command -v python3)" --no-managed-python >/dev/null
    fi

    "$uv_bin" pip install --python "$py" "py-rattler==$PY_RATTLER_VERSION" >/dev/null

    "$py" - "$CUDA_PRIVATE_PREFIX" "$CUDA_RATTLER_CHANNEL" "$CUDA12_VERSION" "$CUDA12_CUBLAS_VERSION" <<'PYCUDA'
import asyncio
import os
import sys
from pathlib import Path
from rattler import VirtualPackage, install, solve

root = Path(sys.argv[1]).resolve()
channel = sys.argv[2]
cuda_version = sys.argv[3]
cublas_version = sys.argv[4]

async def main():
    records = await solve(
        channels=[channel],
        specs=[f"cuda-minimal-build {cuda_version}", f"libcublas-dev {cublas_version}"],
        virtual_packages=VirtualPackage.detect(),
    )
    root.mkdir(parents=True, exist_ok=True)
    await install(records=records, target_prefix=str(root))

asyncio.run(main())

import platform as _platform
_arch = _platform.machine() + "-linux"
candidates = [root / "bin" / "nvcc", root / "targets" / _arch / "bin" / "nvcc"]
candidates += list(root.glob("targets/*/bin/nvcc"))
nvcc = next((p for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)
if nvcc is None:
    raise SystemExit("nvcc not found after rattler installation")
stable = root / "bin" / "nvcc"
if stable != nvcc:
    stable.parent.mkdir(parents=True, exist_ok=True)
    if stable.exists() or stable.is_symlink():
        stable.unlink()
    stable.symlink_to(nvcc)
print(stable)
PYCUDA

    [[ -x "$CUDA_PRIVATE_PREFIX/bin/nvcc" ]] || {
        tui_error "private CUDA bootstrap completed but nvcc is unavailable"
        return 1
    }
    printf '%s\n' "$CUDA_PRIVATE_PREFIX/bin/nvcc"
}

configure_cuda_env_from_nvcc() {
    local nvcc_path="${1:?nvcc path required}"
    local cuda_root
    cuda_root="$(cd "$(dirname "$nvcc_path")/.." && pwd -P)"
    local arch_linux; arch_linux="$(uname -m)-linux"
    local target="$cuda_root/targets/$arch_linux"
    # Fall back to any present target subdirectory if the arch-specific one doesn't exist.
    if [[ ! -d "$target" ]]; then
        local t; t="$(find "$cuda_root/targets" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)"
        [[ -n "$t" ]] && target="$t"
    fi

    export CUDA_ROOT="$cuda_root"
    export CUDA_HOME="$cuda_root"
    export CUDA_PATH="$cuda_root"
    export CUDACXX="$nvcc_path"
    export PATH="$(dirname "$nvcc_path"):$PATH"

    local lib_path="$cuda_root/lib64:$cuda_root/lib"
    local inc_path="$cuda_root/include"
    if [[ -d "$target" ]]; then
        lib_path="$lib_path:$target/lib:$target/lib/stubs"
        inc_path="$inc_path:$target/include"
    fi
    export LD_LIBRARY_PATH="$lib_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$lib_path${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CPATH="$inc_path${CPATH:+:$CPATH}"
}

ensure_nvcc() {
    [[ -n "${GPU_MODEL:-}" ]] || eval "$(detect_gpu)"
    local want_cuda12="false"
    gpu_needs_cuda12 && want_cuda12="true"

    local -a candidates=()
    [[ -n "${CUDA_HOME:-}" ]] && candidates+=("$CUDA_HOME/bin/nvcc")
    [[ -n "${CUDA_PATH:-}" ]] && candidates+=("$CUDA_PATH/bin/nvcc")
    [[ -n "${CUDA_ROOT:-}" ]] && candidates+=("$CUDA_ROOT/bin/nvcc")
    [[ -n "${CUDACXX:-}" ]] && candidates+=("$CUDACXX")
    candidates+=(
        "$CUDA_PRIVATE_PREFIX/bin/nvcc"
        /usr/local/cuda-12.9/bin/nvcc
        /usr/local/cuda-12.8/bin/nvcc
        /usr/local/cuda-12.6/bin/nvcc
        /usr/local/cuda-12.5/bin/nvcc
        /usr/local/cuda-12.4/bin/nvcc
        /usr/local/cuda-12.3/bin/nvcc
        /usr/local/cuda-12.2/bin/nvcc
        /usr/local/cuda-12.1/bin/nvcc
        /usr/local/cuda-12.0/bin/nvcc
        /usr/local/cuda/bin/nvcc
    )
    command -v nvcc &>/dev/null && candidates+=("$(command -v nvcc)")

    local candidate release major
    for candidate in "${candidates[@]}"; do
        [[ -x "$candidate" ]] || continue
        release="$("$candidate" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | tail -n1)"
        major="${release:-0}"
        if [[ "$want_cuda12" == "true" && "$major" -ge 13 ]]; then
            continue
        fi
        configure_cuda_env_from_nvcc "$candidate"
        printf '%s\n' "$candidate"
        return 0
    done

    local bootstrapped=""
    if bootstrapped="$(bootstrap_private_cuda_rattler 2>/dev/null)"; then
        configure_cuda_env_from_nvcc "$bootstrapped"
        printf '%s\n' "$bootstrapped"
        return 0
    fi

    install_cuda_toolkit "$want_cuda12"
    [[ -x "$CUDA_PRIVATE_PREFIX/bin/nvcc" ]] || {
        tui_error "Failed to provision a usable nvcc"
        return 1
    }
    configure_cuda_env_from_nvcc "$CUDA_PRIVATE_PREFIX/bin/nvcc"
    printf '%s\n' "$CUDA_PRIVATE_PREFIX/bin/nvcc"
}

ensure_build_deps() {
    local os="$1"
    tui_step "Checking build dependencies…"
    case "$os" in
        linux)
            local pkgs=()
            command -v git    &>/dev/null || pkgs+=(git)
            command -v cmake  &>/dev/null || pkgs+=(cmake)
            command -v make   &>/dev/null || pkgs+=(make)
            command -v gcc    &>/dev/null || pkgs+=(gcc g++)
            if (( ${#pkgs[@]} > 0 )); then
                if command -v apt-get &>/dev/null; then
                    sudo apt-get install -y "${pkgs[@]}" build-essential 2>/dev/null
                elif command -v dnf &>/dev/null; then
                    sudo dnf install -y "${pkgs[@]}" gcc-c++ 2>/dev/null
                else
                    tui_warn "Cannot auto-install: ${pkgs[*]}. Install manually."
                fi
            fi
            ;;
        mac)
            xcode-select --install 2>/dev/null || true
            command -v cmake &>/dev/null || brew install cmake 2>/dev/null || true
            ;;
    esac
    tui_success "Build dependencies ready ✓"
}

clone_or_update_llamacpp() {
    if [[ -d "$LLAMACPP_DIR/.git" ]]; then
        tui_step "Updating llama.cpp…"
        git -C "$LLAMACPP_DIR" pull --ff-only 2>/dev/null || true
    else
        tui_step "Cloning llama.cpp…"
        git clone --depth=1 "$LLAMACPP_REPO" "$LLAMACPP_DIR"
    fi
    tui_success "llama.cpp source ready ✓"
}

# ── Build strategies ───────────────────────────────────────────────────────────
# Arguments:
#   $1  cuda_arch   — value for -DCMAKE_CUDA_ARCHITECTURES (e.g. "70", "native")
#   $2  cuda_root   — path to CUDA toolkit prefix (may be empty to rely on PATH)
build_with_cuda() {
    local cuda_arch="${1:-native}"
    local cuda_root="${2:-}"

    tui_step "Building llama.cpp with CUDA support (arch=$cuda_arch)…"

    local cmake_extra_args=()

    # Use GGML_CUDA (LLAMA_CUDA is deprecated as of recent llama.cpp versions).
    cmake_extra_args+=(-DGGML_CUDA=ON)
    cmake_extra_args+=(-DCMAKE_CUDA_ARCHITECTURES="$cuda_arch")

    if [[ -n "$cuda_root" ]]; then
        cmake_extra_args+=(-DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc")
        cmake_extra_args+=(-DCUDAToolkit_ROOT="$cuda_root")
    fi

    # Extra performance flags for legacy architectures (sm_70 Volta and
    # sm_61/sm_60 Pascal).  GGML_CUDA_FA_ALL_QUANTS pre-compiles all
    # quantisation variants for flash-attention (a build-coverage flag);
    # GGML_CUDA_PEER_MAX_BATCH_SIZE improves multi-GPU throughput on NVLink
    # systems.  Both are recommended by the V100/P40 deployment guide.
    if [[ "$cuda_arch" != "native" ]]; then
        cmake_extra_args+=(-DGGML_CUDA_FA_ALL_QUANTS=ON)
        cmake_extra_args+=(-DGGML_CUDA_PEER_MAX_BATCH_SIZE=8192)
    fi

    cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" \
        "${cmake_extra_args[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$LLAMA_INSTALL_PREFIX" 2>&1
    cmake --build "$LLAMACPP_DIR/build" --config Release -j"$(nproc)" 2>&1
}

build_with_metal() {
    tui_step "Building llama.cpp with Apple Metal support…"
    cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" \
        -DLLAMA_METAL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$LLAMA_INSTALL_PREFIX" 2>&1
    cmake --build "$LLAMACPP_DIR/build" --config Release -j"$(sysctl -n hw.logicalcpu)" 2>&1
}

build_with_rocm() {
    tui_step "Building llama.cpp with ROCm/HIP support…"
    cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" \
        -DLLAMA_HIPBLAS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$LLAMA_INSTALL_PREFIX" 2>&1
    cmake --build "$LLAMACPP_DIR/build" --config Release -j"$(nproc)" 2>&1
}

build_cpu_optimised() {
    local os="$1"
    local cpu_flags=""

    tui_step "Building llama.cpp with CPU optimisations…"

    case "$os" in
        linux)
            # Detect best CPU extension set
            if grep -q "avx512" /proc/cpuinfo 2>/dev/null; then
                cpu_flags="-DLLAMA_AVX512=ON -DLLAMA_AVX2=ON -DLLAMA_AVX=ON"
            elif grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
                cpu_flags="-DLLAMA_AVX2=ON -DLLAMA_AVX=ON"
            elif grep -q "avx" /proc/cpuinfo 2>/dev/null; then
                cpu_flags="-DLLAMA_AVX=ON"
            fi
            local nthreads
            nthreads=$(nproc)
            ;;
        mac)
            cpu_flags="-DLLAMA_NATIVE=ON"
            local nthreads
            nthreads=$(sysctl -n hw.logicalcpu)
            ;;
        *)
            local nthreads=2
            ;;
    esac

    # shellcheck disable=SC2086
    cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" \
        $cpu_flags \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$LLAMA_INSTALL_PREFIX" 2>&1
    cmake --build "$LLAMACPP_DIR/build" --config Release -j"$nthreads" 2>&1
}

# ── Main build entry ───────────────────────────────────────────────────────────
build_llamacpp() {
    eval "$(detect_gpu)"
    local os
    os=$(detect_os)

    ensure_build_deps "$os"
    clone_or_update_llamacpp

    local build_success=false

    if [[ "$GPU_VENDOR" == "nvidia" ]]; then
        local nvcc_path=""
        nvcc_path="$(ensure_nvcc 2>/dev/null || true)"
        if [[ -n "$nvcc_path" && -x "$nvcc_path" ]]; then
            local cuda_arch
            cuda_arch=$(cuda_arch_for_gpu)
            local cuda_root="${CUDA_ROOT:-$(cd "$(dirname "$nvcc_path")/.." && pwd -P)}"
            build_with_cuda "$cuda_arch" "$cuda_root" && build_success=true || true
        else
            tui_warn "No usable nvcc found — will attempt CPU-only build."
        fi
    fi

    if [[ "$GPU_VENDOR" == "apple" ]] && ! $build_success; then
        build_with_metal && build_success=true || true
    fi

    if [[ "$GPU_VENDOR" == "amd" ]] && ! $build_success; then
        if command -v hipcc &>/dev/null; then
            build_with_rocm && build_success=true || true
        else
            tui_warn "AMD GPU found but hipcc not in PATH — skipping ROCm build."
        fi
    fi

    if ! $build_success; then
        build_cpu_optimised "$os" && build_success=true
    fi

    if $build_success; then
        # Add bin to PATH for this session
        export PATH="$LLAMACPP_BIN:$PATH"
        tui_success "llama.cpp built at $LLAMACPP_BIN ✓"
        echo ""
        echo "Add to PATH permanently:"
        echo "  export PATH=\"$LLAMACPP_BIN:\$PATH\""
    else
        tui_error "llama.cpp build failed. See output above."
        return 1
    fi
}

run_benchmark() {
    local model_path="${1:?model_path required}"
    local n_tokens="${2:-128}"
    local threads="${3:-$(nproc 2>/dev/null || echo 4)}"

    tui_step "Running llama.cpp benchmark on $model_path…"
    if ! command -v llama-bench &>/dev/null && [[ -f "$LLAMACPP_BIN/llama-bench" ]]; then
        export PATH="$LLAMACPP_BIN:$PATH"
    fi

    llama-bench -m "$model_path" -n "$n_tokens" -t "$threads" 2>&1
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-build}" in
        build)     build_llamacpp ;;
        bench)     run_benchmark "${2:?}" "${3:-128}" "${4:-}" ;;
        *)         tui_error "Usage: $0 {build|bench <model_path>}" ; exit 1 ;;
    esac
fi
