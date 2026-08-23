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
build_with_cuda() {
    tui_step "Building llama.cpp with CUDA support…"
    cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" \
        -DLLAMA_CUDA=ON \
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
        if command -v nvcc &>/dev/null; then
            build_with_cuda && build_success=true || true
        else
            tui_warn "NVIDIA GPU found but nvcc not in PATH — skipping CUDA build."
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
