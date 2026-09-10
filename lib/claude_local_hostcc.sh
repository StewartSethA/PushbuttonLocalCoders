#!/usr/bin/env bash
# Select/provision a host compiler compatible with CUDA 12.8/12.9.
# Intended to be sourced by install-claude-local.sh and the installed shim.

claude_local_hostcc_major() {
    local cc="$1" v
    [[ -x "$cc" ]] || command -v "$cc" >/dev/null 2>&1 || return 1
    v="$($cc -dumpfullversion -dumpversion 2>/dev/null | head -1)" || return 1
    printf '%s\n' "${v%%.*}"
}

claude_local_bootstrap_micromamba() {
    local state="$1" mm="$1/bin/micromamba" platform
    if [[ -x "$mm" ]]; then printf '%s\n' "$mm"; return 0; fi
    case "$(uname -m)" in
        x86_64|amd64) platform=linux-64 ;;
        aarch64|arm64) platform=linux-aarch64 ;;
        *) echo "[claude-local] unsupported Linux architecture for private host compiler: $(uname -m)" >&2; return 1 ;;
    esac
    mkdir -p "$state/bin"
    echo "[claude-local] Bootstrapping micromamba for a CUDA-compatible host compiler..." >&2
    curl -fsSL "https://micro.mamba.pm/api/micromamba/$platform/latest" \
      | tar -xj -C "$state/bin" --strip-components=1 bin/micromamba
    chmod +x "$mm"
    printf '%s\n' "$mm"
}

claude_local_prepare_hostcc() {
    local state="${CLAUDE_LOCAL_STATE:-$HOME/.local/share/pushbutton/claude-local}"
    local cxx cc major hostprefix mm

    # Honor an explicitly selected compatible compiler first.
    cc="${CC:-$(command -v gcc 2>/dev/null || true)}"
    cxx="${CXX:-$(command -v g++ 2>/dev/null || true)}"
    if [[ -n "$cc" && -n "$cxx" ]]; then
        major="$(claude_local_hostcc_major "$cc" 2>/dev/null || true)"
        if [[ "$major" =~ ^[0-9]+$ ]] && (( major >= 6 && major <= 14 )); then
            export CC="$cc" CXX="$cxx" NVCC_CCBIN="${NVCC_CCBIN:-$cc}" CUDAHOSTCXX="${CUDAHOSTCXX:-$cxx}"
            return 0
        fi
    fi

    # Prefer a distro-provided compatibility compiler without touching alternatives.
    if command -v gcc-14 >/dev/null 2>&1 && command -v g++-14 >/dev/null 2>&1; then
        export CC="$(command -v gcc-14)" CXX="$(command -v g++-14)"
        export NVCC_CCBIN="$CC" CUDAHOSTCXX="$CXX"
        echo "[claude-local] Using GCC 14 host compiler for CUDA: $CC" >&2
        return 0
    fi

    # New distros such as Ubuntu 26.04 can default to GCC 15 while CUDA 12.8/12.9
    # support GCC <=14. Keep this private and independent of the host's APT state.
    hostprefix="$state/gcc-14"
    if [[ ! -x "$hostprefix/bin/gcc" || ! -x "$hostprefix/bin/g++" ]]; then
        mm="$(claude_local_bootstrap_micromamba "$state")" || return 1
        echo "[claude-local] Installing private GCC 14 host compiler (system compiler unchanged)..." >&2
        MAMBA_ROOT_PREFIX="$state/micromamba-root" "$mm" create -y -p "$hostprefix" \
          -c conda-forge "gcc=14" "gxx=14"
    fi

    [[ -x "$hostprefix/bin/gcc" && -x "$hostprefix/bin/g++" ]] || {
        echo "[claude-local] private GCC 14 installation did not provide gcc/g++" >&2
        return 1
    }
    export CC="$hostprefix/bin/gcc" CXX="$hostprefix/bin/g++"
    export NVCC_CCBIN="$CC" CUDAHOSTCXX="$CXX"
    echo "[claude-local] Using private GCC 14 host compiler: $CC" >&2
}
