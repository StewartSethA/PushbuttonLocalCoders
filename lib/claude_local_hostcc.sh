#!/usr/bin/env bash
# Select/provision a modern host compiler compatible with CUDA 12.8/12.9.
# Intended to be sourced by install-claude-local.sh and runtime launchers.
#
# CUDA accepts some much older GCC versions, but current llama.cpp dependencies
# (notably bundled BoringSSL) can exercise compiler bugs in old toolchains.  Use
# GCC 11+ for the entire C/C++ build, while keeping <=14 for CUDA 12.x support.

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
    curl -fL --show-error "https://micro.mamba.pm/api/micromamba/$platform/latest" \
      | tar -xj -C "$state/bin" --strip-components=1 bin/micromamba
    chmod +x "$mm"
    printf '%s\n' "$mm"
}

claude_local_use_hostcc() {
    local cc="$1" cxx="$2" major
    major="$(claude_local_hostcc_major "$cc" 2>/dev/null || true)"
    [[ "$major" =~ ^[0-9]+$ ]] || return 1
    (( major >= 11 && major <= 14 )) || return 1
    export CC="$cc" CXX="$cxx" NVCC_CCBIN="$cc" CUDAHOSTCXX="$cxx"
    return 0
}

claude_local_prepare_hostcc() {
    local state="${CLAUDE_LOCAL_STATE:-$HOME/.local/share/pushbutton/claude-local}"
    local cc cxx major hostprefix mm

    # Respect an explicitly selected complete C/C++ toolchain when it is both
    # modern enough for current dependencies and accepted by CUDA 12.x.
    cc="${CC:-$(command -v gcc 2>/dev/null || true)}"
    cxx="${CXX:-$(command -v g++ 2>/dev/null || true)}"
    if [[ -n "$cc" && -n "$cxx" ]] && claude_local_use_hostcc "$cc" "$cxx"; then
        return 0
    fi

    if [[ -n "$cc" ]]; then
        major="$(claude_local_hostcc_major "$cc" 2>/dev/null || true)"
        if [[ "$major" =~ ^[0-9]+$ ]] && (( major < 11 )); then
            echo "[claude-local] GCC $major is too old for a reliable current llama.cpp/BoringSSL build; selecting GCC 14 privately." >&2
        elif [[ "$major" =~ ^[0-9]+$ ]] && (( major > 14 )); then
            echo "[claude-local] GCC $major is newer than the CUDA 12.x supported host range; selecting GCC 14 privately." >&2
        fi
    fi

    # Prefer a distro-provided GCC 14 without changing alternatives or system
    # defaults.  Unlike the old logic, use it for ordinary C/C++ too, not only nvcc.
    if command -v gcc-14 >/dev/null 2>&1 && command -v g++-14 >/dev/null 2>&1; then
        cc="$(command -v gcc-14)"; cxx="$(command -v g++-14)"
        claude_local_use_hostcc "$cc" "$cxx" || return 1
        echo "[claude-local] Using GCC 14 for llama.cpp and CUDA host compilation: $CC" >&2
        return 0
    fi

    # Keep a private compiler independent of the host's APT state.
    hostprefix="$state/gcc-14"
    if [[ ! -x "$hostprefix/bin/gcc" || ! -x "$hostprefix/bin/g++" ]]; then
        mm="$(claude_local_bootstrap_micromamba "$state")" || return 1
        echo "[claude-local] Installing private GCC 14 compiler (system compiler unchanged)..." >&2
        MAMBA_ROOT_PREFIX="$state/micromamba-root" "$mm" create -y -p "$hostprefix" \
          -c conda-forge "gcc=14" "gxx=14"
    fi

    [[ -x "$hostprefix/bin/gcc" && -x "$hostprefix/bin/g++" ]] || {
        echo "[claude-local] private GCC 14 installation did not provide gcc/g++" >&2
        return 1
    }
    claude_local_use_hostcc "$hostprefix/bin/gcc" "$hostprefix/bin/g++" || return 1
    echo "[claude-local] Using private GCC 14 for llama.cpp and CUDA host compilation: $CC" >&2
}
