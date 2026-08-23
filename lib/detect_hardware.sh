#!/usr/bin/env bash
# detect_hardware.sh — Detect GPU, CPU, VRAM and system RAM
# Outputs shell variables that other scripts can source.

set -euo pipefail

# ── OS detection ──────────────────────────────────────────────────────────────
detect_os() {
    case "$(uname -s)" in
        Linux*)  echo "linux"  ;;
        Darwin*) echo "mac"    ;;
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        *)       echo "unknown" ;;
    esac
}

# ── CPU detection ─────────────────────────────────────────────────────────────
detect_cpu() {
    local os
    os=$(detect_os)
    local cpu_model cpu_cores cpu_threads

    case "$os" in
        linux)
            cpu_model=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "Unknown CPU")
            cpu_cores=$(grep -c "^processor" /proc/cpuinfo 2>/dev/null || echo 1)
            cpu_threads=$cpu_cores
            ;;
        mac)
            cpu_model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown CPU")
            cpu_cores=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 1)
            cpu_threads=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 1)
            ;;
        *)
            cpu_model="Unknown CPU"
            cpu_cores=1
            cpu_threads=1
            ;;
    esac

    echo "CPU_MODEL=\"$cpu_model\""
    echo "CPU_CORES=$cpu_cores"
    echo "CPU_THREADS=$cpu_threads"
}

detect_cpu_capabilities() {
    local os
    os=$(detect_os)
    local caps=()

    if [[ "$os" == "linux" ]] && [[ -r /proc/cpuinfo ]]; then
        grep -qi "avx512" /proc/cpuinfo 2>/dev/null && caps+=("avx512")
        grep -qi "avx2" /proc/cpuinfo 2>/dev/null && caps+=("avx2")
        grep -qi "avx" /proc/cpuinfo 2>/dev/null && caps+=("avx")
        grep -qi "sse4_2" /proc/cpuinfo 2>/dev/null && caps+=("sse4_2")
    elif [[ "$os" == "mac" ]]; then
        caps+=("neon")
        if [[ "$(uname -m)" == "arm64" ]]; then
            caps+=("apple-silicon")
        fi
    fi

    if (( ${#caps[@]} == 0 )); then
        caps+=("baseline")
    fi

    echo "CPU_CAPABILITIES=\"$(IFS=,; echo "${caps[*]}")\""
}

# ── RAM detection ─────────────────────────────────────────────────────────────
detect_ram() {
    local os
    os=$(detect_os)
    local ram_mb=0

    case "$os" in
        linux)
            ram_mb=$(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)
            ;;
        mac)
            ram_mb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576 ))
            ;;
    esac

    echo "RAM_MB=$ram_mb"
    echo "RAM_GB=$(( ram_mb / 1024 ))"
}

# ── GPU detection ─────────────────────────────────────────────────────────────
detect_gpu() {
    local gpu_vendor="none"
    local gpu_model="None"
    local vram_mb=0
    local gpu_count=0

    # NVIDIA via nvidia-smi
    if command -v nvidia-smi &>/dev/null; then
        local smi_out
        smi_out=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
        if [[ -n "$smi_out" ]]; then
            gpu_vendor="nvidia"
            gpu_count=$(echo "$smi_out" | wc -l)
            gpu_model=$(echo "$smi_out" | head -1 | cut -d',' -f1 | xargs)
            # Sum VRAM across all GPUs
            vram_mb=$(echo "$smi_out" | awk -F',' '{sum += $2} END {printf "%d", sum}')
        fi
    fi

    # AMD via rocm-smi
    if [[ "$gpu_vendor" == "none" ]] && command -v rocm-smi &>/dev/null; then
        local rocm_out
        rocm_out=$(rocm-smi --showproductname --showmeminfo vram 2>/dev/null || true)
        if [[ -n "$rocm_out" ]]; then
            gpu_vendor="amd"
            gpu_count=1
            gpu_model=$(echo "$rocm_out" | grep -i "card\|GPU" | head -1 | xargs || echo "AMD GPU")
            # rocm-smi VRAM total in bytes → MB
            local vram_bytes
            vram_bytes=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -i "total" | awk '{print $NF}' | head -1 || echo 0)
            vram_mb=$(( vram_bytes / 1048576 ))
        fi
    fi

    # Apple Metal (Mac)
    if [[ "$gpu_vendor" == "none" ]] && [[ "$(detect_os)" == "mac" ]]; then
        if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
            gpu_vendor="apple"
            gpu_count=1
            gpu_model=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | cut -d: -f2 | xargs || echo "Apple GPU")
            # On Apple Silicon, GPU shares system RAM; report total RAM as VRAM
            vram_mb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576 ))
        fi
    fi

    echo "GPU_VENDOR=\"$gpu_vendor\""
    echo "GPU_MODEL=\"$gpu_model\""
    echo "GPU_COUNT=$gpu_count"
    echo "VRAM_MB=$vram_mb"
    echo "VRAM_GB=$(( vram_mb / 1024 ))"
}

detect_apple_silicon() {
    if [[ "$(detect_os)" == "mac" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        echo "APPLE_SILICON=1"
    else
        echo "APPLE_SILICON=0"
    fi
}

detect_cuda() {
    local nvcc_path=""
    local version="0.0"
    local provider="none"

    if [[ -n "${CUDA_HOME:-}" ]] && [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
        nvcc_path="${CUDA_HOME}/bin/nvcc"
        provider="env"
    elif [[ -n "${CUDA_PATH:-}" ]] && [[ -x "${CUDA_PATH}/bin/nvcc" ]]; then
        nvcc_path="${CUDA_PATH}/bin/nvcc"
        provider="env"
    elif command -v nvcc &>/dev/null; then
        nvcc_path="$(command -v nvcc)"
        provider="system"
    fi

    if [[ -n "$nvcc_path" ]]; then
        version="$("$nvcc_path" --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | tail -n1)"
        [[ -z "$version" ]] && version="0.0"
        echo "CUDA_AVAILABLE=1"
    else
        echo "CUDA_AVAILABLE=0"
    fi

    echo "CUDA_VERSION=\"$version\""
    echo "CUDA_NVCC_PATH=\"$nvcc_path\""
    echo "CUDA_PROVIDER=\"$provider\""
}

# ── MCDRAM detection (Intel Xeon Phi / HBM) ──────────────────────────────────
detect_mcdram() {
    local mcdram_mb=0
    if command -v numactl &>/dev/null; then
        # HBM typically shows as a high-bandwidth NUMA node
        local hbm_kb
        hbm_kb=$(numactl --hardware 2>/dev/null | awk '/available:/{nodes=$2} /node [0-9]+ size:/{size=$4} /node [0-9]+ free:/{free=$4; if (size > 0) sum += size} END {print sum*1024}' || echo 0)
        mcdram_mb=$(( hbm_kb / 1024 ))
    fi
    echo "MCDRAM_MB=$mcdram_mb"
}

# ── Summarise usable inference memory ─────────────────────────────────────────
# Returns the "best" contiguous memory pool available for model inference:
#   VRAM > MCDRAM > System RAM (leaving 4 GB headroom for the OS)
detect_inference_memory() {
    eval "$(detect_gpu)"
    eval "$(detect_ram)"
    eval "$(detect_mcdram)"

    local inference_mb
    local memory_type

    if (( VRAM_MB > 0 )); then
        inference_mb=$VRAM_MB
        memory_type="vram"
    elif (( MCDRAM_MB > 0 )); then
        inference_mb=$MCDRAM_MB
        memory_type="mcdram"
    else
        # Leave 4 GB headroom for OS
        inference_mb=$(( RAM_MB - 4096 ))
        [[ $inference_mb -lt 0 ]] && inference_mb=0
        memory_type="ram"
    fi

    echo "INFERENCE_MB=$inference_mb"
    echo "INFERENCE_GB=$(( inference_mb / 1024 ))"
    echo "MEMORY_TYPE=\"$memory_type\""
}

# ── Full hardware report ───────────────────────────────────────────────────────
detect_all() {
    local os
    os=$(detect_os)
    echo "OS=\"$os\""
    detect_cpu
    detect_cpu_capabilities
    detect_ram
    detect_gpu
    detect_apple_silicon
    detect_cuda
    detect_mcdram
    detect_inference_memory
}

detect_hardware_profile() {
    eval "$(detect_all)"
    echo "HW_PROFILE_OS=\"$OS\""
    echo "HW_PROFILE_GPU_VENDOR=\"$GPU_VENDOR\""
    echo "HW_PROFILE_GPU_MODEL=\"$GPU_MODEL\""
    echo "HW_PROFILE_GPU_COUNT=$GPU_COUNT"
    echo "HW_PROFILE_VRAM_GB=$VRAM_GB"
    echo "HW_PROFILE_RAM_GB=$RAM_GB"
    echo "HW_PROFILE_INFERENCE_GB=$INFERENCE_GB"
    echo "HW_PROFILE_MEMORY_TYPE=\"$MEMORY_TYPE\""
    echo "HW_PROFILE_CUDA_AVAILABLE=$CUDA_AVAILABLE"
    echo "HW_PROFILE_CUDA_VERSION=\"$CUDA_VERSION\""
    echo "HW_PROFILE_CUDA_PROVIDER=\"$CUDA_PROVIDER\""
    echo "HW_PROFILE_APPLE_SILICON=$APPLE_SILICON"
    echo "HW_PROFILE_CPU_CAPABILITIES=\"$CPU_CAPABILITIES\""
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    detect_hardware_profile
fi
