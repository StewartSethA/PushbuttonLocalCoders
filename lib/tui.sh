#!/usr/bin/env bash
# tui.sh — Minimal TUI helpers: coloured output, progress bars, spinners,
#           and a live GPU/CPU monitor.

set -euo pipefail

# ── Colour codes ───────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    GREEN='\033[0;32m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'
else
    RED='' YELLOW='' GREEN='' CYAN='' BOLD='' DIM='' RESET=''
fi

tui_step()    { echo -e "${CYAN}▶ ${1}${RESET}"; }
tui_success() { echo -e "${GREEN}✓ ${1}${RESET}"; }
tui_warn()    { echo -e "${YELLOW}⚠ ${1}${RESET}"; }
tui_error()   { echo -e "${RED}✗ ${1}${RESET}" >&2; }
tui_info()    { echo -e "${DIM}  ${1}${RESET}"; }
tui_header()  { echo -e "\n${BOLD}${CYAN}══ ${1} ══${RESET}\n"; }

# ── Progress bar ───────────────────────────────────────────────────────────────
# tui_progress_bar <current> <total> [label]
tui_progress_bar() {
    local current="${1:-0}"
    local total="${2:-100}"
    local label="${3:-}"
    local width=40

    (( total == 0 )) && total=1
    local filled=$(( current * width / total ))
    local empty=$(( width - filled ))

    local bar
    bar="["
    bar+=$(printf '%0.s█' $(seq 1 $filled) 2>/dev/null || printf '%.0s#' $(seq 1 $filled))
    bar+=$(printf '%0.s░' $(seq 1 $empty)  2>/dev/null || printf '%.0s-' $(seq 1 $empty))
    bar+="]"

    local pct=$(( current * 100 / total ))
    printf "\r${CYAN}%s${RESET} %3d%%  %s" "$bar" "$pct" "$label"
    [[ $current -ge $total ]] && echo ""
}

# ── Spinner ────────────────────────────────────────────────────────────────────
# Usage: tui_spinner_start "label"
# Call tui_spinner_stop to clean up
_SPINNER_PID=""
tui_spinner_start() {
    local label="${1:-Working…}"
    local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    (
        local i=0
        while true; do
            printf "\r${CYAN}%s${RESET} %s  " "${frames[$((i % ${#frames[@]}))]}" "$label"
            sleep 0.1
            (( i++ ))
        done
    ) &
    _SPINNER_PID=$!
    disown "$_SPINNER_PID" 2>/dev/null || true
}

tui_spinner_stop() {
    if [[ -n "$_SPINNER_PID" ]] && kill -0 "$_SPINNER_PID" 2>/dev/null; then
        kill "$_SPINNER_PID" 2>/dev/null || true
        wait "$_SPINNER_PID" 2>/dev/null || true
    fi
    _SPINNER_PID=""
    printf "\r\033[K"  # clear line
}

# ── GPU/CPU monitor ────────────────────────────────────────────────────────────
# Prints a live one-shot snapshot; call in a loop or watch for a live view.
monitor_snapshot() {
    local os
    os=$(uname -s | tr '[:upper:]' '[:lower:]')

    echo -e "\n${BOLD}${CYAN}── Hardware Monitor ─────────────────────────────────${RESET}"

    # CPU utilisation
    local cpu_pct="N/A"
    case "$os" in
        linux)
            cpu_pct=$(top -bn1 | grep "^%Cpu\|^Cpu" | awk '{print 100 - $8"%"}' 2>/dev/null || echo "N/A")
            ;;
        darwin)
            cpu_pct=$(top -l1 -n0 2>/dev/null | grep "CPU usage" | awk '{print $3}' || echo "N/A")
            ;;
    esac
    echo -e "  ${CYAN}CPU util :${RESET} $cpu_pct"

    # RAM utilisation
    local ram_used_mb=0 ram_total_mb=0
    case "$os" in
        linux)
            eval "$(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print "ram_total_mb=" t/1024 "\nram_used_mb=" (t-a)/1024}' /proc/meminfo 2>/dev/null || echo "ram_total_mb=0; ram_used_mb=0")"
            ;;
        darwin)
            ram_total_mb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576 ))
            ram_used_mb=$(( ram_total_mb - $(vm_stat 2>/dev/null | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} END{printf "%d", (f+i)*4096/1048576}' || echo 0) ))
            ;;
    esac
    echo -e "  ${CYAN}RAM      :${RESET} ${ram_used_mb} MB / ${ram_total_mb} MB used"

    # NVIDIA GPU
    if command -v nvidia-smi &>/dev/null; then
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
                              --format=csv,noheader,nounits 2>/dev/null || true)
        if [[ -n "$gpu_info" ]]; then
            echo -e "  ${CYAN}GPU(s)   :${RESET}"
            local i=0
            while IFS=',' read -r gname gutil gmem_used gmem_total; do
                gname=$(echo "$gname" | xargs)
                gutil=$(echo "$gutil" | xargs)
                gmem_used=$(echo "$gmem_used" | xargs)
                gmem_total=$(echo "$gmem_total" | xargs)
                printf "    [GPU %d] %-30s  util: %3s%%  VRAM: %s / %s MB\n" \
                    "$i" "$gname" "$gutil" "$gmem_used" "$gmem_total"
                (( i++ ))
            done <<< "$gpu_info"
        fi
    elif command -v rocm-smi &>/dev/null; then
        echo -e "  ${CYAN}GPU(s)   :${RESET} (AMD — run \`rocm-smi\` for details)"
    else
        echo -e "  ${CYAN}GPU(s)   :${RESET} None detected / driver not installed"
    fi

    echo -e "${BOLD}${CYAN}─────────────────────────────────────────────────────${RESET}\n"
}

# Live monitor loop (Ctrl-C to quit)
monitor_live() {
    local interval="${1:-2}"
    tui_info "Press Ctrl-C to exit monitor…"
    while true; do
        clear
        monitor_snapshot
        sleep "$interval"
    done
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-snapshot}" in
        live)     monitor_live "${2:-2}" ;;
        snapshot) monitor_snapshot ;;
        *)        monitor_snapshot ;;
    esac
fi
