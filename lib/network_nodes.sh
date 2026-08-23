#!/usr/bin/env bash
# network_nodes.sh — Discover and monitor networked boxes for GPU/CPU utilisation.
# Reads an IP list from ~/.config/pushbutton/nodes.txt (one IP per line).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tui.sh"

NODES_FILE="${NODES_FILE:-$HOME/.config/pushbutton/nodes.txt}"
SSH_USER="${SSH_USER:-$USER}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
SSH_TIMEOUT="${SSH_TIMEOUT:-5}"

# ── Node file management ───────────────────────────────────────────────────────
ensure_nodes_file() {
    if [[ ! -f "$NODES_FILE" ]]; then
        mkdir -p "$(dirname "$NODES_FILE")"
        cat > "$NODES_FILE" <<'EOF'
# PushbuttonLocalCoders — Remote Node List
# One IP address or hostname per line.  Lines starting with # are ignored.
# Example:
#   192.168.1.10
#   192.168.1.11
#   my-gpu-box.local
EOF
        tui_info "Node list created: $NODES_FILE"
    fi
}

read_nodes() {
    ensure_nodes_file
    grep -v '^\s*#' "$NODES_FILE" | grep -v '^\s*$' || true
}

add_node() {
    local ip="$1"
    ensure_nodes_file
    if grep -qF "$ip" "$NODES_FILE" 2>/dev/null; then
        tui_warn "Node $ip already in list."
    else
        echo "$ip" >> "$NODES_FILE"
        tui_success "Added $ip to $NODES_FILE"
    fi
}

remove_node() {
    local ip="$1"
    ensure_nodes_file
    if grep -qF "$ip" "$NODES_FILE" 2>/dev/null; then
        local tmp
        tmp=$(mktemp)
        grep -vF "$ip" "$NODES_FILE" > "$tmp"
        mv "$tmp" "$NODES_FILE"
        tui_success "Removed $ip from $NODES_FILE"
    else
        tui_warn "Node $ip not found in list."
    fi
}

# ── Remote query ───────────────────────────────────────────────────────────────
query_node() {
    local ip="$1"

    local remote_script
    remote_script=$(cat <<'REMOTE'
#!/usr/bin/env bash
cpu_pct=$(top -bn1 2>/dev/null | grep -E "^%?Cpu" | awk '{print 100-$8}' || echo "N/A")
ram_info=$(free -m 2>/dev/null | awk '/Mem:/{printf "%d/%d MB", $3, $2}' || echo "N/A")
if command -v nvidia-smi &>/dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
                          --format=csv,noheader,nounits 2>/dev/null | \
               awk -F',' '{printf "%s util=%s%% VRAM=%s/%sMB", $1, $2, $3, $4}' || echo "none")
else
    gpu_info="none"
fi
printf "cpu=%s ram=%s gpu=%s\n" "$cpu_pct" "$ram_info" "$gpu_info"
REMOTE
)

    local result
    result=$(ssh -o ConnectTimeout="$SSH_TIMEOUT" \
                 -o StrictHostKeyChecking=no \
                 -o BatchMode=yes \
                 -i "$SSH_KEY" \
                 "${SSH_USER}@${ip}" \
                 "bash -s" <<< "$remote_script" 2>/dev/null) || result="UNREACHABLE"

    echo "$ip|$result"
}

# ── Monitor all nodes ──────────────────────────────────────────────────────────
monitor_nodes_snapshot() {
    local nodes
    mapfile -t nodes < <(read_nodes)

    if (( ${#nodes[@]} == 0 )); then
        tui_warn "No nodes configured. Add entries to $NODES_FILE"
        return 0
    fi

    tui_header "Network Node Monitor"
    printf "%-22s  %-12s  %-20s  %s\n" "Node" "CPU%" "RAM" "GPU"
    printf '%0.s─' {1..90}; echo ""

    for ip in "${nodes[@]}"; do
        local info
        info=$(query_node "$ip")
        local node_ip="${info%%|*}"
        local data="${info#*|}"
        if [[ "$data" == "UNREACHABLE" ]]; then
            printf "%-22s  %-12s\n" "$node_ip" "UNREACHABLE"
        else
            local cpu ram gpu
            cpu=$(echo "$data" | grep -oP '(?<=cpu=)\S+' || echo "N/A")
            ram=$(echo "$data" | grep -oP '(?<=ram=)[^\s]+\s+[^\s]+\s+[^\s]+' || echo "N/A")
            gpu=$(echo "$data" | grep -oP '(?<=gpu=).*' || echo "N/A")
            printf "%-22s  %-12s  %-20s  %s\n" "$node_ip" "$cpu%" "$ram" "$gpu"
        fi
    done
    echo ""
}

monitor_nodes_live() {
    local interval="${1:-5}"
    tui_info "Monitoring nodes every ${interval}s — Ctrl-C to stop"
    while true; do
        clear
        monitor_nodes_snapshot
        sleep "$interval"
    done
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-snapshot}" in
        add)       add_node "${2:?IP required}"    ;;
        remove)    remove_node "${2:?IP required}" ;;
        list)      read_nodes                       ;;
        snapshot)  monitor_nodes_snapshot           ;;
        live)      monitor_nodes_live "${2:-5}"     ;;
        *)         monitor_nodes_snapshot           ;;
    esac
fi
