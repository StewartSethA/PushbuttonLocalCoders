#!/usr/bin/env bash
# Qwen3.6 RTX 3090 remaining-tests harness.
# Revision: 2026-07-30-r15-claude-tools
# Strict-local: every script-controlled persistent/cache/temp/Docker-data artifact stays under $(pwd -P).
# This is a continuation harness: it reuses the llama.cpp build and model caches produced by r12/r11.
# No set -e, no pipefail, no broad kill/pkill/killall.

SCRIPT_REVISION="2026-07-30-r15-claude-tools"
ROOT="$(pwd -P)"
MODEL_SIZE="${MODEL_SIZE:-35b}"
TASK="${TASK:-help}"
USE_SYSTEM_DOCKER="${USE_SYSTEM_DOCKER:-0}"
ALLOW_LOW_SPACE="${ALLOW_LOW_SPACE:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"

# Physical GPU controls.
SINGLE_GPU_ID="${SINGLE_GPU_ID:-0}"
COMPARE_GPU_ID="${COMPARE_GPU_ID:-1}"
GPU_PAIR="${GPU_PAIR:-0,1}"

# Existing local images/build.
LLAMA_IMAGE="${LLAMA_IMAGE:-qwen36-llama-buildenv:sm86-r7-https}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.26.0-cu129-ubuntu2404}"
VLLM_PORT="${VLLM_PORT:-18081}"
VLLM_ALIAS="${VLLM_ALIAS:-qwen3.6-${MODEL_SIZE}-vllm}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
NATIVE_CTX="${NATIVE_CTX:-262144}"
TARGET_DEPTH="${TARGET_DEPTH:-253952}"
REPS="${REPS:-2}"
PP_TOKENS="${PP_TOKENS:-4096}"
TG_TOKENS="${TG_TOKENS:-256}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 8)}"
FORCE="${FORCE:-0}"

# llama.cpp remaining tests.
LLAMA_TEST_QUANT="${LLAMA_TEST_QUANT:-UD-IQ4_NL_XL}"
LLAMA_COMPARE_DEPTHS="${LLAMA_COMPARE_DEPTHS:-0,8192,32768,65536,131072,196608,253952}"
LLAMA_BATCH_DEPTHS="${LLAMA_BATCH_DEPTHS:-0,65536,253952}"
LLAMA_EXT_DEPTHS="${LLAMA_EXT_DEPTHS:-0,8192,32768,65536,131072,196608,253952}"
LLAMA_KV_LIST="${LLAMA_KV_LIST:-q8_0/q8_0,q8_0/q4_0,q5_1/q5_1,iq4_nl/iq4_nl,q4_1/q4_1,q4_0/q4_0,f16/f16}"
LLAMA_BATCH_LIST="${LLAMA_BATCH_LIST:-2048/512,4096/512,4096/1024,8192/1024}"
LLAMA_FULL_MATRIX="${LLAMA_FULL_MATRIX:-0}"

# Thermal diagnosis. The max-depth llama-bench invocation normally reserves the full KV
# allocation up front, so thermal-depth begins with high VRAM occupancy even at depth 0.
THERMAL_GPU_ID="${THERMAL_GPU_ID:-1}"
THERMAL_DEPTHS="${THERMAL_DEPTHS:-0,32768,65536,131072,253952}"
THERMAL_BATCH="${THERMAL_BATCH:-4096}"
THERMAL_UBATCH="${THERMAL_UBATCH:-1024}"
THERMAL_REPS="${THERMAL_REPS:-1}"
THERMAL_COLD_MAX_C="${THERMAL_COLD_MAX_C:-55}"
THERMAL_COOL_TIMEOUT_S="${THERMAL_COOL_TIMEOUT_S:-900}"
THERMAL_COOL_POLL_S="${THERMAL_COOL_POLL_S:-5}"
THERMAL_SHORT_ROUNDS="${THERMAL_SHORT_ROUNDS:-3}"
THERMAL_SHORT_PP="${THERMAL_SHORT_PP:-16384}"
THERMAL_SHORT_TG="${THERMAL_SHORT_TG:-1024}"
THERMAL_BATCH_LIST="${THERMAL_BATCH_LIST:-2048/512,4096/512,4096/1024,8192/1024,8192/2048}"

# vLLM remaining tests.
VLLM_TEST_MODEL="${VLLM_TEST_MODEL:-}"
VLLM_TEST_QUANT="${VLLM_TEST_QUANT:-}"
VLLM_TEST_KV="${VLLM_TEST_KV:-fp8_e4m3}"
VLLM_UTIL="${VLLM_UTIL:-0.94}"
VLLM_VRAM_UTILS="${VLLM_VRAM_UTILS:-0.60,0.55,0.50,0.45,0.40}"
VLLM_VRAM_FINE="${VLLM_VRAM_FINE:-1}"
VLLM_VRAM_FINE_ITERS="${VLLM_VRAM_FINE_ITERS:-3}"
VLLM_VRAM_SEQS="${VLLM_VRAM_SEQS:-1}"
VLLM_VRAM_BATCHED_TOKENS="${VLLM_VRAM_BATCHED_TOKENS:-8192}"
VLLM_VRAM_VERIFY_OUTPUT="${VLLM_VRAM_VERIFY_OUTPUT:-64}"
TRAINING_MARGIN_MIB="${TRAINING_MARGIN_MIB:-1536}"
VLLM_EXT_KV_LIST="${VLLM_EXT_KV_LIST:-fp8_e4m3,auto,fp8_e5m2}"
VLLM_EXT_BATCH_LIST="${VLLM_EXT_BATCH_LIST:-4096,8192,16384,32768}"
VLLM_EXT_DEPTHS="${VLLM_EXT_DEPTHS:-8192,65536,131072,253952}"
VLLM_EXT_OUTPUT="${VLLM_EXT_OUTPUT:-256}"
VLLM_EXT_SEQS="${VLLM_EXT_SEQS:-1}"
VLLM_EXT_BASE_BATCH="${VLLM_EXT_BASE_BATCH:-8192}"

# Local-only runtime/cache/Docker paths.
LOCAL_HOME="$ROOT/runtime/home"
LOCAL_TMP="$ROOT/tmp"
LOCAL_CACHE="$ROOT/cache"
LOCAL_DOCKER="$ROOT/docker-runtime"
LOCAL_DOCKER_SOCKET="$LOCAL_DOCKER/docker.sock"
LOCAL_DOCKER_HOST="unix://$LOCAL_DOCKER_SOCKET"
DOCKER_CONFIG="$ROOT/docker-config"
LOCAL_DOCKER_LOG="$ROOT/results/logs/local-dockerd.log"

mkdir -p "$ROOT/results/targeted" "$ROOT/results/telemetry" "$ROOT/results/vram" \
  "$ROOT/results/logs" "$ROOT/results/vllm-r13" "$ROOT/models/hf" "$ROOT/models/llama-cache" \
  "$LOCAL_HOME" "$LOCAL_TMP" "$LOCAL_CACHE/xdg" "$LOCAL_CACHE/torch" \
  "$LOCAL_CACHE/torchinductor" "$LOCAL_CACHE/triton" "$LOCAL_CACHE/cuda" \
  "$LOCAL_CACHE/vllm" "$LOCAL_CACHE/numba" "$ROOT/runtime/xdg-config" \
  "$ROOT/runtime/xdg-data" "$ROOT/runtime/xdg-state" "$DOCKER_CONFIG" "$LOCAL_DOCKER"

export TMPDIR="$LOCAL_TMP" TMP="$LOCAL_TMP" TEMP="$LOCAL_TMP"
export HOME="$LOCAL_HOME"
export XDG_CACHE_HOME="$LOCAL_CACHE/xdg"
export XDG_CONFIG_HOME="$ROOT/runtime/xdg-config"
export XDG_DATA_HOME="$ROOT/runtime/xdg-data"
export XDG_STATE_HOME="$ROOT/runtime/xdg-state"
export HF_HOME="$ROOT/models/hf"
export HUGGINGFACE_HUB_CACHE="$ROOT/models/hf/hub"
export TRANSFORMERS_CACHE="$ROOT/models/hf/transformers"
export TORCH_HOME="$LOCAL_CACHE/torch"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE/torchinductor"
export TRITON_CACHE_DIR="$LOCAL_CACHE/triton"
export CUDA_CACHE_PATH="$LOCAL_CACHE/cuda"
export VLLM_CACHE_ROOT="$LOCAL_CACHE/vllm"
export NUMBA_CACHE_DIR="$LOCAL_CACHE/numba"
export PYTHONPYCACHEPREFIX="$LOCAL_CACHE/python"
export DOCKER_CONFIG

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
warn() { printf '\nWARNING: %s\n' "$*" >&2; }
csv_words() { printf '%s' "$1" | tr ',' ' '; }

if [ "$MODEL_SIZE" = "27b" ]; then
  LLAMA_REPO="${LLAMA_REPO:-unsloth/Qwen3.6-27B-GGUF}"
  LLAMA_PREFIX="${LLAMA_PREFIX:-Qwen3.6-27B}"
  [ "$LLAMA_TEST_QUANT" = "UD-IQ4_NL_XL" ] && LLAMA_TEST_QUANT="Q4_K_M"
  VLLM_TEST_MODEL="${VLLM_TEST_MODEL:-Lorbus/Qwen3.6-27B-int4-AutoRound}"
  VLLM_TEST_QUANT="${VLLM_TEST_QUANT:-auto_round}"
else
  LLAMA_REPO="${LLAMA_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
  LLAMA_PREFIX="${LLAMA_PREFIX:-Qwen3.6-35B-A3B}"
  VLLM_TEST_MODEL="${VLLM_TEST_MODEL:-nchapman/Qwen3.6-35B-A3B-int4-AutoRound}"
  VLLM_TEST_QUANT="${VLLM_TEST_QUANT:-auto_round}"
fi

llama_file() { printf '%s-%s.gguf' "$LLAMA_PREFIX" "$1"; }

docker_cmd() {
  if [ "$USE_SYSTEM_DOCKER" = "1" ]; then
    sudo env HOME="$LOCAL_HOME" TMPDIR="$LOCAL_TMP" DOCKER_CONFIG="$DOCKER_CONFIG" docker "$@"
  else
    sudo env HOME="$LOCAL_HOME" TMPDIR="$LOCAL_TMP" DOCKER_CONFIG="$DOCKER_CONFIG" DOCKER_HOST="$LOCAL_DOCKER_HOST" docker "$@"
  fi
}

docker_exists() { docker_cmd inspect "$1" >/dev/null 2>&1; }
docker_running() { [ "$(docker_cmd inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]; }
docker_remove() { docker_exists "$1" && docker_cmd rm -f "$1" >/dev/null 2>&1 || true; }

check_tools() {
  local missing="" cmd
  for cmd in docker nvidia-smi python3 curl; do command -v "$cmd" >/dev/null 2>&1 || missing="$missing $cmd"; done
  if [ "$USE_SYSTEM_DOCKER" != "1" ]; then
    command -v dockerd >/dev/null 2>&1 || missing="$missing dockerd"
    command -v nvidia-container-runtime >/dev/null 2>&1 || missing="$missing nvidia-container-runtime"
  fi
  [ -z "$missing" ] && return 0
  warn "Missing host commands:$missing. r13 will not install system packages because that would write outside the current directory."
  return 1
}

check_space() {
  local free_gb
  free_gb="$(df -Pk "$ROOT" | awk 'NR==2{printf "%d",$4/1024/1024}')"
  log "Current-directory filesystem: ${free_gb:-unknown} GiB free."
  if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ] 2>/dev/null && [ "$ALLOW_LOW_SPACE" != "1" ]; then
    warn "Less than $MIN_FREE_GB GiB free; use ALLOW_LOW_SPACE=1 only if intentional."
    return 1
  fi
}

start_local_docker() {
  if docker_cmd info >/dev/null 2>&1; then return 0; fi
  [ "$USE_SYSTEM_DOCKER" = "1" ] && { warn "System Docker is unavailable."; return 1; }
  local runtime ready=0
  runtime="$(command -v nvidia-container-runtime 2>/dev/null)"
  rm -f "$LOCAL_DOCKER_SOCKET" "$LOCAL_DOCKER/dockerd.pid" 2>/dev/null
  log "Starting dedicated current-directory Docker daemon."
  sudo env HOME="$LOCAL_HOME" TMPDIR="$LOCAL_TMP" DOCKER_CONFIG="$DOCKER_CONFIG" \
    nohup dockerd --data-root "$LOCAL_DOCKER/data" --exec-root "$LOCAL_DOCKER/exec" \
      --pidfile "$LOCAL_DOCKER/dockerd.pid" --host "$LOCAL_DOCKER_HOST" \
      --bridge none --iptables=false --ip-forward=false --ip-masq=false \
      --add-runtime "nvidia=$runtime" --log-level error > "$LOCAL_DOCKER_LOG" 2>&1 &
  for _ in $(seq 1 90); do docker_cmd info >/dev/null 2>&1 && { ready=1; break; }; sleep 1; done
  [ "$ready" = "1" ] && return 0
  warn "Local Docker daemon did not start. See $LOCAL_DOCKER_LOG"
  return 1
}

container_env=(
  -e HOME=/workspace/runtime/home -e TMPDIR=/workspace/tmp -e TMP=/workspace/tmp -e TEMP=/workspace/tmp
  -e XDG_CACHE_HOME=/workspace/cache/xdg -e XDG_CONFIG_HOME=/workspace/runtime/xdg-config
  -e XDG_DATA_HOME=/workspace/runtime/xdg-data -e XDG_STATE_HOME=/workspace/runtime/xdg-state
  -e HF_HOME=/workspace/models/hf -e HUGGINGFACE_HUB_CACHE=/workspace/models/hf/hub
  -e TRANSFORMERS_CACHE=/workspace/models/hf/transformers -e LLAMA_CACHE=/workspace/models/llama-cache
  -e TORCH_HOME=/workspace/cache/torch -e TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor
  -e TRITON_CACHE_DIR=/workspace/cache/triton -e CUDA_CACHE_PATH=/workspace/cache/cuda
  -e VLLM_CACHE_ROOT=/workspace/cache/vllm -e NUMBA_CACHE_DIR=/workspace/cache/numba
)

ensure_llama() {
  docker_cmd image inspect "$LLAMA_IMAGE" >/dev/null 2>&1 || { warn "Missing local image $LLAMA_IMAGE. Run the successful r12 build once; r13 intentionally does not rebuild it."; return 1; }
  [ -x "$ROOT/build/llama.cpp/bin/llama-bench" ] || { warn "Missing $ROOT/build/llama.cpp/bin/llama-bench."; return 1; }
}

ensure_vllm() {
  if ! docker_cmd image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    log "Pulling vLLM image into the current-directory Docker data-root: $VLLM_IMAGE"
    docker_cmd pull "$VLLM_IMAGE" || return 1
  fi
}

telemetry_start() {
  local outfile="$1"
  (
    printf '%s\n' 'timestamp,index,temp_c,power_w,power_limit_w,util_pct,mem_used_mib,mem_free_mib,graphics_mhz,sm_mhz,memory_mhz'
    exec nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.free,clocks.current.graphics,clocks.current.sm,clocks.current.memory --format=csv,noheader,nounits -l 1
  ) > "$outfile" 2>&1 &
  TELEMETRY_PID=$!
  sleep 1
}

telemetry_stop() {
  if [ -n "${TELEMETRY_PID:-}" ] && kill -0 "$TELEMETRY_PID" 2>/dev/null; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  TELEMETRY_PID=""
}

telemetry_summary() {
  python3 - "$1" "$2" <<'PYTEL'
import csv,statistics,sys
src,dst=sys.argv[1:]; rows=[]
try:
  for r in csv.DictReader(open(src,newline='',encoding='utf-8',errors='replace')):
    try: rows.append(dict(g=int(r['index']),temp=float(r['temp_c']),power=float(r['power_w']),util=float(r['util_pct']),used=float(r['mem_used_mib']),free=float(r['mem_free_mib']),sm=float(r['sm_mhz'])))
    except: pass
except: pass
fields=['gpu','samples','max_temp_c','avg_power_w','max_power_w','avg_util_pct','max_used_mib','min_free_mib','min_sm_mhz','avg_sm_mhz']; out=[]
for g in sorted({r['g'] for r in rows}):
  a=[r for r in rows if r['g']==g]
  out.append(dict(gpu=g,samples=len(a),max_temp_c=max(x['temp'] for x in a),avg_power_w=statistics.mean(x['power'] for x in a),max_power_w=max(x['power'] for x in a),avg_util_pct=statistics.mean(x['util'] for x in a),max_used_mib=max(x['used'] for x in a),min_free_mib=min(x['free'] for x in a),min_sm_mhz=min(x['sm'] for x in a),avg_sm_mhz=statistics.mean(x['sm'] for x in a)))
with open(dst,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
PYTEL
}

thermal_query_supports() {
  local field="$1"
  nvidia-smi --query-gpu="index,$field" --format=csv,noheader,nounits >/dev/null 2>&1
}

thermal_telemetry_start() {
  local outfile="$1"
  local fields="timestamp,index,pstate,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.free,clocks.current.graphics,clocks.current.sm,clocks.current.memory"
  local f
  for f in temperature.memory \
           clocks_event_reasons.active \
           clocks_event_reasons.sw_power_cap \
           clocks_event_reasons.sw_thermal_slowdown \
           clocks_event_reasons.hw_thermal_slowdown \
           clocks_event_reasons.hw_power_brake_slowdown; do
    thermal_query_supports "$f" && fields="$fields,$f"
  done
  (
    printf '%s\n' "$fields"
    exec nvidia-smi --query-gpu="$fields" --format=csv,noheader,nounits -l 1
  ) > "$outfile" 2>&1 &
  TELEMETRY_PID=$!
  sleep 1
}

thermal_telemetry_summary() {
  python3 - "$1" "$2" <<'PYTHERMSUM'
import csv,sys,statistics
src,dst=sys.argv[1:]
rows=[]
try:
    for r in csv.DictReader(open(src,newline='',encoding='utf-8',errors='replace')):
        def num(k):
            try:return float((r.get(k) or '').strip())
            except:return None
        try:g=int(float(r.get('index','')))
        except:continue
        rows.append((g,r,num))
except Exception:
    pass

def active(v):
    s=(v or '').strip().lower()
    return s == 'active' or s in {'1','yes','true'}

def vals(a,k):
    z=[]
    for _,r,num in a:
        x=num(k)
        if x is not None:z.append(x)
    return z

reason_fields=[
 'clocks_event_reasons.sw_power_cap',
 'clocks_event_reasons.sw_thermal_slowdown',
 'clocks_event_reasons.hw_thermal_slowdown',
 'clocks_event_reasons.hw_power_brake_slowdown',
]
fields=['gpu','samples','start_temp_c','max_temp_c','start_mem_temp_c','max_mem_temp_c',
        'avg_power_busy_w','max_power_w','avg_gpu_util_pct','avg_mem_util_pct','max_used_mib','min_free_mib',
        'start_sm_mhz','avg_sm_busy_mhz','min_sm_busy_mhz','end_sm_mhz','pstates',
        'sw_power_cap_busy_pct','sw_thermal_busy_pct','hw_thermal_busy_pct','hw_power_brake_busy_pct']
out=[]
for g in sorted({g for g,_,_ in rows}):
    a=[x for x in rows if x[0]==g]
    busy=[x for x in a if (x[2]('utilization.gpu') or 0)>=50]
    use=busy or a
    temp=vals(a,'temperature.gpu'); mt=vals(a,'temperature.memory'); power=vals(use,'power.draw')
    gu=vals(a,'utilization.gpu'); mu=vals(a,'utilization.memory'); used=vals(a,'memory.used'); free=vals(a,'memory.free')
    sm=vals(use,'clocks.current.sm'); allsm=vals(a,'clocks.current.sm')
    pst=[]
    for _,r,_ in a:
        x=(r.get('pstate') or '').strip()
        if x and x not in pst:pst.append(x)
    den=max(1,len(busy))
    rp=[]
    for rf in reason_fields:
        rp.append(100*sum(active(r.get(rf)) for _,r,_ in busy)/den if busy else 0.0)
    out.append(dict(
      gpu=g,samples=len(a),start_temp_c=temp[0] if temp else '',max_temp_c=max(temp) if temp else '',
      start_mem_temp_c=mt[0] if mt else '',max_mem_temp_c=max(mt) if mt else '',
      avg_power_busy_w=statistics.mean(power) if power else '',max_power_w=max(vals(a,'power.draw')) if vals(a,'power.draw') else '',
      avg_gpu_util_pct=statistics.mean(gu) if gu else '',avg_mem_util_pct=statistics.mean(mu) if mu else '',
      max_used_mib=max(used) if used else '',min_free_mib=min(free) if free else '',
      start_sm_mhz=allsm[0] if allsm else '',avg_sm_busy_mhz=statistics.mean(sm) if sm else '',
      min_sm_busy_mhz=min(sm) if sm else '',end_sm_mhz=allsm[-1] if allsm else '',pstates='|'.join(pst),
      sw_power_cap_busy_pct=rp[0],sw_thermal_busy_pct=rp[1],hw_thermal_busy_pct=rp[2],hw_power_brake_busy_pct=rp[3]))
with open(dst,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
PYTHERMSUM
}

cool_until() {
  local gpu="$1" limit="$2" timeout="$3" now start elapsed
  start="$(date +%s)"
  while :; do
    now="$(nvidia-smi -i "$gpu" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9.')"
    [ -n "$now" ] || { warn "Could not read GPU $gpu temperature."; return 1; }
    if python3 - "$now" "$limit" <<'PYCOOL'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)
PYCOOL
    then
      log "GPU $gpu cold-start threshold satisfied: ${now} C <= ${limit} C."
      return 0
    fi
    elapsed=$(( $(date +%s) - start ))
    if [ "$elapsed" -ge "$timeout" ]; then
      warn "GPU $gpu is still ${now} C after the configured cool-down window; refusing to label this a cold-start run. Raise --cold-max-temp or rerun later."
      return 2
    fi
    printf '\rCooling GPU %s: %s C (target <= %s C)   ' "$gpu" "$now" "$limit"
    sleep "$THERMAL_COOL_POLL_S"
  done
}

thermal_depth_run() {
  ensure_llama || return 1
  cool_until "$THERMAL_GPU_ID" "$THERMAL_COLD_MAX_C" "$THERMAL_COOL_TIMEOUT_S" || return $?
  local id="thermal_depth_gpu${THERMAL_GPU_ID}" raw="$ROOT/results/targeted/thermal_depth_gpu${THERMAL_GPU_ID}.jsonl"
  local out="$ROOT/results/targeted/thermal_depth_gpu${THERMAL_GPU_ID}.csv" logf="$ROOT/results/logs/thermal_depth_gpu${THERMAL_GPU_ID}.log"
  local telem="$ROOT/results/telemetry/thermal_depth_gpu${THERMAL_GPU_ID}.csv" tsum="$ROOT/results/telemetry/thermal_depth_gpu${THERMAL_GPU_ID}_summary.csv" rc
  log "Cold high-VRAM depth progression | GPU=$THERMAL_GPU_ID | depths=$THERMAL_DEPTHS | b$THERMAL_BATCH/ub$THERMAL_UBATCH"
  thermal_telemetry_start "$telem"
  docker_cmd run --rm --network host --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    -e CUDA_VISIBLE_DEVICES="$THERMAL_GPU_ID" "${container_env[@]}" -v "$ROOT:/workspace" -w /workspace "$LLAMA_IMAGE" \
    /workspace/build/llama.cpp/bin/llama-bench --hf-repo "$LLAMA_REPO" --hf-file "$(llama_file "$LLAMA_TEST_QUANT")" --offline \
    -p "$PP_TOKENS" -n "$TG_TOKENS" -d "$THERMAL_DEPTHS" -b "$THERMAL_BATCH" -ub "$THERMAL_UBATCH" -r "$THERMAL_REPS" \
    -ctk q8_0 -ctv q8_0 -ngl 999 -sm none -fa on --load-mode mmap -t "$THREADS" -o jsonl \
    > "$raw.tmp" 2> "$logf"
  rc=$?
  thermal_telemetry_stop 2>/dev/null || telemetry_stop
  mv "$raw.tmp" "$raw" 2>/dev/null || true
  thermal_telemetry_summary "$telem" "$tsum"
  [ "$rc" -eq 0 ] || { warn "$id returned rc=$rc; see $logf"; return "$rc"; }
  parse_llama_jsonl "$raw" "$out" "$THERMAL_GPU_ID" "$LLAMA_TEST_QUANT" q8_0 q8_0 "$THERMAL_BATCH" "$THERMAL_UBATCH" || return 2
  print_curve "$out"
  local corr="$ROOT/results/telemetry/thermal_depth_gpu${THERMAL_GPU_ID}_correlated.csv"
  printf '\nPer-test thermal/throttle correlation:\n'
  correlate_depth_telemetry "$raw" "$telem" "$THERMAL_GPU_ID" "$corr" || true
  printf '\nThermal/throttle summary:\n'; column -s, -t < "$tsum" 2>/dev/null || cat "$tsum"
  printf '\nRaw telemetry: %s\nCorrelated telemetry: %s\n' "$telem" "$corr"
}

# Alias so the thermal helpers use the same stop implementation.
thermal_telemetry_stop() { telemetry_stop; }

correlate_depth_telemetry() {
  python3 - "$1" "$2" "$3" "$4" <<'PYCORR'
import csv,json,sys,datetime,statistics
raw,telem,gpu,dst=sys.argv[1:]; gpu=int(gpu)
localtz=datetime.datetime.now().astimezone().tzinfo

def parse_test(s):
    if not s:return None
    try:return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).astimezone(localtz)
    except:return None

def parse_nv(s):
    s=(s or '').strip()
    for fmt in ('%Y/%m/%d %H:%M:%S.%f','%Y/%m/%d %H:%M:%S','%Y-%m-%d %H:%M:%S.%f','%Y-%m-%d %H:%M:%S'):
        try:return datetime.datetime.strptime(s,fmt).replace(tzinfo=localtz)
        except:pass
    return None

def num(r,k):
    try:return float((r.get(k) or '').strip())
    except:return None

def active(v):
    s=(v or '').strip().lower();return s=='active' or s in {'1','yes','true'}
tr=[]
for r in csv.DictReader(open(telem,newline='',encoding='utf-8',errors='replace')):
    try:
        if int(float(r.get('index',''))) != gpu:continue
    except:continue
    t=parse_nv(r.get('timestamp'))
    if t:tr.append((t,r))
tr.sort(key=lambda x:x[0])
tests=[]
for line in open(raw,encoding='utf-8',errors='replace'):
    line=line.strip()
    if not line.startswith('{'):continue
    try:r=json.loads(line)
    except:continue
    t=parse_test(r.get('test_time'))
    try:ts=float(r.get('avg_ts') or 0)
    except:continue
    if not t or ts<=0:continue
    p=int(r.get('n_prompt') or 0);n=int(r.get('n_gen') or 0)
    kind='PP' if p>0 and n==0 else ('TG' if n>0 and p==0 else 'OTHER')
    tests.append((t,r,kind,ts))
tests.sort(key=lambda x:x[0])
reason_fields=[('sw_power','clocks_event_reasons.sw_power_cap'),('sw_thermal','clocks_event_reasons.sw_thermal_slowdown'),('hw_thermal','clocks_event_reasons.hw_thermal_slowdown'),('hw_power_brake','clocks_event_reasons.hw_power_brake_slowdown')]
out=[]
prev=tr[0][0] if tr else None
for end,r,kind,ts in tests:
    if prev is None: prev=end-datetime.timedelta(seconds=1)
    a=[x for t,x in tr if prev <= t <= end]
    if not a:
        # nearest few samples if timestamp boundaries do not align perfectly
        near=sorted(tr,key=lambda x:abs((x[0]-end).total_seconds()))[:3];a=[x[1] for x in near]
    def vs(k):return [x for x in (num(z,k) for z in a) if x is not None]
    temp=vs('temperature.gpu');mt=vs('temperature.memory');sm=vs('clocks.current.sm');power=vs('power.draw');used=vs('memory.used');free=vs('memory.free');gu=vs('utilization.gpu');mu=vs('utilization.memory')
    busy=[z for z in a if (num(z,'utilization.gpu') or 0)>=50];den=max(1,len(busy))
    rr={name:100*sum(active(z.get(field)) for z in busy)/den if busy else 0 for name,field in reason_fields}
    out.append(dict(depth=int(r.get('n_depth') or 0),kind=kind,tok_s=ts,end_time=end.isoformat(),samples=len(a),
      start_temp_c=temp[0] if temp else '',max_temp_c=max(temp) if temp else '',start_mem_temp_c=mt[0] if mt else '',max_mem_temp_c=max(mt) if mt else '',
      avg_sm_mhz=statistics.mean(sm) if sm else '',min_sm_mhz=min(sm) if sm else '',avg_power_w=statistics.mean(power) if power else '',max_power_w=max(power) if power else '',
      avg_gpu_util_pct=statistics.mean(gu) if gu else '',avg_mem_util_pct=statistics.mean(mu) if mu else '',max_used_mib=max(used) if used else '',min_free_mib=min(free) if free else '',
      sw_power_active_pct=rr['sw_power'],sw_thermal_active_pct=rr['sw_thermal'],hw_thermal_active_pct=rr['hw_thermal'],hw_power_brake_active_pct=rr['hw_power_brake']))
    prev=end
fields=list(out[0]) if out else ['depth','kind','tok_s']
with open(dst,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
print(f"{'DEPTH':>8} {'K':>2} {'tok/s':>9} {'Tmax':>6} {'SMavg':>8} {'Pavg':>7} {'VRAM':>7} {'Pcap%':>6} {'SWth%':>6} {'HWth%':>6}")
for x in out:
    def f(k,n=0):
        try:return float(x[k])
        except:return 0.0
    print(f"{x['depth']:8d} {x['kind']:>2} {f('tok_s'):9.2f} {f('max_temp_c'):6.1f} {f('avg_sm_mhz'):8.0f} {f('avg_power_w'):7.1f} {f('max_used_mib'):7.0f} {f('sw_power_active_pct'):6.1f} {f('sw_thermal_active_pct'):6.1f} {f('hw_thermal_active_pct'):6.1f}")
PYCORR
}

append_short_result() {
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" <<'PYSHORTAPP'
import csv,json,os,sys
src,dst,roundno,b,ub,kind=sys.argv[1:]
rows=[]
for line in open(src,encoding='utf-8',errors='replace'):
    line=line.strip()
    if not line.startswith('{'):continue
    try:r=json.loads(line)
    except:continue
    try:ts=float(r.get('avg_ts') or 0)
    except:continue
    if ts<=0:continue
    p=int(r.get('n_prompt') or 0);n=int(r.get('n_gen') or 0)
    if kind=='pp' and not (p>0 and n==0):continue
    if kind=='tg' and not (n>0 and p==0):continue
    rows.append(dict(round=int(roundno),kind=kind,batch=int(b),ubatch=int(ub),n_prompt=p,n_gen=n,n_depth=int(r.get('n_depth') or 0),tok_s=ts,test_time=r.get('test_time','')))
if not rows:raise SystemExit(2)
fields=list(rows[0]); exists=os.path.exists(dst) and os.path.getsize(dst)>0
with open(dst,'a',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields)
    if not exists:w.writeheader()
    w.writerows(rows)
PYSHORTAPP
}

short_summary() {
  python3 - "$1" "$2" <<'PYSHORTSUM'
import csv,statistics,sys
src,dst=sys.argv[1:];rows=list(csv.DictReader(open(src,newline='',encoding='utf-8')));out=[]
for kind in ('pp','tg'):
  keys=sorted({(int(r['batch']),int(r['ubatch'])) for r in rows if r['kind']==kind})
  for b,u in keys:
    a=[r for r in rows if r['kind']==kind and int(r['batch'])==b and int(r['ubatch'])==u]
    vals=[float(r['tok_s']) for r in a]
    rr=sorted((int(r['round']),float(r['tok_s'])) for r in a)
    out.append(dict(kind=kind,batch=b,ubatch=u,samples=len(vals),avg_tok_s=statistics.mean(vals),best_tok_s=max(vals),worst_tok_s=min(vals),first_tok_s=rr[0][1],last_tok_s=rr[-1][1],last_vs_first=rr[-1][1]/rr[0][1] if rr[0][1] else 0))
fields=['kind','batch','ubatch','samples','avg_tok_s','best_tok_s','worst_tok_s','first_tok_s','last_tok_s','last_vs_first']
with open(dst,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
print(f"{'KIND':<4} {'B/UB':>12} {'AVG':>10} {'BEST':>10} {'WORST':>10} {'LAST/FIRST':>12}")
for r in sorted(out,key=lambda x:(x['kind'],-x['avg_tok_s'])):print(f"{r['kind']:<4} {r['batch']:>5}/{r['ubatch']:<5} {r['avg_tok_s']:10.2f} {r['best_tok_s']:10.2f} {r['worst_tok_s']:10.2f} {r['last_vs_first']:12.3f}")
PYSHORTSUM
}

task_thermal_short() {
  ensure_llama || return 1
  cool_until "$THERMAL_GPU_ID" "$THERMAL_COLD_MAX_C" "$THERMAL_COOL_TIMEOUT_S" || return $?
  local out="$ROOT/results/targeted/THERMAL_SHORT_GPU${THERMAL_GPU_ID}.csv" sum="$ROOT/results/targeted/THERMAL_SHORT_GPU${THERMAL_GPU_ID}_SUMMARY.csv"
  local telem="$ROOT/results/telemetry/thermal_short_gpu${THERMAL_GPU_ID}.csv" tsum="$ROOT/results/telemetry/thermal_short_gpu${THERMAL_GPU_ID}_summary.csv"
  local round pair b ub raw logf rc order
  rm -f "$out" "$sum"
  log "Repeated short-sequence thermal/batch test | GPU=$THERMAL_GPU_ID | rounds=$THERMAL_SHORT_ROUNDS | PP=$THERMAL_SHORT_PP | TG=$THERMAL_SHORT_TG"
  thermal_telemetry_start "$telem"
  for round in $(seq 1 "$THERMAL_SHORT_ROUNDS"); do
    order="$(csv_words "$THERMAL_BATCH_LIST")"
    if [ $((round%2)) -eq 0 ]; then order="$(printf '%s\n' $order | tac | tr '\n' ' ')"; fi
    for pair in $order; do
      b="${pair%/*}";ub="${pair#*/}"
      raw="$ROOT/results/targeted/.thermal_short_pp_r${round}_b${b}_u${ub}.jsonl";logf="$ROOT/results/logs/thermal_short_pp_r${round}_b${b}_u${ub}.log"
      docker_cmd run --rm --network host --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
        -e CUDA_VISIBLE_DEVICES="$THERMAL_GPU_ID" "${container_env[@]}" -v "$ROOT:/workspace" -w /workspace "$LLAMA_IMAGE" \
        /workspace/build/llama.cpp/bin/llama-bench --hf-repo "$LLAMA_REPO" --hf-file "$(llama_file "$LLAMA_TEST_QUANT")" --offline \
        -p "$THERMAL_SHORT_PP" -n 0 -d 0 -b "$b" -ub "$ub" -r 1 --no-warmup -ctk q8_0 -ctv q8_0 -ngl 999 -sm none -fa on --load-mode mmap -t "$THREADS" -o jsonl > "$raw" 2> "$logf"
      rc=$?; [ "$rc" -eq 0 ] && append_short_result "$raw" "$out" "$round" "$b" "$ub" pp || warn "PP r$round b$b/ub$ub failed rc=$rc; see $logf"
    done
    # A long depth-0 TG run after each PP round measures whether short-context decode also decays as the card heats.
    b=4096;ub=1024;raw="$ROOT/results/targeted/.thermal_short_tg_r${round}.jsonl";logf="$ROOT/results/logs/thermal_short_tg_r${round}.log"
    docker_cmd run --rm --network host --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
      -e CUDA_VISIBLE_DEVICES="$THERMAL_GPU_ID" "${container_env[@]}" -v "$ROOT:/workspace" -w /workspace "$LLAMA_IMAGE" \
      /workspace/build/llama.cpp/bin/llama-bench --hf-repo "$LLAMA_REPO" --hf-file "$(llama_file "$LLAMA_TEST_QUANT")" --offline \
      -p 0 -n "$THERMAL_SHORT_TG" -d 0 -b "$b" -ub "$ub" -r 1 --no-warmup -ctk q8_0 -ctv q8_0 -ngl 999 -sm none -fa on --load-mode mmap -t "$THREADS" -o jsonl > "$raw" 2> "$logf"
    rc=$?; [ "$rc" -eq 0 ] && append_short_result "$raw" "$out" "$round" "$b" "$ub" tg || warn "TG round $round failed rc=$rc; see $logf"
  done
  thermal_telemetry_stop
  thermal_telemetry_summary "$telem" "$tsum"
  short_summary "$out" "$sum"
  printf '\nThermal/throttle summary:\n'; column -s, -t < "$tsum" 2>/dev/null || cat "$tsum"
  printf '\nRaw telemetry: %s\nShort-run summary: %s\n' "$telem" "$sum"
}

task_thermal_suite() {
  task_thermal_depth || return $?
  log "Depth progression finished. Cooling back to the same starting threshold before the short-workload comparison."
  task_thermal_short
}

task_thermal_depth() { thermal_depth_run; }

parse_llama_jsonl() {
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" <<'PYLL'
import csv,json,sys
src,dst,gpu,quant,k,v,b,ub=sys.argv[1:]; rows=[]
for line in open(src,encoding='utf-8',errors='replace'):
  line=line.strip()
  if not line.startswith('{'): continue
  try:r=json.loads(line)
  except:continue
  try:
    if float(r.get('avg_ts') or 0)<=0:continue
  except:continue
  rows.append({'gpu':gpu,'quant':quant,'type_k':k,'type_v':v,'batch':b,'ubatch':ub,**r})
if not rows: raise SystemExit(2)
fields=[]
for r in rows:
  for x in r:
    if x not in fields:fields.append(x)
with open(dst,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
PYLL
}

print_curve() {
  python3 - "$1" <<'PYCURVE'
import csv,sys
d={}
for r in csv.DictReader(open(sys.argv[1],newline='',encoding='utf-8')):
  try:dep=int(float(r.get('n_depth') or 0));p=int(float(r.get('n_prompt') or 0));n=int(float(r.get('n_gen') or 0));ts=float(r.get('avg_ts') or 0)
  except:continue
  z=d.setdefault(dep,{})
  if p>0 and n==0:z['pp']=ts
  if n>0 and p==0:z['tg']=ts
print(f"{'DEPTH':>10} {'PP tok/s':>12} {'TG tok/s':>12}");print('-'*38)
for dep,z in sorted(d.items()):print(f"{dep:10,d} {z.get('pp',float('nan')):12.2f} {z.get('tg',float('nan')):12.2f}")
PYCURVE
}

llama_run() {
  local phase="$1" gpu="$2" quant="$3" k="$4" v="$5" batch="$6" ubatch="$7" depths="$8"
  local id raw out logf telem tsum rc
  id="${phase}_gpu${gpu}_${quant}_${k}_${v}_b${batch}_u${ubatch}"
  raw="$ROOT/results/targeted/${id}.jsonl"; out="$ROOT/results/targeted/${id}.csv"; logf="$ROOT/results/logs/${id}.log"
  telem="$ROOT/results/telemetry/${id}.csv"; tsum="$ROOT/results/telemetry/${id}_summary.csv"
  if [ "$FORCE" != "1" ] && [ -s "$out" ]; then log "Resume $id"; print_curve "$out"; return 0; fi
  log "llama.cpp | physical GPU $gpu | $quant | KV $k/$v | b$batch/ub$ubatch | depths=$depths"
  telemetry_start "$telem"
  docker_cmd run --rm --network host --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    -e CUDA_VISIBLE_DEVICES="$gpu" "${container_env[@]}" -v "$ROOT:/workspace" -w /workspace "$LLAMA_IMAGE" \
    /workspace/build/llama.cpp/bin/llama-bench --hf-repo "$LLAMA_REPO" --hf-file "$(llama_file "$quant")" --offline \
    -p "$PP_TOKENS" -n "$TG_TOKENS" -d "$depths" -b "$batch" -ub "$ubatch" -r "$REPS" \
    -ctk "$k" -ctv "$v" -ngl 999 -sm none -fa on --load-mode mmap -t "$THREADS" -o jsonl \
    > "$raw.tmp" 2> "$logf"
  rc=$?
  telemetry_stop
  mv "$raw.tmp" "$raw" 2>/dev/null || true
  telemetry_summary "$telem" "$tsum"
  if [ "$rc" -ne 0 ]; then warn "$id returned rc=$rc. See $logf"; return "$rc"; fi
  parse_llama_jsonl "$raw" "$out" "$gpu" "$quant" "$k" "$v" "$batch" "$ubatch" || { warn "Could not parse $raw"; return 2; }
  print_curve "$out"
  printf '\nTelemetry summary:\n'; column -s, -t < "$tsum" 2>/dev/null || cat "$tsum"
}

task_gpu_compare() {
  ensure_llama || return 1
  llama_run gpucompare "$COMPARE_GPU_ID" "$LLAMA_TEST_QUANT" q8_0 q8_0 4096 1024 "$LLAMA_COMPARE_DEPTHS" || return 1
  local base="$ROOT/results/canonical/ctx_g1_${LLAMA_TEST_QUANT}_q8_0_q8_0_none_b4096_u1024_m0_l999.csv"
  local other="$ROOT/results/targeted/gpucompare_gpu${COMPARE_GPU_ID}_${LLAMA_TEST_QUANT}_q8_0_q8_0_b4096_u1024.csv"
  if [ -s "$base" ] && [ -s "$other" ] && [ "$COMPARE_GPU_ID" != "0" ]; then
    python3 - "$base" "$other" "$COMPARE_GPU_ID" "$ROOT/results/targeted/GPU_COMPARE.csv" <<'PYCOMP'
import csv,sys
base,new,gpu,out=sys.argv[1:]
def curve(fn):
 d={}
 for r in csv.DictReader(open(fn,newline='',encoding='utf-8')):
  try:dep=int(float(r.get('n_depth') or 0));p=int(float(r.get('n_prompt') or 0));n=int(float(r.get('n_gen') or 0));ts=float(r.get('avg_ts') or 0)
  except:continue
  z=d.setdefault(dep,{})
  if p>0 and n==0:z['pp']=ts
  if n>0 and p==0:z['tg']=ts
 return d
a,b=curve(base),curve(new); rows=[]
for d in sorted(set(a)&set(b)):
 rows.append({'depth':d,'gpu0_pp':a[d].get('pp',0),'gpu0_tg':a[d].get('tg',0),f'gpu{gpu}_pp':b[d].get('pp',0),f'gpu{gpu}_tg':b[d].get('tg',0),'pp_ratio':b[d].get('pp',0)/a[d].get('pp',1),'tg_ratio':b[d].get('tg',0)/a[d].get('tg',1)})
with open(out,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['depth']);w.writeheader();w.writerows(rows)
print('\nGPU1/GPU0 comparison ratios:')
for r in rows:print(f"depth={r['depth']:6d}  PP={r['pp_ratio']:.3f}x  TG={r['tg_ratio']:.3f}x")
PYCOMP
  else
    warn "Existing GPU-0 baseline not found at $base; the GPU-$COMPARE_GPU_ID curve and telemetry are still valid."
  fi
}

choose_llama_batch() {
  python3 - "$ROOT/results/targeted" "$SINGLE_GPU_ID" "$LLAMA_TEST_QUANT" "$1" <<'PYBW'
import csv,glob,os,re,sys
root,gpu,q,out=sys.argv[1:]; cand=[]
for fn in glob.glob(os.path.join(root,f'llamabatch_gpu{gpu}_{q}_q8_0_q8_0_b*_u*.csv')):
 m=re.search(r'_b(\d+)_u(\d+)\.csv$',fn)
 if not m:continue
 d={}
 for r in csv.DictReader(open(fn,newline='',encoding='utf-8')):
  try:dep=int(float(r.get('n_depth') or 0));p=int(float(r.get('n_prompt') or 0));n=int(float(r.get('n_gen') or 0));ts=float(r.get('avg_ts') or 0)
  except:continue
  z=d.setdefault(dep,{})
  if p>0 and n==0:z['pp']=ts
  if n>0 and p==0:z['tg']=ts
 score=[]
 for z in d.values():
  pp,tg=z.get('pp',0),z.get('tg',0)
  if pp>0 and tg>0:score.append((32768+1024)/(32768/pp+1024/tg))
 if score:cand.append((sum(score)/len(score),int(m.group(1)),int(m.group(2))))
if not cand:raise SystemExit(2)
best=max(cand);open(out,'w').write(f'{best[1]} {best[2]}\n');print(f'Batch winner: b{best[1]}/ub{best[2]}  score={best[0]:.2f}')
PYBW
}

task_llama_extended() {
  ensure_llama || return 1
  local pair b ub kv k v win="$ROOT/results/targeted/LLAMA_BATCH_WINNER_GPU${SINGLE_GPU_ID}.txt"
  log "Stage 1: batch/microbatch sweep on physical GPU $SINGLE_GPU_ID"
  for pair in $(csv_words "$LLAMA_BATCH_LIST"); do b="${pair%/*}";ub="${pair#*/}";llama_run llamabatch "$SINGLE_GPU_ID" "$LLAMA_TEST_QUANT" q8_0 q8_0 "$b" "$ub" "$LLAMA_BATCH_DEPTHS" || true;done
  choose_llama_batch "$win" || return 1
  read -r b ub < "$win"
  log "Stage 2: KV datatype sweep on winning b$b/ub$ub"
  for kv in $(csv_words "$LLAMA_KV_LIST"); do k="${kv%/*}";v="${kv#*/}";llama_run llamakv "$SINGLE_GPU_ID" "$LLAMA_TEST_QUANT" "$k" "$v" "$b" "$ub" "$LLAMA_EXT_DEPTHS" || true;done
  if [ "$LLAMA_FULL_MATRIX" = "1" ]; then
    log "Stage 3: requested full batch × KV matrix"
    for pair in $(csv_words "$LLAMA_BATCH_LIST"); do b="${pair%/*}";ub="${pair#*/}";for kv in $(csv_words "$LLAMA_KV_LIST"); do k="${kv%/*}";v="${kv#*/}";llama_run llamamatrix "$SINGLE_GPU_ID" "$LLAMA_TEST_QUANT" "$k" "$v" "$b" "$ub" "$LLAMA_EXT_DEPTHS" || true;done;done
  fi
}

vllm_stop() { docker_remove qwen36-vllm-r13; }

vllm_start() {
  local util="$1" kv="$2" batched="$3" seqs="$4" model="$5" quant="$6" ready=0
  local -a qargs=() kvargs=() arargs=()
  case "$quant" in auto_round)qargs=(--quantization auto_round);;awq)qargs=(--quantization awq_marlin);;auto|'')qargs=();;*)qargs=(--quantization "$quant");;esac
  [ "$kv" != "auto" ] && kvargs=(--kv-cache-dtype "$kv")
  if ! nvidia-smi topo -m 2>/dev/null | grep -Eq 'NV[1-9]'; then arargs=(--disable-custom-all-reduce); fi
  vllm_stop
  log "vLLM TP=2 | GPUs=$GPU_PAIR | util=$util | KV=$kv | max-batched=$batched | seqs=$seqs"
  docker_cmd run -d --network host --name qwen36-vllm-r13 --gpus all --ipc=host --shm-size=32g \
    --ulimit memlock=-1 --ulimit stack=67108864 -e CUDA_VISIBLE_DEVICES="$GPU_PAIR" \
    -e NCCL_CUMEM_ENABLE=0 -e VLLM_NO_USAGE_STATS=1 -e OMP_NUM_THREADS=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${container_env[@]}" \
    -v "$ROOT:/workspace" -w /workspace --entrypoint vllm "$VLLM_IMAGE" serve "$model" \
      --served-model-name "$VLLM_ALIAS" --host 127.0.0.1 --port "$VLLM_PORT" --api-key local \
      --tensor-parallel-size 2 --dtype "$VLLM_DTYPE" --max-model-len "$NATIVE_CTX" \
      --max-num-seqs "$seqs" --max-num-batched-tokens "$batched" --gpu-memory-utilization "$util" \
      --language-model-only --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
      --reasoning-parser "$VLLM_REASONING_PARSER" --enable-auto-tool-choice --tool-call-parser "$VLLM_TOOL_CALL_PARSER" \
      "${arargs[@]}" "${qargs[@]}" "${kvargs[@]}" >/dev/null 2>&1 || return 1
  for _ in $(seq 1 300); do
    docker_running qwen36-vllm-r13 || break
    if curl -fsS -H 'Authorization: Bearer local' "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then ready=1;break;fi
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    local f="$ROOT/results/logs/vllm-r13-failed-$(date +%s).log"
    docker_exists qwen36-vllm-r13 && docker_cmd logs qwen36-vllm-r13 > "$f" 2>&1 || true
    warn "vLLM failed to become healthy. See $f"
    tail -n 50 "$f" >&2 2>/dev/null || true
    vllm_stop
    return 1
  fi
}

vllm_bench() {
  local phase="$1" kv="$2" util="$3" batched="$4" inlen="$5" outlen="$6" model="$7" quant="$8"
  local safe id raw csvf logf rc
  safe="$(printf '%s' "$model" | tr '/:.' '_')"
  id="${phase}_${safe}_${quant}_${kv}_u${util//./p}_bt${batched}_i${inlen}_o${outlen}"
  raw="$ROOT/results/vllm-r13/${id}.json";csvf="$ROOT/results/vllm-r13/${id}.csv";logf="$ROOT/results/logs/${id}.log"
  if [ "$FORCE" != "1" ] && [ -s "$csvf" ]; then log "Resume $id";return 0;fi
  log "vLLM bench | input=$inlen output=$outlen"
  docker_cmd exec qwen36-vllm-r13 vllm bench serve --backend openai-chat --base-url "http://127.0.0.1:${VLLM_PORT}" \
    --endpoint /v1/chat/completions --header 'authorization=Bearer local' --model "$VLLM_ALIAS" --tokenizer "$model" \
    --dataset-name random --num-prompts 1 --input-len "$inlen" --output-len "$outlen" --max-concurrency 1 \
    --request-rate inf --ignore-eos --disable-tqdm --save-result --save-detailed \
    --result-dir /workspace/results/vllm-r13 --result-filename "$(basename "$raw")" > "$logf" 2>&1
  rc=$?
  [ "$rc" -ne 0 ] && { warn "vLLM benchmark rc=$rc; see $logf";return "$rc"; }
  python3 - "$raw" "$csvf" "$phase" "$kv" "$util" "$batched" "$inlen" "$outlen" "$model" "$quant" <<'PYVI'
import csv,json,sys
src,dst,phase,kv,util,bt,inp,out,model,quant=sys.argv[1:]
obj=json.load(open(src,encoding='utf-8'));obj=obj[-1] if isinstance(obj,list) else obj
def n(*ks):
 for k in ks:
  try:
   if obj.get(k) is not None:return float(obj[k])
  except:pass
 return 0.0
tin=n('total_input_tokens','total_input');tout=n('total_output_tokens','total_output');ttft=n('mean_ttft_ms');tpot=n('mean_tpot_ms');pp=tin/(ttft/1000) if ttft>0 else 0;gt=1000/tpot if tpot>0 else n('output_throughput')
r={'phase':phase,'kv':kv,'util':util,'max_num_batched_tokens':bt,'input_len':inp,'output_len':out,'model':model,'quant':quant,'completed':n('completed'),'failed':n('failed'),'total_input_tokens':tin,'total_output_tokens':tout,'pp_proxy':pp,'gt_single':gt,'output_throughput':n('output_throughput'),'total_token_throughput':n('total_token_throughput'),'mean_ttft_ms':ttft,'mean_tpot_ms':tpot,'mean_e2e_latency_ms':n('mean_e2e_latency_ms','mean_e2el_ms')}
with open(dst,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerow(r)
PYVI
}

gpu_snapshot() {
  local label="$1" file="$2"
  { printf 'label,index,name,memory_used_mib,memory_free_mib,temp_c,power_w,util_pct\n';nvidia-smi --query-gpu=index,name,memory.used,memory.free,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits | awk -v L="$label" -F', *' '{print L "," $1 "," $2 "," $3 "," $4 "," $5 "," $6 "," $7}'; } > "$file"
}

record_vram() {
  python3 - "$1" "$2" "$3" "$4" "$GPU_PAIR" "$ROOT/results/vram/VRAM_FLOOR.csv" <<'PYVR'
import csv,os,sys
util,status,before,after,pair,dst=sys.argv[1:];ids={int(x) for x in pair.split(',')}
def ld(fn):
 d={}
 try:
  for r in csv.DictReader(open(fn,newline='',encoding='utf-8')):
   try:d[int(r['index'])]=r
   except:pass
 except:pass
 return d
b,a=ld(before),ld(after);fields=['util','status','gpu','used_before_mib','free_before_mib','used_after_mib','free_after_mib','delta_used_mib'];exists=os.path.exists(dst) and os.path.getsize(dst)>0
with open(dst,'a',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fields)
 if not exists:w.writeheader()
 for g in sorted(ids):
  def n(r,k):
   try:return float(r.get(k) or 0)
   except:return 0
  x,y=b.get(g,{}),a.get(g,{})
  w.writerow(dict(util=util,status=status,gpu=g,used_before_mib=n(x,'memory_used_mib'),free_before_mib=n(x,'memory_free_mib'),used_after_mib=n(y,'memory_used_mib'),free_after_mib=n(y,'memory_free_mib'),delta_used_mib=n(y,'memory_used_mib')-n(x,'memory_used_mib')))
PYVR
}

vram_try() {
  local util="$1" before="$ROOT/results/vram/before_${util//./p}.csv" after="$ROOT/results/vram/after_${util//./p}.csv"
  vllm_stop;gpu_snapshot before "$before"
  if vllm_start "$util" "$VLLM_TEST_KV" "$VLLM_VRAM_BATCHED_TOKENS" "$VLLM_VRAM_SEQS" "$VLLM_TEST_MODEL" "$VLLM_TEST_QUANT";then
    sleep 2;gpu_snapshot after "$after";docker_cmd logs qwen36-vllm-r13 > "$ROOT/results/vram/server_${util//./p}.log" 2>&1 || true;record_vram "$util" success "$before" "$after";vllm_stop;return 0
  fi
  gpu_snapshot failed "$after";record_vram "$util" fail "$before" "$after";return 1
}

task_vllm_vram() {
  ensure_vllm || return 1
  rm -f "$ROOT/results/vram/VRAM_FLOOR.csv"
  local util last_success="" first_fail="" best="" hi lo mid i
  log "TP=2 native-262K VRAM floor | GPUs=$GPU_PAIR | seqs=$VLLM_VRAM_SEQS | KV=$VLLM_TEST_KV"
  for util in $(csv_words "$VLLM_VRAM_UTILS");do
    if vram_try "$util";then last_success="$util";best="$util";else first_fail="$util";[ -n "$last_success" ] && break;fi
  done
  if [ "$VLLM_VRAM_FINE" = "1" ] && [ -n "$last_success" ] && [ -n "$first_fail" ];then
    hi="$last_success";lo="$first_fail"
    for i in $(seq 1 "$VLLM_VRAM_FINE_ITERS");do mid="$(python3 -c "lo=float('$lo');hi=float('$hi');print(f'{(lo+hi)/2:.4f}')")";log "Fine VRAM probe $i: $mid";if vram_try "$mid";then hi="$mid";best="$mid";else lo="$mid";fi;done
  fi
  [ -z "$best" ] && { warn "No tested utilization fit native 262K.";return 1; }
  log "Lowest successful startup utilization: $best; now verifying a real near-native request."
  if vllm_start "$best" "$VLLM_TEST_KV" "$VLLM_VRAM_BATCHED_TOKENS" "$VLLM_VRAM_SEQS" "$VLLM_TEST_MODEL" "$VLLM_TEST_QUANT";then
    local telem="$ROOT/results/telemetry/vllm_vram_verify_${best//./p}.csv" tsum="$ROOT/results/telemetry/vllm_vram_verify_${best//./p}_summary.csv"
    telemetry_start "$telem";vllm_bench "vramverify" "$VLLM_TEST_KV" "$best" "$VLLM_VRAM_BATCHED_TOKENS" "$TARGET_DEPTH" "$VLLM_VRAM_VERIFY_OUTPUT" "$VLLM_TEST_MODEL" "$VLLM_TEST_QUANT" || true;telemetry_stop;telemetry_summary "$telem" "$tsum";vllm_stop
    python3 - "$tsum" "$GPU_PAIR" "$TRAINING_MARGIN_MIB" "$best" <<'PYHEAD'
import csv,sys
fn,pair,margin,best=sys.argv[1:];ids={int(x) for x in pair.split(',')};margin=float(margin);mins={}
try:
 for r in csv.DictReader(open(fn,newline='',encoding='utf-8')):
  g=int(r['gpu'])
  if g in ids:mins[g]=float(r['min_free_mib'])
except:pass
print(f'\nLowest successful configured utilization: {best}')
for g in sorted(mins):print(f'GPU {g} minimum free VRAM during 254K verification: {mins[g]:.0f} MiB')
if mins:
 safe=max(0,min(mins.values())-margin);print(f'Conservative per-GPU training coexistence budget after {margin:.0f} MiB reserve: ~{safe:.0f} MiB');print('A TP=2 training job must keep its PEAK allocation on EACH GPU below the smaller per-GPU budget. Compute contention is not included.')
PYHEAD
  fi
  printf 'VRAM floor CSV: %s\n' "$ROOT/results/vram/VRAM_FLOOR.csv"
}

choose_vllm_kv() {
  python3 - "$ROOT/results/vllm-r13" "$1" <<'PYVK'
import csv,glob,os,sys
root,out=sys.argv[1:];by={}
for fn in glob.glob(os.path.join(root,'vextkv_*.csv')):
 try:r=next(csv.DictReader(open(fn,newline='',encoding='utf-8')));pp=float(r['pp_proxy']);tg=float(r['gt_single']);p=float(r['input_len']);g=float(r['output_len'])
 except:continue
 if pp>0 and tg>0:by.setdefault(r['kv'],[]).append((p+g)/(p/pp+g/tg))
if not by:raise SystemExit(2)
best=max(by,key=lambda k:sum(by[k])/len(by[k]));open(out,'w').write(best+'\n')
print('KV weighted scores:')
for k,v in sorted(by.items(),key=lambda x:sum(x[1])/len(x[1]),reverse=True):print(f'  {k}: {sum(v)/len(v):.2f}')
print('KV winner:',best)
PYVK
}

task_vllm_extended() {
  ensure_vllm || return 1
  local kv bt depth winner="$ROOT/results/targeted/VLLM_KV_WINNER.txt"
  log "Stage 1: vLLM TP=2 KV dtype sweep at native context"
  for kv in $(csv_words "$VLLM_EXT_KV_LIST");do
    if vllm_start "$VLLM_UTIL" "$kv" "$VLLM_EXT_BASE_BATCH" "$VLLM_EXT_SEQS" "$VLLM_TEST_MODEL" "$VLLM_TEST_QUANT";then
      for depth in $(csv_words "$VLLM_EXT_DEPTHS");do vllm_bench vextkv "$kv" "$VLLM_UTIL" "$VLLM_EXT_BASE_BATCH" "$depth" "$VLLM_EXT_OUTPUT" "$VLLM_TEST_MODEL" "$VLLM_TEST_QUANT" || break;done
    fi
    vllm_stop
  done
  choose_vllm_kv "$winner" || return 1;read -r kv < "$winner"
  log "Stage 2: vLLM max-num-batched-tokens sweep using KV=$kv"
  for bt in $(csv_words "$VLLM_EXT_BATCH_LIST");do
    if vllm_start "$VLLM_UTIL" "$kv" "$bt" "$VLLM_EXT_SEQS" "$VLLM_TEST_MODEL" "$VLLM_TEST_QUANT";then
      for depth in $(csv_words "$VLLM_EXT_DEPTHS");do vllm_bench vextbatch "$kv" "$VLLM_UTIL" "$bt" "$depth" "$VLLM_EXT_OUTPUT" "$VLLM_TEST_MODEL" "$VLLM_TEST_QUANT" || break;done
    fi
    vllm_stop
  done
  python3 - "$ROOT/results/vllm-r13" "$ROOT/results/targeted/VLLM_EXTENDED_SUMMARY.csv" <<'PYVS'
import csv,glob,os,sys
root,out=sys.argv[1:];rows=[]
for fn in glob.glob(os.path.join(root,'vext*.csv')):
 try:r=next(csv.DictReader(open(fn,newline='',encoding='utf-8')));pp=float(r['pp_proxy']);tg=float(r['gt_single']);p=float(r['input_len']);g=float(r['output_len'])
 except:continue
 r['effective_request_tps']=(p+g)/(p/pp+g/tg) if pp>0 and tg>0 else 0;r['source']=os.path.basename(fn);rows.append(r)
rows.sort(key=lambda r:(float(r['input_len']),r['effective_request_tps']),reverse=True);fields=['source','kv','max_num_batched_tokens','input_len','pp_proxy','gt_single','output_throughput','mean_ttft_ms','mean_tpot_ms','effective_request_tps']
with open(out,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print('Summary:',out)
PYVS
}

task_claude_smoke() {
  ensure_vllm || return 1
  vllm_start "$VLLM_UTIL" "$VLLM_TEST_KV" "$VLLM_VRAM_BATCHED_TOKENS" 1 "$VLLM_TEST_MODEL" "$VLLM_TEST_QUANT" || return 1
  local dir="$ROOT/results/claude-smoke" work="$ROOT/results/claude-smoke/work" claude_bin rc
  mkdir -p "$dir" "$work" "$LOCAL_HOME/.claude/plugins/cache"
  [ -f "$work/README.md" ] || printf "%s\n" "Isolated Claude Code local-vLLM smoke test." > "$work/README.md"
  [ -d "$work/.git" ] || git -C "$work" init -q >/dev/null 2>&1 || true
  claude_bin="$(command -v claude 2>/dev/null || true)"
  [ -z "$claude_bin" ] && [ -x "$ROOT/runtime/home/.local/bin/claude" ] && claude_bin="$ROOT/runtime/home/.local/bin/claude"
  if [ -z "$claude_bin" ];then warn "Claude Code binary not found.";vllm_stop;return 1;fi
  "$claude_bin" --version > "$dir/claude-version.txt" 2>&1 || true
  python3 - "$VLLM_PORT" "$VLLM_ALIAS" > "$dir/direct-stream.sse" 2> "$dir/direct-stream.stderr" <<'PYSSE'
import json,sys,urllib.request
port,model=sys.argv[1:];body=json.dumps({'model':model,'max_tokens':64,'stream':True,'messages':[{'role':'user','content':'Reply exactly STREAM_OK'}]}).encode();req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/messages',data=body,headers={'content-type':'application/json','x-api-key':'local','authorization':'Bearer local'})
with urllib.request.urlopen(req,timeout=300) as r:
 for line in r:print(line.decode('utf-8','replace').rstrip(),flush=True)
PYSSE
  local dbg="$dir/claude-debug.log" out="$dir/claude.stdout.jsonl" err="$dir/claude.stderr.log"
  ( cd "$work" && env \
    HOME="$LOCAL_HOME" CLAUDE_CONFIG_DIR="$LOCAL_HOME/.claude" TMPDIR="$LOCAL_TMP" \
    ANTHROPIC_BASE_URL="http://127.0.0.1:${VLLM_PORT}" ANTHROPIC_API_KEY=local ANTHROPIC_AUTH_TOKEN=local \
    ANTHROPIC_MODEL="$VLLM_ALIAS" ANTHROPIC_DEFAULT_OPUS_MODEL="$VLLM_ALIAS" ANTHROPIC_DEFAULT_SONNET_MODEL="$VLLM_ALIAS" ANTHROPIC_DEFAULT_HAIKU_MODEL="$VLLM_ALIAS" \
    CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 CLAUDE_CODE_ATTRIBUTION_HEADER=0 \
    CLAUDE_CODE_AUTO_COMPACT_WINDOW="$NATIVE_CTX" CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=88 CLAUDE_CODE_MAX_OUTPUT_TOKENS=24576 \
    "$claude_bin" --model "$VLLM_ALIAS" -p 'Reply with exactly VLLM_CLAUDE_OK and do not call tools.' --max-turns 1 --output-format stream-json --verbose --debug-file "$dbg" > "$out" 2> "$err" )
  rc=$?;printf '%s\n' "$rc" > "$dir/claude.rc"
  printf 'Claude smoke rc=%s\nstdout=%s\nstderr=%s\ndebug=%s\n' "$rc" "$out" "$err" "$dbg"
  if [ "$rc" -ne 0 ] || ! grep -q VLLM_CLAUDE_OK "$out" 2>/dev/null;then
    warn "Claude smoke failed; unlike the old smoke test, the debug file should contain the actual client-side diagnostics."
    printf '%s\n' '--- stdout tail ---';tail -n 30 "$out" 2>/dev/null || true
    printf '%s\n' '--- stderr tail ---';tail -n 30 "$err" 2>/dev/null || true
    printf '%s\n' '--- debug tail ---';tail -n 80 "$dbg" 2>/dev/null || true
  fi
  vllm_stop
  return "$rc"
}

usage() {
  cat <<'EOFUSAGE'
Usage:
  ./qwen36-r15-claude-tools.sh TASK [options]

TASKS
  gpu-compare      Re-run the winning llama.cpp progressive depth curve on the other physical GPU; log clocks/thermals/VRAM.
  thermal-depth    Cold-start, full-KV-allocation progressive depth test with clocks, VRAM and throttle/event flags.
  thermal-short    Cold-start repeated short PP batch/ubatch optimization plus depth-0 TG heat-decay probes.
  thermal-suite    Run thermal-depth, cool back to the same threshold, then thermal-short.
  llama-extended   Staged llama.cpp batch+microbatch sweep, then expanded KV datatype sweep on one GPU.
  vllm-vram        TP=2 native-262K gpu_memory_utilization floor search, then a real 254K request with VRAM telemetry.
  vllm-extended    TP=2 KV dtype sweep, then max-num-batched-tokens sweep. No C2/C3 test by default.
  claude-smoke     Direct /v1/messages streaming test plus Claude Code --debug-file smoke test.
  remaining        Run gpu-compare, vllm-vram, llama-extended, vllm-extended in that order.
  help

COMMON OVERRIDES
  --single-gpu N                 GPU used by llama-extended (default 0)
  --compare-gpu N                GPU used by gpu-compare (default 1)
  --gpu-pair A,B                 physical GPUs used by vLLM TP=2 (default 0,1)
  --force                        rerun even if a targeted result already exists

THERMAL OVERRIDES
  --thermal-gpu N                physical GPU to diagnose (default 1)
  --cold-max-temp C              require GPU core <= C before each cold-start phase (default 55)
  --cool-timeout S               maximum local cool-down polling time (default 900)
  --thermal-depths LIST          default 0,32768,65536,131072,253952
  --thermal-rounds N             repeated short-test rounds (default 3)
  --thermal-batch-list LIST      default 2048/512,4096/512,4096/1024,8192/1024,8192/2048
  --thermal-pp-tokens N          PP-only tokens per short batch test (default 16384)
  --thermal-tg-tokens N          TG-only tokens after each PP round (default 1024)

LLAMA OVERRIDES
  --llama-quant Q
  --llama-depths LIST            e.g. 0,8192,32768,65536,131072,196608,253952
  --llama-kv-list LIST           default q8/q8,q8/q4,q5_1,q4_1,iq4_nl,q4_0,f16
  --llama-batch-list LIST        default 2048/512,4096/512,4096/1024,8192/1024
  --llama-full-matrix            after staged search, run full KV x batch matrix

VLLM OVERRIDES
  --vllm-kv TYPE                 KV used by vllm-vram (default fp8_e4m3)
  --vllm-util X                  utilization used by vllm-extended (default .94)
  --vllm-vram-utils LIST         coarse floor list (r14 default .60,.55,.50,.45,.40; .60 already worked on this host)
  --vllm-vram-fine 0|1          binary-refine first success/failure bracket (default 1)
  --vllm-seqs N                  max-num-seqs for VRAM/extended tests (default 1)
  --vllm-kv-list LIST            default fp8_e4m3,auto,fp8_e5m2
  --vllm-batch-list LIST         max-num-batched-tokens; default 4096,8192,16384,32768
  --vllm-depths LIST             default 8192,65536,131072,253952
  --training-margin-mib N        reserve subtracted from measured min-free VRAM (default 1536)

All persistent data, caches, result files, telemetry, and Docker image layers remain under the CURRENT directory.
EOFUSAGE
}

parse_cli() {
  if [ $# -gt 0 ] && [[ "$1" != -* ]];then TASK="$1";shift;fi
  while [ $# -gt 0 ];do
    case "$1" in
      --single-gpu)SINGLE_GPU_ID="$2";shift 2;;
      --compare-gpu)COMPARE_GPU_ID="$2";shift 2;;
      --gpu-pair)GPU_PAIR="$2";shift 2;;
      --force)FORCE=1;shift;;
      --thermal-gpu)THERMAL_GPU_ID="$2";shift 2;;
      --cold-max-temp)THERMAL_COLD_MAX_C="$2";shift 2;;
      --cool-timeout)THERMAL_COOL_TIMEOUT_S="$2";shift 2;;
      --thermal-depths)THERMAL_DEPTHS="$2";shift 2;;
      --thermal-rounds)THERMAL_SHORT_ROUNDS="$2";shift 2;;
      --thermal-batch-list)THERMAL_BATCH_LIST="$2";shift 2;;
      --thermal-pp-tokens)THERMAL_SHORT_PP="$2";shift 2;;
      --thermal-tg-tokens)THERMAL_SHORT_TG="$2";shift 2;;
      --llama-quant)LLAMA_TEST_QUANT="$2";shift 2;;
      --llama-depths)LLAMA_COMPARE_DEPTHS="$2";LLAMA_EXT_DEPTHS="$2";shift 2;;
      --llama-kv-list)LLAMA_KV_LIST="$2";shift 2;;
      --llama-batch-list)LLAMA_BATCH_LIST="$2";shift 2;;
      --llama-full-matrix)LLAMA_FULL_MATRIX=1;shift;;
      --vllm-kv)VLLM_TEST_KV="$2";shift 2;;
      --vllm-util)VLLM_UTIL="$2";shift 2;;
      --vllm-vram-utils)VLLM_VRAM_UTILS="$2";shift 2;;
      --vllm-vram-fine)VLLM_VRAM_FINE="$2";shift 2;;
      --vllm-seqs)VLLM_VRAM_SEQS="$2";VLLM_EXT_SEQS="$2";shift 2;;
      --vllm-kv-list)VLLM_EXT_KV_LIST="$2";shift 2;;
      --vllm-batch-list)VLLM_EXT_BATCH_LIST="$2";shift 2;;
      --vllm-depths)VLLM_EXT_DEPTHS="$2";shift 2;;
      --training-margin-mib)TRAINING_MARGIN_MIB="$2";shift 2;;
      --help|-h)TASK=help;shift;;
      *)warn "Unknown option: $1";TASK=help;shift;;
    esac
  done
}

main() {
  parse_cli "$@"
  printf '\n=== %s ===\nscript: %s\nroot:   %s\ntask:   %s\n' "$SCRIPT_REVISION" "$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || printf '%s' "${BASH_SOURCE[0]}")" "$ROOT" "$TASK"
  [ "$TASK" = help ] && { usage;return 0; }
  check_tools || return 1
  check_space || return 1
  start_local_docker || return 1
  case "$TASK" in
    gpu-compare)task_gpu_compare;;
    thermal-depth)task_thermal_depth;;
    thermal-short)task_thermal_short;;
    thermal-suite)task_thermal_suite;;
    llama-extended)task_llama_extended;;
    vllm-vram)task_vllm_vram;;
    vllm-extended)task_vllm_extended;;
    claude-smoke)task_claude_smoke;;
    remaining)
      task_gpu_compare || return 1
      task_vllm_vram || true
      SINGLE_GPU_ID="$COMPARE_GPU_ID" task_llama_extended || true
      task_vllm_extended || true
      ;;
    *)warn "Unknown task: $TASK";usage;return 64;;
  esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ];then main "$@";fi
