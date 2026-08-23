#!/usr/bin/env bash
# llama-pushbutton.sh
# Pushbutton llama.cpp installer / runner / quant+KV+max-context sweep.
# Defaults are intentionally local: source, builds, HF/llama caches and results
# all live below --cache-dir (default: current working directory).

VERSION="2026-08-21.36"
ORIG_ARGC=$#
SCRIPT_START_EPOCH="$(date +%s)"
RUN_ID="${LLAMA_PUSHBUTTON_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
export LLAMA_PUSHBUTTON_RUN_ID="$RUN_ID"

BACKEND="auto"
MODEL="qwen3.8"
QUANT="auto"
KV="auto"
CTX="auto"
MODE=""
CACHE_DIR="$PWD"
LLAMA_REF="auto"
FIT_MARGIN="1024"
PREFLIGHT_OVERHEAD="0"
GPU_BUSY_ABORT=1
GPU_BUSY_MIN_MIB=2048
GPU_BUSY_MIN_PCT=15
MIN_CTX="65536"
REPS="1"
PORT="8080"
HOST="127.0.0.1"
THREADS="auto"
THREADS_BATCH="auto"
BATCH="auto"
UBATCH="auto"
GPUS=""
UPDATE=1
REBUILD=0
FULL_DEPTH=0
CURVE=1
CURVE_CAP=262144
CURVE_DEPTHS="auto"
CURVE_PP_SIZES="auto"   # deprecated: progressive sweep measures PP while filling
CURVE_PP=4096
CPU_PROBE_PP=256
CPU_TUNE=1
CPU_DEEP=0
CPU_NUMA="auto"
CPU_PLACEMENT="auto"
CPU_PLACEMENT_EFFECTIVE="mmap"
CPU_LOAD_MODE_EFFECTIVE="mmap"
CPU_BLAS="auto"
CPU_BLAS_EFFECTIVE="off"
CPU_TUNE_TG=16
CPU_TUNE_TG_CONFIRM=64
CPU_TUNE_PP=256
CPU_TUNED=0
CPU_BEST_PROBE_TG=0
CPU_FAMILY_GATE_TG=16
CPU_FAMILY_GATE_RATIO="0.20"
CPU_ABLATIONS="auto"
CPU_ABL_PP=256
CPU_ABL_TG=64
CPU_ABL_REPS=3
CPU_FINAL_REPS=3
SCREEN_PP=""
SCREEN_TG=""
SCREEN_PP_SD=""
SCREEN_TG_SD=""
SCREEN_NGL=""
CPU_FINAL_PP=""
CPU_FINAL_TG=""
CPU_NUMA_EFFECTIVE="disabled"
THREADS_WAS_AUTO=0
THREADS_BATCH_WAS_AUTO=0
BATCH_WAS_AUTO=0
UBATCH_WAS_AUTO=0
PROBE_PP_EXPLICIT=0
CURVE_TG=128
DEEP_START=4096
EDGE_RESERVE=16
CACHE_REPLAY_TOLERANCE=16
TARGET_CTX=262144
BATCH_TUNE=1
FAST_V100=0
LOCAL_CUDA_INSTALL=1
CPU_LAST=0
CPU_ONLY=0
MODEL_EXPLICIT=0
MODE_EXPLICIT=0
MEMORY_PROFILE=1
MEMBW_ARRAY_MIB=512
MEMBW_REPS=7
MEMBW_THREADS="auto"
AUTO_SUCCESS_QUANTS=3
PREFETCH=1
PREFETCH_JOBS=3
PREFETCH_HEARTBEAT=10
PROBE_RANK=3
CUDA_PY_MINOR="12.9"
CUDA_ENV_ROOT=""
MICROMAMBA_BIN=""
SPEC="none"
VISION=0
PROMPT=""
LIST_MODELS=0
DRY_RUN=0
COMPACTION_ANALYSIS=1
COMPACTION_RETAINS="4096,8192,16384,32768,65536"
COMPACTION_PP_SCALES="1.0"
COMPACTION_EXTRA_MS="0"
LAST_PROGRESSIVE_TSV=""

usage() {
    cat <<'USAGE'
llama-pushbutton.sh — build llama.cpp locally and run/sweep current GGUFs

No arguments:
  Screen in likely-fastest-first order:
    qwen3.6:35b    = Qwen3.6-35B-A3B (3B active MoE)
    nemotron-3.5   = NVIDIA-Nemotron-3.5-Lightning-30B-A3B (3B active MoE)
    qwen3.8        = Qwen3.8-27B dense
    qwen3.6:27b    = Qwen3.6-27B dense
  Staged search, fastest model families first:
    1. Query HF file sizes only; reject any weight quant that cannot fit even the 64K usable
       floor with the lowest-memory requested KV. No GGUF payload is downloaded for those.
    2. Frontload one conservative probe quant, prefetch other model probes in background, and
       run one cheap pp4K/tg128 probe per model family.
    3. Refine weight/KV quants only for the fastest model. Each weight/KV pair gets the highest
       useful context it can actually hold (262K+ preferred; 64K hard floor). At a fixed context,
       once lower-memory KV cannot fit, all higher-memory KV pairs are pruned for that context.
    4. Tune batch/ubatch with shallow PP tests, then run ONE progressive cached-prefix depth sweep.
       Each interval is filled once by PP and gets a TG128 spot measurement at its endpoint.
  No repeated 0->N prefill sweep and no continuous 258K-token decode are performed.

Core options:
  --version                       print script identity and exit
  --backend auto|cuda|cpu        auto picks CUDA whenever an NVIDIA GPU is visible
  --cpu-only                     run the independent CPU sweep directly (no CUDA first);
                                 defaults to --model all --mode sweep, shallow unless
                                 --cpu-deep is also supplied
  --model NAME|all               aliases above; "all" preserves sweep order
  --quant auto|all|list|Q[,Q..]  queried live from the model's HF GGUF repo
  --kv auto|all|TYPE|K,V[;K,V]   examples: q8_0  or  q8_0,q4_0
  --ctx auto|max|N               server/cli context selection
  --target-ctx N|native          preferred context ceiling (default 262144; candidates may fall back to >=64K)
  --mode sweep|bench|server|cli  no args => sweep; with args => server
  --cache-dir DIR                default: current working directory
  --gpus 0,1,...                 sets CUDA_VISIBLE_DEVICES for this run

Tuning / reproducibility:
  --threads N                    generation threads (CPU auto-tuner otherwise chooses)
  --threads-batch N              prompt-processing threads (CPU auto-tuner otherwise chooses)
  --batch N                      auto: 2048
  --ubatch N                     auto: CUDA 512; CPU auto-tunes 128/256/512
  --cpu-numa auto|disabled|distribute|isolate|numactl
                                 auto => distribute on a multi-NUMA CPU
  --cpu-placement auto|mmap|interleave-none
                                 auto => numactl --interleave=all + load-mode none on a
                                 multi-NUMA CPU when numactl is available; otherwise mmap
  --cpu-blas auto|on|off         main-run baseline; auto/off = native no-BLAS
  --cpu-ablations auto|off         final CPU winner only: native + OpenBLAS when
                                 installed + AVX2-off ISA ablation (default auto)
  --cpu-ablation-pp N             PP tokens/variant (default 256)
  --cpu-ablation-tg N             TG tokens/variant (default 64)
  --cpu-ablation-reps N           timed reps/variant (default 3)
  --no-memory-profile             skip host DIMM inventory + sustained memory-bandwidth test
  --memory-array-mib N            MiB PER array for memory benchmark (default 512; 3 arrays)
  --memory-reps N                 timed repetitions/kernel/thread-count (default 7)
  --memory-threads auto|N,N,...   default auto = 1, half physical, physical, logical
  --no-cpu-tune                  disable cheap CPU thread/ubatch auto-tuning
  --cpu-deep                     allow the final progressive 64K-262K CPU sweep
                                 (off by default; CPU sweep otherwise remains shallow)
  --cpu-tune-tg N                coarse TG tokens/thread candidate (default 16);
                                 top two are confirmed at TG64
  --cpu-tune-pp N                prompt tokens per PP-thread/ubatch trial (default 256)
  --fit-margin MiB               free VRAM target per GPU (default 1024)
  --preflight-overhead MiB       optional extra conservative allowance before download (default 0; exact fit decides borderlines)
  --allow-busy-gpu                 continue despite substantial pre-existing GPU VRAM use
  --no-prefetch                   disable background HF prefetch; -hf may download synchronously
  --prefetch-jobs N              max background model downloads announced/scheduled together (default 3)
  --prefetch-heartbeat SEC       status interval while GPU is waiting for a model download (default 10)
  --min-ctx N                    hard minimum usable max context (default 65536)
  --reps N                       shallow-screen repetitions (default 1)
  --curve-cap N                  compatibility alias for --target-ctx N
  --probe-pp N                   GPU/general shallow-screen PP tokens (default 4096)
  --cpu-probe-pp N               CPU family/refinement PP tokens (default 256)
  --tg-probe N                   TG tokens sampled at each checkpoint (default 128)
  --no-compaction-analysis       skip zero-cost post-processing of the measured deep curve
  --compaction-retains N,N,...   retained contexts to analyze (default 4K,8K,16K,32K,64K)
  --compaction-pp-scales X,...   hypothetical PP multipliers for policy sensitivity (default 1.0)
  --compaction-extra-ms N        fixed extra cost per compaction cycle, beyond rebuild (default 0)
  --depths auto|N,N,...          deep checkpoints (default 4K,8K,16K,32K,64K,128K,192K,edge)
  --edge-reserve N               spare tokens kept beyond deepest TG run (default 16)
  --cache-replay-tolerance N     small tail replay allowed without warning (default 16)
  --pp-sizes ...                 deprecated/ignored; PP is measured during the one fill
  --no-curve                     screen only; do not run the winner's deep sweep
  --no-batch-tune                skip shallow batch/ubatch tuning of the winner
  --full-depth                   compatibility alias: ensure final fitted-depth point
  --llama-ref REF                auto: CUDA=origin/master, CPU=885c5bb (b10428);
                                 CPU pin avoids the current b10429+ server regression
  --no-update                    do not git fetch an existing llama.cpp clone
  --rebuild                      delete and rebuild the selected build dir
  --fast-v100                    force FP16 cuBLAS compute on V100 (faster PP,
                                 less conservative numerically)
  --no-micromamba-cuda           do not create a private micromamba CUDA toolkit when
                                 an NVIDIA GPU is present but nvcc is unavailable
  --no-uv-cuda                   compatibility alias for --no-micromamba-cuda
  --no-local-cuda                compatibility alias for --no-micromamba-cuda
  --no-cpu-last                  with no arguments, skip the final independent CPU
                                 model/quant/KV sweep
  --spec none|mtp                server/cli only; baseline default is none
  --vision                       enable/download Qwen multimodal projector
                                 (text-only default saves VRAM/disk/context)
  --host ADDR                    server bind address (default 127.0.0.1)
  --port N                       server port (default 8080)
  --prompt TEXT                  initial llama-cli prompt
  --dry-run                      print actions/commands where practical
  --list-models                  show aliases and repos
  -h, --help

Examples:
  ./llama-pushbutton.sh
  ./llama-pushbutton.sh --cpu-only
  ./llama-pushbutton.sh --cpu-only --model nemotron-3.5
  ./llama-pushbutton.sh --model qwen3.8 --quant list
  ./llama-pushbutton.sh --model qwen3.6:27b --quant UD-Q4_K_XL \
      --kv q8_0,q4_0 --ctx auto --backend cuda --mode server
  ./llama-pushbutton.sh --model nemotron-3.5 --backend cpu \
      --quant UD-Q4_K_M --kv q8_0 --ctx auto --mode server
  ./llama-pushbutton.sh --backend cuda --model all --quant all --kv all \
      --mode sweep

Notes:
  * Sweep preflights live HF GGUF byte sizes before payload download. The model-family probe
    uses a lower member of the top metadata-feasible quant frontier for fast first response;
    higher-quality candidates download in the background while shallow GPU inference runs.
  * KV pairs are ordered by actual cache bytes. Failure to fit at a lower-memory KV precision
    prunes all higher-memory KV pairs for the same weight quant/context.
  * Sweep mode treats --target-ctx as the preferred ceiling, not an all-or-nothing gate.
    A higher-precision weight/KV combination may fall back through useful context tiers,
    but anything below --min-ctx (default 65536) is rejected. CUDA candidates must
    keep model tensors off the CPU at the context they claim.
  * CPU benchmarks are warmed and PP/TG are timed separately so --threads-batch is
    actually honored. CPU auto-tuning calibrates the first model, cheaply rechecks TG threads per family, and re-tunes the
    selected CPU winner. On multi-NUMA hosts, auto placement prefers numactl interleave
    + load-mode none + --numa distribute, avoiding stale mmap page placement.
  * CPU auto quant probing favors conventional K-quants on AVX2-class CPUs. OpenBLAS is
    disabled in auto mode for a stable TG-oriented baseline; --cpu-blas on is explicit.
    CPU metadata gating uses MemAvailable before any large GGUF download.
  * Deep characterization is one work-efficient progressive pass. llama-server prompt
    caching extends one deterministic token prefix from 4K to the edge. Each checkpoint
    reports true interval/cumulative PP plus a TG128 spot sample at that occupied context.
    There is no second PP pass and no continuous decode across the gaps.
  * CUDA build-tool bootstrap uses micromamba only when nvcc is otherwise unavailable.
    It creates a private NVIDIA CUDA 12.9.1 prefix below --cache-dir; no conda executable,
    NVIDIA .run installer, driver install, shell activation, or /usr/local mutation.
  * PyTorch is not required by llama.cpp. Existing uv/Torch environments remain untouched.
  * Host-memory profiling runs once per RUN_ID. 1T stays because it costs seconds
    and separates per-core behavior from memory-channel saturation.
  * Full 1920X TG thread scaling runs only for competitive families; obvious losers
    are rejected by a cheap physical-core TG gate before costly PP work.
  * CPU build ablations run only on the final winner and cache by commit/config.
    OpenBLAS targets PP; AVX2-off measures ISA value against native -march.
  * CPU uses the explicit HF prefetch/cache path too; an uncached quant is downloaded
    before fitting instead of being mislabeled NO_USABLE_FIT/FIT_FAIL.
  * If uv is absent, the official standalone uv installer is bootstrapped under
    $PWD/.llama-pushbutton/bin; no $HOME write or shell-profile modification is needed.
  * Sweep mode builds only llama-fit-params + llama-bench; server/CLI/UI build on demand.
  * On completion, every primary result table for this RUN_ID is collected into one
    paste-ready all-results-$RUN_ID.txt; all-results-latest.txt is refreshed too.
  * Results are flushed after every fit/screen/depth point so Ctrl-C still leaves
    useful TSV/JSONL output.
  * After the final progressive sweep, a zero-inference compaction analysis integrates the
    measured PP/TG curve and reports throughput-optimal reset/trigger policies. This adds
    negligible wall time and does not alter the benchmark.
USAGE
}

err() { printf 'ERROR: %s\n' "$*" >&2; }
warn() { printf 'WARN:  %s\n' "$*" >&2; }
info() { printf '\n==> %s\n' "$*"; }
print_cmd() {
    printf 'COMMAND:' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
}
show_capture_on_failure() {
    local rc="$1" stdout_file="$2" stderr_file="$3"
    printf '\n========== COMMAND FAILURE ==========\n' >&2
    printf 'COMMAND EXIT STATUS: %s\n' "$rc" >&2
    printf '%s\n' "----- STDOUT replay: $stdout_file -----" >&2
    if [ -s "$stdout_file" ]; then
        cat "$stdout_file" >&2
    else
        printf '%s\n' '[no stdout was emitted]' >&2
    fi
    printf '%s\n' '----- end STDOUT -----' >&2
    printf '%s\n' "----- STDERR replay: $stderr_file -----" >&2
    if [ -s "$stderr_file" ]; then
        cat "$stderr_file" >&2
    else
        printf '%s\n' '[no stderr was emitted]' >&2
    fi
    printf '%s\n' '----- end STDERR -----' >&2
    printf '=====================================\n\n' >&2
}
need_arg() { [ -n "${2:-}" ] || { err "$1 requires an argument"; exit 2; }; }

while [ $# -gt 0 ]; do
    case "$1" in
        --backend) need_arg "$1" "${2:-}"; BACKEND="$2"; shift 2 ;;
        --cpu-only) CPU_ONLY=1; shift ;;
        --model) need_arg "$1" "${2:-}"; MODEL="$2"; MODEL_EXPLICIT=1; shift 2 ;;
        --quant) need_arg "$1" "${2:-}"; QUANT="$2"; shift 2 ;;
        --kv) need_arg "$1" "${2:-}"; KV="$2"; shift 2 ;;
        --ctx) need_arg "$1" "${2:-}"; CTX="$2"; shift 2 ;;
        --target-ctx) need_arg "$1" "${2:-}"; TARGET_CTX="$2"; shift 2 ;;
        --mode) need_arg "$1" "${2:-}"; MODE="$2"; MODE_EXPLICIT=1; shift 2 ;;
        --cache-dir) need_arg "$1" "${2:-}"; CACHE_DIR="$2"; shift 2 ;;
        --llama-ref) need_arg "$1" "${2:-}"; LLAMA_REF="$2"; shift 2 ;;
        --fit-margin) need_arg "$1" "${2:-}"; FIT_MARGIN="$2"; shift 2 ;;
        --preflight-overhead) need_arg "$1" "${2:-}"; PREFLIGHT_OVERHEAD="$2"; shift 2 ;;
        --allow-busy-gpu) GPU_BUSY_ABORT=0; shift ;;
        --no-prefetch) PREFETCH=0; shift ;;
        --prefetch-jobs) need_arg "$1" "${2:-}"; PREFETCH_JOBS="$2"; shift 2 ;;
        --prefetch-heartbeat) need_arg "$1" "${2:-}"; PREFETCH_HEARTBEAT="$2"; shift 2 ;;
        --min-ctx) need_arg "$1" "${2:-}"; MIN_CTX="$2"; shift 2 ;;
        --reps) need_arg "$1" "${2:-}"; REPS="$2"; shift 2 ;;
        --curve-cap) need_arg "$1" "${2:-}"; CURVE_CAP="$2"; shift 2 ;;
        --probe-pp) need_arg "$1" "${2:-}"; CURVE_PP="$2"; PROBE_PP_EXPLICIT=1; shift 2 ;;
        --cpu-probe-pp) need_arg "$1" "${2:-}"; CPU_PROBE_PP="$2"; shift 2 ;;
        --tg-probe) need_arg "$1" "${2:-}"; CURVE_TG="$2"; shift 2 ;;
        --no-compaction-analysis) COMPACTION_ANALYSIS=0; shift ;;
        --compaction-retains) need_arg "$1" "${2:-}"; COMPACTION_RETAINS="$2"; shift 2 ;;
        --compaction-pp-scales) need_arg "$1" "${2:-}"; COMPACTION_PP_SCALES="$2"; shift 2 ;;
        --compaction-extra-ms) need_arg "$1" "${2:-}"; COMPACTION_EXTRA_MS="$2"; shift 2 ;;
        --depths) need_arg "$1" "${2:-}"; CURVE_DEPTHS="$2"; shift 2 ;;
        --edge-reserve) need_arg "$1" "${2:-}"; EDGE_RESERVE="$2"; shift 2 ;;
        --cache-replay-tolerance) need_arg "$1" "${2:-}"; CACHE_REPLAY_TOLERANCE="$2"; shift 2 ;;
        --pp-sizes) need_arg "$1" "${2:-}"; CURVE_PP_SIZES="$2"; shift 2 ;;
        --no-batch-tune) BATCH_TUNE=0; shift ;;
        --no-curve) CURVE=0; shift ;;
        --host) need_arg "$1" "${2:-}"; HOST="$2"; shift 2 ;;
        --port) need_arg "$1" "${2:-}"; PORT="$2"; shift 2 ;;
        --threads) need_arg "$1" "${2:-}"; THREADS="$2"; shift 2 ;;
        --threads-batch) need_arg "$1" "${2:-}"; THREADS_BATCH="$2"; shift 2 ;;
        --batch) need_arg "$1" "${2:-}"; BATCH="$2"; shift 2 ;;
        --ubatch) need_arg "$1" "${2:-}"; UBATCH="$2"; shift 2 ;;
        --cpu-numa) need_arg "$1" "${2:-}"; CPU_NUMA="$2"; shift 2 ;;
        --cpu-placement) need_arg "$1" "${2:-}"; CPU_PLACEMENT="$2"; shift 2 ;;
        --cpu-blas) need_arg "$1" "${2:-}"; CPU_BLAS="$2"; shift 2 ;;
        --cpu-ablations) need_arg "$1" "${2:-}"; CPU_ABLATIONS="$2"; shift 2 ;;
        --cpu-ablation-pp) need_arg "$1" "${2:-}"; CPU_ABL_PP="$2"; shift 2 ;;
        --cpu-ablation-tg) need_arg "$1" "${2:-}"; CPU_ABL_TG="$2"; shift 2 ;;
        --cpu-ablation-reps) need_arg "$1" "${2:-}"; CPU_ABL_REPS="$2"; shift 2 ;;
        --no-memory-profile) MEMORY_PROFILE=0; shift ;;
        --memory-array-mib) need_arg "$1" "${2:-}"; MEMBW_ARRAY_MIB="$2"; shift 2 ;;
        --memory-reps) need_arg "$1" "${2:-}"; MEMBW_REPS="$2"; shift 2 ;;
        --memory-threads) need_arg "$1" "${2:-}"; MEMBW_THREADS="$2"; shift 2 ;;
        --no-cpu-tune) CPU_TUNE=0; shift ;;
        --cpu-deep) CPU_DEEP=1; shift ;;
        --cpu-tune-tg) need_arg "$1" "${2:-}"; CPU_TUNE_TG="$2"; shift 2 ;;
        --cpu-tune-pp) need_arg "$1" "${2:-}"; CPU_TUNE_PP="$2"; shift 2 ;;
        --gpus) need_arg "$1" "${2:-}"; GPUS="$2"; shift 2 ;;
        --spec) need_arg "$1" "${2:-}"; SPEC="$2"; shift 2 ;;
        --vision) VISION=1; shift ;;
        --prompt) need_arg "$1" "${2:-}"; PROMPT="$2"; shift 2 ;;
        --no-update) UPDATE=0; shift ;;
        --rebuild) REBUILD=1; shift ;;
        --full-depth) FULL_DEPTH=1; CURVE=1; shift ;;
        --fast-v100) FAST_V100=1; shift ;;
        --no-micromamba-cuda|--no-uv-cuda|--no-local-cuda) LOCAL_CUDA_INSTALL=0; shift ;;
        --no-cpu-last) CPU_LAST=-1; shift ;;
        --list-models) LIST_MODELS=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --version) printf '%s\n' "llama-pushbutton.sh $VERSION"; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        *) err "unknown argument: $1"; usage >&2; exit 2 ;;
    esac
done

if [ "$CPU_ONLY" -eq 1 ]; then
    BACKEND="cpu"
    CPU_LAST=-1
    [ "$MODEL_EXPLICIT" -eq 0 ] && MODEL="all"
    [ "$MODE_EXPLICIT" -eq 0 ] && MODE="sweep"
fi

if [ "$ORIG_ARGC" -eq 0 ]; then
    MODE="sweep"
    MODEL="all"
    [ "$CPU_LAST" -ne -1 ] && CPU_LAST=1
elif [ -z "$MODE" ]; then
    MODE="server"
fi

case "$BACKEND" in auto|cuda|cpu) ;; *) err "bad --backend: $BACKEND"; exit 2 ;; esac
case "$MODE" in sweep|bench|server|cli) ;; *) err "bad --mode: $MODE"; exit 2 ;; esac
case "$SPEC" in none|mtp) ;; *) err "bad --spec: $SPEC"; exit 2 ;; esac
case "$CPU_NUMA" in auto|disabled|distribute|isolate|numactl) ;; *) err "bad --cpu-numa: $CPU_NUMA"; exit 2 ;; esac
case "$CPU_PLACEMENT" in auto|mmap|interleave-none) ;; *) err "bad --cpu-placement: $CPU_PLACEMENT"; exit 2 ;; esac
case "$CPU_BLAS" in auto|on|off) ;; *) err "bad --cpu-blas: $CPU_BLAS"; exit 2 ;; esac
case "$CTX" in auto|max) ;; *) [[ "$CTX" =~ ^[0-9]+$ ]] || { err "--ctx must be auto|max|integer"; exit 2; } ;; esac
if [ "$CURVE_CAP" != "262144" ] && [ "$TARGET_CTX" = "262144" ]; then TARGET_CTX="$CURVE_CAP"; fi
case "$TARGET_CTX" in native) ;; *) [[ "$TARGET_CTX" =~ ^[0-9]+$ ]] && [ "$TARGET_CTX" -ge 4096 ] || { err "--target-ctx must be native or an integer >= 4096"; exit 2; } ;; esac
[[ "$MIN_CTX" =~ ^[0-9]+$ ]] && [ "$MIN_CTX" -ge 4096 ] || { err "--min-ctx must be an integer >= 4096"; exit 2; }
[[ "$CURVE_PP" =~ ^[0-9]+$ ]] && [ "$CURVE_PP" -ge 128 ] || { err "--probe-pp must be an integer >= 128"; exit 2; }
[[ "$CPU_PROBE_PP" =~ ^[0-9]+$ ]] && [ "$CPU_PROBE_PP" -ge 128 ] || { err "--cpu-probe-pp must be an integer >= 128"; exit 2; }
[[ "$CPU_TUNE_TG" =~ ^[0-9]+$ ]] && [ "$CPU_TUNE_TG" -ge 8 ] || { err "--cpu-tune-tg must be an integer >= 8"; exit 2; }
[[ "$CPU_TUNE_PP" =~ ^[0-9]+$ ]] && [ "$CPU_TUNE_PP" -ge 128 ] || { err "--cpu-tune-pp must be an integer >= 128"; exit 2; }
case "$CPU_ABLATIONS" in auto|off) ;; *) err "--cpu-ablations must be auto or off"; exit 2 ;; esac
[[ "$CPU_ABL_PP" =~ ^[0-9]+$ ]] && [ "$CPU_ABL_PP" -ge 64 ] || { err "--cpu-ablation-pp must be >= 64"; exit 2; }
[[ "$CPU_ABL_TG" =~ ^[0-9]+$ ]] && [ "$CPU_ABL_TG" -ge 16 ] || { err "--cpu-ablation-tg must be >= 16"; exit 2; }
[[ "$CPU_ABL_REPS" =~ ^[0-9]+$ ]] && [ "$CPU_ABL_REPS" -ge 2 ] || { err "--cpu-ablation-reps must be >= 2"; exit 2; }
[[ "$MEMBW_ARRAY_MIB" =~ ^[0-9]+$ ]] && [ "$MEMBW_ARRAY_MIB" -ge 64 ] || { err "--memory-array-mib must be an integer >= 64"; exit 2; }
[[ "$MEMBW_REPS" =~ ^[0-9]+$ ]] && [ "$MEMBW_REPS" -ge 3 ] || { err "--memory-reps must be an integer >= 3"; exit 2; }
if [ "$MEMBW_THREADS" != "auto" ] && ! [[ "$MEMBW_THREADS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    err "--memory-threads must be auto or a comma-separated list of positive integers"; exit 2
fi
[[ "$CURVE_TG" =~ ^[0-9]+$ ]] && [ "$CURVE_TG" -ge 1 ] || { err "--tg-probe must be an integer >= 1"; exit 2; }
[[ "$COMPACTION_EXTRA_MS" =~ ^[0-9]+([.][0-9]+)?$ ]] || { err "--compaction-extra-ms must be a nonnegative number"; exit 2; }
if [ "$TARGET_CTX" != "native" ] && [ "$MIN_CTX" -gt "$TARGET_CTX" ]; then err "--min-ctx cannot exceed --target-ctx"; exit 2; fi
if [ "$CURVE_PP_SIZES" != "auto" ]; then warn "--pp-sizes is deprecated and ignored: progressive PP is measured during the single cached-prefix fill"; fi

CACHE_DIR="$(mkdir -p "$CACHE_DIR" 2>/dev/null && cd "$CACHE_DIR" && pwd)" || { err "cannot create/use cache dir"; exit 1; }
WORK="$CACHE_DIR/.llama-pushbutton"
SRC="$WORK/llama.cpp"
RESULTS="$CACHE_DIR/llama-results"
mkdir -p "$WORK" "$RESULTS" "$CACHE_DIR/huggingface/hub"

# Keep all model/cache artifacts under the chosen root. Current llama.cpp checks
# LLAMA_CACHE before HF_HUB_CACHE, so point both at the SAME standard HF Hub
# cache; background huggingface_hub prefetches are then immediately reusable by -hf.
export HF_HOME="$CACHE_DIR/huggingface"
export HF_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export LLAMA_CACHE="$HF_HUB_CACHE"
export XDG_CACHE_HOME="$CACHE_DIR/.cache"
export UV_CACHE_DIR="$WORK/uv-cache"

if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# Self-identify every run so stale/copied scripts are obvious in pasted logs.
SELF_PATH="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || printf '%s' "${BASH_SOURCE[0]}")"
if command -v sha256sum >/dev/null 2>&1 && [ -f "$SELF_PATH" ]; then
    SELF_SHA256="$(sha256sum "$SELF_PATH" | awk '{print $1}')"
else
    SELF_SHA256="unavailable"
fi
printf 'llama-pushbutton.sh version=%s\n' "$VERSION" >&2
printf 'script_path=%s\n' "$SELF_PATH" >&2
printf 'script_sha256=%s\n' "$SELF_SHA256" >&2

model_meta() {
    # fields: alias, GGUF repo, filename prefix, total layers, native context, class,
    #         context-growing full-attention layers, KV heads, KV head dimension.
    # The last three fields are used only for a metadata-only *lower-bound* VRAM
    # preflight. Exact llama.cpp fitting still decides borderline candidates.
    local a="${1,,}"
    case "$a" in
        qwen3.8|qwen3.8:27b|qwen3.8-27b|qwen38)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "qwen3.8" "unsloth/Qwen3.8-27B-GGUF" "Qwen3.8-27B" "64" "262144" "dense" "16" "4" "256" ;;
        qwen3.6:27b|qwen3.6-27b|qwen36:27b|qwen3.6-27)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "qwen3.6:27b" "unsloth/Qwen3.6-27B-GGUF" "Qwen3.6-27B" "64" "262144" "dense" "16" "4" "256" ;;
        qwen3.6:36|qwen3.6:35b|qwen3.6:35b-a3b|qwen3.6-35b-a3b|qwen36)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "qwen3.6:35b" "unsloth/Qwen3.6-35B-A3B-GGUF" "Qwen3.6-35B-A3B" "40" "262144" "moe" "10" "2" "256" ;;
        nemotron|nemotron-3.5|nemotron-3.5-lightning|nemotron3.5)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "nemotron-3.5" "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF" "NVIDIA-Nemotron-3.5-Lightning-30B-A3B" "52" "262144" "moe" "6" "2" "128" ;;
        *) return 1 ;;
    esac
}

show_models() {
    printf '%-16s %-58s %s\n' ALIAS HF_REPO GGUF_PREFIX
    local m meta
    for m in qwen3.6:35b nemotron-3.5 qwen3.8 qwen3.6:27b; do
        meta="$(model_meta "$m")"
        printf '%-16s %-58s %s\n' "$(printf '%s' "$meta" | cut -f1)" "$(printf '%s' "$meta" | cut -f2)" "$(printf '%s' "$meta" | cut -f3)"
    done
}

if [ "$LIST_MODELS" -eq 1 ]; then show_models; exit 0; fi

have() { command -v "$1" >/dev/null 2>&1; }
wall_est() { printf 'WALL ESTIMATE: %s | %s\n' "$1" "$2" >&2; }

memory_thread_list() {
    if [ "$MEMBW_THREADS" != "auto" ]; then
        printf '%s\n' "$MEMBW_THREADS" | tr ',' '\n' | awk -v max="$logical" '$1>=1 && $1<=max && !seen[$1]++' | paste -sd, -
        return
    fi
    local half=$(( (physical + 1) / 2 ))
    printf '%s\n' 1 "$half" "$physical" "$logical" \
        | awk -v max="$logical" '$1>=1 && $1<=max && !seen[$1]++' | paste -sd, -
}

capture_dimm_inventory() {
    local raw="$1" out="$2" meta="$3" source="unavailable"
    : > "$out"
    printf 'locator\tbank_locator\tsize\ttype\tform_factor\tspeed\tconfigured_speed\tmanufacturer\tpart_number\trank\ttotal_width\tdata_width\n' > "$out"

    if have dmidecode; then
        if dmidecode --type 17 > "$raw" 2>/dev/null; then
            source="dmidecode"
        elif [ "$(id -u)" -eq 0 ] && dmidecode --type 17 > "$raw" 2>/dev/null; then
            source="dmidecode-root"
        elif have sudo && sudo -n dmidecode --type 17 > "$raw" 2>/dev/null; then
            source="sudo-n-dmidecode"
        fi
    fi

    if [[ "$source" == *dmidecode* ]]; then
        python3 - "$raw" "$out" "$meta" "$source" <<'PYDIMM'
import re, sys, csv
raw, out, meta, source = sys.argv[1:5]
text = open(raw, encoding="utf-8", errors="replace").read()
blocks = re.split(r'(?=Handle\s+0x[0-9A-Fa-f]+,\s+DMI type 17\b)', text)
rows=[]
for b in blocks:
    if not re.search(r'DMI type 17\b', b):
        continue
    fields={}
    for ln in b.splitlines():
        m=re.match(r'\s*([^:]+):\s*(.*?)\s*$', ln)
        if m:
            fields[m.group(1).strip()] = m.group(2).strip()
    size=fields.get("Size","")
    if not size or "No Module Installed" in size:
        continue
    cfg=(fields.get("Configured Memory Speed") or
         fields.get("Configured Clock Speed") or "")
    rows.append({
        "locator":fields.get("Locator",""),
        "bank_locator":fields.get("Bank Locator",""),
        "size":size,
        "type":fields.get("Type",""),
        "form_factor":fields.get("Form Factor",""),
        "speed":fields.get("Speed",""),
        "configured_speed":cfg,
        "manufacturer":fields.get("Manufacturer",""),
        "part_number":fields.get("Part Number","").strip(),
        "rank":fields.get("Rank",""),
        "total_width":fields.get("Total Width",""),
        "data_width":fields.get("Data Width",""),
    })
with open(out,"w",newline="",encoding="utf-8") as f:
    cols=["locator","bank_locator","size","type","form_factor","speed","configured_speed","manufacturer","part_number","rank","total_width","data_width"]
    w=csv.DictWriter(f,fieldnames=cols,delimiter="\t",lineterminator="\n")
    w.writeheader(); w.writerows(rows)

def size_mib(s):
    m=re.match(r'([\d.]+)\s*(MB|GB|TB)\b',s,re.I)
    if not m: return 0
    v=float(m.group(1)); u=m.group(2).upper()
    return int(round(v*({"MB":1,"GB":1024,"TB":1024*1024}[u])))

speeds=[]
for r in rows:
    m=re.search(r'(\d+)\s*(?:MT/s|MHz)',r["configured_speed"])
    if m: speeds.append(int(m.group(1)))
total=sum(size_mib(r["size"]) for r in rows)
rated=[]
for r in rows:
    m=re.search(r'(\d+)\s*(?:MT/s|MHz)',r["speed"])
    if m: rated.append(int(m.group(1)))
raw_min=min(speeds) if speeds else None
raw_max=max(speeds) if speeds else None
rated_med=sorted(rated)[len(rated)//2] if rated else None
inferred=""
if raw_min and raw_max and raw_min==raw_max and rated_med and abs(2*raw_min-rated_med) <= max(2,int(rated_med*0.03)):
    inferred=str(2*raw_min)
with open(meta,"w",encoding="utf-8") as f:
    f.write(f"dimm_info_source\t{source}\n")
    f.write(f"populated_dimms\t{len(rows)}\n")
    f.write(f"dimm_total_mib\t{total}\n")
    f.write(f"configured_speed_min_raw\t{raw_min if raw_min else ''}\n")
    f.write(f"configured_speed_max_raw\t{raw_max if raw_max else ''}\n")
    f.write(f"inferred_ddr_effective_mtps\t{inferred}\n")
PYDIMM
        return 0
    fi

    # Raw-only fallback when SMBIOS access is unavailable.
    if have lshw; then
        lshw -class memory -sanitize > "$raw" 2>/dev/null || true
        [ -s "$raw" ] && source="lshw-raw-only"
    else
        : > "$raw"
    fi
    {
        printf 'dimm_info_source\t%s\n' "$source"
        printf 'populated_dimms\t\n'
        printf 'dimm_total_mib\t\n'
        printf 'configured_speed_min_raw\t\n'
        printf 'configured_speed_max_raw\t\n'
        printf 'inferred_ddr_effective_mtps\t\n'
    } > "$meta"
    return 1
}

ensure_memory_bandwidth_tool() {
    # WALL <1s cached / seconds to compile: dependent locals must be assigned sequentially in Bash.
    local tool_dir csrc bin cc
    tool_dir="$WORK/tools"
    csrc="$tool_dir/host-membw-v1.c"
    bin="$tool_dir/host-membw-v1"
    mkdir -p "$tool_dir" || return 1
    if [ ! -s "$csrc" ]; then
        cat > "$csrc" <<'CMEM'
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_s(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC_RAW
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int cmp_double(const void *a, const void *b) {
    double x=*(const double*)a, y=*(const double*)b;
    return (x>y)-(x<y);
}
static void emit_stats(int threads, const char *kernel, int reps, int array_mib,
                       size_t n, double nominal_bytes, double *bw) {
    double sorted[128], mean=0.0, var=0.0, best=0.0;
    if (reps > 128) reps=128;
    for (int i=0;i<reps;i++) { sorted[i]=bw[i]; mean+=bw[i]; if (bw[i]>best) best=bw[i]; }
    mean/=reps;
    for (int i=0;i<reps;i++) { double d=bw[i]-mean; var+=d*d; }
    var = reps>1 ? var/(reps-1) : 0.0;
    qsort(sorted,reps,sizeof(double),cmp_double);
    double median = reps&1 ? sorted[reps/2] : 0.5*(sorted[reps/2-1]+sorted[reps/2]);
    double sd=sqrt(var), cv=mean ? 100.0*sd/mean : 0.0;
    printf("%d\t%s\t%d\t%d\t%zu\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
           threads,kernel,reps,array_mib,n,median,best,mean,sd,cv);
}
int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,"usage: %s THREADS ARRAY_MIB REPS\n",argv[0]); return 2;
    }
    int threads=atoi(argv[1]), array_mib=atoi(argv[2]), reps=atoi(argv[3]);
    if (threads<1 || array_mib<16 || reps<3 || reps>128) return 2;
    size_t bytes=(size_t)array_mib*1024u*1024u;
    size_t n=bytes/sizeof(double);
    double *a=NULL,*b=NULL,*c=NULL;
    if (posix_memalign((void**)&a,64,n*sizeof(double)) ||
        posix_memalign((void**)&b,64,n*sizeof(double)) ||
        posix_memalign((void**)&c,64,n*sizeof(double))) {
        fprintf(stderr,"allocation failed for 3 x %d MiB\n",array_mib); return 3;
    }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i=0;i<n;i++) { a[i]=1.0; b[i]=2.0+(double)(i&7)*1e-6; c[i]=0.5; }

    volatile double guard=0.0;
    double sum=0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static) num_threads(threads)
    for (size_t i=0;i<n;i++) sum += b[i];
    guard += sum;
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i=0;i<n;i++) a[i]=b[i];
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i=0;i<n;i++) a[i]=b[i]+3.0*c[i];
    guard += a[n/2];

    double bw[128];
    for (int r=0;r<reps;r++) {
        sum=0.0;
        double t0=wall_s();
        #pragma omp parallel for reduction(+:sum) schedule(static) num_threads(threads)
        for (size_t i=0;i<n;i++) sum += b[i];
        double dt=wall_s()-t0;
        bw[r]=(double)(n*sizeof(double))/dt/1e9;
        guard += sum*1e-300;
    }
    emit_stats(threads,"READ",reps,array_mib,n,(double)(n*sizeof(double)),bw);

    for (int r=0;r<reps;r++) {
        double t0=wall_s();
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (size_t i=0;i<n;i++) a[i]=b[i];
        double dt=wall_s()-t0;
        bw[r]=(double)(2*n*sizeof(double))/dt/1e9;
        guard += a[(size_t)r%n]*1e-300;
    }
    emit_stats(threads,"COPY",reps,array_mib,n,(double)(2*n*sizeof(double)),bw);

    for (int r=0;r<reps;r++) {
        double t0=wall_s();
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (size_t i=0;i<n;i++) a[i]=b[i]+3.0*c[i];
        double dt=wall_s()-t0;
        bw[r]=(double)(3*n*sizeof(double))/dt/1e9;
        guard += a[(size_t)(r*131u)%n]*1e-300;
    }
    emit_stats(threads,"TRIAD",reps,array_mib,n,(double)(3*n*sizeof(double)),bw);

    fprintf(stderr,"guard=%.17g\n",(double)guard);
    free(a); free(b); free(c);
    return 0;
}
CMEM
    fi

    if [ ! -x "$bin" ]; then
        cc=""
        have gcc && cc="$(command -v gcc)"
        [ -n "$cc" ] || { have cc && cc="$(command -v cc)"; }
        [ -n "$cc" ] || { warn "no C compiler available for host memory-bandwidth test"; return 1; }
        # stdout is reserved for the executable path because caller uses command substitution.
        info "Building tiny host memory-bandwidth helper (one-time)" >&2
        if ! "$cc" -O3 -march=native -mtune=native -ffast-math -fopenmp "$csrc" -lm -o "$bin"; then
            warn "could not compile OpenMP memory-bandwidth helper; memory bandwidth test skipped"
            rm -f "$bin"
            return 1
        fi
    fi
    printf '%s\n' "$bin"
}

run_memory_profile() {
    [ "$MEMORY_PROFILE" -eq 1 ] || return 0
    local summary="$RESULTS/memory-profile-$RUN_ID.tsv"
    local dimms="$RESULTS/memory-dimms-$RUN_ID.tsv"
    local dmi_raw="$RESULTS/memory-dmi-$RUN_ID.raw.txt"
    local dmi_meta="$RESULTS/memory-dmi-$RUN_ID.meta.tsv"
    local bw="$RESULTS/memory-bandwidth-$RUN_ID.tsv"

    # CUDA parent and CPU-last child share RUN_ID; benchmark only once.
    if [ -s "$summary" ] && [ -s "$bw" ]; then
        info "Host memory profile already exists for RUN_ID=$RUN_ID; reusing it"
        printf 'memory_profile=%s\nmemory_dimms=%s\nmemory_bandwidth=%s\n' "$summary" "$dimms" "$bw" >&2
        return 0
    fi

    info "Host DIMM inventory + sustained memory bandwidth"
    # WALL ~10-30s: 1T is cheap and separates core/cache speed from channel saturation.
    wall_est "~10-30 s typical" "DIMM inventory + 1/half/core/SMT READ/COPY/TRIAD"
    capture_dimm_inventory "$dmi_raw" "$dimms" "$dmi_meta" || true

    local avail_mib actual_array_mib="$MEMBW_ARRAY_MIB"
    avail_mib="$(awk '/MemAvailable:/ {printf "%d",$2/1024}' /proc/meminfo 2>/dev/null)"
    if [[ "$avail_mib" =~ ^[0-9]+$ ]] && [ "$avail_mib" -gt 0 ]; then
        local safe_per_array=$(( avail_mib / 8 ))
        if [ "$safe_per_array" -lt "$actual_array_mib" ]; then
            actual_array_mib="$safe_per_array"
        fi
    fi
    if [ "$actual_array_mib" -lt 64 ]; then
        warn "insufficient free RAM for >=64 MiB/array host-memory test; bandwidth test skipped"
        : > "$bw"
    else
        local tool=""
        if ! tool="$(ensure_memory_bandwidth_tool)"; then
            warn "host memory-bandwidth helper unavailable; skipping all bandwidth thread points"
            tool=""
        fi
        printf 'run_id\tscript_version\thost\tcpu\tphysical_cores\tlogical_threads\tnuma_nodes\tthreads\tkernel\treps\tarray_mib_per_array\telements\tmedian_GB_s\tbest_GB_s\tmean_GB_s\tsd_GB_s\tcv_percent\tplacement\n' > "$bw"
        if [ -n "$tool" ] && [ -x "$tool" ] && [ "$DRY_RUN" -eq 0 ]; then
            local tlist t prefix_desc="local"
            tlist="$(memory_thread_list)"
            local -a mprefix=()
            if [ "$NUMA_NODES" -gt 1 ] && have numactl; then
                mprefix=(numactl --interleave=all)
                prefix_desc="numactl-interleave-all"
            fi
            # WALL ~10-30s total: one validated helper serves all thread points.
            for t in ${tlist//,/ }; do
                [ "$t" -le "$logical" ] || continue
                tmpbw="$RESULTS/.memory-bandwidth-$RUN_ID.$t.tmp"
                printf 'MEMORY BW: threads=%s arrays=3x%sMiB reps=%s placement=%s\n' "$t" "$actual_array_mib" "$MEMBW_REPS" "$prefix_desc" >&2
                OMP_PROC_BIND=spread OMP_PLACES=cores "${mprefix[@]}" "$tool" "$t" "$actual_array_mib" "$MEMBW_REPS" > "$tmpbw" 2> >(tee "$tmpbw.stderr" >&2)
                rc=$?
                if [ "$rc" -eq 0 ]; then
                    while IFS=$'\t' read -r tt kernel reps arr n med best mean sd cv; do
                        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                            "$RUN_ID" "$VERSION" "$(hostname 2>/dev/null || printf unknown)" "$cpu_model" "$physical" "$logical" "$NUMA_NODES" \
                            "$tt" "$kernel" "$reps" "$arr" "$n" "$med" "$best" "$mean" "$sd" "$cv" "$prefix_desc" >> "$bw"
                    done < "$tmpbw"
                else
                    warn "host memory-bandwidth helper failed for threads=$t"
                fi
                rm -f "$tmpbw" "$tmpbw.stderr"
            done
        fi
    fi

    local board_vendor board_name bios_version mem_total_mib governor dmi_source dimm_count dimm_total_mib cfg_min cfg_max cfg_eff
    board_vendor="$(cat /sys/class/dmi/id/board_vendor 2>/dev/null || true)"
    board_name="$(cat /sys/class/dmi/id/board_name 2>/dev/null || true)"
    bios_version="$(cat /sys/class/dmi/id/bios_version 2>/dev/null || true)"
    mem_total_mib="$(awk '/MemTotal:/ {printf "%d",$2/1024}' /proc/meminfo 2>/dev/null)"
    governor="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true)"
    dmi_source="$(awk -F'\t' '$1=="dimm_info_source"{print $2}' "$dmi_meta" 2>/dev/null)"
    dimm_count="$(awk -F'\t' '$1=="populated_dimms"{print $2}' "$dmi_meta" 2>/dev/null)"
    dimm_total_mib="$(awk -F'\t' '$1=="dimm_total_mib"{print $2}' "$dmi_meta" 2>/dev/null)"
    cfg_min="$(awk -F'\t' '$1=="configured_speed_min_raw"{print $2}' "$dmi_meta" 2>/dev/null)"
    cfg_max="$(awk -F'\t' '$1=="configured_speed_max_raw"{print $2}' "$dmi_meta" 2>/dev/null)"
    cfg_eff="$(awk -F'\t' '$1=="inferred_ddr_effective_mtps"{print $2}' "$dmi_meta" 2>/dev/null)"

    printf 'run_id\tscript_version\thost\tcpu\tphysical_cores\tlogical_threads\tnuma_nodes\tmem_total_mib\tboard_vendor\tboard_name\tbios_version\tcpu_governor\tdimm_info_source\tpopulated_dimms\tdimm_total_mib\tconfigured_speed_min_raw\tconfigured_speed_max_raw\tinferred_ddr_effective_mtps\tbandwidth_array_mib_per_array\tbandwidth_reps\tbandwidth_threads\tbandwidth_file\tdimm_file\n' > "$summary"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$RUN_ID" "$VERSION" "$(hostname 2>/dev/null || printf unknown)" "$cpu_model" "$physical" "$logical" "$NUMA_NODES" "$mem_total_mib" \
        "$board_vendor" "$board_name" "$bios_version" "$governor" "$dmi_source" "$dimm_count" "$dimm_total_mib" "$cfg_min" "$cfg_max" "$cfg_eff" \
        "$actual_array_mib" "$MEMBW_REPS" "$(memory_thread_list)" "$bw" "$dimms" >> "$summary"

    printf '\nHOST MEMORY PROFILE:\n' >&2
    cat "$summary" >&2
    if [ -s "$dimms" ]; then
        printf '\nPOPULATED DIMMS:\n' >&2
        cat "$dimms" >&2
    fi
    if [ -s "$bw" ]; then
        printf '\nMEMORY BANDWIDTH (GB/s, requested bytes; median/best/mean/SD/CV):\n' >&2
        cat "$bw" >&2
    fi
    if [ "$dmi_source" = "unavailable" ] || [ "$dmi_source" = "lshw-raw-only" ] || [ -z "$dmi_source" ]; then
        warn "per-DIMM SMBIOS details unavailable. For slot/part/rank data, run 'sudo -v' once before the script; it will then use sudo -n dmidecode without prompting."
    fi
    printf 'memory_profile=%s\nmemory_dimms=%s\nmemory_bandwidth=%s\n' "$summary" "$dimms" "$bw" >&2
}

install_deps() {
    local missing=0 x
    for x in git cmake ninja curl jq python3 tar bzip2; do have "$x" || missing=1; done
    have c++ || missing=1
    [ "$missing" -eq 0 ] && return 0
    [ "$DRY_RUN" -eq 1 ] && { info "Would install build dependencies"; return 0; }

    info "Installing build dependencies"
    local SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        have sudo || { err "sudo is required to install missing packages"; return 1; }
        SUDO=sudo
    fi
    if have apt-get; then
        $SUDO apt-get update || return 1
        $SUDO apt-get install -y build-essential cmake ninja-build git curl jq python3 pkg-config ca-certificates bzip2 || return 1
    elif have dnf; then
        $SUDO dnf install -y gcc gcc-c++ make cmake ninja-build git curl jq python3 pkgconf-pkg-config ca-certificates bzip2 || return 1
    elif have pacman; then
        $SUDO pacman -Sy --needed --noconfirm base-devel cmake ninja git curl jq python3 pkgconf ca-certificates bzip2 || return 1
    else
        err "unsupported package manager; install git cmake ninja curl jq python3 tar bzip2 and a C++ compiler"
        return 1
    fi
}

# HF quant inventory deliberately happens before model downloads.  `-hf` on most
# llama.cpp tools (including llama-fit-params) invokes the common download handler,
# so using fit-params itself as the first gate can download a huge GGUF merely to
# discover that it cannot fit.  The Hub tree API gives us filename + byte size only.
hf_inventory() {
    local repo="$1" prefix="$2"
    local out="$WORK/hf-inventory-$(printf '%s' "$repo" | tr '/:' '__').tsv"
    if [ -s "$out" ]; then printf '%s\n' "$out"; return 0; fi
    local tmp="$out.tmp.$$"
    info "HF metadata inventory (no model weights): $repo" >&2
    HF_INV_REPO="$repo" HF_INV_PREFIX="$prefix" HF_INV_OUT="$tmp" python3 - <<'PYINV'
import json, os, re, sys, urllib.request, urllib.parse
repo=os.environ['HF_INV_REPO']; prefix=os.environ['HF_INV_PREFIX']; out=os.environ['HF_INV_OUT']
token=os.environ.get('HF_TOKEN','')
url=f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true&expand=false&limit=1000"
items=[]; pages=0
while url:
    req=urllib.request.Request(url, headers={"User-Agent":"llama-pushbutton/16"})
    if token: req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            data=json.load(r); link=r.headers.get('Link','')
    except Exception as e:
        print(f"ERROR: Hugging Face tree metadata request failed: {e}", file=sys.stderr); sys.exit(2)
    pages += 1
    if not isinstance(data, list):
        print("ERROR: unexpected Hugging Face tree response", file=sys.stderr); sys.exit(2)
    items.extend(data)
    nxt=None
    for part in link.split(','):
        if 'rel="next"' in part:
            m=re.search(r'<([^>]+)>', part)
            if m: nxt=urllib.parse.urljoin(url,m.group(1))
    url=nxt

groups={}
pre=prefix.lower()+'-'
for it in items:
    path=it.get('path') or it.get('rfilename') or ''
    base=path.rsplit('/',1)[-1]
    low=base.lower()
    if not low.endswith('.gguf'): continue
    if any(x in low for x in ('mmproj','mtp','dflash','eagle')): continue
    stem=base[:-5]
    if not stem.lower().startswith(pre): continue
    q=stem[len(prefix)+1:]
    q=re.sub(r'-\d{5}-of-\d{5}$','',q,flags=re.I)
    if not q: continue
    size=it.get('size')
    if size is None and isinstance(it.get('lfs'),dict): size=it['lfs'].get('size')
    try: size=int(size)
    except Exception: size=0
    k=q.lower()
    g=groups.setdefault(k,{'q':q,'size':0,'parts':0,'files':[]})
    g['size'] += size; g['parts'] += 1; g['files'].append(path)
if not groups:
    print(f"ERROR: no matching GGUF files found for prefix {prefix} in {repo}", file=sys.stderr); sys.exit(3)
with open(out,'w',encoding='utf-8') as f:
    for g in sorted(groups.values(), key=lambda x:(-x['size'],x['q'].lower())):
        f.write(f"{g['q']}\t{g['size']}\t{g['parts']}\t{';'.join(g['files'])}\n")
print(f"HF metadata: {len(groups)} quant(s), {pages} page(s); no GGUF payload downloaded", file=sys.stderr)
PYINV
    local rc=$?
    if [ "$rc" -ne 0 ]; then rm -f "$tmp"; return "$rc"; fi
    mv "$tmp" "$out"
    printf '%s\n' "$out"
}

available_quants() {
    local repo="$1" prefix="$2" inv
    inv="$(hf_inventory "$repo" "$prefix")" || return 1
    cut -f1 "$inv"
}

quant_size_bytes() {
    local repo="$1" prefix="$2" want="$3" inv
    inv="$(hf_inventory "$repo" "$prefix")" || return 1
    awk -F'\t' -v w="${want,,}" 'tolower($1)==w {print $2; exit}' "$inv"
}

quant_parts_count() {
    local repo="$1" prefix="$2" want="$3" inv
    inv="$(hf_inventory "$repo" "$prefix")" || return 1
    awk -F'\t' -v w="${want,,}" 'tolower($1)==w {print $3; exit}' "$inv"
}

resolve_quant() {
    local want="$1"; shift
    printf '%s\n' "$@" | awk -v w="${want,,}" 'tolower($0)==w {print; exit}'
}

if [ "$QUANT" = "list" ]; then
    install_deps || exit 1
    if [ "$MODEL" = "all" ]; then
        models=(qwen3.6:35b nemotron-3.5 qwen3.8 qwen3.6:27b)
    else
        models=("$MODEL")
    fi
    for m in "${models[@]}"; do
        meta="$(model_meta "$m")" || { err "unknown model alias: $m"; exit 2; }
        repo="$(printf '%s' "$meta" | cut -f2)"; prefix="$(printf '%s' "$meta" | cut -f3)"
        info "$m — $repo"
        available_quants "$repo" "$prefix" || exit 1
    done
    exit 0
fi

install_deps || exit 1

# Backend selection. GPU presence and compiler presence are deliberately separate:
# a machine with a working NVIDIA driver must never silently fall back to CPU merely
# because nvcc is absent from PATH. If needed, install a private CUDA 12.9.1 build
# toolkit with micromamba below the cache root. No conda executable, py-rattler,
# NVIDIA .run installer, driver install, shell activation, or /usr/local mutation.
# CUDA 12.9 covers V100/A100/RTX 3090.
gpu_probe_names=""
if have nvidia-smi; then
    if [ -n "$GPUS" ]; then
        gpu_probe_names="$(nvidia-smi -i "$GPUS" --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    else
        gpu_probe_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    fi
fi

info "Backend probe"
printf 'nvidia_smi=%s\n' "$(command -v nvidia-smi 2>/dev/null || printf 'NOT_FOUND')"
if [ -n "$gpu_probe_names" ]; then
    printf 'visible_nvidia_gpus:\n%s\n' "$gpu_probe_names" | sed '2,$s/^/  - /'
else
    printf 'visible_nvidia_gpus: NONE\n'
fi
printf 'nvcc_on_PATH=%s\n' "$(command -v nvcc 2>/dev/null || printf 'NOT_FOUND')"
printf 'micromamba_on_PATH=%s\n' "$(command -v micromamba 2>/dev/null || printf 'NOT_FOUND')"

if [ "$BACKEND" = "auto" ]; then
    if [ -n "$gpu_probe_names" ]; then
        BACKEND="cuda"
        printf 'auto_backend=cuda (NVIDIA GPU is visible; compiler will be located or provisioned with micromamba)\n'
    else
        BACKEND="cpu"
        printf 'auto_backend=cpu (no usable NVIDIA GPU reported by nvidia-smi)\n'
    fi
fi

get_micromamba() {
    local mm="$WORK/bin/micromamba"
    local platform url archive tmp

    if have micromamba; then
        command -v micromamba
        return 0
    fi
    if [ -x "$mm" ]; then
        printf '%s\n' "$mm"
        return 0
    fi

    case "$(uname -s 2>/dev/null)-$(uname -m 2>/dev/null)" in
        Linux-x86_64)  platform="linux-64" ;;
        Linux-aarch64|Linux-arm64) platform="linux-aarch64" ;;
        Linux-ppc64le) platform="linux-ppc64le" ;;
        *)
            err "No micromamba on PATH and automatic micromamba bootstrap does not support $(uname -s)-$(uname -m)."
            return 1 ;;
    esac

    have curl || { err "curl is required to bootstrap micromamba"; return 1; }
    have tar  || { err "tar is required to bootstrap micromamba"; return 1; }

    mkdir -p "$WORK/bin" "$WORK/downloads"
    url="https://micro.mamba.pm/api/micromamba/$platform/latest"
    archive="$WORK/downloads/micromamba-$platform.tar.bz2"
    tmp="$(mktemp -d "$WORK/micromamba-extract.XXXXXX")" || return 1

    printf '\n==> micromamba is not installed; fetching the official standalone micromamba binary\n' >&2
    printf 'micromamba_url=%s\n' "$url" >&2
    printf 'micromamba_path=%s\n' "$mm" >&2
    print_cmd curl -fL --retry 3 "$url" -o "$archive"
    if ! curl -fL --retry 3 "$url" -o "$archive"; then
        rm -rf "$tmp"
        return 1
    fi
    print_cmd tar -xjf "$archive" -C "$tmp" bin/micromamba
    if ! tar -xjf "$archive" -C "$tmp" bin/micromamba; then
        rm -rf "$tmp"
        return 1
    fi
    cp "$tmp/bin/micromamba" "$mm" || { rm -rf "$tmp"; return 1; }
    chmod 0755 "$mm" || { rm -rf "$tmp"; return 1; }
    rm -rf "$tmp"

    [ -x "$mm" ] || { err "micromamba bootstrap completed but $mm is not executable"; return 1; }
    "$mm" --version >&2 || return 1
    printf '%s\n' "$mm"
}

bootstrap_micromamba_cuda() {
    local root="$WORK/cuda-12.9.1"
    local mm
    local channel="https://conda.anaconda.org/nvidia/label/cuda-12.9.1"

    if [ -x "$root/bin/nvcc" ]; then
        CUDA_ENV_ROOT="$root"
        printf '%s\n' "$root/bin/nvcc"
        return 0
    fi

    mm="$(get_micromamba)" || {
        err "CUDA GPU is visible but nvcc is absent and micromamba could not be located/provisioned."
        err "Provide micromamba or a system CUDA 12.x toolkit. No .run-installer or CPU-fallback path will be used."
        return 1
    }
    MICROMAMBA_BIN="$mm"
    export MAMBA_ROOT_PREFIX="$WORK/micromamba-root"
    export MAMBARC="$WORK/mambarc"
    mkdir -p "$MAMBA_ROOT_PREFIX"
    : > "$MAMBARC"

    printf '\n==> CUDA compiler is missing; provisioning private CUDA 12.9.1 with micromamba\n' >&2
    printf 'micromamba=%s\n' "$mm" >&2
    printf 'mamba_root_prefix=%s\n' "$MAMBA_ROOT_PREFIX" >&2
    printf 'mambarc=%s (empty; ignores user channel configuration)\n' "$MAMBARC" >&2
    printf 'cuda_prefix=%s\n' "$root" >&2
    printf 'cuda_channel=%s\n' "$channel" >&2
    printf 'NOTE: no conda executable, no py-rattler, no .run file, no driver install, no sudo, no shell activation.\n' >&2

    # --override-channels prevents any ~/.condarc/default channels from changing
    # the solve. The NVIDIA versioned label pins the CUDA family; conda-forge only
    # supplies ordinary host dependencies if required.
    local -a cmd=("$mm" create -y -p "$root" --override-channels
        -c "$channel" -c conda-forge
        "cuda-minimal-build=12.9.1" "libcublas-dev=12.9.1.4")
    print_cmd "${cmd[@]}"
    "${cmd[@]}" 1>&2 || return 1

    if [ ! -x "$root/bin/nvcc" ]; then
        err "micromamba CUDA installation completed, but $root/bin/nvcc is unavailable."
        err "Refusing to hide the environment contents or fall back to CPU."
        printf '%s\n' '----- micromamba package list -----' >&2
        "$mm" list -p "$root" >&2 || true
        printf '%s\n' '----- candidate nvcc paths -----' >&2
        find "$root" \( -type f -o -type l \) -name nvcc 2>/dev/null | sort >&2 || true
        printf '%s\n' '----- top-level CUDA prefix -----' >&2
        find "$root" -maxdepth 2 -mindepth 1 -print 2>/dev/null | sort | head -n 200 >&2 || true
        return 1
    fi

    printf 'nvcc=%s\n' "$root/bin/nvcc" >&2
    "$root/bin/nvcc" --version >&2 || return 1
    CUDA_ENV_ROOT="$root"
    printf '%s\n' "$root/bin/nvcc"
}

find_nvcc() {
    local want12="${1:-0}" x
    # V100 must use CUDA 12.x. Prefer the private reproducible toolkit, then
    # already-installed system toolkits.
    if [ "$want12" -eq 1 ]; then
        for x in "$WORK"/cuda-12.9.1/bin/nvcc /usr/local/cuda-12*/bin/nvcc /opt/cuda-12*/bin/nvcc; do
            [ -x "$x" ] && { printf '%s\n' "$x"; return 0; }
        done
    fi
    if have nvcc; then
        x="$(command -v nvcc)"
        if [ "$want12" -eq 0 ] || "$x" --version 2>/dev/null | grep -q 'release 12\.'; then
            printf '%s\n' "$x"; return 0
        fi
    fi
    for x in "$WORK"/cuda-12.9.1/bin/nvcc /usr/local/cuda-*/bin/nvcc /opt/cuda*/bin/nvcc; do
        [ -x "$x" ] || continue
        if [ "$want12" -eq 0 ] || "$x" --version 2>/dev/null | grep -q 'release 12\.'; then
            printf '%s\n' "$x"; return 0
        fi
    done
    return 1
}

if [ "$BACKEND" = "cuda" ]; then
    have nvidia-smi || { err "CUDA backend selected but nvidia-smi is unavailable"; exit 1; }
    [ -n "$gpu_probe_names" ] || { err "CUDA backend selected but nvidia-smi reports no usable selected GPU"; exit 1; }
    want_cuda12=0
    printf '%s\n' "$gpu_probe_names" | grep -qi 'V100' && want_cuda12=1
    NVCC_BIN="$(find_nvcc "$want_cuda12" || true)"
    if [ -z "$NVCC_BIN" ] && [ "$LOCAL_CUDA_INSTALL" -eq 1 ]; then
        NVCC_BIN="$(bootstrap_micromamba_cuda)" || exit 1
    fi
    if [ -z "$NVCC_BIN" ]; then
        err "NVIDIA GPU detected, but no usable nvcc was found. Refusing to silently benchmark CPU."
        err "Allow the private micromamba CUDA bootstrap (default), provide a system CUDA 12.x toolkit, or use --backend cpu explicitly."
        exit 1
    fi
    cuda_root_pre="$(cd "$(dirname "$NVCC_BIN")/.." && pwd)"
    cuda_target_pre="$cuda_root_pre/targets/x86_64-linux"
    cuda_lib_path="$cuda_root_pre/lib64:$cuda_root_pre/lib:$cuda_target_pre/lib:$cuda_target_pre/lib/stubs"
    cuda_inc_path="$cuda_root_pre/include:$cuda_target_pre/include"
    export PATH="$(dirname "$NVCC_BIN"):$PATH"
    export CUDA_HOME="$cuda_root_pre"
    export CUDA_PATH="$cuda_root_pre"
    export CUDACXX="$NVCC_BIN"
    export LD_LIBRARY_PATH="$cuda_lib_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$cuda_lib_path${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CPATH="$cuda_inc_path${CPATH:+:$CPATH}"
    printf 'selected_nvcc=%s\n' "$NVCC_BIN"
    printf 'cuda_root=%s\n' "$cuda_root_pre"
    "$NVCC_BIN" --version | tail -n 2 || true
fi

cpu_model="$(awk -F: '/model name/ {sub(/^[ \t]+/,"",$2); print $2; exit}' /proc/cpuinfo 2>/dev/null)"
logical="$(nproc 2>/dev/null || echo 1)"
physical="$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
[ "$physical" -gt 0 ] 2>/dev/null || physical="$logical"

[ "$THREADS" = "auto" ] && THREADS_WAS_AUTO=1
[ "$THREADS_BATCH" = "auto" ] && THREADS_BATCH_WAS_AUTO=1
[ "$BATCH" = "auto" ] && BATCH_WAS_AUTO=1
[ "$UBATCH" = "auto" ] && UBATCH_WAS_AUTO=1

if [ "$THREADS" = "auto" ]; then
    case "$cpu_model" in
        *"Threadripper 1920X"*) THREADS=12 ;;
        *) THREADS="$physical" ;;
    esac
fi
if [ "$THREADS_BATCH" = "auto" ]; then
    case "$cpu_model" in
        *"Threadripper 1920X"*) THREADS_BATCH=24 ;;
        *) THREADS_BATCH="$logical" ;;
    esac
fi
if [ "$BATCH" = "auto" ]; then BATCH=2048; fi
if [ "$UBATCH" = "auto" ]; then UBATCH=512; fi

NUMA_NODES="$(find /sys/devices/system/node -maxdepth 1 -type d -name 'node[0-9]*' 2>/dev/null | wc -l)"
[ "${NUMA_NODES:-0}" -gt 0 ] 2>/dev/null || NUMA_NODES=1
if [ "$BACKEND" = "cpu" ]; then
    case "$cpu_model" in
        *"Threadripper 19"*"X"*|*"Threadripper 29"*"X"*)
            if [ "$NUMA_NODES" -eq 1 ]; then
                warn "multi-die Threadripper is exposed by firmware/Linux as ONE NUMA node; llama.cpp --numa cannot split topology the OS does not expose. This can be intentional UMA/distributed-memory firmware mode."
            fi ;;
    esac
    case "$CPU_NUMA" in
        auto)
            if [ "$NUMA_NODES" -gt 1 ]; then CPU_NUMA_EFFECTIVE="distribute"; else CPU_NUMA_EFFECTIVE="disabled"; fi ;;
        *) CPU_NUMA_EFFECTIVE="$CPU_NUMA" ;;
    esac

    # Benchmark baseline: BLAS stays off unless explicitly requested. It can help
    # some PP paths but is not universally neutral for CPU inference.
    case "$CPU_BLAS" in
        on)
            if have pkg-config && { pkg-config --exists openblas 2>/dev/null || pkg-config --exists openblas64 2>/dev/null; }; then
                CPU_BLAS_EFFECTIVE="on"
            else
                err "--cpu-blas on requested, but pkg-config cannot find OpenBLAS development files"
                exit 1
            fi ;;
        auto|off) CPU_BLAS_EFFECTIVE="off" ;;
    esac

    if [ "$CPU_ABLATIONS" = "auto" ] && ! { have pkg-config && { pkg-config --exists openblas 2>/dev/null || pkg-config --exists openblas64 2>/dev/null; }; }; then
        # WALL ~1-3m first CPU run only: install the small dev package so the OpenBLAS ablation is real.
        wall_est "~1-3 min first CPU run only" "install OpenBLAS development package for PP ablation"
        SUDO_OB=""
        if [ "$(id -u)" -ne 0 ]; then
            if have sudo; then SUDO_OB=sudo; else warn "sudo missing; OpenBLAS ablation will be skipped"; fi
        fi
        if [ "$(id -u)" -eq 0 ] || [ -n "$SUDO_OB" ]; then
            if have apt-get; then
                $SUDO_OB apt-get update && $SUDO_OB apt-get install -y libopenblas-dev || warn "OpenBLAS install failed"
            elif have dnf; then
                $SUDO_OB dnf install -y openblas-devel || warn "OpenBLAS install failed"
            elif have pacman; then
                $SUDO_OB pacman -Sy --needed --noconfirm openblas || warn "OpenBLAS install failed"
            else
                warn "package manager unsupported for automatic OpenBLAS install"
            fi
        fi
    fi

    case "$CPU_PLACEMENT" in
        auto)
            if [ "$NUMA_NODES" -gt 1 ] && have numactl && [ "$CPU_NUMA_EFFECTIVE" = "distribute" ]; then
                CPU_PLACEMENT_EFFECTIVE="interleave-none"
            else
                CPU_PLACEMENT_EFFECTIVE="mmap"
            fi ;;
        *) CPU_PLACEMENT_EFFECTIVE="$CPU_PLACEMENT" ;;
    esac
    if [ "$CPU_PLACEMENT_EFFECTIVE" = "interleave-none" ]; then
        if ! have numactl; then
            warn "--cpu-placement interleave-none requested but numactl is unavailable; falling back to mmap"
            CPU_PLACEMENT_EFFECTIVE="mmap"
        else
            CPU_LOAD_MODE_EFFECTIVE="none"
        fi
    fi
    if [ "$CPU_PLACEMENT_EFFECTIVE" = "mmap" ] && [ "$NUMA_NODES" -gt 1 ] && [ "$CPU_NUMA_EFFECTIVE" = "distribute" ]; then
        warn "CPU uses mmap+NUMA distribute. Previously cached GGUF pages may retain nonideal placement; interleave-none avoids this when numactl is available."
    fi

    [ "$PROBE_PP_EXPLICIT" -eq 0 ] && CURVE_PP="$CPU_PROBE_PP"
fi

GPU_COUNT=0
TOTAL_VRAM=0
FREE_VRAM=0
ARCHS=""
HAS_V100=0
# NVCC_BIN is selected during backend probing/bootstrap above.
NVCC_BIN="${NVCC_BIN:-}"

if [ "$BACKEND" = "cuda" ]; then
    smi_query() {
        local field="$1"
        if [ -n "$GPUS" ]; then
            nvidia-smi -i "$GPUS" --query-gpu="$field" --format=csv,noheader,nounits 2>/dev/null
        else
            nvidia-smi --query-gpu="$field" --format=csv,noheader,nounits 2>/dev/null
        fi
    }
    GPU_COUNT="$(smi_query name | wc -l)"
    [ "$GPU_COUNT" -gt 0 ] || { err "nvidia-smi reports no selected GPUs"; exit 1; }
    TOTAL_VRAM="$(smi_query memory.total | awk '{s+=$1} END{printf "%d",s}')"
    FREE_VRAM="$(smi_query memory.free | awk '{s+=$1} END{printf "%d",s}')"
    [ "${FREE_VRAM:-0}" -gt 0 ] 2>/dev/null || FREE_VRAM="$TOTAL_VRAM"

    USED_VRAM=$(( TOTAL_VRAM - FREE_VRAM ))
    [ "$USED_VRAM" -ge 0 ] || USED_VRAM=0

    gpu_compute_processes() {
        if [ -n "$GPUS" ]; then
            nvidia-smi -i "$GPUS" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true
        else
            nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true
        fi
    }

    # This script has not loaded a model yet. Substantial use here is external,
    # so classify it as GPU_BUSY instead of corrupting metadata feasibility.
    busy_threshold_mib=$(( TOTAL_VRAM * GPU_BUSY_MIN_PCT / 100 ))
    [ "$busy_threshold_mib" -lt "$GPU_BUSY_MIN_MIB" ] && busy_threshold_mib="$GPU_BUSY_MIN_MIB"
    if [ "$GPU_BUSY_ABORT" -eq 1 ] && [ "$USED_VRAM" -gt "$busy_threshold_mib" ]; then
        printf '\nGPU BUSY: selected GPU(s) have %s MiB / %s MiB already in use; only %s MiB is free.\n' \
            "$USED_VRAM" "$TOTAL_VRAM" "$FREE_VRAM" >&2
        printf 'This run has not loaded a model yet: this is pre-existing occupancy, NOT a model-fit result.\n' >&2
        printf 'Compute processes reported by nvidia-smi:\n' >&2
        procs="$(gpu_compute_processes)"
        if [ -n "$procs" ]; then
            printf '  PID\tPROCESS\tUSED_MiB\n' >&2
            printf '%s\n' "$procs" | sed 's/, */\t/g; s/^/  /' >&2
        else
            printf '  (none reported; graphics/display or another NVIDIA client may hold VRAM)\n' >&2
        fi
        if have ollama; then
            printf '\nOllama residency (if any):\n' >&2
            ollama ps 2>/dev/null >&2 || true
            printf 'If Ollama owns the VRAM, unload it with: ollama stop <model-name>\n' >&2
        fi
        printf '\nFree the GPU and rerun. Use --allow-busy-gpu only if contention is intentional.\n' >&2
        exit 3
    fi
    caps="$(smi_query compute_cap)"
    if [ -z "$caps" ]; then
        caps="$(smi_query name | awk '
            /V100/ {print "7.0"; next} /A100/ {print "8.0"; next}
            /RTX 3090/ {print "8.6"; next}')"
    fi
    if printf '%s\n' "$caps" | grep -q '^7\.0'; then HAS_V100=1; fi
    ARCHS="$(printf '%s\n' "$caps" | awk -F. 'NF>=2 {print $1 $2}' | sort -un | paste -sd';' -)"
    [ -n "$ARCHS" ] || ARCHS="native"

    [ -x "$NVCC_BIN" ] || { err "selected nvcc vanished or is not executable: $NVCC_BIN"; exit 1; }
    nvcc_major="$($NVCC_BIN --version 2>/dev/null | sed -nE 's/.*release ([0-9]+)\..*/\1/p' | tail -1)"
    if [ "$HAS_V100" -eq 1 ] && [ "${nvcc_major:-0}" -ge 13 ]; then
        alt_nvcc="$(find_nvcc 1 || true)"
        if [ -z "$alt_nvcc" ] && [ "$LOCAL_CUDA_INSTALL" -eq 1 ]; then alt_nvcc="$(bootstrap_micromamba_cuda)" || exit 1; fi
        [ -n "$alt_nvcc" ] || { err "V100 requires CUDA 12.x; no CUDA 12 nvcc is available"; exit 1; }
        NVCC_BIN="$alt_nvcc"
        cuda_root_pre="$(cd "$(dirname "$NVCC_BIN")/.." && pwd)"
        cuda_target_pre="$cuda_root_pre/targets/x86_64-linux"
        cuda_lib_path="$cuda_root_pre/lib64:$cuda_root_pre/lib:$cuda_target_pre/lib:$cuda_target_pre/lib/stubs"
        cuda_inc_path="$cuda_root_pre/include:$cuda_target_pre/include"
        export PATH="$(dirname "$NVCC_BIN"):$PATH"
        export CUDA_HOME="$cuda_root_pre"
        export CUDA_PATH="$cuda_root_pre"
        export CUDACXX="$NVCC_BIN"
        export LD_LIBRARY_PATH="$cuda_lib_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export LIBRARY_PATH="$cuda_lib_path${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export CPATH="$cuda_inc_path${CPATH:+:$CPATH}"
        info "V100 detected: using CUDA 12.x compiler $NVCC_BIN"
    fi
    [ "$GPU_COUNT" -gt 1 ] && export CUDA_SCALE_LAUNCH_QUEUES=4x
    [ "$FAST_V100" -eq 1 ] && [ "$HAS_V100" -eq 1 ] && export GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1
fi

# Host memory topology/bandwidth is independent of llama.cpp and is measured
# before model inference. A shared RUN_ID prevents re-running it in CPU-last child.
run_memory_profile

# Clone/update and select a reproducible source revision.
if [ ! -d "$SRC/.git" ]; then
    info "Cloning llama.cpp into $SRC"
    [ "$DRY_RUN" -eq 1 ] || git clone https://github.com/ggml-org/llama.cpp.git "$SRC" || exit 1
fi

if [ "$DRY_RUN" -eq 0 ]; then
    if [ "$LLAMA_REF" = "auto" ] && [ "$BACKEND" = "cpu" ]; then
        # WALL ~0s: b10428 is pinned, so skip network fetch when commit already exists locally.
        TARGET_REF="885c5bb"
        if [ "$UPDATE" -eq 1 ] && ! git -C "$SRC" cat-file -e "$TARGET_REF^{commit}" 2>/dev/null; then
            info "Fetching llama.cpp (CPU pin not present locally)"
            git -C "$SRC" fetch --prune origin || exit 1
        fi
    else
        if [ "$UPDATE" -eq 1 ]; then
            info "Fetching llama.cpp"
            git -C "$SRC" fetch --prune origin || exit 1
        fi
    fi
    if [ "$LLAMA_REF" = "auto" ]; then
        if [ "$BACKEND" = "cpu" ]; then
            TARGET_REF="885c5bb"
        else
            TARGET_REF="origin/master"
        fi
    else
        TARGET_REF="$LLAMA_REF"
    fi
    git -C "$SRC" checkout -q --detach "$TARGET_REF" || { err "cannot checkout llama.cpp ref $TARGET_REF"; exit 1; }
    COMMIT="$(git -C "$SRC" rev-parse --short=10 HEAD)"
else
    TARGET_REF="$LLAMA_REF"
    COMMIT="dryrun"
fi

build_tag="$BACKEND-$COMMIT-statictools"
[ "$BACKEND" = "cpu" ] && build_tag="$build_tag-blas${CPU_BLAS_EFFECTIVE}"
[ "$BACKEND" = "cuda" ] && build_tag="$build_tag-$(printf '%s' "$ARCHS" | tr ';' '_')"
BUILD="$WORK/build-$build_tag"

if [ "$REBUILD" -eq 1 ] && [ -d "$BUILD" ]; then
    info "Removing $BUILD"
    [ "$DRY_RUN" -eq 1 ] || rm -rf "$BUILD"
fi

required_targets=(llama-fit-params llama-bench)
required_bins=("$BUILD/bin/llama-fit-params" "$BUILD/bin/llama-bench")
if [ "$MODE" = "server" ]; then
    required_targets+=(llama-server)
    required_bins+=("$BUILD/bin/llama-server")
elif [ "$MODE" = "cli" ]; then
    required_targets+=(llama-cli)
    required_bins+=("$BUILD/bin/llama-cli")
fi
need_build=0
for b in "${required_bins[@]}"; do [ -x "$b" ] || need_build=1; done

if [ "$need_build" -eq 1 ]; then
    info "Configuring llama.cpp ($BACKEND, ref=$TARGET_REF, commit=$COMMIT)"
    # WALL first build only: static host tools avoid libllama-*-impl.so loader failures.
    cmake_args=(-S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF)
    if [ "$BACKEND" = "cuda" ]; then
        cuda_root="$(cd "$(dirname "$NVCC_BIN")/.." && pwd)"
        cmake_args+=(
            -DGGML_CUDA=ON
            "-DCMAKE_CUDA_ARCHITECTURES=$ARCHS"
            "-DCMAKE_CUDA_COMPILER=$NVCC_BIN"
            "-DCUDAToolkit_ROOT=$cuda_root"
            "-DCMAKE_PREFIX_PATH=$cuda_root;$cuda_root/targets/x86_64-linux"
            "-DCMAKE_BUILD_RPATH=$cuda_root/lib64;$cuda_root/lib;$cuda_root/targets/x86_64-linux/lib"
            -DGGML_CUDA_FA_ALL_QUANTS=ON
            -DGGML_CUDA_PEER_MAX_BATCH_SIZE=8192
        )
    else
        cmake_args+=(-DGGML_CUDA=OFF)
        if [ "$CPU_BLAS_EFFECTIVE" = "on" ]; then
            cmake_args+=(-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS)
        fi
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'cmake'; printf ' %q' "${cmake_args[@]}"; printf '\n'
    else
        cmake "${cmake_args[@]}" || exit 1
        info "Building required llama.cpp targets: ${required_targets[*]}"
        # WALL benchmark sweep avoids server/CLI/Svelte UI; interactive tools build only on demand.
        cmake --build "$BUILD" -j "$logical" --target "${required_targets[@]}" || exit 1
    fi
fi

FIT="$BUILD/bin/llama-fit-params"
BENCH="$BUILD/bin/llama-bench"
SERVER="$BUILD/bin/llama-server"
CLI="$BUILD/bin/llama-cli"

audit_runtime_binary() {
    local b="$1" name lddout rc
    name="$(basename "$b")"
    [ -x "$b" ] || { err "missing built binary: $b"; return 1; }

    # WALL <1s/binary: catch unresolved loader dependencies before downloading GGUFs.
    if have ldd; then
        lddout="$(ldd "$b" 2>&1 || true)"
        if printf '%s\n' "$lddout" | grep -q 'not found'; then
            err "ABI audit failed for $name: unresolved shared library"
            printf '%s\n' "$lddout" >&2
            return 1
        fi
    fi

    "$b" --help >/dev/null 2>"$RESULTS/.abi-$RUN_ID-$name.stderr"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        err "runtime smoke test failed for $name (rc=$rc)"
        cat "$RESULTS/.abi-$RUN_ID-$name.stderr" >&2 || true
        return 1
    fi
    rm -f "$RESULTS/.abi-$RUN_ID-$name.stderr"
    return 0
}

if [ "$DRY_RUN" -eq 0 ]; then
    # WALL <5s total: audit exactly the binaries this mode needs before downloading GGUFs.
    wall_est "<5 s" "llama.cpp ABI/runtime audit before model payload downloads"
    for b in "${required_bins[@]}"; do
        audit_runtime_binary "$b" || {
            err "llama.cpp build is unusable; aborting before model planning/downloads"
            err "build_dir=$BUILD"
            exit 1
        }
    done
fi

info "Hardware profile"
printf 'backend=%s cpu=%s threads=%s/%s batch=%s ubatch=%s\n' "$BACKEND" "${cpu_model:-unknown}" "$THREADS" "$THREADS_BATCH" "$BATCH" "$UBATCH"
if [ "$CPU_ONLY" -eq 1 ]; then
    printf 'cpu_only=YES workflow=independent-model-quant-kv-sweep deep=%s\n' "$CPU_DEEP"
fi
if [ "$BACKEND" = "cpu" ]; then
    printf 'cpu_numa_nodes=%s cpu_numa=%s cpu_placement=%s cpu_load_mode=%s cpu_tune=%s cpu_probe_pp=%s cpu_blas=%s\n' "$NUMA_NODES" "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" "$CPU_LOAD_MODE_EFFECTIVE" "$CPU_TUNE" "$CURVE_PP" "$CPU_BLAS_EFFECTIVE"
fi
if [ "$BACKEND" = "cuda" ]; then
    printf 'gpus=%s total_vram_MiB=%s free_vram_MiB=%s used_vram_MiB=%s cuda_arches=%s nvcc=%s cuda_root=%s\n' "$GPU_COUNT" "$TOTAL_VRAM" "$FREE_VRAM" "${USED_VRAM:-0}" "$ARCHS" "$NVCC_BIN" "${cuda_root_pre:-unknown}"
fi
printf 'cache_root=%s llama_ref=%s commit=%s\n' "$CACHE_DIR" "$TARGET_REF" "$COMMIT"
if [ "$BACKEND" = "cuda" ]; then
    printf 'planning_vram_MiB=%s runtime_free_vram_MiB=%s\n' "$TOTAL_VRAM" "$FREE_VRAM"
fi
RUN_MANIFEST="$RESULTS/run-$RUN_ID-$BACKEND.manifest.tsv"
printf 'run_id\tscript_version\tbackend\tcommit\tcpu\tphysical_cores\tlogical_threads\tnuma_nodes\tcpu_numa\tcpu_placement\tcpu_load_mode\tcpu_blas\tthreads\tthreads_batch\tbatch\tubatch\n' > "$RUN_MANIFEST"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$RUN_ID" "$VERSION" "$BACKEND" "$COMMIT" "${cpu_model:-unknown}" "$physical" "$logical" "$NUMA_NODES" "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" "$CPU_LOAD_MODE_EFFECTIVE" "$CPU_BLAS_EFFECTIVE" "$THREADS" "$THREADS_BATCH" "$BATCH" "$UBATCH" >> "$RUN_MANIFEST"
printf 'run_manifest=%s\n' "$RUN_MANIFEST" >&2

# In sweep mode, auto considers the live HF inventory in descending GGUF byte size.
# The metadata-only VRAM prefilter below rejects impossible candidates before any
# model payload download. We stop after a few successful quant families, rather than
# hard-coding Q4 candidates that might all be too large at 262k on a 24 GiB GPU.
auto_quants() {
    local repo="$1" prefix="$2"; shift 2
    local -a avail=("$@")

    if [ "$MODE" = "sweep" ] && [ "$BACKEND" != "cpu" ]; then
        local inv
        inv="$(hf_inventory "$repo" "$prefix")" || return 1
        cut -f1 "$inv"
        return 0
    fi

    local -a prefs
    if [ "$BACKEND" = "cpu" ]; then
        # Quality-oriented CPU ladder with conventional K-quants first.
        prefs=(
            Q6_K UD-Q6_K
            Q5_K_M UD-Q5_K_M Q5_K_S UD-Q5_K_S
            Q4_K_M UD-Q4_K_M UD-Q4_K_XL Q4_K_S UD-Q4_K_S Q4_1 Q4_0
            UD-Q3_K_XL Q3_K_M UD-Q3_K_M Q3_K_S UD-Q3_K_S
            MXFP4_MOE UD-IQ4_NL IQ4_NL UD-IQ4_XS IQ4_XS Q8_0
        )
    else
        prefs=(UD-Q4_K_XL UD-Q4_K_M Q4_K_M Q5_K_M Q5_K_S Q6_K Q8_0 Q4_K_S Q3_K_M Q3_K_S)
    fi

    local p q picked=0
    declare -A seen=()
    for p in "${prefs[@]}"; do
        q="$(resolve_quant "$p" "${avail[@]}")"; [ -n "$q" ] || continue
        [ -n "${seen[${q,,}]:-}" ] && continue
        seen[${q,,}]=1
        printf '%s\n' "$q"
        picked=$((picked+1))
        if [ "$BACKEND" = "cpu" ]; then
            [ "$picked" -ge 10 ] && break
        elif [ "$MODE" != "sweep" ]; then
            [ "$picked" -ge 3 ] && break
        fi
    done
    if [ "$BACKEND" = "cpu" ] && [ "$picked" -lt 10 ]; then
        for q in "${avail[@]}"; do
            [ -n "${seen[${q,,}]:-}" ] && continue
            printf '%s\n' "$q"
            picked=$((picked+1)); [ "$picked" -ge 10 ] && break
        done
    fi
    [ "$picked" -gt 0 ] || { [ "${#avail[@]}" -gt 0 ] && printf '%s\n' "${avail[0]}"; }
}

quant_set() {
    local repo="$1" prefix="$2" spec="$3"
    mapfile -t aq < <(available_quants "$repo" "$prefix")
    [ "${#aq[@]}" -gt 0 ] || { err "no model GGUF quants discovered for $repo"; return 1; }
    case "${spec,,}" in
        auto) auto_quants "$repo" "$prefix" "${aq[@]}" ;;
        all) printf '%s\n' "${aq[@]}" ;;
        *)
            IFS=',' read -r -a wants <<< "$spec"
            for w in "${wants[@]}"; do
                q="$(resolve_quant "$w" "${aq[@]}")"
                [ -n "$q" ] || { err "quant '$w' is not currently present in $repo (try --quant list)"; return 1; }
                printf '%s\n' "$q"
            done ;;
    esac
}

# GGML storage bytes per 32 scalar values for the KV cache types we screen.
kv_block_bytes32() {
    case "${1,,}" in
        f32) printf '%s\n' 128 ;;
        f16|bf16) printf '%s\n' 64 ;;
        q8_0) printf '%s\n' 34 ;;
        q5_0) printf '%s\n' 22 ;;
        q4_0) printf '%s\n' 18 ;;
        *) return 1 ;;
    esac
}

metadata_vram_preflight() {
    # Return 0 = worth exact fitting, 1 = reject.
    # capacity_mode=physical answers "can this hardware ever fit it?"
    # capacity_mode=runtime answers "can it fit with memory free right now?"
    local alias="$1" repo="$2" prefix="$3" quant="$4" kvk="$5" kvv="$6" ctx="$7" meta="$8"
    local wbytes parts attn kvheads hdim kb vb kvbytes reserve_bytes overhead_bytes need_bytes capacity_bytes
    local capacity_label verbose="${9:-1}" capacity_mode="${10:-runtime}"
    wbytes="$(quant_size_bytes "$repo" "$prefix" "$quant")" || return 0
    parts="$(quant_parts_count "$repo" "$prefix" "$quant")" || parts='?'
    [ -n "$wbytes" ] && [ "$wbytes" -gt 0 ] 2>/dev/null || { warn "no byte size for $quant; cannot metadata-prefilter"; return 0; }
    attn="$(printf '%s' "$meta" | cut -f7)"; kvheads="$(printf '%s' "$meta" | cut -f8)"; hdim="$(printf '%s' "$meta" | cut -f9)"
    kb="$(kv_block_bytes32 "$kvk")" || { warn "unknown KV storage size for $kvk; exact fit required"; return 0; }
    vb="$(kv_block_bytes32 "$kvv")" || { warn "unknown KV storage size for $kvv; exact fit required"; return 0; }
    kvbytes=$(( ctx * attn * kvheads * hdim * (kb + vb) / 32 ))
    overhead_bytes=$(( PREFLIGHT_OVERHEAD * 1048576 ))

    if [ "$BACKEND" = "cuda" ]; then
        reserve_bytes=$(( FIT_MARGIN * GPU_COUNT * 1048576 ))
        if [ "$capacity_mode" = "physical" ]; then
            capacity_bytes=$(( TOTAL_VRAM * 1048576 ))
            capacity_label="physical VRAM"
        else
            capacity_bytes=$(( FREE_VRAM * 1048576 ))
            capacity_label="free VRAM now"
        fi
    else
        local avail total reserve_mib
        avail="$(cpu_mem_available_mib 2>/dev/null || echo 0)"
        total="$(cpu_mem_total_mib 2>/dev/null || echo 0)"
        if ! [[ "$avail" =~ ^[0-9]+$ ]] || [ "$avail" -le 0 ]; then
            warn "cannot read MemAvailable for CPU metadata preflight; allowing exact fit"
            return 0
        fi
        reserve_mib=$(( total / 20 ))
        [ "$reserve_mib" -lt 4096 ] && reserve_mib=4096
        [ "$FIT_MARGIN" -gt "$reserve_mib" ] && reserve_mib="$FIT_MARGIN"
        reserve_bytes=$(( reserve_mib * 1048576 ))
        capacity_bytes=$(( avail * 1048576 ))
        capacity_label="MemAvailable"
    fi

    need_bytes=$(( wbytes + kvbytes + reserve_bytes + overhead_bytes ))
    if [ "$verbose" -ne 0 ]; then
        python3 - "$quant" "$parts" "$wbytes" "$kvbytes" "$reserve_bytes" "$overhead_bytes" "$capacity_bytes" "$capacity_label" <<'PYPRE'
import sys
q,parts=sys.argv[1],sys.argv[2]
w,kv,res,ovh,cap=map(int,sys.argv[3:8])
label=sys.argv[8]
g=lambda x:x/(1024**3)
need=w+kv+res+ovh
status='REJECT' if need>cap else 'PLAUSIBLE'
print(f"PREFLIGHT: {q} parts={parts} model={g(w):.2f} GiB + KV={g(kv):.2f} GiB + reserve={g(res):.2f} GiB + allowance={g(ovh):.2f} GiB = {g(need):.2f} GiB; {label}={g(cap):.2f} GiB => {status}", file=sys.stderr)
PYPRE
    fi
    [ "$need_bytes" -le "$capacity_bytes" ]
}

kv_set() {
    local spec="$1"
    case "${spec,,}" in
        auto)
            if [ "$BACKEND" = "cpu" ]; then
                printf '%s\n' 'f16,f16' 'q8_0,q8_0' 'q4_0,q4_0'
            elif [ "$TOTAL_VRAM" -gt 49152 ]; then
                printf '%s\n' 'f16,f16' 'q8_0,q8_0' 'q8_0,q5_0' 'q5_0,q5_0' 'q8_0,q4_0' 'q4_0,q4_0'
            else
                printf '%s\n' 'q8_0,q8_0' 'q8_0,q5_0' 'q5_0,q5_0' 'q8_0,q4_0' 'q4_0,q4_0'
            fi ;;
        all)
            printf '%s\n' 'f16,f16' 'q8_0,q8_0' 'q8_0,q5_0' 'q8_0,q4_0' 'q5_0,q5_0' 'q4_0,q4_0' ;;
        *)
            # Semicolon separates multiple asymmetric pairs. A bare TYPE means K=V.
            IFS=';' read -r -a pairs <<< "$spec"
            for pair in "${pairs[@]}"; do
                if [[ "$pair" == *,* ]]; then printf '%s\n' "$pair"; else printf '%s,%s\n' "$pair" "$pair"; fi
            done ;;
    esac
}

sanitize() { printf '%s' "$1" | tr '/:;, ' '______' | tr -cd 'A-Za-z0-9_.-'; }

quant_files() {
    local repo="$1" prefix="$2" want="$3" inv
    inv="$(hf_inventory "$repo" "$prefix")" || return 1
    awk -F'\t' -v w="${want,,}" 'tolower($1)==w {print $4; exit}' "$inv"
}

kv_pair_bytes32() {
    local pair="$1" k v kb vb
    IFS=',' read -r k v <<< "$pair"
    kb="$(kv_block_bytes32 "$k")" || return 1
    vb="$(kv_block_bytes32 "$v")" || return 1
    printf '%s\n' "$((kb+vb))"
}

sort_kv_pairs_low_to_high() {
    local pair n
    for pair in "$@"; do
        n="$(kv_pair_bytes32 "$pair")" || n=999999
        printf '%09d\t%s\n' "$n" "$pair"
    done | sort -n -k1,1 | cut -f2-
}

# A cheap necessary-condition plan.  If a weight quant cannot fit with the
# smallest requested KV representation, there is no reason to download it or
# test any higher-memory KV representation at this context.
plausible_quants_for_min_kv() {
    local alias="$1" repo="$2" prefix="$3" target="$4" meta="$5" min_pair="$6"; shift 6
    local q kvk kvv
    IFS=',' read -r kvk kvv <<< "$min_pair"
    for q in "$@"; do
        if metadata_vram_preflight "$alias" "$repo" "$prefix" "$q" "$kvk" "$kvv" "$target" "$meta" 0; then
            printf '%s\n' "$q"
        fi
    done
}

choose_probe_quant() {
    local -a a=("$@")
    [ "${#a[@]}" -gt 0 ] || return 1

    if [ "$BACKEND" = "cpu" ]; then
        # Speed-oriented model-family probe; avoid exotic formats when a normal
        # K-quant is available on AVX2-class CPUs.
        local -a prefs=(Q5_K_S UD-Q5_K_S Q4_K_XL UD-Q4_K_XL Q5_K_M UD-Q5_K_M Q4_K_M UD-Q4_K_M Q4_K_S UD-Q4_K_S Q4_0 Q4_1 UD-Q3_K_XL Q3_K_M UD-Q3_K_M Q3_K_S UD-Q3_K_S Q6_K UD-Q6_K MXFP4_MOE UD-IQ4_NL IQ4_NL)
        local p q
        for p in "${prefs[@]}"; do
            q="$(resolve_quant "$p" "${a[@]}")"
            [ -n "$q" ] && { printf '%s\n' "$q"; return 0; }
        done
        printf '%s\n' "${a[0]}"
        return 0
    fi

    local idx=$(( PROBE_RANK - 1 ))
    [ "$idx" -lt "${#a[@]}" ] || idx=$((${#a[@]}-1))
    printf '%s\n' "${a[$idx]}"
}

HF_PREFETCH_PY=""
UV_BIN=""
declare -A PREFETCH_PID=()
declare -A PREFETCH_LOG=()
declare -A PREFETCH_REPO=()
declare -A PREFETCH_QUANT=()

ensure_local_uv() {
    local local_uv="$WORK/bin/uv" installer="$WORK/downloads/uv-install.sh"
    if have uv; then
        command -v uv
        return 0
    fi
    if [ -x "$local_uv" ]; then
        printf '%s\n' "$local_uv"
        return 0
    fi
    have curl || { err "curl is required to bootstrap local uv"; return 1; }
    mkdir -p "$WORK/bin" "$WORK/downloads" "$UV_CACHE_DIR" || return 1
    # WALL usually seconds, first run only: official uv standalone installer, contained under $PWD.
    wall_est "~5-20 s first run only" "bootstrap local uv for Hugging Face prefetch"
    info "uv is absent; bootstrapping official standalone uv under $WORK/bin" >&2
    if ! curl -fLsS --retry 3 https://astral.sh/uv/install.sh -o "$installer"; then
        err "failed to download official uv installer"
        return 1
    fi
    # UV_UNMANAGED_INSTALL prevents PATH/profile edits and self-update management.
    if ! env UV_UNMANAGED_INSTALL="$WORK/bin" UV_NO_MODIFY_PATH=1 sh "$installer" 1>&2; then
        err "local uv bootstrap failed"
        return 1
    fi
    [ -x "$local_uv" ] || { err "uv installer completed but $local_uv is missing"; return 1; }
    printf '%s\n' "$local_uv"
}

ensure_hf_prefetch_env() {
    [ "$PREFETCH" -eq 1 ] || return 0
    [ -n "$HF_PREFETCH_PY" ] && [ -x "$HF_PREFETCH_PY" ] && return 0
    UV_BIN="$(ensure_local_uv)" || {
        err "cannot provision uv for background Hugging Face prefetch"
        return 1
    }
    local envdir="$WORK/hf-prefetch-py"
    HF_PREFETCH_PY="$envdir/bin/python"
    mkdir -p "$UV_CACHE_DIR"
    if [ ! -x "$HF_PREFETCH_PY" ]; then
        info "Creating local uv environment for Hugging Face prefetch"
        print_cmd "$UV_BIN" venv --python python3 "$envdir"
        "$UV_BIN" venv --python python3 "$envdir" || return 1
    fi
    if ! "$HF_PREFETCH_PY" -c 'import huggingface_hub' >/dev/null 2>&1; then
        print_cmd "$UV_BIN" pip install --python "$HF_PREFETCH_PY" huggingface_hub
        "$UV_BIN" pip install --python "$HF_PREFETCH_PY" huggingface_hub || return 1
    fi
}

prefetch_key() { sanitize "$1__$3"; }

prefetch_quant_start() {
    local repo="$1" prefix="$2" quant="$3"
    [ "$PREFETCH" -eq 1 ] || return 0
    ensure_hf_prefetch_env || return 1
    local files key log py="$HF_PREFETCH_PY" done_marker fail_marker
    files="$(quant_files "$repo" "$prefix" "$quant")" || return 1
    [ -n "$files" ] || { err "no GGUF files found for prefetch $repo:$quant"; return 1; }
    key="$(prefetch_key "$repo" "$prefix" "$quant")"
    [ -n "${PREFETCH_PID[$key]:-}" ] && return 0
    log="$RESULTS/prefetch-$(sanitize "$repo")-$(sanitize "$quant").log"
    done_marker="$log.done"; fail_marker="$log.failed"
    rm -f "$done_marker" "$fail_marker"
    PREFETCH_LOG[$key]="$log"
    PREFETCH_REPO[$key]="$repo"
    PREFETCH_QUANT[$key]="$quant"
    printf 'PREFETCH START: %-58s %-14s (HF cache=%s)\n' "$repo" "$quant" "$HF_HUB_CACHE" >&2
    (
        export HF_HUB_DISABLE_PROGRESS_BARS=1
        export HF_HUB_CACHE HF_HOME HF_TOKEN="${HF_TOKEN:-}"
        if have ionice; then
            ionice -c3 nice -n 15 "$py" - "$repo" "$files" "$HF_HUB_CACHE" <<'PYHF'
import os, sys, time
from huggingface_hub import hf_hub_download
repo, files, cache = sys.argv[1:4]
token=os.environ.get("HF_TOKEN") or None
paths=[x for x in files.split(';') if x]
t0=time.time()
for i,fn in enumerate(paths,1):
    print(f"download {i}/{len(paths)} {fn}", flush=True)
    p=hf_hub_download(repo_id=repo, filename=fn, cache_dir=cache, token=token)
    print(f"cached {p}", flush=True)
print(f"complete seconds={time.time()-t0:.3f}", flush=True)
PYHF
        else
            nice -n 15 "$py" - "$repo" "$files" "$HF_HUB_CACHE" <<'PYHF'
import os, sys, time
from huggingface_hub import hf_hub_download
repo, files, cache = sys.argv[1:4]
token=os.environ.get("HF_TOKEN") or None
paths=[x for x in files.split(';') if x]
t0=time.time()
for i,fn in enumerate(paths,1):
    print(f"download {i}/{len(paths)} {fn}", flush=True)
    p=hf_hub_download(repo_id=repo, filename=fn, cache_dir=cache, token=token)
    print(f"cached {p}", flush=True)
print(f"complete seconds={time.time()-t0:.3f}", flush=True)
PYHF
        fi
        rc=$?
        if [ "$rc" -eq 0 ]; then : > "$done_marker"; else printf '%s\n' "$rc" > "$fail_marker"; fi
        exit "$rc"
    ) >"$log" 2>&1 &
    PREFETCH_PID[$key]=$!
}

prefetch_quant_ready() {
    # 0 = successfully complete; 1 = still running/not started; 2 = failed.
    local repo="$1" prefix="$2" quant="$3" key pid log rc
    key="$(prefetch_key "$repo" "$prefix" "$quant")"
    pid="${PREFETCH_PID[$key]:-}"
    [ "$pid" = "done" ] && return 0
    [ "$pid" = "failed" ] && return 2
    [ -n "$pid" ] || return 1
    log="${PREFETCH_LOG[$key]:-}"
    if [ -f "$log.done" ]; then
        wait "$pid" 2>/dev/null || true
        PREFETCH_PID[$key]="done"
        printf 'PREFETCH DONE:  %-58s %-14s\n' "$repo" "$quant" >&2
        return 0
    fi
    if [ -f "$log.failed" ]; then
        wait "$pid" 2>/dev/null; rc=$?
        PREFETCH_PID[$key]="failed"
        printf 'PREFETCH FAILED: %s:%s rc=%s\n' "$repo" "$quant" "$rc" >&2
        [ -s "$log" ] && cat "$log" >&2
        return 2
    fi
    return 1
}

prefetch_status_all() {
    local key p repo quant log elapsed last active=0
    for key in "${!PREFETCH_PID[@]}"; do
        p="${PREFETCH_PID[$key]}"
        case "$p" in done|failed|'') continue ;; esac
        repo="${PREFETCH_REPO[$key]:-$key}"; quant="${PREFETCH_QUANT[$key]:-?}"; log="${PREFETCH_LOG[$key]:-}"
        if [ -n "$log" ] && [ -f "$log.done" ]; then continue; fi
        if [ -n "$log" ] && [ -f "$log.failed" ]; then continue; fi
        elapsed="$(ps -o etime= -p "$p" 2>/dev/null | xargs || true)"
        last="$(tail -n 1 "$log" 2>/dev/null | tr '\r' '\n' | tail -n 1 | cut -c1-180)"
        printf 'PREFETCH ACTIVE: %-42s %-14s pid=%-7s elapsed=%-10s %s\n' "$repo" "$quant" "$p" "${elapsed:-?}" "${last:-[waiting for downloader output]}" >&2
        active=$((active+1))
    done
    [ "$active" -gt 0 ] || printf 'PREFETCH STATUS: no background downloads are currently active.\n' >&2
}

prefetch_quant_wait() {
    local repo="$1" prefix="$2" quant="$3"
    [ "$PREFETCH" -eq 1 ] || return 0
    local key pid log rc waited=0
    key="$(prefetch_key "$repo" "$prefix" "$quant")"
    pid="${PREFETCH_PID[$key]:-}"
    [ "$pid" = "done" ] && return 0
    [ "$pid" = "failed" ] && return 1
    [ -n "$pid" ] || { prefetch_quant_start "$repo" "$prefix" "$quant" || return 1; pid="${PREFETCH_PID[$key]:-}"; }
    [ -n "$pid" ] || return 0
    printf '%s WAIT: required payload is not ready yet: %s:%s. Inference is paused until this download completes.\n' "${BACKEND^^}" "$repo" "$quant" >&2
    while :; do
        prefetch_quant_ready "$repo" "$prefix" "$quant"; rc=$?
        [ "$rc" -eq 0 ] && return 0
        [ "$rc" -eq 2 ] && return 1
        if [ "$waited" -eq 0 ] || [ $((waited % PREFETCH_HEARTBEAT)) -eq 0 ]; then
            prefetch_status_all
            printf 'NEXT %s TASK: shallow fit/PP%s/TG%s probe for %s:%s as soon as its payload is ready.\n' "${BACKEND^^}" "$CURVE_PP" "$CURVE_TG" "$repo" "$quant" >&2
        fi
        sleep 1
        waited=$((waited+1))
    done
}

prefetch_wait_any_alias() {
    # Args are aliases. Echo the first completed alias, preserving argument order
    # as the tie-breaker. Prints a heartbeat while all remaining payloads are in flight.
    local -a aliases=("$@")
    local a repo prefix q rc waited=0
    while :; do
        for a in "${aliases[@]}"; do
            repo="${M_REPO[$a]}"; prefix="${M_PREFIX[$a]}"; q="${M_PROBE[$a]}"
            prefetch_quant_ready "$repo" "$prefix" "$q"; rc=$?
            if [ "$rc" -eq 0 ]; then printf '%s\n' "$a"; return 0; fi
            if [ "$rc" -eq 2 ]; then
                printf 'PREFETCH NOTE: planned probe failed for %s (%s); scheduler will still select it so fallback logic can run.\n' "$a" "$q" >&2
                printf '%s\n' "$a"; return 0
            fi
        done
        if [ "$waited" -eq 0 ] || [ $((waited % PREFETCH_HEARTBEAT)) -eq 0 ]; then
            printf '%s IDLE / NETWORK WAIT: none of the remaining model probes is cached yet; the first completed payload will run next.\n' "${BACKEND^^}" >&2
            prefetch_status_all
        fi
        sleep 1
        waited=$((waited+1))
    done
}

prefetch_background_cleanup() {
    local k p
    for k in "${!PREFETCH_PID[@]}"; do
        p="${PREFETCH_PID[$k]}"
        [[ "$p" =~ ^[0-9]+$ ]] || continue
        kill "$p" 2>/dev/null || true
    done
}
cleanup_on_signal() { prefetch_background_cleanup; exit 130; }
trap cleanup_on_signal INT TERM
trap prefetch_background_cleanup EXIT

cpu_mem_available_mib() {
    awk '/^MemAvailable:/ {print int($2/1024); exit}' /proc/meminfo
}

cpu_mem_total_mib() {
    awk '/^MemTotal:/ {print int($2/1024); exit}' /proc/meminfo
}

cpu_estimate_mib() {
    # Print estimated Host model+context+compute MiB for an explicit CPU context.
    # llama-fit-params --fit-print performs memory planning without relying on
    # the dedicated-device fitter (which intentionally does not fit host RAM).
    local repo="$1" quant="$2" kvk="$3" kvv="$4" ctx="$5" stem="$6"
    local out="$stem.cpu-mem-${ctx}.txt" log="$stem.cpu-mem-${ctx}.log" rc need
    local -a cmd=("$FIT" -hf "$repo:$quant" -c "$ctx" -ngl 0 -ctk "$kvk" -ctv "$kvv" -fa on -b "$BATCH" -ub "$UBATCH" -t "$THREADS" --fit-print on)
    print_cmd "${cmd[@]}"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'Host 0 0 0\n' > "$out"
        printf '0\n'
        return 0
    fi
    # Show BOTH stdout and stderr live. stdout is also retained for parsing.
    # The pipeline writes normal tool stdout to our stderr so command-substitution
    # callers still receive only this function's final machine-readable value.
    "${cmd[@]}" 2> >(tee "$log" >&2) | tee "$out" >&2
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        warn "CPU memory estimate failed at ctx=$ctx"
        show_capture_on_failure "$rc" "$out" "$log"
        return 1
    fi
    need="$(awk '$1=="Host" && NF>=4 {print $2+$3+$4; exit}' "$out")"
    if ! [[ "$need" =~ ^[0-9]+$ ]]; then
        warn "could not parse Host memory estimate at ctx=$ctx"
        show_capture_on_failure 0 "$out" "$log"
        return 1
    fi
    printf '%s\n' "$need"
}

cpu_fit_one() {
    local repo="$1" quant="$2" kvk="$3" kvv="$4" stem="$5"
    local out="$stem.fit.args" avail total reserve budget target minc need
    local lo hi mid ctx_try best=0 step=512

    avail="$(cpu_mem_available_mib)"
    total="$(cpu_mem_total_mib)"
    [[ "$avail" =~ ^[0-9]+$ ]] && [[ "$total" =~ ^[0-9]+$ ]] || {
        err "cannot read MemAvailable/MemTotal from /proc/meminfo for CPU fitting"
        return 1
    }

    # Leave the larger of the user fit margin, 4 GiB, or 5% of physical RAM.
    reserve=$(( total / 20 ))
    [ "$reserve" -lt 4096 ] && reserve=4096
    [ "$FIT_MARGIN" -gt "$reserve" ] && reserve="$FIT_MARGIN"
    budget=$(( avail - reserve ))
    [ "$budget" -gt 0 ] || {
        err "CPU memory budget is <= 0 MiB (available=$avail reserve=$reserve)"
        return 1
    }

    if [ "$CTX" = "auto" ] || [ "$CTX" = "max" ]; then
        target="$CURVE_CAP"
        minc="$MIN_CTX"
    else
        target="$CTX"
        minc="$CTX"
    fi

    printf '\n==> CPU RAM fit: MemAvailable=%s MiB, reserve=%s MiB, usable=%s MiB, target_ctx=%s\n' \
        "$avail" "$reserve" "$budget" "$target" >&2

    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%s\n' "-c $target -ngl 0" > "$out"
        printf '%s\n' "$out"
        return 0
    fi

    # Explicit context is a hard request: estimate it once and fail visibly if
    # it exceeds the current RAM budget.
    if [ "$CTX" != "auto" ] && [ "$CTX" != "max" ]; then
        need="$(cpu_estimate_mib "$repo" "$quant" "$kvk" "$kvv" "$target" "$stem")" || return 1
        printf 'CPU estimate: ctx=%s requires=%s MiB budget=%s MiB\n' "$target" "$need" "$budget" >&2
        if [ "$need" -gt "$budget" ]; then
            err "requested CPU context $target is estimated to need ${need} MiB > ${budget} MiB usable RAM"
            return 1
        fi
        printf '%s\n' "-c $target -ngl 0" > "$out"
        printf '%s\n' "$out"
        return 0
    fi

    # First try the cap. Most roomy CPU systems finish in one estimate.
    need="$(cpu_estimate_mib "$repo" "$quant" "$kvk" "$kvv" "$target" "$stem")" || return 1
    printf 'CPU estimate: ctx=%s requires=%s MiB budget=%s MiB\n' "$target" "$need" "$budget" >&2
    if [ "$need" -le "$budget" ]; then
        best="$target"
    else
        # Check the minimum before binary searching, then search in 512-token
        # increments for the largest estimated-safe context.
        need="$(cpu_estimate_mib "$repo" "$quant" "$kvk" "$kvv" "$minc" "$stem")" || return 1
        printf 'CPU estimate: ctx=%s requires=%s MiB budget=%s MiB\n' "$minc" "$need" "$budget" >&2
        if [ "$need" -gt "$budget" ]; then
            err "even minimum CPU context $minc is estimated to need ${need} MiB > ${budget} MiB usable RAM"
            return 1
        fi
        best="$minc"
        lo=$(( (minc + step - 1) / step ))
        hi=$(( target / step ))
        while [ "$lo" -le "$hi" ]; do
            mid=$(( (lo + hi) / 2 ))
            [ "$mid" -lt 1 ] && mid=1
            ctx_try=$(( mid * step ))
            need="$(cpu_estimate_mib "$repo" "$quant" "$kvk" "$kvv" "$ctx_try" "$stem")" || return 1
            printf 'CPU estimate: ctx=%s requires=%s MiB budget=%s MiB\n' "$ctx_try" "$need" "$budget" >&2
            if [ "$need" -le "$budget" ]; then
                best="$ctx_try"
                lo=$(( mid + 1 ))
            else
                hi=$(( mid - 1 ))
            fi
        done
    fi

    [ "$best" -ge "$minc" ] || { err "CPU context fitting found no usable context"; return 1; }
    printf 'CPU fitted context: %s tokens\n' "$best" >&2
    printf '%s\n' "-c $best -ngl 0" > "$out"
    printf '%s\n' "$out"
}

fit_one() {
    local repo="$1" quant="$2" kvk="$3" kvv="$4" stem="$5"
    local out="$stem.fit.args" log="$stem.fit.log" start_ctx min_fit rc
    if [ "$BACKEND" = "cpu" ]; then
        if cpu_fit_one "$repo" "$quant" "$kvk" "$kvv" "$stem"; then
            return 0
        fi
        warn "CPU fit failed for $repo:$quant KV=$kvk/$kvv; the estimator output above is the actual failure"
        return 1
    fi
    if [ "$CTX" = "auto" ] || [ "$CTX" = "max" ]; then start_ctx=0; min_fit="$MIN_CTX"; else start_ctx="$CTX"; min_fit="$CTX"; fi

    # Do NOT pass --no-mmproj here. llama-fit-params does not expose the
    # multimodal-projector CLI option; fit-params itself does not use mmproj.
    local -a cmd=("$FIT" -hf "$repo:$quant" -c "$start_ctx" -ctk "$kvk" -ctv "$kvv" -fa on -b "$BATCH" -ub "$UBATCH" -t "$THREADS" --fit-target "$FIT_MARGIN" --fit-ctx "$min_fit")
    print_cmd "${cmd[@]}"

    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%s\n' '-c 0 -ngl 0' > "$out"
        printf '%s\n' "$out"
        return 0
    fi

    # stderr is NEVER hidden: show it live and retain an identical per-run log.
    # stdout must stay separate because llama-fit-params emits machine-consumable
    # fitted CLI arguments there.
    # Show BOTH stdout and stderr live. stdout is also retained for parsing.
    # The pipeline writes normal tool stdout to our stderr so command-substitution
    # callers still receive only this function's final machine-readable value.
    "${cmd[@]}" 2> >(tee "$log" >&2) | tee "$out" >&2
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        warn "fit failed for $repo:$quant KV=$kvk/$kvv (raw output follows)"
        show_capture_on_failure "$rc" "$out" "$log"
        if grep -Eq 'error while loading shared libraries|cannot open shared object file|symbol lookup error' "$log" "$out" 2>/dev/null; then
            err "runtime loader failure is a BUILD error, not a quant/context fit result; aborting"
            exit 127
        fi
        return 1
    fi
    if ! grep -Eq '(^|[[:space:]])-c[[:space:]]+[0-9]+' "$out"; then
        warn "llama-fit-params exited 0 but did not emit a parsable -c argument"
        show_capture_on_failure 0 "$out" "$log"
        return 1
    fi
    printf '%s\n' "$out"
}

fit_ctx() {
    awk '{for(i=1;i<=NF;i++) if($i=="-c" && (i+1)<=NF){print $(i+1); exit}}' "$1"
}

bench_args_from_fit() {
    # llama-bench accepts ngl/ts/ot but not -c. Remove only the fitted context.
    sed -E 's/(^|[[:space:]])-c[[:space:]]+[0-9]+//g' "$1"
}


cpu_numa_args() {
    CPU_NUMA_ARGS=()
    if [ "$BACKEND" = "cpu" ] && [ "$CPU_NUMA_EFFECTIVE" != "disabled" ]; then
        CPU_NUMA_ARGS=(--numa "$CPU_NUMA_EFFECTIVE")
    fi
}

cpu_command_prefix() {
    CPU_CMD_PREFIX=()
    if [ "$BACKEND" = "cpu" ] && [ "$CPU_PLACEMENT_EFFECTIVE" = "interleave-none" ]; then
        CPU_CMD_PREFIX=(numactl --interleave=all)
    fi
}

result_meta_tsv() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
        "$RUN_ID" "$VERSION" "$BACKEND" "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" \
        "$CPU_BLAS_EFFECTIVE" "$THREADS" "$THREADS_BATCH" "$BATCH" "$UBATCH"
}

cpu_tg_thread_candidates_csv() {
    # WALL ~15-30s competitive MoE: sub-core points can reveal memory saturation/congestion headroom.
    local half=$(( (physical + 1) / 2 ))
    local two_thirds=$(( (physical * 2 + 2) / 3 ))
    local four_fifths=$(( (physical * 4 + 4) / 5 ))
    [ "$half" -lt 1 ] && half=1
    [ "$two_thirds" -lt 1 ] && two_thirds=1
    [ "$four_fifths" -lt 1 ] && four_fifths=1
    if [[ "$cpu_model" == *"Threadripper 19"*"X"* ]]; then
        printf '%s\n' "$half" "$two_thirds" "$four_fifths" "$physical" "$(( physical + half ))" "$logical"             | awk -v max="$logical" '$1>0 && $1<=max && !seen[$1]++' | paste -sd, -
    else
        printf '%s\n' "$half" "$two_thirds" "$physical" "$logical"             | awk -v max="$logical" '$1>0 && $1<=max && !seen[$1]++' | paste -sd, -
    fi
}

cpu_pp_thread_candidates_csv() {
    # WALL saved ~25-35%: PP starts near ~2/3 cores; very-low-core PP has little practical value.
    local two_thirds=$(( (physical * 2 + 2) / 3 ))
    local four_fifths=$(( (physical * 4 + 4) / 5 ))
    [ "$two_thirds" -lt 1 ] && two_thirds=1
    [ "$four_fifths" -lt 1 ] && four_fifths=1
    if [[ "$cpu_model" == *"Threadripper 19"*"X"* ]]; then
        printf '%s\n' "$two_thirds" "$four_fifths" "$physical" "$logical"             | awk -v max="$logical" '$1>0 && $1<=max && !seen[$1]++' | paste -sd, -
    else
        printf '%s\n' "$two_thirds" "$physical" "$logical"             | awk -v max="$logical" '$1>0 && $1<=max && !seen[$1]++' | paste -sd, -
    fi
}

cpu_retune_tg_threads_for_model() {
    # After the first full CPU calibration, cheaply re-check ONLY decode threads
    # for each subsequent model family. This keeps family selection fair without
    # repeating the more expensive PP-thread/ubatch sweep four times.
    [ "$BACKEND" = "cpu" ] || return 0
    [ "$CPU_TUNE" -eq 1 ] || return 0
    [ "$THREADS_WAS_AUTO" -eq 1 ] || return 0
    local repo="$1" quant="$2" kvk="$3" kvv="$4" fitfile="$5" stem="$6"
    local barg="$stem.cpu-family-tg.args" json="$stem.cpu-family-tg.json" log="$stem.cpu-family-tg.log"
    local tcsv rc best top2 confirm_json confirm_log
    bench_args_from_fit "$fitfile" > "$barg"
    cpu_numa_args
    cpu_command_prefix
    tcsv="$(cpu_tg_thread_candidates_csv)"
    wall_est "~15-30 s if competitive; skipped for obvious losers" "TG16 all candidates + TG64 top-two"
    local -a cmd=(-hf "$repo:$quant" -p 0 -n "$CPU_TUNE_TG" -d 0 -r 1 --progress
        -b "$BATCH" -ub "$UBATCH" -t "$tcsv" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
    cmd+=("${CPU_NUMA_ARGS[@]}")
    printf 'CPU FAMILY TG TUNE: model_quant=%s candidates=%s placement=%s load_mode=%s\n' "$quant" "$tcsv" "$CPU_PLACEMENT_EFFECTIVE" "$CPU_LOAD_MODE_EFFECTIVE" >&2
    printf 'CPU FAMILY TG COMMAND: xargs ' >&2
    printf '%q ' "${CPU_CMD_PREFIX[@]}" "$BENCH" "${cmd[@]}" >&2
    printf '< %q\n' "$barg" >&2
    if [ "$DRY_RUN" -eq 1 ]; then return 0; fi
    xargs "${CPU_CMD_PREFIX[@]}" "$BENCH" "${cmd[@]}" < "$barg" > "$json" 2> >(tee "$log" >&2)
    rc=$?
    if [ "$rc" -ne 0 ]; then
        warn "CPU per-model decode-thread tune failed; retaining threads=$THREADS"
        show_capture_on_failure "$rc" "$json" "$log"
        return "$rc"
    fi
    jq -r --argjson n "$CPU_TUNE_TG" '.[]? | select(.n_prompt==0 and .n_gen==$n) | "CPU FAMILY TG COARSE: threads=\\(.n_threads) tps=\\(.avg_ts)"' "$json" >&2 || true
    top2="$(jq -r --argjson n "$CPU_TUNE_TG" '[.[]? | select(.n_prompt==0 and .n_gen==$n)] | sort_by(.avg_ts) | reverse | .[:2] | map(.n_threads|tostring) | join(",")' "$json")"
    if [[ "$top2" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
        confirm_json="$stem.cpu-family-tg-confirm.json"; confirm_log="$stem.cpu-family-tg-confirm.log"
        local -a ccmd=(-hf "$repo:$quant" -p 0 -n "$CPU_TUNE_TG_CONFIRM" -d 0 -r 1 --progress
            -b "$BATCH" -ub "$UBATCH" -t "$top2" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
        ccmd+=("${CPU_NUMA_ARGS[@]}")
        xargs "${CPU_CMD_PREFIX[@]}" "$BENCH" "${ccmd[@]}" < "$barg" > "$confirm_json" 2> >(tee "$confirm_log" >&2)
        if [ "$?" -eq 0 ]; then
            jq -r --argjson n "$CPU_TUNE_TG_CONFIRM" '.[]? | select(.n_prompt==0 and .n_gen==$n) | "CPU FAMILY TG CONFIRM: threads=\\(.n_threads) tps=\\(.avg_ts)"' "$confirm_json" >&2 || true
            best="$(jq -r --argjson n "$CPU_TUNE_TG_CONFIRM" '[.[]? | select(.n_prompt==0 and .n_gen==$n)] | sort_by(.avg_ts) | last | .n_threads // empty' "$confirm_json")"
        fi
    fi
    if [[ "$best" =~ ^[0-9]+$ ]]; then
        THREADS="$best"
        printf 'CPU FAMILY TG SELECTED: threads=%s\n' "$THREADS" >&2
    fi
}

cpu_family_gate_tg() {
    # WALL ~2s winner / ~20s 0.8-t/s loser: reject >5x-slower families before PP/thread sweeps.
    local repo="$1" quant="$2" kvk="$3" kvv="$4" fitfile="$5" stem="$6"
    local barg="$stem.cpu-gate.args" json="$stem.cpu-gate.json" log="$stem.cpu-gate.log" gate=""
    bench_args_from_fit "$fitfile" > "$barg"
    cpu_numa_args; cpu_command_prefix
    wall_est "~2-25 s depending on model" "physical-core TG16 family gate"
    local -a cmd=(-hf "$repo:$quant" -p 0 -n "$CPU_FAMILY_GATE_TG" -d 0 -r 1 --progress
        -b "$BATCH" -ub "$UBATCH" -t "$physical" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
    cmd+=("${CPU_NUMA_ARGS[@]}")
    if [ "$DRY_RUN" -eq 1 ]; then printf '0\n'; return 0; fi
    xargs "${CPU_CMD_PREFIX[@]}" "$BENCH" "${cmd[@]}" < "$barg" > "$json" 2> >(tee "$log" >&2) || return 1
    gate="$(jq -r --argjson n "$CPU_FAMILY_GATE_TG" '[.[]? | select(.n_prompt==0 and .n_gen==$n)] | .[0].avg_ts // empty' "$json")"
    printf '%s\n' "$gate"
}

cpu_autotune_once() {
    [ "$BACKEND" = "cpu" ] || return 0
    [ "$CPU_TUNE" -eq 1 ] || return 0
    [ "$CPU_TUNED" -eq 0 ] || return 0

    local repo="$1" quant="$2" kvk="$3" kvv="$4" fitfile="$5" stem="$6"
    local barg="$stem.cpu-tune.args" tcsv ppcsv json log rc best top2 confirm_json confirm_log
    bench_args_from_fit "$fitfile" > "$barg"
    cpu_numa_args
    cpu_command_prefix
    tcsv="$(cpu_tg_thread_candidates_csv)"
    ppcsv="$(cpu_pp_thread_candidates_csv)"

    info "CPU AUTOTUNE: one-time hardware/runtime calibration on $quant KV=$kvk/$kvv"
    printf 'CPU AUTOTUNE METHOD: NUMA=%s placement=%s load_mode=%s candidates=%s warmup=ON model=CPU-only.\n' "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" "$CPU_LOAD_MODE_EFFECTIVE" "$tcsv" >&2

    if [ "$THREADS_WAS_AUTO" -eq 1 ]; then
        json="$stem.cpu-tune-tg.json"; log="$stem.cpu-tune-tg.log"
        # WALL ~15-30s winner-class: coarse full sweep + longer top-two confirmation.
        wall_est "~15-30 s if winner-class" "decode-thread scaling"
        local -a cmd=(-hf "$repo:$quant" -p 0 -n "$CPU_TUNE_TG" -d 0 -r 1 --progress
            -b "$BATCH" -ub "$UBATCH" -t "$tcsv" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
        cmd+=("${CPU_NUMA_ARGS[@]}")
        printf 'CPU TUNE TG COMMAND: xargs ' >&2; printf '%q ' "$BENCH" "${cmd[@]}" >&2; printf '< %q\n' "$barg" >&2
        if [ "$DRY_RUN" -eq 0 ]; then
            xargs "${CPU_CMD_PREFIX[@]}" "$BENCH" "${cmd[@]}" < "$barg" > "$json" 2> >(tee "$log" >&2)
            rc=$?
            if [ "$rc" -eq 0 ]; then
                jq -r --argjson n "$CPU_TUNE_TG" '.[]? | select(.n_prompt==0 and .n_gen==$n) | "CPU TUNE TG COARSE: threads=\(.n_threads) tps=\(.avg_ts)"' "$json" >&2 || true
                top2="$(jq -r --argjson n "$CPU_TUNE_TG" '[.[]? | select(.n_prompt==0 and .n_gen==$n)] | sort_by(.avg_ts) | reverse | .[:2] | map(.n_threads|tostring) | join(",")' "$json")"
                if [[ "$top2" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
                    confirm_json="$stem.cpu-tune-tg-confirm.json"; confirm_log="$stem.cpu-tune-tg-confirm.log"
                    local -a ccmd=(-hf "$repo:$quant" -p 0 -n "$CPU_TUNE_TG_CONFIRM" -d 0 -r 1 --progress
                        -b "$BATCH" -ub "$UBATCH" -t "$top2" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
                    ccmd+=("${CPU_NUMA_ARGS[@]}")
                    xargs "${CPU_CMD_PREFIX[@]}" "$BENCH" "${ccmd[@]}" < "$barg" > "$confirm_json" 2> >(tee "$confirm_log" >&2)
                    if [ "$?" -eq 0 ]; then
                        jq -r --argjson n "$CPU_TUNE_TG_CONFIRM" '.[]? | select(.n_prompt==0 and .n_gen==$n) | "CPU TUNE TG CONFIRM: threads=\(.n_threads) tps=\(.avg_ts)"' "$confirm_json" >&2 || true
                        best="$(jq -r --argjson n "$CPU_TUNE_TG_CONFIRM" '[.[]? | select(.n_prompt==0 and .n_gen==$n)] | sort_by(.avg_ts) | last | .n_threads // empty' "$confirm_json")"
                    fi
                fi
                [[ "$best" =~ ^[0-9]+$ ]] && THREADS="$best"
            else
                warn "CPU decode-thread autotune failed; keeping threads=$THREADS"
                show_capture_on_failure "$rc" "$json" "$log"
            fi
        fi
    fi

    if [ "$THREADS_BATCH_WAS_AUTO" -eq 1 ]; then
        json="$stem.cpu-tune-ppthreads.json"; log="$stem.cpu-tune-ppthreads.log"
        # WALL ~45-90s winner-class: omit very-low-core PP points with little upside.
        wall_est "~45-90 s typical on winner-class MoE" "PP-thread scaling"
        local -a cmd=(-hf "$repo:$quant" -p "$CPU_TUNE_PP" -n 0 -d 0 -r 1 --progress
            -b 2048 -ub 512 -t "$ppcsv" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
        cmd+=("${CPU_NUMA_ARGS[@]}")
        printf 'CPU TUNE PP-THREAD COMMAND: xargs ' >&2; printf '%q ' "$BENCH" "${cmd[@]}" >&2; printf '< %q\n' "$barg" >&2
        if [ "$DRY_RUN" -eq 0 ]; then
            xargs "${CPU_CMD_PREFIX[@]}" "$BENCH" "${cmd[@]}" < "$barg" > "$json" 2> >(tee "$log" >&2)
            rc=$?
            if [ "$rc" -eq 0 ]; then
                jq -r --argjson p "$CPU_TUNE_PP" '.[]? | select(.n_prompt==$p and .n_gen==0) | "CPU TUNE PP: threads=\(.n_threads)  ubatch=\(.n_ubatch)  tps=\(.avg_ts)"' "$json" >&2 || true
                best="$(jq -r --argjson p "$CPU_TUNE_PP" '[.[]? | select(.n_prompt==$p and .n_gen==0)] | sort_by(.avg_ts) | last | .n_threads // empty' "$json")"
                [[ "$best" =~ ^[0-9]+$ ]] && THREADS_BATCH="$best"
            else
                warn "CPU PP-thread autotune failed; keeping threads-batch=$THREADS_BATCH"
                show_capture_on_failure "$rc" "$json" "$log"
            fi
        fi
    fi

    if [ "$UBATCH_WAS_AUTO" -eq 1 ]; then
        json="$stem.cpu-tune-ubatch.json"; log="$stem.cpu-tune-ubatch.log"
        local -a cmd=(-hf "$repo:$quant" -p "$CPU_TUNE_PP" -n 0 -d 0 -r 1 --progress
            -b "$BATCH" -ub 128,256,512 -t "$THREADS_BATCH" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
        cmd+=("${CPU_NUMA_ARGS[@]}")
        printf 'CPU TUNE UBATCH COMMAND: xargs ' >&2; printf '%q ' "$BENCH" "${cmd[@]}" >&2; printf '< %q\n' "$barg" >&2
        if [ "$DRY_RUN" -eq 0 ]; then
            xargs "${CPU_CMD_PREFIX[@]}" "$BENCH" "${cmd[@]}" < "$barg" > "$json" 2> >(tee "$log" >&2)
            rc=$?
            if [ "$rc" -eq 0 ]; then
                jq -r --argjson p "$CPU_TUNE_PP" '.[]? | select(.n_prompt==$p and .n_gen==0) | "CPU TUNE UBATCH: ubatch=\(.n_ubatch)  threads=\(.n_threads)  tps=\(.avg_ts)"' "$json" >&2 || true
                best="$(jq -r --argjson p "$CPU_TUNE_PP" '[.[]? | select(.n_prompt==$p and .n_gen==0)] | sort_by(.avg_ts) | last | .n_ubatch // empty' "$json")"
                [[ "$best" =~ ^[0-9]+$ ]] && UBATCH="$best"
            else
                warn "CPU ubatch autotune failed; keeping ubatch=$UBATCH"
                show_capture_on_failure "$rc" "$json" "$log"
            fi
        fi
    fi

    CPU_TUNED=1
    printf 'CPU AUTOTUNE SELECTED: numa=%s placement=%s load_mode=%s decode_threads=%s pp_threads=%s batch=%s ubatch=%s\n' \
        "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" "$CPU_LOAD_MODE_EFFECTIVE" "$THREADS" "$THREADS_BATCH" "$BATCH" "$UBATCH" >&2
}

curve_ladder() {
    # Usage: curve_ladder MAX SPEC INCLUDE_ZERO
    # Emits a sorted comma list, always including the exact cap when possible.
    local max="$1" spec="$2" include_zero="$3"
    local -a raw=() out=()
    local n x found
    if [ "$spec" = "auto" ]; then
        raw=(512 2048 8192 32768 65536 131072 196608 262144)
        if [ "$include_zero" -eq 1 ]; then
            raw=(0 "${raw[@]}")
        else
            # 41 tokens mirrors the supplied Ollama Qwen3.8 baseline while the
            # larger points provide stable prompt-processing measurements.
            raw=(41 "${raw[@]}")
        fi
    else
        IFS=',' read -r -a raw <<< "$spec"
    fi
    for x in "${raw[@]}"; do
        [[ "$x" =~ ^[0-9]+$ ]] || { err "bad curve point: $x"; return 1; }
        [ "$x" -le "$max" ] || continue
        found=0; for n in "${out[@]}"; do [ "$n" -eq "$x" ] && found=1; done
        [ "$found" -eq 0 ] && out+=("$x")
    done
    # Exact edge is useful even when it is not one of the standard powers/steps.
    found=0; for n in "${out[@]}"; do [ "$n" -eq "$max" ] && found=1; done
    [ "$found" -eq 0 ] && out+=("$max")
    printf '%s\n' "${out[@]}" | sort -n -u | paste -sd, -
}

bench_shallow_one() {
    local repo="$1" quant="$2" kvk="$3" kvv="$4" fitfile="$5" stem="$6"
    local barg="$stem.bench.args" json="$stem.shallow.json" log="$stem.shallow.log" rc
    bench_args_from_fit "$fitfile" > "$barg"
    local -a base=(-hf "$repo:$quant" -p "$CURVE_PP" -n "$CURVE_TG" -d 0 -r "$REPS" -b "$BATCH" -ub "$UBATCH" -t "$THREADS" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
    printf 'COMMAND: xargs ' >&2; printf '%q ' "$BENCH" "${base[@]}" >&2; printf '< %q\n' "$barg" >&2
    printf 'FITTED ARGS: %s\n' "$(cat "$barg")" >&2
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%s\n' '[]' > "$json"
        return 0
    fi
    xargs "$BENCH" "${base[@]}" < "$barg" > "$json" 2> >(tee "$log" >&2)
    rc=$?
    if [ "$rc" -ne 0 ]; then
        warn "shallow benchmark failed"
        show_capture_on_failure "$rc" "$json" "$log"
        return "$rc"
    fi
}

bench_curves_one() {
    local repo="$1" quant="$2" kvk="$3" kvv="$4" fitfile="$5" stem="$6" ctx="$7"
    local barg="$stem.bench.args"
    local depth_json="$stem.depth-curve.json" depth_log="$stem.depth-curve.log"
    local pp_json="$stem.pp-curve.json" pp_log="$stem.pp-curve.log" rc
    local cap="$ctx" max_depth depth_csv pp_csv
    bench_args_from_fit "$fitfile" > "$barg"

    [ "$cap" -gt "$CURVE_CAP" ] && cap="$CURVE_CAP"
    [ "$cap" -ge 512 ] || return 1

    # A depth-D PP probe needs room for the configured prompt test.
    max_depth=$((cap - CURVE_PP))
    [ "$max_depth" -lt 0 ] && max_depth=0
    depth_csv="$(curve_ladder "$max_depth" "$CURVE_DEPTHS" 1)" || return 1
    pp_csv="$(curve_ladder "$cap" "$CURVE_PP_SIZES" 0)" || return 1

    # --full-depth historically meant "test the edge". The normal curve already
    # includes max_depth exactly; retain the option as a compatibility alias.
    printf '%s\n' "$depth_csv" > "$stem.depths.txt"
    printf '%s\n' "$pp_csv" > "$stem.pp-sizes.txt"

    local -a common=(-hf "$repo:$quant" -r "$REPS" -b "$BATCH" -ub "$UBATCH" -t "$THREADS" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)

    printf 'COMMAND: xargs ' >&2; printf '%q ' "$BENCH" "${common[@]}" -p "$CURVE_PP" -n "$CURVE_TG" -d "$depth_csv" >&2; printf '< %q\n' "$barg" >&2
    printf 'FITTED ARGS: %s\n' "$(cat "$barg")" >&2
    printf 'COMMAND: xargs ' >&2; printf '%q ' "$BENCH" "${common[@]}" -p "$pp_csv" -n "$CURVE_TG" -d 0 >&2; printf '< %q\n' "$barg" >&2
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%s\n' '[]' > "$depth_json"
        printf '%s\n' '[]' > "$pp_json"
        return 0
    fi

    info "Context-depth curve: pp${CURVE_PP}/tg${CURVE_TG} @ d=${depth_csv}"
    xargs "$BENCH" "${common[@]}" -p "$CURVE_PP" -n "$CURVE_TG" -d "$depth_csv" < "$barg" > "$depth_json" 2> >(tee "$depth_log" >&2)
    rc=$?
    if [ "$rc" -ne 0 ]; then
        warn "context-depth benchmark failed"
        show_capture_on_failure "$rc" "$depth_json" "$depth_log"
        return "$rc"
    fi

    info "Full-prompt PP curve: p=${pp_csv} @ d=0"
    # llama-bench always has a TG test as well; the parser below ignores the
    # redundant tg128 row and keeps only PP rows from this invocation.
    xargs "$BENCH" "${common[@]}" -p "$pp_csv" -n "$CURVE_TG" -d 0 < "$barg" > "$pp_json" 2> >(tee "$pp_log" >&2)
    rc=$?
    if [ "$rc" -ne 0 ]; then
        warn "full-prompt benchmark failed"
        show_capture_on_failure "$rc" "$pp_json" "$pp_log"
        return "$rc"
    fi
}

parse_bench() {
    local json="$1" field="$2"
    case "$field" in
        pp) jq -r --argjson p "$CURVE_PP" '[.[]? | select(.n_prompt==$p and .n_gen==0 and .n_depth==0) | .avg_ts][0] // ""' "$json" 2>/dev/null ;;
        tg) jq -r --argjson n "$CURVE_TG" '[.[]? | select(.n_prompt==0 and .n_gen==$n and .n_depth==0) | .avg_ts][0] // ""' "$json" 2>/dev/null ;;
        ngl) jq -r '[.[]? | .n_gpu_layers][0] // ""' "$json" 2>/dev/null ;;
    esac
}

append_curves_tsv() {
    local alias="$1" repo="$2" quant="$3" kvk="$4" kvv="$5" maxctx="$6" stem="$7" out="$8"
    local depth_json="$stem.depth-curve.json" pp_json="$stem.pp-curve.json"
    [ -s "$depth_json" ] && jq -r \
        --arg model "$alias" --arg repo "$repo" --arg quant "$quant" --arg kvk "$kvk" --arg kvv "$kvv" \
        --arg maxctx "$maxctx" --arg backend "$BACKEND" --arg commit "$COMMIT" \
        '.[] | select((.n_prompt > 0 and .n_gen == 0) or (.n_prompt == 0 and .n_gen > 0)) |
         [$model,$repo,$quant,$kvk,$kvv,$maxctx,"depth",(.n_depth|tostring),(.n_prompt|tostring),(.n_gen|tostring),(.avg_ts|tostring),(.stddev_ts|tostring),(.n_gpu_layers|tostring),$backend,$commit] | @tsv' \
        "$depth_json" >> "$out"
    [ -s "$pp_json" ] && jq -r \
        --arg model "$alias" --arg repo "$repo" --arg quant "$quant" --arg kvk "$kvk" --arg kvv "$kvv" \
        --arg maxctx "$maxctx" --arg backend "$BACKEND" --arg commit "$COMMIT" \
        '.[] | select(.n_prompt > 0 and .n_gen == 0 and .n_depth == 0) |
         [$model,$repo,$quant,$kvk,$kvv,$maxctx,"full_prompt",(.n_depth|tostring),(.n_prompt|tostring),(.n_gen|tostring),(.avg_ts|tostring),(.stddev_ts|tostring),(.n_gpu_layers|tostring),$backend,$commit] | @tsv' \
        "$pp_json" >> "$out"
}


model_target_ctx() {
    local meta="$1" native
    native="$(printf '%s' "$meta" | cut -f5)"
    if [ "$TARGET_CTX" = "native" ]; then printf '%s\n' "$native"; else printf '%s\n' "$TARGET_CTX"; fi
}


context_tiers_for_cap() {
    # Useful maxima only. We deliberately do not spend time below MIN_CTX.
    # The exact cap is always tried first; then a short descending ladder.
    local cap="$1" x
    local -a raw=("$cap" 1048576 524288 262144 196608 131072 65536 "$MIN_CTX")
    for x in "${raw[@]}"; do
        [[ "$x" =~ ^[0-9]+$ ]] || continue
        [ "$x" -le "$cap" ] || continue
        [ "$x" -ge "$MIN_CTX" ] || continue
        printf '%s\n' "$x"
    done | sort -nr -u
}

metadata_best_ctx() {
    # Highest useful context tier that passes the metadata-only lower bound.
    # Prints 0 when even MIN_CTX is impossible. No model payload is downloaded.
    local alias="$1" repo="$2" prefix="$3" quant="$4" kvk="$5" kvv="$6" cap="$7" meta="$8"
    local c
    while read -r c; do
        [ -n "$c" ] || continue
        if metadata_vram_preflight "$alias" "$repo" "$prefix" "$quant" "$kvk" "$kvv" "$c" "$meta" 0 physical; then
            printf '%s\n' "$c"
            return 0
        fi
    done < <(context_tiers_for_cap "$cap")
    printf '0\n'
    return 1
}

fit_pair_best_ctx() {
    # Try the largest useful context first and fall back only as far as MIN_CTX.
    # Stage-0 theory used physical VRAM; llama-fit-params below is authoritative
    # for the memory actually available at execution time.
    # stdout on success: actual_ctx<TAB>fit_file<TAB>ngl
    # rc=2 means every useful context spilled/failed residency; at MIN_CTX this
    # proves all higher-memory KV pairs are also unusable for this weight quant.
    local alias="$1" repo="$2" prefix="$3" quant="$4" kvk="$5" kvv="$6" cap="$7" meta="$8" layers="$9" stem="${10}"
    local c oldctx fitfile frc ngl tried=0
    while read -r c; do
        [ -n "$c" ] || continue
        # Once this GGUF has been selected/downloaded, metadata is advisory only.
        # Exact fitting is cheap and authoritative, so always test the requested
        # context tiers from the current hard cap downward.  v16 incorrectly
        # reused the conservative metadata gate here and could therefore skip
        # 262K without ever asking llama-fit-params whether it really fits.
        tried=1
        printf 'TRY CONTEXT: %-14s %-14s KV=%-11s ctx=%s\n' "$alias" "$quant" "$kvk/$kvv" "$c" >&2
        oldctx="$CTX"; CTX="$c"
        fitfile="$(fit_one "$repo" "$quant" "$kvk" "$kvv" "$stem.ctx-$c")"; frc=$?
        CTX="$oldctx"
        if [ "$frc" -ne 0 ]; then
            printf 'FIT ERROR at ctx=%s is not treated as a memory proof; stopping this pair without pruning higher KV.\n' "$c" >&2
            return 1
        fi
        ngl="$(fit_ngl "$fitfile")"
        if [ "$BACKEND" = "cuda" ] && ! gpu_resident_at_target "$fitfile" "$layers"; then
            printf 'SPILL at ctx=%s; trying the next lower useful context tier.\n' "$c" >&2
            continue
        fi
        printf 'FIT CONTEXT: %s %s KV=%s/%s -> %s tokens fully resident\n' "$alias" "$quant" "$kvk" "$kvv" "$c" >&2
        printf '%s\t%s\t%s\n' "$c" "$fitfile" "$ngl"
        return 0
    done < <(context_tiers_for_cap "$cap")
    [ "$tried" -eq 1 ] && return 2
    return 2
}

fit_ngl() {
    awk '{for(i=1;i<=NF;i++) if($i=="-ngl" && (i+1)<=NF){print $(i+1); exit}}' "$1"
}

gpu_resident_at_target() {
    # Decide whether llama-fit-params found a fully device-resident placement.
    #
    # llama-fit-params can legitimately emit `-ngl -1` when a requested fixed
    # context fits without reducing GPU offload. At the low-level llama model
    # API, negative n_gpu_layers means all layers. In the common CLI, -1 is also
    # the auto/default sentinel while fitting is enabled. Since this file is the
    # OUTPUT of a successful exact fit, a negative -ngl with no explicit =CPU
    # tensor overrides means the fitter did not need to spill model tensors.
    #
    # v16 incorrectly required ^[0-9]+$, so every valid `-ngl -1` result was
    # mislabeled as SPILL.
    local fitfile="$1" layers="$2" ngl need
    [ "$BACKEND" = "cuda" ] || return 0

    if grep -Eq '=CPU([,"]|$)' "$fitfile"; then
        printf 'RESIDENCY CHECK: explicit =CPU tensor override found -> SPILL\n' >&2
        return 1
    fi

    ngl="$(fit_ngl "$fitfile")"
    need=$((layers + 1))

    case "$ngl" in
        all)
            printf 'RESIDENCY CHECK: -ngl all and no CPU overrides -> FULL GPU\n' >&2
            return 0
            ;;
        -[0-9]*)
            printf 'RESIDENCY CHECK: -ngl %s returned by successful fit and no CPU overrides -> FULL GPU\n' "$ngl" >&2
            return 0
            ;;
        ''|*[!0-9]*)
            printf 'RESIDENCY CHECK: unrecognized -ngl value %q -> NOT accepting as full GPU\n' "$ngl" >&2
            return 1
            ;;
        *)
            if [ "$ngl" -ge "$need" ]; then
                printf 'RESIDENCY CHECK: -ngl %s >= required %s and no CPU overrides -> FULL GPU\n' "$ngl" "$need" >&2
                return 0
            fi
            printf 'RESIDENCY CHECK: -ngl %s < required %s -> SPILL\n' "$ngl" "$need" >&2
            return 1
            ;;
    esac
}

screen_one() {
    # One cheap low-context PP/TG screen.
    # Results return through globals so CPU tuning persists in this shell.
    SCREEN_PP=""; SCREEN_TG=""; SCREEN_PP_SD=""; SCREEN_TG_SD=""; SCREEN_NGL=""
    # CPU is special: PP and TG must be separate because llama-bench has only one
    # -t control. This lets PP use THREADS_BATCH and TG use THREADS, and we leave
    # warmup enabled so mmap first-touch/storage latency is not counted as PP.
    local repo="$1" quant="$2" kvk="$3" kvv="$4" fitfile="$5" stem="$6"
    local barg="$stem.screen.args" rc pp tg ngl
    bench_args_from_fit "$fitfile" > "$barg"

    if [ "$BACKEND" = "cpu" ]; then
        cpu_autotune_once "$repo" "$quant" "$kvk" "$kvv" "$fitfile" "$stem" || true
        cpu_numa_args
        cpu_command_prefix
        local ppjson="$stem.cpu-pp.json" pplog="$stem.cpu-pp.log"
        local tgjson="$stem.cpu-tg.json" tglog="$stem.cpu-tg.log"
        local -a ppcmd=(-hf "$repo:$quant" -p "$CURVE_PP" -n 0 -d 0 -r "$REPS" --progress
            -b "$BATCH" -ub "$UBATCH" -t "$THREADS_BATCH" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
        local -a tgcmd=(-hf "$repo:$quant" -p 0 -n "$CURVE_TG" -d 0 -r "$REPS" --progress
            -b "$BATCH" -ub "$UBATCH" -t "$THREADS" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
        ppcmd+=("${CPU_NUMA_ARGS[@]}"); tgcmd+=("${CPU_NUMA_ARGS[@]}")
        printf 'CPU SCREEN: warmup=ON PP%s threads=%s; TG%s threads=%s; NUMA=%s placement=%s load_mode=%s BLAS=%s\n' \
            "$CURVE_PP" "$THREADS_BATCH" "$CURVE_TG" "$THREADS" "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" "$CPU_LOAD_MODE_EFFECTIVE" "$CPU_BLAS_EFFECTIVE" >&2
        printf 'COMMAND PP: xargs ' >&2; printf '%q ' "$BENCH" "${ppcmd[@]}" >&2; printf '< %q\n' "$barg" >&2
        if [ "$DRY_RUN" -eq 0 ]; then
            xargs "${CPU_CMD_PREFIX[@]}" "$BENCH" "${ppcmd[@]}" < "$barg" > "$ppjson" 2> >(tee "$pplog" >&2)
            rc=$?
            [ "$rc" -eq 0 ] || { warn "CPU PP screen failed"; show_capture_on_failure "$rc" "$ppjson" "$pplog"; return "$rc"; }
        else
            printf '[]\n' > "$ppjson"
        fi
        printf 'COMMAND TG: xargs ' >&2; printf '%q ' "$BENCH" "${tgcmd[@]}" >&2; printf '< %q\n' "$barg" >&2
        if [ "$DRY_RUN" -eq 0 ]; then
            xargs "${CPU_CMD_PREFIX[@]}" "$BENCH" "${tgcmd[@]}" < "$barg" > "$tgjson" 2> >(tee "$tglog" >&2)
            rc=$?
            [ "$rc" -eq 0 ] || { warn "CPU TG screen failed"; show_capture_on_failure "$rc" "$tgjson" "$tglog"; return "$rc"; }
        else
            printf '[]\n' > "$tgjson"
        fi
        pp="$(jq -r --argjson p "$CURVE_PP" '[.[]? | select(.n_prompt==$p and .n_gen==0 and .n_depth==0) | .avg_ts][0] // ""' "$ppjson")"
        tg="$(jq -r --argjson n "$CURVE_TG" '[.[]? | select(.n_prompt==0 and .n_gen==$n and .n_depth==0) | .avg_ts][0] // ""' "$tgjson")"
        SCREEN_PP_SD="$(jq -r --argjson p "$CURVE_PP" '[.[]? | select(.n_prompt==$p and .n_gen==0 and .n_depth==0) | .stddev_ts][0] // ""' "$ppjson")"
        SCREEN_TG_SD="$(jq -r --argjson n "$CURVE_TG" '[.[]? | select(.n_prompt==0 and .n_gen==$n and .n_depth==0) | .stddev_ts][0] // ""' "$tgjson")"
        ngl="$(jq -r '[.[]? | .n_gpu_layers][0] // 0' "$tgjson")"
        [ -n "$pp" ] && [ -n "$tg" ] || { warn "could not parse CPU shallow benchmark"; cat "$ppjson" "$tgjson" >&2; return 1; }
        SCREEN_PP="$pp"; SCREEN_TG="$tg"; SCREEN_NGL="$ngl"
        return 0
    fi

    local json="$stem.screen.json" log="$stem.screen.log"
    local -a base=(-hf "$repo:$quant" -p "$CURVE_PP" -n "$CURVE_TG" -d 0 -r "$REPS" --no-warmup --progress
        -b "$BATCH" -ub "$UBATCH" -t "$THREADS" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
    printf 'COMMAND: xargs ' >&2; printf '%q ' "$BENCH" "${base[@]}" >&2; printf '< %q\n' "$barg" >&2
    printf 'FITTED ARGS: %s\n' "$(cat "$barg")" >&2
    if [ "$DRY_RUN" -eq 1 ]; then SCREEN_PP=0; SCREEN_TG=0; SCREEN_NGL=0; return 0; fi
    xargs "$BENCH" "${base[@]}" < "$barg" > "$json" 2> >(tee "$log" >&2)
    rc=$?
    if [ "$rc" -ne 0 ]; then
        warn "shallow benchmark failed"
        show_capture_on_failure "$rc" "$json" "$log"
        return "$rc"
    fi
    pp="$(jq -r --argjson p "$CURVE_PP" '[.[]? | select(.n_prompt==$p and .n_gen==0 and .n_depth==0) | .avg_ts][0] // ""' "$json")"
    tg="$(jq -r --argjson n "$CURVE_TG" '[.[]? | select(.n_prompt==0 and .n_gen==$n and .n_depth==0) | .avg_ts][0] // ""' "$json")"
    SCREEN_PP_SD="$(jq -r --argjson p "$CURVE_PP" '[.[]? | select(.n_prompt==$p and .n_gen==0 and .n_depth==0) | .stddev_ts][0] // ""' "$json")"
    SCREEN_TG_SD="$(jq -r --argjson n "$CURVE_TG" '[.[]? | select(.n_prompt==0 and .n_gen==$n and .n_depth==0) | .stddev_ts][0] // ""' "$json")"
    ngl="$(jq -r '[.[]? | .n_gpu_layers][0] // ""' "$json")"
    [ -n "$pp" ] && [ -n "$tg" ] || { warn "could not parse shallow benchmark"; cat "$json" >&2; return 1; }
    SCREEN_PP="$pp"; SCREEN_TG="$tg"; SCREEN_NGL="$ngl"
    return 0
}

progressive_depth_csv() {
    local target="$1" max_depth=$((target - CURVE_TG - EDGE_RESERVE)) spec="$CURVE_DEPTHS" x found
    local -a raw out=()
    [ "$max_depth" -ge 512 ] || return 1
    if [ "$spec" = "auto" ]; then raw=(4096 8192 16384 32768 65536 131072 196608); else IFS=',' read -r -a raw <<< "$spec"; fi
    for x in "${raw[@]}"; do
        [[ "$x" =~ ^[0-9]+$ ]] || { err "bad depth point: $x"; return 1; }
        [ "$x" -le "$max_depth" ] || continue
        found=0; for n in "${out[@]}"; do [ "$n" -eq "$x" ] && found=1; done
        [ "$found" -eq 0 ] && out+=("$x")
    done
    found=0; for n in "${out[@]}"; do [ "$n" -eq "$max_depth" ] && found=1; done
    [ "$found" -eq 0 ] && out+=("$max_depth")
    printf '%s\n' "${out[@]}" | sort -n -u | paste -sd, -
}

parse_fit_args_array() {
    local f="$1"
    FIT_ARGS_ARRAY=()
    mapfile -d '' -t FIT_ARGS_ARRAY < <(python3 - "$f" <<'PYFIT'
import shlex, sys
text = open(sys.argv[1], encoding='utf-8').read()
for a in shlex.split(text):
    sys.stdout.buffer.write(a.encode() + b'\0')
PYFIT
)
}

progressive_server_sweep() {
    # Deep sweep using ONE growing prompt cache. Each request extends the previous
    # deterministic token prefix; server timings expose the newly evaluated suffix.
    # Generated branches are discarded automatically when the next longer prefix
    # is submitted to the same slot with cache_prompt=true.
    local alias="$1" repo="$2" quant="$3" kvk="$4" kvv="$5" fitfile="$6" stem="$7" target="$8"
    local depths port server_log raw_jsonl out_tsv pid rc
    depths="$(progressive_depth_csv "$target")" || return 1
    port="$PORT"
    server_log="$stem.progressive-server.log"
    raw_jsonl="$stem.progressive-responses.jsonl"
    out_tsv="$stem.progressive.tsv"
    LAST_PROGRESSIVE_TSV="$out_tsv"
    : > "$raw_jsonl"
    printf 'model\tquant\tkv_k\tkv_v\ttarget_ctx\tdepth\tcache_n\tunique_new_tokens\treplay_tokens\tnew_pp_tokens\tsegment_pp_tps\tcumulative_pp_tps\ttg_tokens\ttg_tps\tprompt_ms\tpredicted_ms\n' > "$out_tsv"

    parse_fit_args_array "$fitfile"
    local load_mode="mmap"
    [ "$BACKEND" = "cpu" ] && load_mode="$CPU_LOAD_MODE_EFFECTIVE"
    local -a cmd=("$SERVER" -hf "$repo:$quant" -ctk "$kvk" -ctv "$kvv" -fa on -b "$BATCH" -ub "$UBATCH" -t "$THREADS" -tb "$THREADS_BATCH" -lm "$load_mode" --fit off --parallel 1 --cache-prompt --no-cache-idle-slots --cache-ram 0 --host 127.0.0.1 --port "$port" --no-ui)
    if [ "$BACKEND" = "cpu" ] && [ "$CPU_NUMA_EFFECTIVE" != "disabled" ]; then cmd+=(--numa "$CPU_NUMA_EFFECTIVE"); fi
    [ "$VISION" -eq 0 ] && cmd+=(--no-mmproj)
    cmd+=("${FIT_ARGS_ARRAY[@]}")
    info "ONE FAST progressive PP+TG128 sweep: $alias | $quant | KV $kvk/$kvv | checkpoints=$depths | target=$target | edge_reserve=$EDGE_RESERVE"
    printf 'METHOD: fill each new context interval ONCE by prompt processing; then sample TG%s at that occupied depth. No continuous-generation fill between checkpoints.\n' "$CURVE_TG" >&2
    cpu_command_prefix
    print_cmd "${CPU_CMD_PREFIX[@]}" "${cmd[@]}"
    if [ "$DRY_RUN" -eq 1 ]; then return 0; fi

    "${CPU_CMD_PREFIX[@]}" "${cmd[@]}" > >(tee "$server_log" >&2) 2>&1 &
    pid=$!
    cleanup_progressive_server() { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }

    local ready=0 i
    for i in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
        if ! kill -0 "$pid" 2>/dev/null; then
            err "llama-server exited before becoming healthy"
            [ -s "$server_log" ] && cat "$server_log" >&2
            cleanup_progressive_server
            return 1
        fi
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then
        err "llama-server never became healthy"
        cleanup_progressive_server
        return 1
    fi

    printf '\n%-10s %-10s %-11s %-8s %-10s %-14s %-14s %-10s %-10s\n' DEPTH CACHE UNIQUE_NEW REPLAY PROCESSED SEGMENT_PP_TPS CUM_PP_TPS TG_TOKENS TG_TPS
    python3 - "127.0.0.1" "$port" "$depths" "$CURVE_TG" "$raw_jsonl" "$out_tsv" "$alias" "$quant" "$kvk" "$kvv" "$target" "$CACHE_REPLAY_TOLERANCE" <<'PYPROG'
import json, sys, urllib.request, urllib.error
host, port, depth_csv, ngen, raw_path, tsv_path, model, quant, kvk, kvv, target, replay_tolerance = sys.argv[1:]
ngen = int(ngen); target = int(target); replay_tolerance = int(replay_tolerance)
base = f"http://{host}:{port}"

def post(path, obj):
    data = json.dumps(obj, separators=(",", ":")).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"HTTP ERROR {e.code}: {body}", file=sys.stderr, flush=True)
        raise

m = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=30).read())
n_vocab = int(m["data"][0].get("meta", {}).get("n_vocab") or 0)
if n_vocab < 2:
    # Fallback: obtain at least one known-valid token and cycle a small range around it.
    tok = post("/tokenize", {"content":"The quick brown fox jumps over the lazy dog.","add_special":False})["tokens"]
    if not tok: raise RuntimeError("could not obtain valid benchmark tokens")
    vocab_tokens = [int(x) for x in tok]
    make_tokens = lambda n: [vocab_tokens[i % len(vocab_tokens)] for i in range(n)]
else:
    # Deterministic pseudo-random in-vocab IDs, analogous to llama-bench's random-token prompt.
    def make_tokens(n):
        mod = n_vocab - 1
        return [1 + ((i * 48271 + 17) % mod) for i in range(n)]

depths = [int(x) for x in depth_csv.split(",") if x]
all_tokens = make_tokens(max(depths))
sum_prompt_n = 0
sum_prompt_ms = 0.0
prev_depth = 0
for depth in depths:
    payload = {
        "prompt": all_tokens[:depth],
        "n_predict": ngen,
        "id_slot": 0,
        "cache_prompt": True,
        "stream": False,
        "temperature": 0.0,
        "seed": 1,
        "ignore_eos": True,
        "repeat_penalty": 1.0,
    }
    resp = post("/completion", payload)
    with open(raw_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(resp, separators=(",", ":")) + "\n")
        f.flush()
    t = resp.get("timings") or {}
    cache_n = int(t.get("cache_n", resp.get("tokens_cached", 0)) or 0)
    prompt_n = int(t.get("prompt_n", 0) or 0)
    prompt_ms = float(t.get("prompt_ms", 0.0) or 0.0)
    seg_pp = float(t.get("prompt_per_second", 0.0) or 0.0)
    pred_n = int(t.get("predicted_n", 0) or 0)
    pred_ms = float(t.get("predicted_ms", 0.0) or 0.0)
    tg = float(t.get("predicted_per_second", 0.0) or 0.0)
    unique_new = depth - prev_depth
    replay_n = max(0, prompt_n - unique_new)
    sum_prompt_n += prompt_n
    sum_prompt_ms += prompt_ms
    # End-to-end cumulative PP should use UNIQUE prompt progress, not count the
    # handful of tail tokens that hybrid/recurrent cache restoration may replay.
    cum_pp = (1000.0 * depth / sum_prompt_ms) if sum_prompt_ms > 0 else 0.0
    if prev_depth:
        cache_short = max(0, prev_depth - cache_n)
        if cache_short > replay_tolerance:
            print(f"WARNING: material cache reuse shortfall at depth {depth}: cache_n={cache_n}, previous_depth={prev_depth}, replay={cache_short} (> tolerance {replay_tolerance})", file=sys.stderr, flush=True)
        elif cache_short > 0:
            reuse_pct = 100.0 * cache_n / prev_depth
            print(f"CACHE REUSE: depth {depth}: reused {cache_n}/{prev_depth} prior tokens ({reuse_pct:.5f}%); replayed {cache_short} tail token(s)", file=sys.stderr, flush=True)
    print(f"{depth:<10d} {cache_n:<10d} {unique_new:<11d} {replay_n:<8d} {prompt_n:<10d} {seg_pp:<14.2f} {cum_pp:<14.2f} {pred_n:<10d} {tg:<10.2f}", flush=True)
    with open(tsv_path, "a", encoding="utf-8") as f:
        f.write("\t".join(map(str,[model,quant,kvk,kvv,target,depth,cache_n,unique_new,replay_n,prompt_n,f"{seg_pp:.6f}",f"{cum_pp:.6f}",pred_n,f"{tg:.6f}",f"{prompt_ms:.3f}",f"{pred_ms:.3f}"])) + "\n")
        f.flush()
    prev_depth = depth
    if resp.get("truncated"):
        print(f"ERROR: server reported context truncation at depth {depth}", file=sys.stderr, flush=True)
        sys.exit(3)
PYPROG
    rc=$?
    cleanup_progressive_server
    if [ "$rc" -ne 0 ]; then
        err "progressive sweep failed with status $rc; completed rows remain in $out_tsv"
        return "$rc"
    fi
    info "Progressive results: $out_tsv"
    return 0
}


analyze_compaction_policy() {
    # Zero-inference post-processing. Uses the measured progressive PP segment times
    # and sampled TG curve to solve a renewal/average-cost compaction policy.
    local curve_tsv="$1" out_tsv="$2"
    [ "$COMPACTION_ANALYSIS" -eq 1 ] || return 0
    [ -s "$curve_tsv" ] || { warn "compaction analysis skipped: progressive TSV is missing/empty: $curve_tsv"; return 0; }
    info "Compaction economics (post-processing only; no additional inference)"
    printf 'retains=%s pp_scales=%s extra_compaction_ms=%s\n' "$COMPACTION_RETAINS" "$COMPACTION_PP_SCALES" "$COMPACTION_EXTRA_MS" >&2
    python3 - "$curve_tsv" "$out_tsv" "$COMPACTION_RETAINS" "$COMPACTION_PP_SCALES" "$COMPACTION_EXTRA_MS" <<'PYCOMPACT'
import csv, math, sys
curve_path, out_path, retain_csv, scale_csv, extra_ms_s = sys.argv[1:]
extra_ms=float(extra_ms_s)
retains=[]
for x in retain_csv.split(','):
    x=x.strip().lower()
    if not x: continue
    mult=1
    if x.endswith('k'): mult=1024; x=x[:-1]
    elif x.endswith('m'): mult=1024*1024; x=x[:-1]
    retains.append(int(float(x)*mult))
scales=[float(x) for x in scale_csv.split(',') if x.strip()]
if not scales or any(x<=0 for x in scales):
    raise SystemExit("compaction PP scales must all be > 0")
rows=[]
with open(curve_path,encoding='utf-8') as f:
    r=csv.DictReader(f,delimiter='\t')
    for x in r:
        try:
            rows.append({
                'depth':int(x['depth']),
                'prompt_ms':float(x['prompt_ms']),
                'tg':float(x['tg_tps']),
            })
        except (KeyError,ValueError):
            pass
rows=sorted((x for x in rows if x['depth']>0 and x['tg']>0),key=lambda x:x['depth'])
if len(rows)<2:
    raise SystemExit("need at least two completed progressive points for compaction analysis")
max_depth=rows[-1]['depth']
# Measured PP segments are [previous_depth, depth] with prompt_ms for that interval.
segments=[]
prev=0
for x in rows:
    if x['depth']>prev:
        segments.append((prev,x['depth'],x['prompt_ms']))
    prev=x['depth']

def rebuild_ms(r, pp_scale):
    total=0.0
    for a,b,ms in segments:
        overlap=max(0,min(r,b)-a)
        if overlap<=0: continue
        total += ms*(overlap/(b-a))/pp_scale
        if b>=r: break
    return total

# Linear interpolation of measured TG t/s; flat outside measured range.
xs=[x['depth'] for x in rows]; ys=[x['tg'] for x in rows]
def tg_at(pos):
    if pos<=xs[0]: return ys[0]
    if pos>=xs[-1]: return ys[-1]
    lo=0; hi=len(xs)-1
    while hi-lo>1:
        m=(lo+hi)//2
        if xs[m]<=pos: lo=m
        else: hi=m
    x0,x1=xs[lo],xs[hi]; y0,y1=ys[lo],ys[hi]
    t=(pos-x0)/(x1-x0)
    return y0+(y1-y0)*t

# Integrate generation seconds with small fixed steps. The measured curve is sparse,
# so 256-token resolution is far finer than the source data and computationally cheap.
step=256
results=[]
for scale in scales:
    for R in sorted(set(retains)):
        if R < xs[0] or R >= max_depth-step:
            continue
        rebuild=rebuild_ms(R,scale)/1000.0 + extra_ms/1000.0
        best=None
        gen_s=0.0
        pos=R
        next_report=R+step
        while pos < max_depth:
            nxt=min(max_depth,pos+step)
            mid=(pos+nxt)/2
            gen_s += (nxt-pos)/max(tg_at(mid),1e-9)
            pos=nxt
            if pos-R < 1024:
                continue
            produced=pos-R
            cycle_s=rebuild+gen_s
            eff_tps=produced/cycle_s if cycle_s>0 else 0
            cand=(eff_tps,pos,cycle_s,gen_s,tg_at(pos))
            if best is None or cand[0]>best[0]: best=cand
        if best:
            eff,H,cycle_s,gen_s,tgh=best
            results.append({
                'pp_scale':scale,'retain_ctx':R,'trigger_ctx':int(H),
                'rebuild_s':rebuild,'generation_s':gen_s,'cycle_s':cycle_s,
                'cycle_new_tokens':int(H-R),'effective_tps':eff,'tg_at_trigger':tgh,
            })
fields=['pp_scale','retain_ctx','trigger_ctx','rebuild_s','generation_s','cycle_s','cycle_new_tokens','effective_tps','tg_at_trigger']
with open(out_path,'w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter='\t'); w.writeheader()
    for x in results:
        y=x.copy()
        for k in ('rebuild_s','generation_s','cycle_s','effective_tps','tg_at_trigger'):
            y[k]=f"{x[k]:.6f}"
        w.writerow(y)
print(f"{'PPx':>5} {'RETAIN':>9} {'TRIGGER':>9} {'REBUILD_s':>10} {'CYCLE_s':>9} {'EFF_TG':>10} {'TG@TRIG':>10}")
for x in results:
    print(f"{x['pp_scale']:>5.2f} {x['retain_ctx']:>9d} {x['trigger_ctx']:>9d} {x['rebuild_s']:>10.2f} {x['cycle_s']:>9.2f} {x['effective_tps']:>10.2f} {x['tg_at_trigger']:>10.2f}")
if results:
    for scale in scales:
        rr=[x for x in results if abs(x['pp_scale']-scale)<1e-12]
        if rr:
            b=max(rr,key=lambda x:x['effective_tps'])
            print(f"OPTIMUM PPx={scale:g}: retain={b['retain_ctx']} trigger≈{b['trigger_ctx']} effective≈{b['effective_tps']:.2f} tok/s")
print("NOTE: throughput-only policy; semantic/information loss from compaction is not modeled.")
PYCOMPACT
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        warn "compaction analysis failed; benchmark results remain valid"
        return "$rc"
    fi
    printf 'compaction_policy=%s\n' "$out_tsv" >&2
}

cpu_openblas_available() {
    have pkg-config && { pkg-config --exists openblas 2>/dev/null || pkg-config --exists openblas64 2>/dev/null; }
}

cpu_ablation_bench_binary() {
    local variant="$1" bdir="$WORK/build-cpu-$COMMIT-ablation-$variant" bin="$bdir/bin/llama-bench"
    local -a args=(-S "$SRC" -B "$bdir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF)
    case "$variant" in
        native)
            if [ "$CPU_BLAS_EFFECTIVE" = "off" ]; then printf '%s\n' "$BENCH"; return 0; fi
            args+=(-DGGML_NATIVE=ON -DGGML_BLAS=OFF) ;;
        openblas)
            cpu_openblas_available || return 2
            args+=(-DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS) ;;
        noavx2)
            # WALL build first run only: AVX2-off keeps AVX/FMA/F16C/BMI2 and native scheduling tune.
            args+=(-DGGML_NATIVE=OFF -DGGML_BLAS=OFF -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=OFF
                   -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON
                   "-DCMAKE_C_FLAGS=-mtune=native" "-DCMAKE_CXX_FLAGS=-mtune=native") ;;
        *) return 2 ;;
    esac
    if [ ! -x "$bin" ]; then
        wall_est "~1-4 min first build; ~0 s cached" "$variant llama-bench build"
        [ "$DRY_RUN" -eq 1 ] && { printf '%s\n' "$bin"; return 0; }
        cmake "${args[@]}" || return 1
        cmake --build "$bdir" -j "$logical" --target llama-bench || return 1
    else
        wall_est "~0 s build (cached)" "$variant llama-bench"
    fi
    printf '%s\n' "$bin"
}

run_cpu_build_ablations() {
    [ "$BACKEND" = "cpu" ] || return 0
    [ "$CPU_ABLATIONS" = "auto" ] || return 0
    local out="$RESULTS/cpu-build-ablations-$RUN_ID.tsv" barg="$base_wstem.cpu-ablation.args"
    local variant abench cached pjson tjson plog tlog pp tg ppsd tgsd rc note expected
    local -a variants=(native openblas noavx2)
    bench_args_from_fit "$W_FIT_FINAL" > "$barg"
    cpu_numa_args; cpu_command_prefix
    # WALL cached ~2-4m total; first run adds ~1-4m per extra build.
    wall_est "~2-4 min cached; ~4-12 min first run" "native/OpenBLAS/AVX2-off winner ablations"
    printf 'run_id\tscript_version\tvariant\tbuild_was_cached\tmodel\tquant\tkv_k\tkv_v\tpp_tokens\tpp_reps\tpp_tps\tpp_sd_tps\ttg_tokens\ttg_reps\ttg_tps\ttg_sd_tps\tdecode_threads\tpp_threads\tbatch\tubatch\tnuma\tplacement\tbench_path\tnote\n' > "$out"

    for variant in "${variants[@]}"; do
        if [ "$variant" = "openblas" ] && ! cpu_openblas_available; then
            printf '%s\t%s\topenblas\tNA\t%s\t%s\t%s\t%s\t%s\t%s\t\t\t%s\t%s\t\t\t%s\t%s\t%s\t%s\t%s\t%s\t\tSKIP: OpenBLAS dev package not found\n' \
                "$RUN_ID" "$VERSION" "$W_ALIAS" "$W_QUANT" "$W_KVK" "$W_KVV" "$CPU_ABL_PP" "$CPU_ABL_REPS" "$CPU_ABL_TG" "$CPU_ABL_REPS" \
                "$THREADS" "$THREADS_BATCH" "$BATCH" "$UBATCH" "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" >> "$out"
            continue
        fi
        expected="$WORK/build-cpu-$COMMIT-ablation-$variant/bin/llama-bench"
        [ "$variant" = "native" ] && [ "$CPU_BLAS_EFFECTIVE" = "off" ] && expected="$BENCH"
        [ -x "$expected" ] && cached=YES || cached=NO
        abench="$(cpu_ablation_bench_binary "$variant")"; rc=$?
        if [ "$rc" -ne 0 ] || [ ! -x "$abench" ]; then
            warn "CPU ablation build skipped/failed: $variant"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t\t%s\t%s\t\t\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tBUILD_FAIL\n' \
                "$RUN_ID" "$VERSION" "$variant" "$cached" "$W_ALIAS" "$W_QUANT" "$W_KVK" "$W_KVV"                 "$CPU_ABL_PP" "$CPU_ABL_REPS" "$CPU_ABL_TG" "$CPU_ABL_REPS" "$THREADS" "$THREADS_BATCH"                 "$BATCH" "$UBATCH" "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" "${abench:-}" >> "$out"
            continue
        fi

        pjson="$RESULTS/cpu-ablation-$RUN_ID-$variant-pp.json"; plog="$RESULTS/cpu-ablation-$RUN_ID-$variant-pp.log"
        tjson="$RESULTS/cpu-ablation-$RUN_ID-$variant-tg.json"; tlog="$RESULTS/cpu-ablation-$RUN_ID-$variant-tg.log"
        local -a ppcmd=(-hf "$W_REPO:$W_QUANT" -p "$CPU_ABL_PP" -n 0 -d 0 -r "$CPU_ABL_REPS" --progress
            -b "$BATCH" -ub "$UBATCH" -t "$THREADS_BATCH" -ctk "$W_KVK" -ctv "$W_KVV" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
        local -a tgcmd=(-hf "$W_REPO:$W_QUANT" -p 0 -n "$CPU_ABL_TG" -d 0 -r "$CPU_ABL_REPS" --progress
            -b "$BATCH" -ub "$UBATCH" -t "$THREADS" -ctk "$W_KVK" -ctv "$W_KVV" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
        ppcmd+=("${CPU_NUMA_ARGS[@]}"); tgcmd+=("${CPU_NUMA_ARGS[@]}")
        xargs "${CPU_CMD_PREFIX[@]}" "$abench" "${ppcmd[@]}" < "$barg" > "$pjson" 2> >(tee "$plog" >&2) || { warn "CPU ablation PP failed: $variant"; continue; }
        xargs "${CPU_CMD_PREFIX[@]}" "$abench" "${tgcmd[@]}" < "$barg" > "$tjson" 2> >(tee "$tlog" >&2) || { warn "CPU ablation TG failed: $variant"; continue; }

        pp="$(jq -r --argjson p "$CPU_ABL_PP" '[.[]? | select(.n_prompt==$p and .n_gen==0)] | .[0].avg_ts // empty' "$pjson")"
        ppsd="$(jq -r --argjson p "$CPU_ABL_PP" '[.[]? | select(.n_prompt==$p and .n_gen==0)] | .[0].stddev_ts // empty' "$pjson")"
        tg="$(jq -r --argjson n "$CPU_ABL_TG" '[.[]? | select(.n_prompt==0 and .n_gen==$n)] | .[0].avg_ts // empty' "$tjson")"
        tgsd="$(jq -r --argjson n "$CPU_ABL_TG" '[.[]? | select(.n_prompt==0 and .n_gen==$n)] | .[0].stddev_ts // empty' "$tjson")"
        note="native -march=native"
        [ "$variant" = "openblas" ] && note="native + OpenBLAS; PP-targeted"
        [ "$variant" = "noavx2" ] && note="AVX/FMA/F16C/BMI2; AVX2 disabled"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$RUN_ID" "$VERSION" "$variant" "$cached" "$W_ALIAS" "$W_QUANT" "$W_KVK" "$W_KVV" \
            "$CPU_ABL_PP" "$CPU_ABL_REPS" "$pp" "$ppsd" "$CPU_ABL_TG" "$CPU_ABL_REPS" "$tg" "$tgsd" \
            "$THREADS" "$THREADS_BATCH" "$BATCH" "$UBATCH" "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" "$abench" "$note" >> "$out"
        printf 'CPU ABLATION RESULT: %-9s PP%s=%s±%s TG%s=%s±%s\n' "$variant" "$CPU_ABL_PP" "$pp" "$ppsd" "$CPU_ABL_TG" "$tg" "$tgsd" >&2
    done
    printf 'cpu_build_ablations=%s\n' "$out" >&2
}

write_backend_run_summary() {
    local status="${1:-complete}" elapsed now summary
    now="$(date +%s)"
    elapsed=$((now-SCRIPT_START_EPOCH))
    summary="$RESULTS/run-summary-$RUN_ID-$BACKEND.tsv"

    # WALL <1s: a tiny backend summary makes CUDA + CPU-child timing visible in the one-file handoff.
    printf 'run_id\tscript_version\tbackend\tcommit\tstatus\telapsed_s\tmodel\tquant\tkv_k\tkv_v\ttarget_ctx\tpp_probe_tps\ttg_probe_tps\tbatch\tubatch\tdecode_threads\tpp_threads\tnuma\tplacement\tblas\tdeep_status\n' > "$summary"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$RUN_ID" "$VERSION" "$BACKEND" "$COMMIT" "$status" "$elapsed" \
        "${W_ALIAS:-}" "${W_QUANT:-}" "${W_KVK:-}" "${W_KVV:-}" "${W_TARGET:-}" \
        "${W_PP:-${CPU_FINAL_PP:-}}" "${W_TG:-${CPU_FINAL_TG:-}}" \
        "$BATCH" "$UBATCH" "$THREADS" "$THREADS_BATCH" \
        "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" "$CPU_BLAS_EFFECTIVE" "${DEEP_SWEEP_STATUS:-}" \
        >> "$summary"
}

append_result_section() {
    local out="$1" section="$2" file="$3"
    [ -n "$file" ] && [ -s "$file" ] || return 0
    printf '\n===== %s =====\n' "$section" >> "$out"
    printf 'source_file\t%s\n' "$(basename "$file")" >> "$out"
    cat "$file" >> "$out"
    case "$(tail -c 1 "$file" 2>/dev/null || true)" in
        '') ;;
        *) printf '\n' >> "$out" ;;
    esac
}

write_all_results() {
    local out="$RESULTS/all-results-$RUN_ID.txt"
    local latest="$RESULTS/all-results-latest.txt"
    local f base section
    local -a files=()
    declare -A seen=()

    # WALL <1s typical: concatenate only result tables, not verbose logs/model payloads.
    wall_est "<1 s" "assemble one paste-ready result file"
    {
        printf 'LLAMA_PUSHBUTTON_ALL_RESULTS\t1\n'
        printf 'run_id\t%s\n' "$RUN_ID"
        printf 'script_version\t%s\n' "$VERSION"
        printf 'generated_at\t%s\n' "$(date -Is 2>/dev/null || date)"
        printf 'host\t%s\n' "$(hostname 2>/dev/null || printf unknown)"
        printf 'cache_root\t%s\n' "$CACHE_DIR"
        printf 'NOTE\tSections retain their native TSV schemas; ===== markers identify each table.\n'
        printf 'NOTE\tThis file intentionally excludes verbose .log/.json/model payloads.\n'
    } > "$out"

    # RUN_ID-named tables include both the CUDA parent and CPU-last child.
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        files+=("$f")
    done < <(find "$RESULTS" -maxdepth 1 -type f \
        \( -name "*$RUN_ID*.tsv" -o -name "*$RUN_ID*.manifest.tsv" \) \
        ! -name "hf-inventory-*" -print 2>/dev/null | sort)

    # A few high-value tables use timestamp stems rather than RUN_ID; include current backend's copies explicitly.
    for f in "${WINNER_AUG:-}" "${LAST_PROGRESSIVE_TSV:-}" "${COMPACTION_TSV:-}"; do
        [ -n "$f" ] && [ -s "$f" ] && files+=("$f")
    done

    for f in "${files[@]}"; do
        [ -s "$f" ] || continue
        [ "$f" = "$out" ] && continue
        [ -n "${seen[$f]:-}" ] && continue
        seen[$f]=1
        base="$(basename "$f")"
        case "$base" in
            run-summary-*)             section="RUN SUMMARY :: $base" ;;
            run-*.manifest.tsv)        section="RUN MANIFEST :: $base" ;;
            memory-profile-*)          section="MEMORY PROFILE :: $base" ;;
            memory-dimms-*)            section="DIMM INVENTORY :: $base" ;;
            memory-bandwidth-*)        section="MEMORY BANDWIDTH :: $base" ;;
            memory-dmi-*.meta.tsv)     section="MEMORY SMBIOS META :: $base" ;;
            plan-*)                    section="MODEL PLAN :: $base" ;;
            probe-*)                   section="MODEL FAMILY PROBES :: $base" ;;
            screen-*)                  section="QUANT / KV SCREEN :: $base" ;;
            frontier-*)                section="PARETO FRONTIER :: $base" ;;
            winner-candidates-*)       section="WINNER CANDIDATES :: $base" ;;
            winner-*)                  section="WINNER :: $base" ;;
            cpu-final-validation-*)    section="CPU FINAL VALIDATION :: $base" ;;
            cpu-build-ablations-*)     section="CPU BUILD ABLATIONS :: $base" ;;
            *.progressive.tsv)         section="PROGRESSIVE CONTEXT CURVE :: $base" ;;
            *.compaction.tsv)          section="COMPACTION ECONOMICS :: $base" ;;
            *)                         section="RESULT TABLE :: $base" ;;
        esac
        append_result_section "$out" "$section" "$f"
    done

    # Stable convenience name: ordinary file copy works across filesystems and paste workflows.
    cp -f "$out" "$latest"
    printf 'all_results=%s\n' "$out"
    printf 'all_results_latest=%s\n' "$latest"
    printf 'copy_paste_command=cat %q\n' "$out"
}

batch_tune_winner() {
    # Cheap PP-only tuning after model/quant/KV selection, before the sole deep sweep.
    # Each candidate is required to fit TARGET context with the same full-GPU constraint.
    local alias="$1" repo="$2" quant="$3" kvk="$4" kvv="$5" layers="$6" target="$7" base_stem="$8"
    [ "$BATCH_TUNE" -eq 1 ] || return 0
    [ "$BACKEND" = "cuda" ] || return 0
    info "Shallow batch tuning for winner (deep sweep has not started yet)"
    local orig_b="$BATCH" orig_ub="$UBATCH" best_b="$BATCH" best_ub="$UBATCH" best_pp=0 pair b ub stem fitfile barg json log pp rc
    local -a pairs=("512,128" "1024,256" "2048,512" "4096,512")
    for pair in "${pairs[@]}"; do
        IFS=',' read -r b ub <<< "$pair"
        BATCH="$b"; UBATCH="$ub"
        stem="$base_stem.batch-${b}-${ub}"
        local oldctx="$CTX"; CTX="$target"
        fitfile="$(fit_one "$repo" "$quant" "$kvk" "$kvv" "$stem")"; rc=$?
        CTX="$oldctx"
        if [ "$rc" -ne 0 ] || ! gpu_resident_at_target "$fitfile" "$layers"; then
            printf 'batch=%s ubatch=%s  REJECT (does not preserve target full-GPU fit)\n' "$b" "$ub" >&2
            continue
        fi
        barg="$stem.args"; bench_args_from_fit "$fitfile" > "$barg"
        json="$stem.json"; log="$stem.log"
        local tune_threads="$THREADS"
        [ "$BACKEND" = "cpu" ] && tune_threads="$THREADS_BATCH"
        local -a cmd=(-hf "$repo:$quant" -p 2048 -n 0 -d 0 -r 1 --progress -b "$b" -ub "$ub" -t "$tune_threads" -ctk "$kvk" -ctv "$kvv" -fa on -lm "$CPU_LOAD_MODE_EFFECTIVE" -o json)
        [ "$BACKEND" != "cpu" ] && cmd+=(--no-warmup)
        if [ "$BACKEND" = "cpu" ] && [ "$CPU_NUMA_EFFECTIVE" != "disabled" ]; then cmd+=(--numa "$CPU_NUMA_EFFECTIVE"); fi
        printf 'COMMAND: xargs ' >&2; printf '%q ' "$BENCH" "${cmd[@]}" >&2; printf '< %q\n' "$barg" >&2
        xargs "${CPU_CMD_PREFIX[@]}" "$BENCH" "${cmd[@]}" < "$barg" > "$json" 2> >(tee "$log" >&2); rc=$?
        if [ "$rc" -ne 0 ]; then show_capture_on_failure "$rc" "$json" "$log"; continue; fi
        pp="$(jq -r '[.[]? | select(.n_prompt==2048 and .n_gen==0) | .avg_ts][0] // 0' "$json")"
        printf 'batch=%s ubatch=%s  pp2048=%s tok/s\n' "$b" "$ub" "$pp" >&2
        if awk -v a="$pp" -v b="$best_pp" 'BEGIN{exit !(a>b)}'; then best_pp="$pp"; best_b="$b"; best_ub="$ub"; fi
    done
    BATCH="$best_b"; UBATCH="$best_ub"
    printf 'SELECTED BATCH: batch=%s ubatch=%s pp2048=%s tok/s\n' "$BATCH" "$UBATCH" "$best_pp" >&2
}

run_interactive() {
    local mode="$1" repo="$2" quant="$3" kvk="$4" kvv="$5"
    local stem="$RESULTS/run-$(sanitize "$repo")-$(sanitize "$quant")-$(sanitize "$kvk-$kvv")"
    local fitfile ctx
    fitfile="$(fit_one "$repo" "$quant" "$kvk" "$kvv" "$stem")" || exit 1
    ctx="$(fit_ctx "$fitfile")"
    info "Pinned context: $ctx tokens; fitted args: $(cat "$fitfile")"

    local load_mode="mmap"
    [ "$BACKEND" = "cpu" ] && load_mode="$CPU_LOAD_MODE_EFFECTIVE"
    local -a common=(-hf "$repo:$quant" -ctk "$kvk" -ctv "$kvv" -fa on -b "$BATCH" -ub "$UBATCH" -t "$THREADS" -tb "$THREADS_BATCH" -lm "$load_mode" --fit off --jinja)
    if [ "$BACKEND" = "cpu" ] && [ "$CPU_NUMA_EFFECTIVE" != "disabled" ]; then common+=(--numa "$CPU_NUMA_EFFECTIVE"); fi
    [ "$VISION" -eq 0 ] && common+=(--no-mmproj)
    if [ "$SPEC" = "mtp" ]; then
        common+=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k f16 --spec-draft-type-v f16)
    fi
    if [ "$mode" = "server" ]; then
        common+=(--host "$HOST" --port "$PORT" --parallel 1)
        if [ "$DRY_RUN" -eq 1 ]; then
            printf 'xargs '; printf '%q ' "$SERVER" "${common[@]}"; printf '< %q\n' "$fitfile"
        else
            cpu_command_prefix
            exec xargs "${CPU_CMD_PREFIX[@]}" "$SERVER" "${common[@]}" < "$fitfile"
        fi
    else
        [ -n "$PROMPT" ] && common+=(-p "$PROMPT")
        if [ "$DRY_RUN" -eq 1 ]; then
            printf 'xargs '; printf '%q ' "$CLI" "${common[@]}"; printf '< %q\n' "$fitfile"
        else
            cpu_command_prefix
            exec xargs "${CPU_CMD_PREFIX[@]}" "$CLI" "${common[@]}" < "$fitfile"
        fi
    fi
}

if [ "$MODEL" = "all" ]; then
    MODELS=(qwen3.6:35b nemotron-3.5 qwen3.8 qwen3.6:27b)
else
    model_meta "$MODEL" >/dev/null || { err "unknown model alias: $MODEL"; show_models; exit 2; }
    MODELS=("$MODEL")
fi

if [ "$MODE" = "server" ] || [ "$MODE" = "cli" ]; then
    [ "${#MODELS[@]}" -eq 1 ] || { err "$MODE mode requires one --model, not all"; exit 2; }
    meta="$(model_meta "${MODELS[0]}")"; repo="$(printf '%s' "$meta" | cut -f2)"; prefix="$(printf '%s' "$meta" | cut -f3)"
    mapfile -t qs < <(quant_set "$repo" "$prefix" "$QUANT") || exit 1
    [ "${#qs[@]}" -eq 1 ] || {
        if [ "$QUANT" = "auto" ]; then
            warn "--quant auto selected ${#qs[@]} sweep candidates; server/cli will use the first: ${qs[0]}"
            qs=("${qs[0]}")
        else
            err "$MODE mode requires exactly one quant"; exit 2
        fi
    }
    mapfile -t kvs < <(kv_set "$KV")
    if [ "$KV" = "auto" ]; then kvs=("${kvs[0]}"); fi
    [ "${#kvs[@]}" -eq 1 ] || { err "$MODE mode requires exactly one KV pair"; exit 2; }
    IFS=',' read -r kvk kvv <<< "${kvs[0]}"
    run_interactive "$MODE" "$repo" "${qs[0]}" "$kvk" "$kvv"
    exit $?
fi

stamp="$(date +%Y%m%d-%H%M%S)"
PROBE_TSV="$RESULTS/probe-$RUN_ID-$BACKEND-$stamp.tsv"
SCREEN_TSV="$RESULTS/screen-$RUN_ID-$BACKEND-$stamp.tsv"
PLAN_TSV="$RESULTS/plan-$RUN_ID-$BACKEND-$stamp.tsv"
FRONTIER_TSV="$RESULTS/frontier-$RUN_ID-$BACKEND-$stamp.tsv"
WINNER_FILE="$RESULTS/winner-$RUN_ID-$BACKEND-$stamp.tsv"
RESULT_META_HEADER=$'run_id\tscript_version\tbackend\tcpu_numa\tcpu_placement\tcpu_blas\tdecode_threads\tpp_threads\tbatch\tubatch'
printf 'model\trepo\tquant\tkv_k\tkv_v\tmax_ctx\tresident\tpp_probe_tps\ttg_probe_tps\tngl\tstatus\tfit_file\t%s\n' "$RESULT_META_HEADER" > "$PROBE_TSV"
printf 'model\trepo\tquant\tkv_k\tkv_v\tmax_ctx\tresident\tpp_probe_tps\ttg_probe_tps\tngl\tstatus\tfit_file\t%s\n' "$RESULT_META_HEADER" > "$SCREEN_TSV"
printf 'model\trepo\tprefix\tpreferred_ctx\tmin_usable_ctx\tmin_kv\tprobe_quant\tprobe_theory_ctx\tplausible_quants\trun_id\tscript_version\tbackend\n' > "$PLAN_TSV"

# KVs are ordered by actual cache bytes. The dominance rule is context-specific:
# if a lower-memory KV cannot fit at context C, higher-memory KVs need not be
# tried at C. The weight quant itself can still fall back to a lower useful tier.
mapfile -t BASE_KVS < <(kv_set "$KV")
mapfile -t BASE_KVS < <(sort_kv_pairs_low_to_high "${BASE_KVS[@]}")
[ "${#BASE_KVS[@]}" -gt 0 ] || { err "no KV candidates"; exit 1; }
MIN_KV_PAIR="${BASE_KVS[0]}"
printf 'KV fit order (low memory -> high memory): %s\n' "${BASE_KVS[*]}" >&2
printf 'hard usable context floor: %s tokens\n' "$MIN_CTX" >&2
printf 'preferred context ceiling: %s\n' "$TARGET_CTX" >&2
printf 'necessary-condition KV pair at the 64K+ floor: %s\n' "$MIN_KV_PAIR" >&2

if [ "$MODEL" = "all" ]; then
    info "Probe order (fast sparse families first): ${MODELS[*]}"
fi

declare -A M_REPO=() M_PREFIX=() M_META=() M_LAYERS=() M_TARGET=() M_PROBE=() M_PLAUSIBLE=() M_PROBE_THEORY_CTX=()
declare -A M_PROBE_FIT=() M_PROBE_PP=() M_PROBE_TG=() M_PROBE_NGL=() M_PROBE_CTX=()
declare -A M_PROBE_THREADS=() M_PROBE_THREADS_BATCH=() M_PROBE_BATCH=() M_PROBE_UBATCH=()

# -------- Stage 0: metadata-only PHYSICAL-capacity theory --------
# No GGUF payloads are downloaded here. Theory uses TOTAL VRAM rather than
# temporary free VRAM. Exact fitting remains authoritative at execution time.
for m in "${MODELS[@]}"; do
    meta="$(model_meta "$m")" || { warn "skipping unknown model $m"; continue; }
    alias="$(printf '%s' "$meta" | cut -f1)"; repo="$(printf '%s' "$meta" | cut -f2)"; prefix="$(printf '%s' "$meta" | cut -f3)"
    layers="$(printf '%s' "$meta" | cut -f4)"; target="$(model_target_ctx "$meta")"
    [ "$MIN_CTX" -le "$target" ] || { printf 'PLAN: %-14s preferred cap %s is below hard floor %s; skip\n' "$alias" "$target" "$MIN_CTX" >&2; continue; }
    mapfile -t qs < <(quant_set "$repo" "$prefix" "$QUANT") || continue
    IFS=',' read -r min_k min_v <<< "$MIN_KV_PAIR"
    plausible=()
    for q in "${qs[@]}"; do
        bctx="$(metadata_best_ctx "$alias" "$repo" "$prefix" "$q" "$min_k" "$min_v" "$target" "$meta" || true)"
        if [[ "$bctx" =~ ^[0-9]+$ ]] && [ "$bctx" -ge "$MIN_CTX" ]; then plausible+=("$q"); fi
    done
    if [ "${#plausible[@]}" -eq 0 ]; then
        printf 'PLAN: %-14s no weight quant can satisfy metadata lower bound at >=%s with KV=%s; ZERO GGUF DOWNLOADS\n' "$alias" "$MIN_CTX" "$MIN_KV_PAIR" >&2
        continue
    fi
    probe="$(choose_probe_quant "${plausible[@]}")" || continue
    probe_ctx="$(metadata_best_ctx "$alias" "$repo" "$prefix" "$probe" "$min_k" "$min_v" "$target" "$meta" || echo 0)"
    M_REPO[$alias]="$repo"; M_PREFIX[$alias]="$prefix"; M_META[$alias]="$meta"; M_LAYERS[$alias]="$layers"; M_TARGET[$alias]="$target"; M_PROBE[$alias]="$probe"; M_PROBE_THEORY_CTX[$alias]="$probe_ctx"
    M_PLAUSIBLE[$alias]="$(IFS=,; echo "${plausible[*]}")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$alias" "$repo" "$prefix" "$target" "$MIN_CTX" "$MIN_KV_PAIR" "$probe" "$probe_ctx" "${M_PLAUSIBLE[$alias]}" "$RUN_ID" "$VERSION" "$BACKEND" >> "$PLAN_TSV"
    qbytes="$(quant_size_bytes "$repo" "$prefix" "$probe" 2>/dev/null || echo 0)"
    python3 - "$alias" "$probe" "$qbytes" "${#plausible[@]}" "$probe_ctx" "$target" <<'PYPLAN'
import sys
m,q,b,n,c,t=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6])
print(f"PLAN: {m:14s} probe={q:14s} size={b/(1024**3):.2f} GiB theory_max_ctx={c} preferred={t} plausible_weight_quants={n}", file=sys.stderr)
PYPLAN
done

[ -s "$PLAN_TSV" ] || { err "metadata planning produced no candidates at the >=$MIN_CTX usable-context floor"; exit 1; }

# -------- Stage 1: one low-context speed probe per model family --------
# Frontload the first probe; prefetch other model probes while its exact fit and
# shallow inference run. After the first probe, whichever payload finishes first
# runs next; preferred model order is only the tie-breaker.
probe_model_family() {
    local alias="$1" repo prefix target layers q kvk kvv stem fitres frc actual_ctx fitfile ngl vals brc pp tg bngl model_meta_line
    local planned_probe pidx=0 probe_passed=0 pi i
    repo="${M_REPO[$alias]}"; prefix="${M_PREFIX[$alias]}"; target="${M_TARGET[$alias]}"; layers="${M_LAYERS[$alias]}"
    model_meta_line="$(model_meta "$alias")"
    IFS=',' read -r -a probe_candidates <<< "${M_PLAUSIBLE[$alias]}"
    planned_probe="${M_PROBE[$alias]}"
    for i in "${!probe_candidates[@]}"; do [ "${probe_candidates[$i]}" = "$planned_probe" ] && pidx="$i" && break; done
    for ((pi=pidx; pi<${#probe_candidates[@]}; pi++)); do
        q="${probe_candidates[$pi]}"; IFS=',' read -r kvk kvv <<< "$MIN_KV_PAIR"
        stem="$RESULTS/$stamp-PROBE-$(sanitize "$alias")-$(sanitize "$q")-$(sanitize "$kvk-$kvv")"
        if [ "$PREFETCH" -eq 1 ]; then
            # WALL network-only when uncached: missing CPU GGUFs must never masquerade as fit failures.
            prefetch_quant_start "$repo" "$prefix" "$q" || return 1
            prefetch_quant_wait "$repo" "$prefix" "$q" || { warn "probe download failed for $alias $q"; continue; }
        fi
        info "MODEL PROBE: $alias | $q | lowest-memory KV $kvk/$kvv | find best ctx >=$MIN_CTX | shallow PP4K/TG128"
        if [ "$BACKEND" = "cuda" ]; then
            printf 'GPU TASK STARTING NOW: exact fit followed immediately by shallow PP%s/TG%s if resident.\n' "$CURVE_PP" "$CURVE_TG" >&2
        else
            printf 'CPU TASK STARTING NOW: RAM fit, one-time CPU autotune if needed, then warmed PP%s/TG%s.\n' "$CURVE_PP" "$CURVE_TG" >&2
        fi
        fitres="$(fit_pair_best_ctx "$alias" "$repo" "$prefix" "$q" "$kvk" "$kvv" "$target" "$model_meta_line" "$layers" "$stem")"; frc=$?
        if [ "$frc" -ne 0 ]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\tNO\t\t\t\tNO_USABLE_FIT\t\t%s\n' "$alias" "$repo" "$q" "$kvk" "$kvv" "$MIN_CTX" "$(result_meta_tsv)" >> "$PROBE_TSV"
            printf 'PROBE FALLBACK: %s %s has no confirmed fully-resident context >=%s; trying smaller metadata-feasible weight quant.\n' "$alias" "$q" "$MIN_CTX" >&2
            continue
        fi
        IFS=$'\t' read -r actual_ctx fitfile ngl <<< "$fitres"
        if [ "$BACKEND" = "cpu" ] && awk -v b="$CPU_BEST_PROBE_TG" 'BEGIN{exit !(b>0)}'; then
            gate_tg="$(cpu_family_gate_tg "$repo" "$q" "$kvk" "$kvv" "$fitfile" "$stem.ctx-$actual_ctx" || true)"
            if [[ "$gate_tg" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v g="$gate_tg" -v b="$CPU_BEST_PROBE_TG" -v r="$CPU_FAMILY_GATE_RATIO" 'BEGIN{exit !(g < b*r)}'; then
                saved_threads="$THREADS"; THREADS="$physical"
                printf '%s\t%s\t%s\t%s\t%s\t%s\tYES\t\t%s\t%s\tEARLY_REJECT_TG\t%s\t%s\n'                     "$alias" "$repo" "$q" "$kvk" "$kvv" "$actual_ctx" "$gate_tg" "$ngl" "$fitfile" "$(result_meta_tsv)" >> "$PROBE_TSV"
                THREADS="$saved_threads"
                printf 'CPU EARLY REJECT: %s TG%s@%sc=%s < 20%% of best %s; skip thread sweep + PP.\n'                     "$alias" "$CPU_FAMILY_GATE_TG" "$physical" "$gate_tg" "$CPU_BEST_PROBE_TG" >&2
                probe_passed=1
                break
            fi
        fi
        if [ "$BACKEND" = "cpu" ] && [ "$CPU_TUNE" -eq 1 ] && [ "$CPU_TUNED" -eq 1 ]; then
            cpu_retune_tg_threads_for_model "$repo" "$q" "$kvk" "$kvv" "$fitfile" "$stem.ctx-$actual_ctx" || true
        fi
        screen_one "$repo" "$q" "$kvk" "$kvv" "$fitfile" "$stem.ctx-$actual_ctx"; brc=$?
        if [ "$brc" -ne 0 ]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\tYES\t\t\t%s\tBENCH_FAIL\t%s\t%s\n' "$alias" "$repo" "$q" "$kvk" "$kvv" "$actual_ctx" "$ngl" "$fitfile" "$(result_meta_tsv)" >> "$PROBE_TSV"
            continue
        fi
        pp="$SCREEN_PP"; tg="$SCREEN_TG"; bngl="$SCREEN_NGL"
        if ! [[ "$pp" =~ ^[0-9]+([.][0-9]+)?$ && "$tg" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            warn "refusing PASS for $alias $q: missing/non-numeric PP/TG"
            printf '%s	%s	%s	%s	%s	%s	YES			%s	BENCH_PARSE_FAIL	%s	%s
' "$alias" "$repo" "$q" "$kvk" "$kvv" "$actual_ctx" "$bngl" "$fitfile" "$(result_meta_tsv)" >> "$PROBE_TSV"
            continue
        fi
        printf '%s	%s	%s	%s	%s	%s	YES	%s	%s	%s	PASS	%s	%s
' "$alias" "$repo" "$q" "$kvk" "$kvv" "$actual_ctx" "$pp" "$tg" "$bngl" "$fitfile" "$(result_meta_tsv)" >> "$PROBE_TSV"
        M_PROBE[$alias]="$q"; M_PROBE_FIT[$alias]="$fitfile"; M_PROBE_PP[$alias]="$pp"; M_PROBE_TG[$alias]="$tg"; M_PROBE_NGL[$alias]="$bngl"; M_PROBE_CTX[$alias]="$actual_ctx"
        M_PROBE_THREADS[$alias]="$THREADS"; M_PROBE_THREADS_BATCH[$alias]="$THREADS_BATCH"; M_PROBE_BATCH[$alias]="$BATCH"; M_PROBE_UBATCH[$alias]="$UBATCH"
        if [ "$BACKEND" = "cpu" ] && awk -v t="$tg" -v b="$CPU_BEST_PROBE_TG" 'BEGIN{exit !(t>b)}'; then CPU_BEST_PROBE_TG="$tg"; fi
        printf 'PROBE RESULT NOW: %-14s quant=%-14s KV=%-11s fitted_max_ctx=%-8s USED_CTX=0 PP%s=%-10s TG%s=%-10s\n' "$alias" "$q" "$kvk/$kvv" "$actual_ctx" "$CURVE_PP" "$pp" "$CURVE_TG" "$tg" >&2
        probe_passed=1; break
    done
    [ "$probe_passed" -eq 1 ] || printf 'MODEL PROBE FAILED: %s exhausted metadata-feasible >=%s candidates.\n' "$alias" "$MIN_CTX" >&2
    return 0
}

probe_aliases=()
for m in "${MODELS[@]}"; do
    meta="$(model_meta "$m" 2>/dev/null || true)"; [ -n "$meta" ] || continue
    alias="$(printf '%s' "$meta" | cut -f1)"; [ -n "${M_PROBE[$alias]:-}" ] || continue
    probe_aliases+=("$alias")
done

first_alias=""
if [ "${#probe_aliases[@]}" -gt 0 ]; then first_alias="${probe_aliases[0]}"; fi

if [ "$PREFETCH" -eq 1 ] && [ -n "$first_alias" ]; then
    info "Frontloading first probe download: $first_alias | ${M_PROBE[$first_alias]}"
    prefetch_quant_start "${M_REPO[$first_alias]}" "${M_PREFIX[$first_alias]}" "${M_PROBE[$first_alias]}" || exit 1
    prefetch_quant_wait  "${M_REPO[$first_alias]}" "${M_PREFIX[$first_alias]}" "${M_PROBE[$first_alias]}" || exit 1
    started=0
    for a in "${probe_aliases[@]:1}"; do
        if [ "$started" -lt "$PREFETCH_JOBS" ]; then
            prefetch_quant_start "${M_REPO[$a]}" "${M_PREFIX[$a]}" "${M_PROBE[$a]}" || exit 1
            started=$((started+1))
        fi
    done
    printf 'PREFETCH OVERLAP: %s background model probe download(s) now overlap first inference experiments.\n' "$started" >&2
fi

if [ -n "$first_alias" ]; then
    probe_model_family "$first_alias"
fi

remaining=("${probe_aliases[@]:1}")
while [ "${#remaining[@]}" -gt 0 ]; do
    if [ "$BACKEND" = "cuda" ] && [ "$PREFETCH" -eq 1 ]; then
        # Ensure any aliases beyond the initial PREFETCH_JOBS are started whenever
        # a slot becomes available. With the default four models/jobs=3, all are
        # already active here.
        active_numeric=0
        for k in "${!PREFETCH_PID[@]}"; do case "${PREFETCH_PID[$k]}" in done|failed|'') ;; *) active_numeric=$((active_numeric+1));; esac; done
        for a in "${remaining[@]}"; do
            [ "$active_numeric" -ge "$PREFETCH_JOBS" ] && break
            key="$(prefetch_key "${M_REPO[$a]}" "${M_PREFIX[$a]}" "${M_PROBE[$a]}")"
            [ -n "${PREFETCH_PID[$key]:-}" ] && continue
            prefetch_quant_start "${M_REPO[$a]}" "${M_PREFIX[$a]}" "${M_PROBE[$a]}" || exit 1
            active_numeric=$((active_numeric+1))
        done
        next_alias="$(prefetch_wait_any_alias "${remaining[@]}")" || exit 1
    else
        next_alias="${remaining[0]}"
    fi
    printf 'SCHEDULER: next model probe is %s (payload ready or selected for fallback).\n' "$next_alias" >&2
    probe_model_family "$next_alias"
    new_remaining=()
    for a in "${remaining[@]}"; do [ "$a" = "$next_alias" ] || new_remaining+=("$a"); done
    remaining=("${new_remaining[@]}")
done

info "Model-probe results (durable already): $PROBE_TSV"
if have column; then column -t -s $'\t' "$PROBE_TSV"; else cat "$PROBE_TSV"; fi
# Model-family selection deliberately follows shallow speed, as requested; every
# contender has already proven at least the hard usable context floor.
model_winner="$(awk -F'	' 'NR>1 && $11=="PASS" && $8 ~ /^[0-9]+([.][0-9]+)?$/ && $9 ~ /^[0-9]+([.][0-9]+)?$/ && ($8+0)>0 && ($9+0)>0 {print}' "$PROBE_TSV" | sort -s -t$'	' -k9,9nr -k8,8nr | head -1)"
[ -n "$model_winner" ] || { err "no model probe achieved a fully-resident context >=$MIN_CTX"; exit 1; }
IFS=$'\t' read -r MW_ALIAS _ _ _ _ MW_CTX _ MW_PP MW_TG _ _ _ <<< "$model_winner"
info "MODEL WINNER FROM CHEAP PROBES: $MW_ALIAS | probe_max_ctx=$MW_CTX | PPprobe=$MW_PP | TGprobe=$MW_TG"

if [ "$BACKEND" = "cpu" ]; then
    [ -n "${M_PROBE_THREADS[$MW_ALIAS]:-}" ] && THREADS="${M_PROBE_THREADS[$MW_ALIAS]}"
    [ -n "${M_PROBE_THREADS_BATCH[$MW_ALIAS]:-}" ] && THREADS_BATCH="${M_PROBE_THREADS_BATCH[$MW_ALIAS]}"
    [ -n "${M_PROBE_BATCH[$MW_ALIAS]:-}" ] && BATCH="${M_PROBE_BATCH[$MW_ALIAS]}"
    [ -n "${M_PROBE_UBATCH[$MW_ALIAS]:-}" ] && UBATCH="${M_PROBE_UBATCH[$MW_ALIAS]}"
    printf 'CPU WINNER TUNING RESTORED: decode_threads=%s pp_threads=%s batch=%s ubatch=%s\n'         "$THREADS" "$THREADS_BATCH" "$BATCH" "$UBATCH" >&2
fi

if [ "$BACKEND" = "cpu" ] && [ "$CPU_TUNE" -eq 1 ] && [ -n "${M_PROBE_FIT[$MW_ALIAS]:-}" ]; then
    info "CPU WINNER PP/UBATCH RETUNE: preserving decode_threads=$THREADS for $MW_ALIAS"
    saved_decode_threads="$THREADS"
    saved_pp_threads="$THREADS_BATCH"
    saved_batch="$BATCH"
    saved_ubatch="$UBATCH"
    saved_threads_was_auto="$THREADS_WAS_AUTO"
    baseline_pp="${M_PROBE_PP[$MW_ALIAS]}"
    baseline_tg="${M_PROBE_TG[$MW_ALIAS]}"

    THREADS_WAS_AUTO=0
    CPU_TUNED=0
    IFS=',' read -r retune_k retune_v <<< "$MIN_KV_PAIR"
    retune_stem="$RESULTS/$stamp-RETUNE-PP-$(sanitize "$MW_ALIAS")-$(sanitize "${M_PROBE[$MW_ALIAS]}")"
    cpu_autotune_once "${M_REPO[$MW_ALIAS]}" "${M_PROBE[$MW_ALIAS]}" "$retune_k" "$retune_v" "${M_PROBE_FIT[$MW_ALIAS]}" "$retune_stem" || true
    THREADS_WAS_AUTO="$saved_threads_was_auto"
    THREADS="$saved_decode_threads"

    if screen_one "${M_REPO[$MW_ALIAS]}" "${M_PROBE[$MW_ALIAS]}" "$retune_k" "$retune_v" "${M_PROBE_FIT[$MW_ALIAS]}" "$retune_stem.measure"; then
        MW_PP="$SCREEN_PP"; MW_TG="$SCREEN_TG"; retune_ngl="$SCREEN_NGL"
        if awk -v new="$MW_TG" -v old="$baseline_tg" 'BEGIN{exit !(new >= old*0.97)}'; then
            M_PROBE_PP[$MW_ALIAS]="$MW_PP"
            M_PROBE_TG[$MW_ALIAS]="$MW_TG"
            M_PROBE_NGL[$MW_ALIAS]="$retune_ngl"
            M_PROBE_THREADS[$MW_ALIAS]="$THREADS"
            M_PROBE_THREADS_BATCH[$MW_ALIAS]="$THREADS_BATCH"
            M_PROBE_BATCH[$MW_ALIAS]="$BATCH"
            M_PROBE_UBATCH[$MW_ALIAS]="$UBATCH"
            printf 'CPU WINNER PP/UBATCH ACCEPTED: PP%s %s -> %s; TG%s %s -> %s; decode_threads=%s pp_threads=%s ubatch=%s\n'                 "$CURVE_PP" "$baseline_pp" "$MW_PP" "$CURVE_TG" "$baseline_tg" "$MW_TG" "$THREADS" "$THREADS_BATCH" "$UBATCH" >&2
        else
            warn "CPU winner PP/ubatch retune degraded TG by >3%; restoring family-probe tuning"
            THREADS="$saved_decode_threads"
            THREADS_BATCH="$saved_pp_threads"
            BATCH="$saved_batch"
            UBATCH="$saved_ubatch"
            MW_PP="$baseline_pp"
            MW_TG="$baseline_tg"
            M_PROBE_PP[$MW_ALIAS]="$baseline_pp"
            M_PROBE_TG[$MW_ALIAS]="$baseline_tg"
            M_PROBE_THREADS[$MW_ALIAS]="$THREADS"
            M_PROBE_THREADS_BATCH[$MW_ALIAS]="$THREADS_BATCH"
            M_PROBE_BATCH[$MW_ALIAS]="$BATCH"
            M_PROBE_UBATCH[$MW_ALIAS]="$UBATCH"
            printf 'CPU WINNER TUNING RESTORED AFTER REGRESSION: decode_threads=%s pp_threads=%s batch=%s ubatch=%s TG%s=%s\n'                 "$THREADS" "$THREADS_BATCH" "$BATCH" "$UBATCH" "$CURVE_TG" "$MW_TG" >&2
        fi
    else
        warn "CPU winner PP/ubatch retune benchmark failed; restoring family-probe tuning"
        THREADS="$saved_decode_threads"
        THREADS_BATCH="$saved_pp_threads"
        BATCH="$saved_batch"
        UBATCH="$saved_ubatch"
        MW_PP="$baseline_pp"
        MW_TG="$baseline_tg"
        M_PROBE_PP[$MW_ALIAS]="$baseline_pp"
        M_PROBE_TG[$MW_ALIAS]="$baseline_tg"
    fi
    CPU_TUNED=1
fi

# -------- Stage 2: quality/context/KV frontier for only the winning model --------
W_META="${M_META[$MW_ALIAS]}"; repo="${M_REPO[$MW_ALIAS]}"; prefix="${M_PREFIX[$MW_ALIAS]}"; target="${M_TARGET[$MW_ALIAS]}"; layers="${M_LAYERS[$MW_ALIAS]}"; probe="${M_PROBE[$MW_ALIAS]}"
IFS=',' read -r -a plausible <<< "${M_PLAUSIBLE[$MW_ALIAS]}"
probe_idx=0
for i in "${!plausible[@]}"; do [ "${plausible[$i]}" = "$probe" ] && probe_idx="$i" && break; done
refine_qs=("$probe")
for ((i=0;i<probe_idx;i++)); do refine_qs+=("${plausible[$i]}"); done
for ((i=probe_idx+1;i<${#plausible[@]};i++)); do refine_qs+=("${plausible[$i]}"); done

if [ "$PREFETCH" -eq 1 ]; then
    started=0
    for ((i=0;i<probe_idx && started<PREFETCH_JOBS;i++)); do
        prefetch_quant_start "$repo" "$prefix" "${plausible[$i]}" || exit 1
        started=$((started+1))
    done
    [ "$started" -gt 0 ] && printf 'PREFETCH OVERLAP: %s higher-quality %s quant(s) downloading while probe KV inference runs.\n' "$started" "$MW_ALIAS" >&2
fi

successful_quant_families=0
have_preferred_ctx=0
for q in "${refine_qs[@]}"; do
    q_had_pass=0
    IFS=',' read -r min_k min_v <<< "$MIN_KV_PAIR"
    theory_min_ctx="$(metadata_best_ctx "$MW_ALIAS" "$repo" "$prefix" "$q" "$min_k" "$min_v" "$target" "$W_META" || true)"
    info "WEIGHT-QUANT USABLE GATE: $MW_ALIAS | $q | minimum KV $min_k/$min_v | theory_max_ctx=${theory_min_ctx:-0} | floor=$MIN_CTX"
    if ! [[ "$theory_min_ctx" =~ ^[0-9]+$ ]] || [ "$theory_min_ctx" -lt "$MIN_CTX" ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\tNO\t\t\t\tPRECHECK_BELOW_MIN_CTX\t\t%s\n' "$MW_ALIAS" "$repo" "$q" "$min_k" "$min_v" "$MIN_CTX" "$(result_meta_tsv)" >> "$SCREEN_TSV"
        printf 'PRUNE ENTIRE KV LADDER: %s cannot reach even %s tokens with %s. ZERO DOWNLOAD if not already cached.\n' "$q" "$MIN_CTX" "$MIN_KV_PAIR" >&2
        continue
    fi
    if [ "$PREFETCH" -eq 1 ]; then
        # WALL 0s cached; otherwise download time. Explicit prefetch distinguishes I/O from CPU FIT_FAIL.
        prefetch_quant_start "$repo" "$prefix" "$q" || exit 1
        prefetch_quant_wait "$repo" "$prefix" "$q" || { warn "prefetch failed for $MW_ALIAS $q"; continue; }
    fi

    prune_remaining_kv=0
    # Exact lower-precision KV results form a hard upper bound for every later,
    # higher-memory KV pair. If q4/q4 tops out at 128K, q5/q5 will never waste
    # time attempting 262K because it cannot possibly exceed q4/q4's ceiling.
    kv_cap="$target"
    for pair in "${BASE_KVS[@]}"; do
        IFS=',' read -r kvk kvv <<< "$pair"
        stem="$RESULTS/$stamp-REFINE-$(sanitize "$MW_ALIAS")-$(sanitize "$q")-$(sanitize "$kvk-$kvv")"
        if [ "$prune_remaining_kv" -eq 1 ]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\tNO\t\t\t\tPRUNED_KV_BELOW_MIN_CTX\t\t%s\n' "$MW_ALIAS" "$repo" "$q" "$kvk" "$kvv" "$MIN_CTX" "$(result_meta_tsv)" >> "$SCREEN_TSV"
            printf 'PRUNE: %-14s %-12s KV=%-11s because lower-memory KV already failed the %s-token floor.\n' "$MW_ALIAS" "$q" "$kvk/$kvv" "$MIN_CTX" >&2
            continue
        fi
        theory_ctx="$(metadata_best_ctx "$MW_ALIAS" "$repo" "$prefix" "$q" "$kvk" "$kvv" "$kv_cap" "$W_META" || true)"
        if ! [[ "$theory_ctx" =~ ^[0-9]+$ ]] || [ "$theory_ctx" -lt "$MIN_CTX" ]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\tNO\t\t\t\tPRECHECK_BELOW_MIN_CTX\t\t%s\n' "$MW_ALIAS" "$repo" "$q" "$kvk" "$kvv" "$MIN_CTX" "$(result_meta_tsv)" >> "$SCREEN_TSV"
            printf 'KV FLOOR FOUND BY METADATA: %s %s cannot reach %s; all higher-memory KV pairs are impossible at the usable floor.\n' "$q" "$kvk/$kvv" "$MIN_CTX" >&2
            prune_remaining_kv=1
            continue
        fi

        # Reuse model probe only when quant/KV matches; its actual max context is already known.
        if [ "$q" = "$probe" ] && [ "$pair" = "$MIN_KV_PAIR" ] && [ -n "${M_PROBE_FIT[$MW_ALIAS]:-}" ]; then
            actual_ctx="${M_PROBE_CTX[$MW_ALIAS]}"; fitfile="${M_PROBE_FIT[$MW_ALIAS]}"; pp="${M_PROBE_PP[$MW_ALIAS]}"; tg="${M_PROBE_TG[$MW_ALIAS]}"; bngl="${M_PROBE_NGL[$MW_ALIAS]}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\tYES\t%s\t%s\t%s\tPASS\t%s\t%s\n' "$MW_ALIAS" "$repo" "$q" "$kvk" "$kvv" "$actual_ctx" "$pp" "$tg" "$bngl" "$fitfile" "$(result_meta_tsv)" >> "$SCREEN_TSV"
            printf 'REUSE PROBE RESULT: %-14s %-12s KV=%-11s max_ctx=%-8s PPprobe=%-10s TGprobe=%-10s\n' "$MW_ALIAS" "$q" "$kvk/$kvv" "$actual_ctx" "$pp" "$tg" >&2
            [ "$actual_ctx" -lt "$kv_cap" ] && { kv_cap="$actual_ctx"; printf 'KV DOMINANCE CAP: all higher-memory KV pairs are now capped at <=%s tokens.\n' "$kv_cap" >&2; }
            [ "$actual_ctx" -ge "$target" ] && have_preferred_ctx=1
            q_had_pass=1
            continue
        fi

        info "KV/CONTEXT FRONTIER: $MW_ALIAS | $q | $kvk/$kvv | theory_max_ctx=$theory_ctx | current_KV_cap=$kv_cap | preferred=$target"
        fitres="$(fit_pair_best_ctx "$MW_ALIAS" "$repo" "$prefix" "$q" "$kvk" "$kvv" "$kv_cap" "$W_META" "$layers" "$stem")"; frc=$?
        if [ "$frc" -eq 1 ]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\tNO\t\t\t\tFIT_FAIL\t\t%s\n' "$MW_ALIAS" "$repo" "$q" "$kvk" "$kvv" "$theory_ctx" "$(result_meta_tsv)" >> "$SCREEN_TSV"
            printf 'FIT_FAIL is not a memory proof; higher KV is NOT pruned.\n' >&2
            continue
        elif [ "$frc" -ne 0 ]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\tNO\t\t\t\tNO_USABLE_CTX\t\t%s\n' "$MW_ALIAS" "$repo" "$q" "$kvk" "$kvv" "$MIN_CTX" "$(result_meta_tsv)" >> "$SCREEN_TSV"
            printf 'EXACT FLOOR: %s %s cannot remain resident at any useful tier >=%s; all higher-memory KV pairs are pruned.\n' "$q" "$kvk/$kvv" "$MIN_CTX" >&2
            prune_remaining_kv=1
            continue
        fi
        IFS=$'\t' read -r actual_ctx fitfile ngl <<< "$fitres"
        screen_one "$repo" "$q" "$kvk" "$kvv" "$fitfile" "$stem.ctx-$actual_ctx"; brc=$?
        if [ "$brc" -ne 0 ]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\tYES\t\t\t%s\tBENCH_FAIL\t%s\t%s\n' "$MW_ALIAS" "$repo" "$q" "$kvk" "$kvv" "$actual_ctx" "$ngl" "$fitfile" "$(result_meta_tsv)" >> "$SCREEN_TSV"
            continue
        fi
        pp="$SCREEN_PP"; tg="$SCREEN_TG"; bngl="$SCREEN_NGL"
        printf '%s	%s	%s	%s	%s	%s	YES	%s	%s	%s	PASS	%s	%s
' "$MW_ALIAS" "$repo" "$q" "$kvk" "$kvv" "$actual_ctx" "$pp" "$tg" "$bngl" "$fitfile" "$(result_meta_tsv)" >> "$SCREEN_TSV"
        printf 'RESULT NOW: %-14s %-12s KV=%-11s max_ctx=%-8s PPprobe=%-10s TGprobe=%-10s ngl=%s\n' "$MW_ALIAS" "$q" "$kvk/$kvv" "$actual_ctx" "$pp" "$tg" "$bngl" >&2
        [ "$actual_ctx" -lt "$kv_cap" ] && { kv_cap="$actual_ctx"; printf 'KV DOMINANCE CAP: all higher-memory KV pairs are now capped at <=%s tokens.\n' "$kv_cap" >&2; }
        [ "$actual_ctx" -ge "$target" ] && have_preferred_ctx=1
        q_had_pass=1
    done

    if [ "$q_had_pass" -eq 1 ]; then
        successful_quant_families=$((successful_quant_families+1))
        if [ "${QUANT,,}" = "auto" ] && [ "$successful_quant_families" -ge "$AUTO_SUCCESS_QUANTS" ] && [ "$have_preferred_ctx" -eq 1 ]; then
            printf 'AUTO QUANT FRONTIER COMPLETE: %s successful weight-quant families for %s AND at least one reaches preferred ctx=%s; smaller fallbacks will not be downloaded/tested.\n' "$successful_quant_families" "$MW_ALIAS" "$target" >&2
            break
        elif [ "${QUANT,,}" = "auto" ] && [ "$successful_quant_families" -ge "$AUTO_SUCCESS_QUANTS" ] && [ "$have_preferred_ctx" -eq 0 ]; then
            printf 'AUTO CONTINUES: %s successful quant families found, but none yet reaches preferred ctx=%s; descending further instead of stopping early.\n' "$successful_quant_families" "$target" >&2
        fi
    fi
done

info "Winning-model quant/KV/context screen (durable already): $SCREEN_TSV"
if have column; then column -t -s $'\t' "$SCREEN_TSV"; else cat "$SCREEN_TSV"; fi

# Build a non-dominated frontier over three things we can measure cheaply here:
# larger GGUF payload (quality/precision proxy), larger usable max context, and TG.
# The default deep-sweep selection remains context-first because 262K+ is preferred;
# the frontier preserves higher-precision 128K/64K alternatives rather than hiding them.
python3 - "$SCREEN_TSV" "$FRONTIER_TSV" "$WORK" "$repo" "$prefix" <<'PYFRONT'
import csv, os, sys
screen,out,work,repo,prefix=sys.argv[1:]
inv=os.path.join(work, 'hf-inventory-'+repo.replace('/','_').replace(':','_')+'.tsv')
sizes={}
try:
    with open(inv,encoding='utf-8') as f:
        for line in f:
            q,b,*_=line.rstrip('\n').split('\t')
            sizes[q.lower()]=int(b)
except FileNotFoundError:
    pass
rows=[]
with open(screen,encoding='utf-8') as f:
    r=csv.DictReader(f,delimiter='\t')
    for x in r:
        if x['status']!='PASS': continue
        x['quant_bytes']=sizes.get(x['quant'].lower(),0)
        kvb={'f32':128,'f16':64,'bf16':64,'q8_0':34,'q5_0':22,'q4_0':18}
        x['kv_bytes32']=kvb.get(x['kv_k'].lower(),0)+kvb.get(x['kv_v'].lower(),0)
        x['ctx_i']=int(x['max_ctx']); x['tg_f']=float(x['tg_probe_tps']); x['pp_f']=float(x['pp_probe_tps'])
        rows.append(x)
front=[]
for a in rows:
    dominated=False
    for b in rows:
        if a is b: continue
        ge=(b['quant_bytes']>=a['quant_bytes'] and b['kv_bytes32']>=a['kv_bytes32'] and b['ctx_i']>=a['ctx_i'] and b['tg_f']>=a['tg_f'])
        gt=(b['quant_bytes']>a['quant_bytes'] or b['kv_bytes32']>a['kv_bytes32'] or b['ctx_i']>a['ctx_i'] or b['tg_f']>a['tg_f'])
        if ge and gt: dominated=True; break
    if not dominated: front.append(a)
front.sort(key=lambda x:(-x['ctx_i'],-x['quant_bytes'],-x['kv_bytes32'],-x['tg_f']))
fields=['model','repo','quant','kv_k','kv_v','max_ctx','pp_probe_tps','tg_probe_tps','quant_bytes','kv_bytes32','fit_file']
with open(out,'w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore'); w.writeheader(); w.writerows(front)
PYFRONT
info "Pareto frontier (higher precision may trade context down, never below $MIN_CTX): $FRONTIER_TSV"
if have column; then column -t -s $'\t' "$FRONTIER_TSV"; else cat "$FRONTIER_TSV"; fi

# Preferred winner: context first (262K+ preferred), then larger weight quant,
# then higher KV precision, then TG. Alternatives remain visible in FRONTIER_TSV.
WINNER_AUG="$RESULTS/winner-candidates-$stamp.tsv"
printf 'row\tquant_bytes\tkv_bytes32\n' > "$WINNER_AUG"
while IFS=$'\t' read -r a_repo a_quant a_k a_v a_ctx a_pp a_tg a_fit; do
    [ "$a_repo" = "repo" ] && continue
    qbytes="$(quant_size_bytes "$a_repo" "$prefix" "$a_quant" 2>/dev/null || echo 0)"
    kvb="$(kv_pair_bytes32 "$a_k,$a_v" 2>/dev/null || echo 0)"
    row="$(awk -F'\t' -v q="$a_quant" -v k="$a_k" -v v="$a_v" -v c="$a_ctx" 'NR>1 && $3==q && $4==k && $5==v && $6==c && $11=="PASS" {for(i=1;i<=12;i++) printf "%s%s",$i,(i<12?OFS:ORS); exit}' OFS='\t' "$SCREEN_TSV")"
    [ -n "$row" ] && printf '%s\t%s\t%s\n' "$row" "$qbytes" "$kvb" >> "$WINNER_AUG"
done < <(tail -n +2 "$FRONTIER_TSV" | cut -f2-8,11)

winner="$(tail -n +2 "$WINNER_AUG" | sort -t$'\t' -k6,6nr -k13,13nr -k14,14nr -k9,9nr | head -1 | cut -f1-12)"
if [ -z "$winner" ]; then
    err "winning model had no quant/KV configuration with a fully-resident max context >=$MIN_CTX"
    exit 1
fi
printf '%s\n' "$winner" > "$WINNER_FILE"
IFS=$'\t' read -r W_ALIAS W_REPO W_QUANT W_KVK W_KVV W_TARGET W_RES W_PP W_TG W_NGL W_STATUS W_FIT <<< "$winner"
W_META="$(model_meta "$W_ALIAS")"; W_LAYERS="$(printf '%s' "$W_META" | cut -f4)"
info "PREFERRED WINNER BEFORE BATCH TUNE: $W_ALIAS | $W_QUANT | KV $W_KVK/$W_KVV | max_ctx=$W_TARGET | PPprobe=$W_PP | TGprobe=$W_TG"
# Tune batch only with shallow PP measurements, so the expensive deep context is
# still filled exactly once. Then refit the winner at the selected batch sizes.
base_wstem="$RESULTS/$stamp-WINNER-$(sanitize "$W_ALIAS")-$(sanitize "$W_QUANT")-$(sanitize "$W_KVK-$W_KVV")"
batch_tune_winner "$W_ALIAS" "$W_REPO" "$W_QUANT" "$W_KVK" "$W_KVV" "$W_LAYERS" "$W_TARGET" "$base_wstem"
oldctx="$CTX"; CTX="$W_TARGET"
W_FIT_FINAL="$(fit_one "$W_REPO" "$W_QUANT" "$W_KVK" "$W_KVV" "$base_wstem.final")" || exit 1
CTX="$oldctx"
if [ "$BACKEND" = "cuda" ] && ! gpu_resident_at_target "$W_FIT_FINAL" "$W_LAYERS"; then
    err "selected batch sizing no longer preserves a full-GPU fit at target context"
    exit 1
fi

if [ "$BACKEND" = "cpu" ]; then
    old_curve_pp="$CURVE_PP"; old_reps="$REPS"
    CURVE_PP=1024; REPS="$CPU_FINAL_REPS"
    # WALL ~3-4m current MoE: spend repeated PP1024/TG only on the final winner.
    wall_est "~3-4 min on current Nemotron-class winner" "PP1024 + TG$CURVE_TG, $CPU_FINAL_REPS timed reps"
    info "CPU FINAL VALIDATION: $W_ALIAS | $W_QUANT | KV $W_KVK/$W_KVV | warmed PP1024/TG$CURVE_TG reps=$CPU_FINAL_REPS"
    if screen_one "$W_REPO" "$W_QUANT" "$W_KVK" "$W_KVV" "$W_FIT_FINAL" "$base_wstem.cpu-final-validation"; then
        CPU_FINAL_PP="$SCREEN_PP"; CPU_FINAL_TG="$SCREEN_TG"
        CPU_FINAL_VALIDATION_TSV="$RESULTS/cpu-final-validation-$RUN_ID.tsv"
        printf 'run_id	script_version	model	quant	kv_k	kv_v	max_ctx	pp_tokens	pp_reps	pp_tps	pp_sd_tps	tg_tokens	tg_reps	tg_tps	tg_sd_tps	decode_threads	pp_threads	batch	ubatch	numa	placement	blas	fit_file
' > "$CPU_FINAL_VALIDATION_TSV"
        printf '%s	%s	%s	%s	%s	%s	%s	1024	%s	%s	%s	%s	%s	%s	%s	%s	%s	%s	%s	%s	%s	%s	%s
'             "$RUN_ID" "$VERSION" "$W_ALIAS" "$W_QUANT" "$W_KVK" "$W_KVV" "$W_TARGET" "$CPU_FINAL_REPS" "$SCREEN_PP" "$SCREEN_PP_SD"             "$CURVE_TG" "$CPU_FINAL_REPS" "$SCREEN_TG" "$SCREEN_TG_SD" "$THREADS" "$THREADS_BATCH" "$BATCH" "$UBATCH"             "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" "$CPU_BLAS_EFFECTIVE" "$W_FIT_FINAL" >> "$CPU_FINAL_VALIDATION_TSV"
        printf 'CPU FINAL VALIDATION RESULT: PP1024=%s±%s TG%s=%s±%s threads=%s/%s batch=%s ubatch=%s
'             "$SCREEN_PP" "$SCREEN_PP_SD" "$CURVE_TG" "$SCREEN_TG" "$SCREEN_TG_SD" "$THREADS" "$THREADS_BATCH" "$BATCH" "$UBATCH" >&2
    fi
    CURVE_PP="$old_curve_pp"; REPS="$old_reps"
    run_cpu_build_ablations
fi

DEEP_SWEEP_STATUS=0
if [ "$BACKEND" = "cpu" ] && [ "$CPU_DEEP" -ne 1 ]; then
    info "CPU deep sweep skipped by default (use --cpu-deep to opt in); fit + warmed shallow PP/TG results remain complete."
    CURVE=0
fi
if [ "$CURVE" -eq 1 ]; then
    DEEP_START_EPOCH="$(date +%s)"
    progressive_server_sweep "$W_ALIAS" "$W_REPO" "$W_QUANT" "$W_KVK" "$W_KVV" "$W_FIT_FINAL" "$base_wstem.final" "$W_TARGET"
    DEEP_SWEEP_STATUS=$?
    DEEP_END_EPOCH="$(date +%s)"
    printf 'deep_sweep_elapsed_s=%s\n' "$((DEEP_END_EPOCH-DEEP_START_EPOCH))" >&2
    [ "$DEEP_SWEEP_STATUS" -eq 0 ] || warn "deep sweep ended with status $DEEP_SWEEP_STATUS; completed depth rows are preserved and the run will continue"
    if [ -n "$LAST_PROGRESSIVE_TSV" ] && [ -s "$LAST_PROGRESSIVE_TSV" ]; then
        COMPACTION_TSV="$base_wstem.final.compaction.tsv"
        analyze_compaction_policy "$LAST_PROGRESSIVE_TSV" "$COMPACTION_TSV" || true
    fi
else
    info "Deep sweep skipped (--no-curve)."
fi

# No-argument pushbutton mode saves CPU for last and performs a genuine
# CPU-specific shallow model/quant/KV funnel. It deliberately does not deep-fill
# CPU context; that remains opt-in via --cpu-deep on an explicit CPU invocation.
if [ "$CPU_LAST" -eq 1 ] && [ "$BACKEND" = "cuda" ]; then
    info "CPU LAST: independent CPU-specific shallow sweep (all model families; no deep CPU fill)"
    printf 'CPU LAST METHOD: RAM-fit + CPU-native quant/KV funnel; warmed PP%s + TG%s; independent PP/TG thread tuning; NUMA-aware; no progressive CPU prefill.\n' "$CPU_PROBE_PP" "$CURVE_TG" >&2
    printf 'CPU LAST NOTE: CPU may choose a different model/weight/KV than CUDA; that is intentional.\n' >&2
    "$SELF_PATH" --backend cpu --model all --quant auto --kv auto --mode sweep --no-curve --no-batch-tune --no-cpu-last --reps 1 \
        --cpu-probe-pp "$CPU_PROBE_PP" --tg-probe "$CURVE_TG" \
        --target-ctx "$TARGET_CTX" --min-ctx "$MIN_CTX" \
        --cache-dir "$CACHE_DIR" || warn "CPU-last independent sweep failed; GPU results above remain valid and durable"
fi

write_backend_run_summary "complete"
write_all_results

info "Run complete"
printf 'screen_results=%s\n' "$SCREEN_TSV"
printf 'frontier=%s\n' "$FRONTIER_TSV"
printf 'winner=%s\n' "$WINNER_FILE"
printf 'winner_config=%s %s KV=%s/%s target_ctx=%s batch=%s ubatch=%s deep_sweep_status=%s\n' "$W_ALIAS" "$W_QUANT" "$W_KVK" "$W_KVV" "$W_TARGET" "$BATCH" "$UBATCH" "$DEEP_SWEEP_STATUS"
if [ "$MEMORY_PROFILE" -eq 1 ]; then
    printf 'memory_profile=%s\nmemory_dimms=%s\nmemory_bandwidth=%s\n'         "$RESULTS/memory-profile-$RUN_ID.tsv" "$RESULTS/memory-dimms-$RUN_ID.tsv" "$RESULTS/memory-bandwidth-$RUN_ID.tsv"
fi
if [ "$BACKEND" = "cpu" ] && [ "$CPU_ABLATIONS" = "auto" ]; then
    printf 'cpu_build_ablations=%s\n' "$RESULTS/cpu-build-ablations-$RUN_ID.tsv"
fi
SCRIPT_END_EPOCH="$(date +%s)"
printf 'run_id=%s script_version=%s backend=%s commit=%s total_elapsed_s=%s\n' "$RUN_ID" "$VERSION" "$BACKEND" "$COMMIT" "$((SCRIPT_END_EPOCH-SCRIPT_START_EPOCH))"
printf 'PASTE THIS FILE: %s\n' "$RESULTS/all-results-$RUN_ID.txt"
if [ "$BACKEND" = "cpu" ]; then
    printf 'cpu_final_tuning=numa:%s placement:%s load_mode:%s decode_threads:%s pp_threads:%s batch:%s ubatch:%s warmup:on deep:%s blas:%s\n' \
        "$CPU_NUMA_EFFECTIVE" "$CPU_PLACEMENT_EFFECTIVE" "$CPU_LOAD_MODE_EFFECTIVE" "$THREADS" "$THREADS_BATCH" "$BATCH" "$UBATCH" "$CPU_DEEP" "$CPU_BLAS_EFFECTIVE"
fi
exit 0

