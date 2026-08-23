#!/usr/bin/env bash
# apple_ablate.sh — Apple Silicon multi-backend ablation lab.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tui.sh"

APPLE_ABLATE_MODE="${APPLE_ABLATE_MODE:-auto}"
APPLE_TIME_BUDGET_MIN="${APPLE_TIME_BUDGET_MIN:-240}"
APPLE_ONLY_MODEL="${APPLE_ONLY_MODEL:-}"
APPLE_ONLY_BACKEND="${APPLE_ONLY_BACKEND:-}"
APPLE_GEN_TOKENS="${APPLE_GEN_TOKENS:-192}"

APPLE_LAB=""
APPLE_SRC=""
APPLE_BIN=""
APPLE_MODELS=""
APPLE_CACHE=""
APPLE_PROMPTS=""
APPLE_LOGS=""
APPLE_RESULTS=""
APPLE_VENV=""
APPLE_CSV=""
APPLE_TSV=""
APPLE_SUMMARY=""
APPLE_START_EPOCH=""
APPLE_PROFILE=""
APPLE_LONG_CTX=""
APPLE_CHIP=""
APPLE_MEM_GB=0
APPLE_GPU_CORES=""
APPLE_OLLAMA_PID=""
APPLE_LLAMA_PID=""
APPLE_OMLX_DIR=""
APPLE_OMLX_VENV=""
APPLE_LLAMA_DIR=""
APPLE_LLAMA_SERVER=""
APPLE_OLLAMA_HOST="${APPLE_OLLAMA_HOST:-127.0.0.1:11439}"
APPLE_DEPTHS=()

declare -Ag APPLE_MLX4=()
declare -Ag APPLE_MLX3=()
declare -Ag APPLE_GGUF=()
declare -Ag APPLE_OLLAMA_GGUF=()
declare -Ag APPLE_OMLX=()

apple_have() { command -v "$1" >/dev/null 2>&1; }
apple_model_enabled() { [[ -z "$APPLE_ONLY_MODEL" || "$APPLE_ONLY_MODEL" == "$1" ]]; }
apple_backend_enabled() { [[ -z "$APPLE_ONLY_BACKEND" || "$APPLE_ONLY_BACKEND" == "$1" ]]; }
apple_time_left() { local e=$(( $(date +%s) - APPLE_START_EPOCH )); (( e < APPLE_TIME_BUDGET_MIN * 60 )); }
apple_budget_guard() {
    apple_time_left || {
        tui_warn "Time budget exhausted; stopping new experiments."
        apple_summarize
        return 1
    }
}

_apple_init_layout() {
    local lab_dir="${1:?lab dir required}"
    APPLE_LAB="$lab_dir"
    APPLE_SRC="$APPLE_LAB/src"
    APPLE_BIN="$APPLE_LAB/bin"
    APPLE_MODELS="$APPLE_LAB/models"
    APPLE_CACHE="$APPLE_LAB/cache"
    APPLE_PROMPTS="$APPLE_LAB/prompts"
    APPLE_LOGS="$APPLE_LAB/logs"
    APPLE_RESULTS="$APPLE_LAB/results"
    APPLE_VENV="$APPLE_LAB/venv"
    APPLE_CSV="$APPLE_RESULTS/results.csv"
    APPLE_TSV="$APPLE_RESULTS/results.tsv"
    APPLE_SUMMARY="$APPLE_RESULTS/summary.tsv"
    APPLE_OMLX_DIR="$APPLE_SRC/omlx"
    APPLE_OMLX_VENV="$APPLE_LAB/omlx-venv"
    APPLE_LLAMA_DIR="$APPLE_SRC/llama.cpp"
    APPLE_LLAMA_SERVER="$APPLE_LLAMA_DIR/build/bin/llama-server"
    APPLE_START_EPOCH="$(date +%s)"

    mkdir -p "$APPLE_SRC" "$APPLE_BIN" "$APPLE_MODELS/ollama" "$APPLE_MODELS/mlx" \
        "$APPLE_CACHE/hf" "$APPLE_CACHE/llama" "$APPLE_PROMPTS" "$APPLE_LOGS" "$APPLE_RESULTS"
    export HF_HOME="$APPLE_CACHE/hf"
    export HUGGINGFACE_HUB_CACHE="$APPLE_CACHE/hf/hub"
    export HF_HUB_CACHE="$APPLE_CACHE/hf/hub"
    export HF_XET_CACHE="$APPLE_CACHE/hf/xet"
    export TRANSFORMERS_CACHE="$APPLE_CACHE/hf/transformers"
    export LLAMA_CACHE="$APPLE_CACHE/llama"
    export OLLAMA_MODELS="$APPLE_MODELS/ollama"
    export OLLAMA_HOST="$APPLE_OLLAMA_HOST"
    export PYTHONUNBUFFERED=1

    APPLE_MLX4[qwen38]='mlx-community/Qwen3.8-27B-4bit'
    APPLE_MLX3[qwen38]='leonsarmiento/Qwen3.8-27B-3bit-mlx'
    APPLE_GGUF[qwen38]='ggml-org/Qwen3.8-27B-GGUF:Q4_K_M'
    APPLE_OLLAMA_GGUF[qwen38]='hf.co/ggml-org/Qwen3.8-27B-GGUF:Q4_K_M'
    APPLE_OMLX[qwen38]=''
    APPLE_MLX4[qwen36]='mlx-community/Qwen3.6-35B-A3B-4bit'
    APPLE_MLX3[qwen36]='andrevp/Qwen3.6-35B-A3B-3bit-MLX'
    APPLE_GGUF[qwen36]='ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M'
    APPLE_OLLAMA_GGUF[qwen36]='qwen3.6:35b-a3b-q4_K_M'
    APPLE_OMLX[qwen36]='qwen3.6:35b-a3b-nvfp4'
    APPLE_MLX4[nemotron]='majentik/Nemotron-3.5-Lightning-30B-A3B-MLX-4bit'
    APPLE_MLX3[nemotron]='majentik/Nemotron-3.5-Lightning-30B-A3B-MLX-3bit'
    APPLE_GGUF[nemotron]='ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:Q4_K_M'
    APPLE_OLLAMA_GGUF[nemotron]='nemotron-3.5-lightning:30b-a3b-q4_K_M'
    APPLE_OMLX[nemotron]='nemotron-3.5-lightning:30b-a3b-nvfp4'
}

apple_append_row() {
    python3 - "$APPLE_CSV" "$APPLE_TSV" "$@" <<'PY'
import csv, os, sys
csvp, tsvp, *row = sys.argv[1:]
header = ["model","backend","quant","optimization","target_depth","actual_prompt_tokens","pp_tps","tg_tokens","tg_tps","e2e_tps","wall_s","peak_gb","status","note"]
for path, delim in ((csvp, ","), (tsvp, "\t")):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=delim)
        if new:
            writer.writerow(header)
        writer.writerow(row)
PY
}

apple_summarize() {
    [[ -s "$APPLE_TSV" ]] || return 0
    python3 - "$APPLE_TSV" "$APPLE_SUMMARY" <<'PY'
import csv, sys
src, dst = sys.argv[1:]
rows = list(csv.DictReader(open(src), delimiter="\t"))
ok = [r for r in rows if r["status"] == "ok"]
def f(r, k):
    try:
        return float(r[k])
    except Exception:
        return -1
with open(dst, "w", newline="") as o:
    w = csv.writer(o, delimiter="\t")
    w.writerow(["model","backend","quant","optimization","runs","best_pp","best_tg","best_e2e","max_ctx"])
    groups = {}
    for row in ok:
        groups.setdefault((row["model"], row["backend"], row["quant"], row["optimization"]), []).append(row)
    for key, rows in sorted(groups.items()):
        w.writerow([*key, len(rows), max(f(x, "pp_tps") for x in rows), max(f(x, "tg_tps") for x in rows), max(f(x, "e2e_tps") for x in rows), max(int(float(x["actual_prompt_tokens"] or 0)) for x in rows)])
PY
}

_apple_write_prompts() {
    python3 - "$APPLE_PROMPTS" "${APPLE_DEPTHS[@]}" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
depths = [int(x) for x in sys.argv[2:]]
payload = """A systems programmer is reviewing a medium-sized codebase containing Python, C++, shell scripts, tests, build files, parsers, network services, and concurrent workers. The goal is to improve correctness and performance without changing externally visible behavior. Check invariants, error paths, resource ownership, complexity, cache behavior, allocation patterns, synchronization, serialization, boundary conditions, and test coverage. Proposed optimizations must preserve semantics and include falsifiable benchmarks. Measurements distinguish startup, prompt processing, autoregressive generation, and end-to-end throughput.\n"""
for depth in depths:
    chars = int(depth * 4.7)
    prefix = f"RUN_DEPTH_{depth}\nAnalyze these repository-review notes and produce technical recommendations.\n"
    body = (payload * ((chars // len(payload)) + 2))[:max(0, chars - len(prefix))]
    (root / f"depth-{depth}.txt").write_text(prefix + body)
PY
}

_apple_write_bench_helpers() {
    cat > "$APPLE_LAB/bench_ollama.py" <<'PY'
import argparse, json, time, urllib.request
p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--prompt', required=True)
p.add_argument('--ctx', type=int, required=True)
p.add_argument('--gen', type=int, default=192)
a = p.parse_args()
payload = {'model': a.model, 'prompt': open(a.prompt).read(), 'stream': False, 'keep_alive': -1, 'options': {'num_ctx': a.ctx, 'num_predict': a.gen, 'temperature': 0, 'seed': 1}}
t = time.perf_counter()
req = urllib.request.Request('http://127.0.0.1:11439/api/generate', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=7200) as r:
    d = json.load(r)
wall = time.perf_counter() - t
pc = d.get('prompt_eval_count', 0)
pd = d.get('prompt_eval_duration', 0) / 1e9
ec = d.get('eval_count', 0)
ed = d.get('eval_duration', 0) / 1e9
print(json.dumps({'prompt_tokens': pc, 'pp_tps': pc / pd if pd else 0, 'tg_tokens': ec, 'tg_tps': ec / ed if ed else 0, 'e2e_tps': (pc + ec) / wall if wall else 0, 'wall_s': wall, 'peak_gb': '', 'status': 'ok'}))
PY
    cat > "$APPLE_LAB/bench_llama.py" <<'PY'
import argparse, json, time, urllib.request
p = argparse.ArgumentParser()
p.add_argument('--prompt', required=True)
p.add_argument('--gen', type=int, default=192)
a = p.parse_args()
payload = {'prompt': open(a.prompt).read(), 'n_predict': a.gen, 'temperature': 0, 'seed': 1, 'cache_prompt': False, 'stream': False}
t = time.perf_counter()
req = urllib.request.Request('http://127.0.0.1:11440/completion', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=7200) as r:
    d = json.load(r)
wall = time.perf_counter() - t
tm = d.get('timings', {})
pc = int(tm.get('prompt_n', 0) or 0)
ec = int(tm.get('predicted_n', 0) or 0)
print(json.dumps({'prompt_tokens': pc, 'pp_tps': float(tm.get('prompt_per_second', 0) or 0), 'tg_tokens': ec, 'tg_tps': float(tm.get('predicted_per_second', 0) or 0), 'e2e_tps': (pc + ec) / wall if wall else 0, 'wall_s': wall, 'peak_gb': '', 'status': 'ok'}))
PY
    cat > "$APPLE_LAB/bench_mlx.py" <<'PY'
import json, re, sys, time, subprocess
model, prompt, gen = sys.argv[1:4]
text = open(prompt).read()
def run(cmd):
    t = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=7200)
    return p, time.perf_counter() - t
p, wall = run([sys.executable, '-m', 'mlx_lm.generate', '--model', model, '--prompt', text, '--max-tokens', gen, '--temp', '0'])
if p.returncode:
    p, wall = run([sys.executable, '-m', 'mlx_vlm.generate', '--model', model, '--prompt', text, '--max-tokens', gen, '--temperature', '0'])
out = p.stdout
print(out, file=sys.stderr)
def one(patterns, default=0.0):
    for pattern in patterns:
        m = re.search(pattern, out, re.I)
        if m:
            return float(m.group(1))
    return default
pc = one([r'Prompt:\s*([0-9.]+)\s*tokens', r'prompt tokens[:=]\s*([0-9.]+)'])
pp = one([r'Prompt:.*?([0-9.]+)\s*tokens-per-sec', r'prompt.*?([0-9.]+)\s*tokens/s'])
ec = one([r'Generation:\s*([0-9.]+)\s*tokens', r'generation tokens[:=]\s*([0-9.]+)'])
tg = one([r'Generation:.*?([0-9.]+)\s*tokens-per-sec', r'generation.*?([0-9.]+)\s*tokens/s'])
peak = one([r'Peak memory:\s*([0-9.]+)\s*GB', r'peak memory.*?([0-9.]+)\s*GB'], '')
print(json.dumps({'prompt_tokens': int(pc), 'pp_tps': pp, 'tg_tokens': int(ec), 'tg_tps': tg, 'e2e_tps': (pc + ec) / wall if wall and pc else 0, 'wall_s': wall, 'peak_gb': peak, 'status': 'ok' if p.returncode == 0 and tg else 'failed'}))
PY
}

apple_check_prereqs() {
    [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || {
        tui_error "Apple Silicon macOS required"
        return 1
    }
    APPLE_CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo Apple-Silicon)"
    APPLE_MEM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
    APPLE_GPU_CORES="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2; exit}' | tr -dc '0-9')"
    [[ -n "$APPLE_GPU_CORES" ]] || APPLE_GPU_CORES=unknown
    if (( APPLE_MEM_GB <= 24 )); then
        APPLE_PROFILE=air24
        APPLE_DEPTHS=(1024 4096 8192)
        APPLE_LONG_CTX=12288
    else
        APPLE_PROFILE=max
        APPLE_DEPTHS=(1024 8192 32768)
        APPLE_LONG_CTX=40960
    fi
    tui_step "Apple hardware profile"
    tui_info "$APPLE_CHIP | RAM=${APPLE_MEM_GB}GB | GPU=$APPLE_GPU_CORES | profile=$APPLE_PROFILE"
    tui_info "Depths: ${APPLE_DEPTHS[*]} | gen=$APPLE_GEN_TOKENS | budget=${APPLE_TIME_BUDGET_MIN}m | mode=$APPLE_ABLATE_MODE"
}

apple_setup_tooling() {
    local lab_dir="${1:?lab dir required}"
    _apple_init_layout "$lab_dir"
    apple_check_prereqs
    _apple_write_prompts
    apple_have git && apple_have cmake && apple_have python3 || {
        tui_error "Need git, cmake, and python3"
        return 1
    }
    if ! apple_have uv; then
        tui_step "Installing private uv"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$APPLE_BIN" sh
        export PATH="$APPLE_BIN:$PATH"
    fi
    [[ -x "$APPLE_VENV/bin/python" ]] || uv venv --python python3 "$APPLE_VENV"
    tui_step "Installing/updating MLX and Hugging Face tooling"
    uv pip install --python "$APPLE_VENV/bin/python" -q -U 'mlx-lm>=0.31' 'mlx-vlm>=0.6' 'huggingface_hub[hf_xet]>=0.34'
    _apple_write_bench_helpers
}

apple_setup_omlx() {
    local lab_dir="${1:-$APPLE_LAB}"
    [[ -n "$lab_dir" ]] || return 1
    _apple_init_layout "$lab_dir"
    apple_backend_enabled omlx || return 0
    if [[ ! -d "$APPLE_OMLX_DIR/.git" ]]; then
        tui_step "Cloning oMLX"
        git clone --depth=1 https://github.com/jundot/omlx.git "$APPLE_OMLX_DIR"
    else
        tui_step "Updating oMLX"
        git -C "$APPLE_OMLX_DIR" pull --ff-only || true
    fi
    [[ -x "$APPLE_OMLX_VENV/bin/python" ]] || uv venv --python python3 "$APPLE_OMLX_VENV"
    uv pip install --python "$APPLE_OMLX_VENV/bin/python" -q -e "$APPLE_OMLX_DIR"
}

apple_setup_llama() {
    local lab_dir="${1:-$APPLE_LAB}"
    [[ -n "$lab_dir" ]] || return 1
    _apple_init_layout "$lab_dir"
    apple_backend_enabled llama || return 0
    if [[ ! -d "$APPLE_LLAMA_DIR/.git" ]]; then
        tui_step "Cloning llama.cpp"
        git clone --depth=1 https://github.com/ggml-org/llama.cpp.git "$APPLE_LLAMA_DIR"
    else
        tui_step "Updating llama.cpp"
        git -C "$APPLE_LLAMA_DIR" pull --ff-only || true
    fi
    tui_step "Building llama.cpp Metal"
    cmake -S "$APPLE_LLAMA_DIR" -B "$APPLE_LLAMA_DIR/build" -DGGML_METAL=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$APPLE_LLAMA_DIR/build" -j "$(sysctl -n hw.ncpu)" --target llama-server >/dev/null
}

apple_jf() { python3 -c 'import json,sys; print(json.load(sys.stdin).get(sys.argv[1],""))' "$1"; }

apple_record() {
    local model="$1" backend="$2" quant="$3" optimization="$4" depth="$5" json_blob="$6" note="${7:-}"
    local pc pp ec tg e2e wall peak status
    pc="$(printf '%s' "$json_blob" | apple_jf prompt_tokens 2>/dev/null || true)"
    pp="$(printf '%s' "$json_blob" | apple_jf pp_tps 2>/dev/null || true)"
    ec="$(printf '%s' "$json_blob" | apple_jf tg_tokens 2>/dev/null || true)"
    tg="$(printf '%s' "$json_blob" | apple_jf tg_tps 2>/dev/null || true)"
    e2e="$(printf '%s' "$json_blob" | apple_jf e2e_tps 2>/dev/null || true)"
    wall="$(printf '%s' "$json_blob" | apple_jf wall_s 2>/dev/null || true)"
    peak="$(printf '%s' "$json_blob" | apple_jf peak_gb 2>/dev/null || true)"
    status="$(printf '%s' "$json_blob" | apple_jf status 2>/dev/null || true)"
    [[ -n "$status" ]] || status=ok
    [[ -n "$tg" && "$tg" != 0 && "$tg" != 0.0 ]] || status=failed
    apple_append_row "$model" "$backend" "$quant" "$optimization" "$depth" "${pc:-0}" "${pp:-0}" "${ec:-0}" "${tg:-0}" "${e2e:-0}" "${wall:-0}" "${peak:-}" "$status" "$note"
}

apple_cleanup() {
    if [[ -n "$APPLE_OLLAMA_PID" ]] && kill -0 "$APPLE_OLLAMA_PID" 2>/dev/null; then
        kill "$APPLE_OLLAMA_PID" 2>/dev/null || true
    fi
    if [[ -n "$APPLE_LLAMA_PID" ]] && kill -0 "$APPLE_LLAMA_PID" 2>/dev/null; then
        kill "$APPLE_LLAMA_PID" 2>/dev/null || true
    fi
    APPLE_OLLAMA_PID=''
    APPLE_LLAMA_PID=''
}

apple_start_ollama() {
    apple_backend_enabled ollama || return 1
    apple_have ollama || {
        tui_warn "ollama not installed; skipping Ollama backend"
        return 1
    }
    curl -fsS "http://$APPLE_OLLAMA_HOST/api/version" >/dev/null 2>&1 && return 0
    tui_step "Starting isolated Ollama at $APPLE_OLLAMA_HOST"
    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=f16 OLLAMA_NUM_PARALLEL=1 ollama serve >"$APPLE_LOGS/ollama-server.log" 2>&1 &
    APPLE_OLLAMA_PID=$!
    for _ in $(seq 1 60); do
        curl -fsS "http://$APPLE_OLLAMA_HOST/api/version" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

apple_run_ollama() {
    local model="$1" tag="$2" quant="$3" optimization="$4"
    [[ -n "$tag" ]] || return 0
    apple_budget_guard || return 0
    tui_step "Ollama pull $tag"
    ollama pull "$tag" >"$APPLE_LOGS/pull-${model}-${quant}.log" 2>&1 || {
        tui_warn "pull failed: $tag"
        return 0
    }
    local depth json_blob
    for depth in "${APPLE_DEPTHS[@]}"; do
        apple_budget_guard || return 0
        json_blob="$("$APPLE_VENV/bin/python" "$APPLE_LAB/bench_ollama.py" --model "$tag" --prompt "$APPLE_PROMPTS/depth-$depth.txt" --ctx "$APPLE_LONG_CTX" --gen "$APPLE_GEN_TOKENS" 2>>"$APPLE_LOGS/ollama-${model}-${quant}.log" || echo '{}')"
        apple_record "$model" ollama "$quant" "$optimization" "$depth" "$json_blob"
    done
}

apple_download_mlx() {
    local repo="$1" dest="$2"
    [[ -f "$dest/config.json" ]] && return 0
    apple_budget_guard || return 1
    "$APPLE_VENV/bin/hf" download "$repo" --local-dir "$dest" >"$APPLE_LOGS/hf-$(basename "$dest").log" 2>&1
}

apple_run_mlx() {
    local model="$1" repo="$2" quant="$3" dest="$APPLE_MODELS/mlx/${1}-${3}"
    apple_download_mlx "$repo" "$dest" || {
        tui_warn "MLX download failed: $repo"
        return 0
    }
    local depth json_blob
    for depth in "${APPLE_DEPTHS[@]}"; do
        apple_budget_guard || return 0
        json_blob="$("$APPLE_VENV/bin/python" "$APPLE_LAB/bench_mlx.py" "$dest" "$APPLE_PROMPTS/depth-$depth.txt" "$APPLE_GEN_TOKENS" 2>"$APPLE_LOGS/mlx-${model}-${quant}-${depth}.log" || echo '{}')"
        apple_record "$model" mlx "$quant" baseline "$depth" "$json_blob"
    done
}

apple_run_omlx() {
    local model="$1" repo="$2" quant="$3" dest="$APPLE_MODELS/mlx/${1}-${3}" log_file="$APPLE_LOGS/omlx-${1}-${3}.log"
    apple_download_mlx "$repo" "$dest" || {
        tui_warn "oMLX model download failed: $repo"
        return 0
    }
    apple_budget_guard || return 0
    "$APPLE_OMLX_VENV/bin/python" "$APPLE_OMLX_DIR/scripts/bench.py" "$dest" --pp "${APPLE_DEPTHS[@]}" --gen "$APPLE_GEN_TOKENS" --batch 2 4 --warmup 1 >"$log_file" 2>&1 || {
        tui_warn "oMLX benchmark failed; see $log_file"
        return 0
    }
    "$APPLE_VENV/bin/python" - "$log_file" "$model" "$quant" "$APPLE_GEN_TOKENS" "$APPLE_CSV" "$APPLE_TSV" <<'PY'
import csv, re, sys, os
log, model, quant, gen, csvp, tsvp = sys.argv[1:]
gen = int(gen)
text = open(log).read().splitlines()
header = ["model","backend","quant","optimization","target_depth","actual_prompt_tokens","pp_tps","tg_tokens","tg_tps","e2e_tps","wall_s","peak_gb","status","note"]
rows = []
single = batch = False
for line in text:
    if 'Single-request' in line:
        single = True; batch = False; continue
    if 'Continuous-batching' in line:
        batch = True; single = False; continue
    if single:
        m = re.match(r'^\s*(\d+)\s+([0-9.]+)ms\s+([0-9.]+)/s\s+([0-9.]+)/s\s+([0-9.]+)G', line)
        if m:
            pp = int(m.group(1)); ttft = float(m.group(2)) / 1000; tg = float(m.group(3)); ppt = float(m.group(4)); mem = float(m.group(5)); wall = ttft + (gen / tg if tg else 0); e2e = (pp + gen) / wall if wall else 0
            rows.append([model,'omlx',quant,'native',pp,pp,ppt,gen,tg,e2e,wall,mem,'ok',''])
    if batch:
        m = re.match(r'^\s*(\d+)\s+([0-9.]+)/s\s+([0-9.]+)/s\s+([0-9.]+)ms', line)
        if m:
            bs = int(m.group(1)); ppt = float(m.group(2)); tg = float(m.group(3)); ttft = float(m.group(4)) / 1000
            rows.append([model,'omlx',quant,f'batch{bs}',0,0,ppt,gen*bs,tg,0,ttft,'','ok','aggregate throughput'])
for path, delim in ((csvp, ','), (tsvp, '\t')):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f, delimiter=delim)
        if new:
            w.writerow(header)
        w.writerows(rows)
PY
}

apple_llama_stop() {
    if [[ -n "$APPLE_LLAMA_PID" ]] && kill -0 "$APPLE_LLAMA_PID" 2>/dev/null; then
        kill "$APPLE_LLAMA_PID" 2>/dev/null || true
    fi
    APPLE_LLAMA_PID=''
}

apple_llama_start() {
    local hf="$1"
    shift
    apple_llama_stop
    "$APPLE_LLAMA_SERVER" -hf "$hf" --host 127.0.0.1 --port 11440 -ngl 999 -c "$APPLE_LONG_CTX" "$@" >"$APPLE_LOGS/llama-server-current.log" 2>&1 &
    APPLE_LLAMA_PID=$!
    for _ in $(seq 1 180); do
        curl -fsS http://127.0.0.1:11440/health >/dev/null 2>&1 && return 0
        kill -0 "$APPLE_LLAMA_PID" 2>/dev/null || break
        sleep 1
    done
    apple_llama_stop
    return 1
}

apple_llama_one() {
    local model="$1" hf="$2" quant="$3" optimization="$4" depth="$5"
    shift 5
    apple_budget_guard || return 0
    apple_llama_start "$hf" "$@" || {
        tui_warn "llama.cpp server start failed: $model $optimization"
        apple_append_row "$model" llama "$quant" "$optimization" "$depth" 0 0 0 0 0 0 '' failed server-start
        return 0
    }
    local json_blob
    json_blob="$("$APPLE_VENV/bin/python" "$APPLE_LAB/bench_llama.py" --prompt "$APPLE_PROMPTS/depth-$depth.txt" --gen "$APPLE_GEN_TOKENS" 2>>"$APPLE_LOGS/llama-${model}-${optimization}-${depth}.log" || echo '{}')"
    apple_record "$model" llama "$quant" "$optimization" "$depth" "$json_blob"
    apple_llama_stop
}

apple_llama_sweep() {
    local model="$1" hf="$2" mid="${APPLE_DEPTHS[1]}" long="${APPLE_DEPTHS[2]}"
    local depth repo ub kv n b
    for depth in "${APPLE_DEPTHS[@]}"; do
        apple_llama_one "$model" "$hf" q4_k_m fa-auto_ub512_kvf16 "$depth" -fa auto -b 2048 -ub 512 -ctk f16 -ctv f16
    done
    [[ "$APPLE_ABLATE_MODE" == quick ]] && return 0
    apple_llama_one "$model" "$hf" q4_k_m fa-on_ub512_kvf16 "$mid" -fa on -b 2048 -ub 512 -ctk f16 -ctv f16
    apple_llama_one "$model" "$hf" q4_k_m fa-off_ub512_kvf16 "$mid" -fa off -b 2048 -ub 512 -ctk f16 -ctv f16
    for ub in 256 1024; do
        apple_llama_one "$model" "$hf" q4_k_m "fa-on_ub${ub}_kvf16" "$mid" -fa on -b 2048 -ub "$ub" -ctk f16 -ctv f16
    done
    for kv in q8_0 q4_0; do
        apple_llama_one "$model" "$hf" q4_k_m "fa-on_ub512_kv${kv}" "$long" -fa on -b 2048 -ub 512 -ctk "$kv" -ctv "$kv"
    done
    if [[ "$model" == qwen38 || "$model" == qwen36 ]]; then
        repo="${hf%%:*}"
        for n in 2 3; do
            apple_llama_one "$model" "$hf" q4_k_m "mtp${n}_fa-on" "$mid" -hfd "$repo:Q4_0" --spec-default --spec-type draft-mtp --spec-draft-n-max "$n" --spec-draft-p-min 0.5 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16
        done
    fi
    if [[ "$APPLE_ABLATE_MODE" == full ]]; then
        for b in 1024 4096; do
            for ub in 512 1024; do
                apple_llama_one "$model" "$hf" q4_k_m "fa-on_b${b}_ub${ub}" "$mid" -fa on -b "$b" -ub "$ub" -ctk f16 -ctv f16
            done
        done
        apple_llama_one "$model" "$hf" q4_k_m ngram-simple "$mid" --spec-type ngram-simple -fa on -b 2048 -ub 512 -ctk f16 -ctv f16
    fi
}

apple_run_ablation() {
    local lab_dir="${1:-$APPLE_LAB}"
    local mode="${2:-$APPLE_ABLATE_MODE}"
    local time_budget_min="${3:-$APPLE_TIME_BUDGET_MIN}"
    local only_model="${4:-$APPLE_ONLY_MODEL}"
    local only_backend="${5:-$APPLE_ONLY_BACKEND}"

    APPLE_ABLATE_MODE="$mode"
    APPLE_TIME_BUDGET_MIN="$time_budget_min"
    APPLE_ONLY_MODEL="$only_model"
    APPLE_ONLY_BACKEND="$only_backend"
    _apple_init_layout "$lab_dir"

    local ollama_ready=0
    trap apple_cleanup EXIT INT TERM
    if apple_backend_enabled ollama && apple_start_ollama; then
        ollama_ready=1
    fi

    local -a models=(qwen38 qwen36 nemotron)
    local model
    for model in "${models[@]}"; do
        apple_model_enabled "$model" || continue
        tui_header "Apple ablation: $model"

        if (( ollama_ready )); then
            if [[ "$APPLE_PROFILE" == max ]]; then
                apple_run_ollama "$model" "${APPLE_OLLAMA_GGUF[$model]}" q4_k_m baseline
                [[ -n "${APPLE_OMLX[$model]}" ]] && apple_run_ollama "$model" "${APPLE_OMLX[$model]}" nvfp4 mlx-engine
            else
                tui_warn "Air24: skipping large Ollama Q4/NVFP4 pulls for $model"
            fi
        fi

        if apple_backend_enabled mlx; then
            apple_run_mlx "$model" "${APPLE_MLX3[$model]}" 3bit
            if [[ "$APPLE_PROFILE" == max || ( "$model" == qwen38 && "$APPLE_ABLATE_MODE" == full ) ]]; then
                apple_run_mlx "$model" "${APPLE_MLX4[$model]}" 4bit
            fi
        fi

        if apple_backend_enabled omlx; then
            apple_run_omlx "$model" "${APPLE_MLX3[$model]}" 3bit
            if [[ "$APPLE_PROFILE" == max ]]; then
                apple_run_omlx "$model" "${APPLE_MLX4[$model]}" 4bit
            fi
        fi

        if apple_backend_enabled llama; then
            if [[ "$APPLE_PROFILE" == max ]]; then
                apple_llama_sweep "$model" "${APPLE_GGUF[$model]}"
            else
                tui_warn "Air24: skipping large llama.cpp Q4 matrix for $model"
            fi
        fi
    done

    apple_summarize
    tui_success "Apple ablation finished"
    tui_info "CSV: $APPLE_CSV"
    tui_info "Summary: $APPLE_SUMMARY"
    tui_info "Logs: $APPLE_LOGS"
}

setup_apple_ablate() {
    local lab_dir="${1:-$PWD/.apple-llm-lab}"
    _apple_init_layout "$lab_dir"
    apple_setup_tooling "$lab_dir"
    apple_setup_omlx "$lab_dir"
    apple_setup_llama "$lab_dir"
    apple_run_ablation "$lab_dir" "$APPLE_ABLATE_MODE" "$APPLE_TIME_BUDGET_MIN" "$APPLE_ONLY_MODEL" "$APPLE_ONLY_BACKEND"
}
