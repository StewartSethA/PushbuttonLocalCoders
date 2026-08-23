#!/usr/bin/env bash
# Apple-Silicon LLM ablation lab. Everything it owns stays under $PWD/.apple-llm-lab.
# Models: Qwen3.8-27B, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning-30B-A3B.
# Backends: Ollama GGUF/MLX, direct MLX, oMLX native benchmark, llama.cpp Metal.

MODE="${MODE:-auto}"                    # quick | auto | full
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-240}"
ONLY_MODEL="${ONLY_MODEL:-}"            # qwen38 | qwen36 | nemotron
ONLY_BACKEND="${ONLY_BACKEND:-}"        # ollama | mlx | omlx | llama
GEN_TOKENS="${GEN_TOKENS:-192}"
LAB="$PWD/.apple-llm-lab"
SRC="$LAB/src"; BIN="$LAB/bin"; MODELS="$LAB/models"; CACHE="$LAB/cache"
PROMPTS="$LAB/prompts"; LOGS="$LAB/logs"; RESULTS="$LAB/results"; VENV="$LAB/venv"
CSV="$RESULTS/results.csv"; TSV="$RESULTS/results.tsv"; SUMMARY="$RESULTS/summary.tsv"
START_EPOCH="$(date +%s)"
mkdir -p "$SRC" "$BIN" "$MODELS/ollama" "$MODELS/mlx" "$CACHE/hf" "$CACHE/llama" "$PROMPTS" "$LOGS" "$RESULTS"
export HF_HOME="$CACHE/hf" HUGGINGFACE_HUB_CACHE="$CACHE/hf/hub" HF_HUB_CACHE="$CACHE/hf/hub"
export HF_XET_CACHE="$CACHE/hf/xet" TRANSFORMERS_CACHE="$CACHE/hf/transformers" LLAMA_CACHE="$CACHE/llama"
export OLLAMA_MODELS="$MODELS/ollama" OLLAMA_HOST="127.0.0.1:11439" PYTHONUNBUFFERED=1

say(){ printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
warn(){ printf '[WARN] %s\n' "$*" >&2; }
have(){ command -v "$1" >/dev/null 2>&1; }
model_enabled(){ [[ -z "$ONLY_MODEL" || "$ONLY_MODEL" == "$1" ]]; }
backend_enabled(){ [[ -z "$ONLY_BACKEND" || "$ONLY_BACKEND" == "$1" ]]; }
time_left(){ local e=$(( $(date +%s)-START_EPOCH )); (( e < TIME_BUDGET_MIN*60 )); }
budget_guard(){ time_left || { warn "Time budget exhausted; stopping new experiments."; summarize; exit 0; }; }

append_row(){
  python3 - "$CSV" "$TSV" "$@" <<'PY'
import csv,os,sys
csvp,tsvp,*row=sys.argv[1:]
h=["model","backend","quant","optimization","target_depth","actual_prompt_tokens","pp_tps","tg_tokens","tg_tps","e2e_tps","wall_s","peak_gb","status","note"]
for p,d in ((csvp,","),(tsvp,"\t")):
    new=not os.path.exists(p)
    with open(p,"a",newline="") as f:
        w=csv.writer(f,delimiter=d)
        if new:w.writerow(h)
        w.writerow(row)
PY
}

summarize(){
  [[ -s "$TSV" ]] || return 0
  python3 - "$TSV" "$SUMMARY" <<'PY'
import csv,sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter="\t")); ok=[r for r in rows if r["status"]=="ok"]
def f(r,k):
    try:return float(r[k])
    except:return -1
with open(dst,"w",newline="") as o:
    w=csv.writer(o,delimiter="\t"); w.writerow(["model","backend","quant","optimization","runs","best_pp","best_tg","best_e2e","max_ctx"])
    g={}
    for r in ok:g.setdefault((r["model"],r["backend"],r["quant"],r["optimization"]),[]).append(r)
    for k,rs in sorted(g.items()):
        w.writerow([*k,len(rs),max(f(x,"pp_tps") for x in rs),max(f(x,"tg_tps") for x in rs),max(f(x,"e2e_tps") for x in rs),max(int(float(x["actual_prompt_tokens"] or 0)) for x in rs)])
print("\n=== TOP TG ===")
for r in sorted(ok,key=lambda x:f(x,"tg_tps"),reverse=True)[:20]:
    print(f'{r["tg_tps"]:>8} TG | {r["pp_tps"]:>8} PP | {r["actual_prompt_tokens"]:>7} ctx | {r["model"]:<9} {r["backend"]:<8} {r["quant"]:<8} {r["optimization"]}')
print("\n=== TOP PP ===")
for r in sorted(ok,key=lambda x:f(x,"pp_tps"),reverse=True)[:20]:
    print(f'{r["pp_tps"]:>8} PP | {r["tg_tps"]:>8} TG | {r["actual_prompt_tokens"]:>7} ctx | {r["model"]:<9} {r["backend"]:<8} {r["quant"]:<8} {r["optimization"]}')
PY
}

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo "Apple Silicon macOS required" >&2; exit 1; }
CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo Apple-Silicon)"
MEM_GB=$(( $(sysctl -n hw.memsize)/1024/1024/1024 ))
GPU_CORES="$(system_profiler SPDisplaysDataType 2>/dev/null|awk -F': ' '/Total Number of Cores/{print $2;exit}'|tr -dc '0-9')"; [[ -n "$GPU_CORES" ]]||GPU_CORES=unknown
if (( MEM_GB<=24 )); then PROFILE=air24; DEPTHS=(1024 4096 8192); LONG_CTX=12288; else PROFILE=max; DEPTHS=(1024 8192 32768); LONG_CTX=40960; fi
say "Hardware: $CHIP | RAM=${MEM_GB}GB | GPU=$GPU_CORES | profile=$PROFILE"
say "Depths: ${DEPTHS[*]} | gen=$GEN_TOKENS | budget=${TIME_BUDGET_MIN}m | mode=$MODE"
say "Everything owned by this run stays under $LAB"

python3 - "$PROMPTS" "${DEPTHS[@]}" <<'PY'
import pathlib,sys
r=pathlib.Path(sys.argv[1]); ds=[int(x) for x in sys.argv[2:]]
p="""A systems programmer is reviewing a medium-sized codebase containing Python, C++, shell scripts, tests, build files, parsers, network services, and concurrent workers. The goal is to improve correctness and performance without changing externally visible behavior. Check invariants, error paths, resource ownership, complexity, cache behavior, allocation patterns, synchronization, serialization, boundary conditions, and test coverage. Proposed optimizations must preserve semantics and include falsifiable benchmarks. Measurements distinguish startup, prompt processing, autoregressive generation, and end-to-end throughput.\n"""
for n in ds:
    chars=int(n*4.7); pre=f"RUN_DEPTH_{n}\nAnalyze these repository-review notes and produce technical recommendations.\n"
    body=(p*((chars//len(p))+2))[:max(0,chars-len(pre))]; (r/f"depth-{n}.txt").write_text(pre+body)
PY

have git && have cmake && have python3 || { echo "Need git, cmake, python3 (Xcode CLT/Homebrew)." >&2; exit 2; }
if ! have uv; then say "Installing private uv"; curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$BIN" sh; export PATH="$BIN:$PATH"; fi
[[ -x "$VENV/bin/python" ]] || uv venv --python python3 "$VENV"
say "Installing/updating local MLX/HF tooling"
uv pip install --python "$VENV/bin/python" -q -U 'mlx-lm>=0.31' 'mlx-vlm>=0.6' 'huggingface_hub[hf_xet]>=0.34'

OMLX_DIR="$SRC/omlx"; OMLX_VENV="$LAB/omlx-venv"
if backend_enabled omlx; then
  if [[ ! -d "$OMLX_DIR/.git" ]]; then say "Cloning oMLX"; git clone --depth=1 https://github.com/jundot/omlx.git "$OMLX_DIR"; else say "Updating oMLX"; git -C "$OMLX_DIR" pull --ff-only || true; fi
  [[ -x "$OMLX_VENV/bin/python" ]] || uv venv --python python3 "$OMLX_VENV"
  say "Installing/updating CWD-local oMLX"
  uv pip install --python "$OMLX_VENV/bin/python" -q -e "$OMLX_DIR"
fi

LLAMA_DIR="$SRC/llama.cpp"; LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
if backend_enabled llama; then
  if [[ ! -d "$LLAMA_DIR/.git" ]]; then say "Cloning llama.cpp"; git clone --depth=1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"; else say "Updating llama.cpp"; git -C "$LLAMA_DIR" pull --ff-only || true; fi
  say "Building current llama.cpp Metal"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_METAL=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$LLAMA_DIR/build" -j "$(sysctl -n hw.ncpu)" --target llama-server >/dev/null
fi

declare -A MLX4 MLX3 GGUF OGG OMLX
MLX4[qwen38]='mlx-community/Qwen3.8-27B-4bit'; MLX3[qwen38]='leonsarmiento/Qwen3.8-27B-3bit-mlx'; GGUF[qwen38]='ggml-org/Qwen3.8-27B-GGUF:Q4_K_M'; OGG[qwen38]='hf.co/ggml-org/Qwen3.8-27B-GGUF:Q4_K_M'; OMLX[qwen38]=''
MLX4[qwen36]='mlx-community/Qwen3.6-35B-A3B-4bit'; MLX3[qwen36]='andrevp/Qwen3.6-35B-A3B-3bit-MLX'; GGUF[qwen36]='ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M'; OGG[qwen36]='qwen3.6:35b-a3b-q4_K_M'; OMLX[qwen36]='qwen3.6:35b-a3b-nvfp4'
MLX4[nemotron]='majentik/Nemotron-3.5-Lightning-30B-A3B-MLX-4bit'; MLX3[nemotron]='majentik/Nemotron-3.5-Lightning-30B-A3B-MLX-3bit'; GGUF[nemotron]='ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:Q4_K_M'; OGG[nemotron]='nemotron-3.5-lightning:30b-a3b-q4_K_M'; OMLX[nemotron]='nemotron-3.5-lightning:30b-a3b-nvfp4'

cat > "$LAB/bench_ollama.py" <<'PY'
import argparse,json,time,urllib.request
p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--prompt',required=True); p.add_argument('--ctx',type=int,required=True); p.add_argument('--gen',type=int,default=192); a=p.parse_args()
payload={'model':a.model,'prompt':open(a.prompt).read(),'stream':False,'keep_alive':-1,'options':{'num_ctx':a.ctx,'num_predict':a.gen,'temperature':0,'seed':1}}
t=time.perf_counter(); req=urllib.request.Request('http://127.0.0.1:11439/api/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=7200) as r:d=json.load(r)
wall=time.perf_counter()-t; pc=d.get('prompt_eval_count',0); pd=d.get('prompt_eval_duration',0)/1e9; ec=d.get('eval_count',0); ed=d.get('eval_duration',0)/1e9
print(json.dumps({'prompt_tokens':pc,'pp_tps':pc/pd if pd else 0,'tg_tokens':ec,'tg_tps':ec/ed if ed else 0,'e2e_tps':(pc+ec)/wall if wall else 0,'wall_s':wall,'peak_gb':'','status':'ok'}))
PY
cat > "$LAB/bench_llama.py" <<'PY'
import argparse,json,time,urllib.request
p=argparse.ArgumentParser(); p.add_argument('--prompt',required=True); p.add_argument('--gen',type=int,default=192); a=p.parse_args()
payload={'prompt':open(a.prompt).read(),'n_predict':a.gen,'temperature':0,'seed':1,'cache_prompt':False,'stream':False}
t=time.perf_counter(); req=urllib.request.Request('http://127.0.0.1:11440/completion',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=7200) as r:d=json.load(r)
wall=time.perf_counter()-t; tm=d.get('timings',{}); pc=int(tm.get('prompt_n',0) or 0); ec=int(tm.get('predicted_n',0) or 0)
print(json.dumps({'prompt_tokens':pc,'pp_tps':float(tm.get('prompt_per_second',0) or 0),'tg_tokens':ec,'tg_tps':float(tm.get('predicted_per_second',0) or 0),'e2e_tps':(pc+ec)/wall if wall else 0,'wall_s':wall,'peak_gb':'','status':'ok'}))
PY
cat > "$LAB/bench_mlx.py" <<'PY'
import json,re,sys,time,subprocess
model,prompt,gen=sys.argv[1:4]; text=open(prompt).read()
def run(cmd):
    t=time.perf_counter(); p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=7200); return p,time.perf_counter()-t
p,wall=run([sys.executable,'-m','mlx_lm.generate','--model',model,'--prompt',text,'--max-tokens',gen,'--temp','0'])
if p.returncode:
    p,wall=run([sys.executable,'-m','mlx_vlm.generate','--model',model,'--prompt',text,'--max-tokens',gen,'--temperature','0'])
out=p.stdout; print(out,file=sys.stderr)
def one(ps,default=0.0):
    for x in ps:
        m=re.search(x,out,re.I)
        if m:return float(m.group(1))
    return default
pc=one([r'Prompt:\s*([0-9.]+)\s*tokens',r'prompt tokens[:=]\s*([0-9.]+)']); pp=one([r'Prompt:.*?([0-9.]+)\s*tokens-per-sec',r'prompt.*?([0-9.]+)\s*tokens/s'])
ec=one([r'Generation:\s*([0-9.]+)\s*tokens',r'generation tokens[:=]\s*([0-9.]+)']); tg=one([r'Generation:.*?([0-9.]+)\s*tokens-per-sec',r'generation.*?([0-9.]+)\s*tokens/s'])
peak=one([r'Peak memory:\s*([0-9.]+)\s*GB',r'peak memory.*?([0-9.]+)\s*GB'],'')
print(json.dumps({'prompt_tokens':int(pc),'pp_tps':pp,'tg_tokens':int(ec),'tg_tps':tg,'e2e_tps':(pc+ec)/wall if wall and pc else 0,'wall_s':wall,'peak_gb':peak,'status':'ok' if p.returncode==0 and tg else 'failed'}))
PY

jf(){ python3 -c 'import json,sys; print(json.load(sys.stdin).get(sys.argv[1],""))' "$1"; }
record(){
  local m="$1" b="$2" q="$3" o="$4" d="$5" js="$6" note="${7:-}" pc pp ec tg e2e wall peak status
  pc="$(printf '%s' "$js"|jf prompt_tokens 2>/dev/null||true)"; pp="$(printf '%s' "$js"|jf pp_tps 2>/dev/null||true)"; ec="$(printf '%s' "$js"|jf tg_tokens 2>/dev/null||true)"; tg="$(printf '%s' "$js"|jf tg_tps 2>/dev/null||true)"; e2e="$(printf '%s' "$js"|jf e2e_tps 2>/dev/null||true)"; wall="$(printf '%s' "$js"|jf wall_s 2>/dev/null||true)"; peak="$(printf '%s' "$js"|jf peak_gb 2>/dev/null||true)"; status="$(printf '%s' "$js"|jf status 2>/dev/null||true)"
  [[ -n "$status" ]]||status=ok; [[ -n "$tg" && "$tg" != 0 && "$tg" != 0.0 ]]||status=failed
  append_row "$m" "$b" "$q" "$o" "$d" "${pc:-0}" "${pp:-0}" "${ec:-0}" "${tg:-0}" "${e2e:-0}" "${wall:-0}" "${peak:-}" "$status" "$note"
  printf '  -> actual=%s  PP=%s  TG=%s  E2E=%s tok/s\n' "${pc:-?}" "${pp:-?}" "${tg:-?}" "${e2e:-?}"
}

OLLAMA_PID=''; LLAMA_PID=''
cleanup(){ [[ -n "$OLLAMA_PID" ]]&&kill "$OLLAMA_PID" 2>/dev/null||true; [[ -n "$LLAMA_PID" ]]&&kill "$LLAMA_PID" 2>/dev/null||true; }; trap cleanup EXIT INT TERM
start_ollama(){
  backend_enabled ollama||return 1; have ollama||{ warn "ollama not installed; skipping Ollama backend"; return 1; }
  curl -fsS "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1&&return 0
  say "Starting isolated Ollama at $OLLAMA_HOST with CWD-local model store"
  OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=f16 OLLAMA_NUM_PARALLEL=1 ollama serve >"$LOGS/ollama-server.log" 2>&1 & OLLAMA_PID=$!
  for _ in $(seq 1 60); do curl -fsS "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1&&return 0; sleep 1; done; return 1
}
run_ollama(){
  local m="$1" tag="$2" q="$3" opt="$4"; [[ -n "$tag" ]]||return; budget_guard; say "Ollama pull $tag"; ollama pull "$tag" >"$LOGS/pull-${m}-${q}.log" 2>&1||{ warn "pull failed: $tag"; return; }
  for d in "${DEPTHS[@]}"; do budget_guard; say "Ollama $m $q $opt @ $d"; js="$("$VENV/bin/python" "$LAB/bench_ollama.py" --model "$tag" --prompt "$PROMPTS/depth-$d.txt" --ctx "$LONG_CTX" --gen "$GEN_TOKENS" 2>>"$LOGS/ollama-${m}-${q}.log"||echo '{}')"; record "$m" ollama "$q" "$opt" "$d" "$js"; done
}

download_mlx(){ local repo="$1" dest="$2"; [[ -f "$dest/config.json" ]]&&return 0; budget_guard; say "Downloading $repo"; "$VENV/bin/hf" download "$repo" --local-dir "$dest" >"$LOGS/hf-$(basename "$dest").log" 2>&1; }
run_mlx(){
  local m="$1" repo="$2" q="$3" dest="$MODELS/mlx/${1}-${3}"; download_mlx "$repo" "$dest"||{ warn "MLX download failed: $repo"; return; }
  for d in "${DEPTHS[@]}"; do budget_guard; say "direct MLX $m $q @ $d"; js="$("$VENV/bin/python" "$LAB/bench_mlx.py" "$dest" "$PROMPTS/depth-$d.txt" "$GEN_TOKENS" 2>"$LOGS/mlx-${m}-${q}-${d}.log"||echo '{}')"; record "$m" mlx "$q" baseline "$d" "$js"; done
}

run_omlx(){
  local m="$1" repo="$2" q="$3" dest="$MODELS/mlx/${1}-${3}" log="$LOGS/omlx-${1}-${3}.log"
  download_mlx "$repo" "$dest" || { warn "oMLX model download failed: $repo"; return; }
  budget_guard
  say "oMLX native benchmark $m $q @ ${DEPTHS[*]} + batch 2/4"
  "$OMLX_VENV/bin/python" "$OMLX_DIR/scripts/bench.py" "$dest" --pp "${DEPTHS[@]}" --gen "$GEN_TOKENS" --batch 2 4 --warmup 1 >"$log" 2>&1 || { warn "oMLX benchmark failed; see $log"; return; }
  "$VENV/bin/python" - "$log" "$m" "$q" "$GEN_TOKENS" "$CSV" "$TSV" <<'PY'
import csv,re,sys,os
log,model,quant,gen,csvp,tsvp=sys.argv[1:]; gen=int(gen); text=open(log).read().splitlines()
header=["model","backend","quant","optimization","target_depth","actual_prompt_tokens","pp_tps","tg_tokens","tg_tps","e2e_tps","wall_s","peak_gb","status","note"]
rows=[]; single=False; batch=False
for line in text:
    if 'Single-request' in line: single=True; batch=False; continue
    if 'Continuous-batching' in line: batch=True; single=False; continue
    if single:
        m=re.match(r'^\s*(\d+)\s+([0-9.]+)ms\s+([0-9.]+)/s\s+([0-9.]+)/s\s+([0-9.]+)G',line)
        if m:
            pp=int(m.group(1)); ttft=float(m.group(2))/1000; tg=float(m.group(3)); ppt=float(m.group(4)); mem=float(m.group(5)); wall=ttft+(gen/tg if tg else 0); e2e=(pp+gen)/wall if wall else 0
            rows.append([model,'omlx',quant,'native',pp,pp,ppt,gen,tg,e2e,wall,mem,'ok',''])
    if batch:
        m=re.match(r'^\s*(\d+)\s+([0-9.]+)/s\s+([0-9.]+)/s\s+([0-9.]+)ms',line)
        if m:
            bs=int(m.group(1)); ppt=float(m.group(2)); tg=float(m.group(3)); ttft=float(m.group(4))/1000
            rows.append([model,'omlx',quant,f'batch{bs}',0,0,ppt,gen*bs,tg,0,ttft,'','ok','aggregate throughput'])
for p,d in ((csvp,','),(tsvp,'\t')):
    new=not os.path.exists(p)
    with open(p,'a',newline='') as f:
        w=csv.writer(f,delimiter=d)
        if new:w.writerow(header)
        w.writerows(rows)
print(f'  -> recorded {len(rows)} oMLX rows')
PY
}

llama_stop(){ [[ -n "$LLAMA_PID" ]]&&kill "$LLAMA_PID" 2>/dev/null||true; LLAMA_PID=''; }
llama_start(){ local hf="$1"; shift; llama_stop; "$LLAMA_SERVER" -hf "$hf" --host 127.0.0.1 --port 11440 -ngl 999 -c "$LONG_CTX" "$@" >"$LOGS/llama-server-current.log" 2>&1 & LLAMA_PID=$!; for _ in $(seq 1 180); do curl -fsS http://127.0.0.1:11440/health >/dev/null 2>&1&&return 0; kill -0 "$LLAMA_PID" 2>/dev/null||break; sleep 1; done; llama_stop; return 1; }
llama_one(){
  local m="$1" hf="$2" q="$3" opt="$4" d="$5"; shift 5; budget_guard; say "llama.cpp $m $q $opt @ $d"
  llama_start "$hf" "$@"||{ warn "server start failed: $m $opt"; append_row "$m" llama "$q" "$opt" "$d" 0 0 0 0 0 0 '' failed server-start; return; }
  js="$("$VENV/bin/python" "$LAB/bench_llama.py" --prompt "$PROMPTS/depth-$d.txt" --gen "$GEN_TOKENS" 2>>"$LOGS/llama-${m}-${opt}-${d}.log"||echo '{}')"; record "$m" llama "$q" "$opt" "$d" "$js"; llama_stop
}
llama_sweep(){
  local m="$1" hf="$2" mid="${DEPTHS[1]}" long="${DEPTHS[2]}"
  for d in "${DEPTHS[@]}"; do llama_one "$m" "$hf" q4_k_m fa-auto_ub512_kvf16 "$d" -fa auto -b 2048 -ub 512 -ctk f16 -ctv f16; done
  [[ "$MODE" == quick ]]&&return
  llama_one "$m" "$hf" q4_k_m fa-on_ub512_kvf16 "$mid" -fa on -b 2048 -ub 512 -ctk f16 -ctv f16
  llama_one "$m" "$hf" q4_k_m fa-off_ub512_kvf16 "$mid" -fa off -b 2048 -ub 512 -ctk f16 -ctv f16
  for ub in 256 1024; do llama_one "$m" "$hf" q4_k_m "fa-on_ub${ub}_kvf16" "$mid" -fa on -b 2048 -ub "$ub" -ctk f16 -ctv f16; done
  for kv in q8_0 q4_0; do llama_one "$m" "$hf" q4_k_m "fa-on_ub512_kv${kv}" "$long" -fa on -b 2048 -ub 512 -ctk "$kv" -ctv "$kv"; done
  if [[ "$m" == qwen38 || "$m" == qwen36 ]]; then
    repo="${hf%%:*}"
    for n in 2 3; do llama_one "$m" "$hf" q4_k_m "mtp${n}_fa-on" "$mid" -hfd "$repo:Q4_0" --spec-default --spec-type draft-mtp --spec-draft-n-max "$n" --spec-draft-p-min 0.5 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16; done
  fi
  if [[ "$MODE" == full ]]; then
    for b in 1024 4096; do for ub in 512 1024; do llama_one "$m" "$hf" q4_k_m "fa-on_b${b}_ub${ub}" "$mid" -fa on -b "$b" -ub "$ub" -ctk f16 -ctv f16; done; done
    llama_one "$m" "$hf" q4_k_m ngram-simple "$mid" --spec-type ngram-simple -fa on -b 2048 -ub 512 -ctk f16 -ctv f16
  fi
}

MS=(qwen38 qwen36 nemotron)
OLLAMA_READY=0
if backend_enabled ollama && start_ollama; then OLLAMA_READY=1; fi

# Finish the highest-priority model across backends before moving to the next one.
# This makes a time-limited afternoon run useful even if later downloads/runs are cut off.
for m in "${MS[@]}"; do
  model_enabled "$m" || continue
  say "========== MODEL: $m =========="

  if (( OLLAMA_READY )); then
    if [[ "$PROFILE" == max ]]; then
      run_ollama "$m" "${OGG[$m]}" q4_k_m baseline
      [[ -n "${OMLX[$m]}" ]] && run_ollama "$m" "${OMLX[$m]}" nvfp4 mlx-engine
    else
      warn "Air24: skipping 19-25GB Ollama Q4/NVFP4 for $m; direct 3-bit MLX is safer."
    fi
  fi

  if backend_enabled mlx; then
    run_mlx "$m" "${MLX3[$m]}" 3bit
    if [[ "$PROFILE" == max ]]; then
      run_mlx "$m" "${MLX4[$m]}" 4bit
    elif [[ "$m" == qwen38 && "$MODE" == full ]]; then
      run_mlx "$m" "${MLX4[$m]}" 4bit
    fi
  fi

  if backend_enabled omlx; then
    run_omlx "$m" "${MLX3[$m]}" 3bit
    if [[ "$PROFILE" == max ]]; then run_omlx "$m" "${MLX4[$m]}" 4bit; fi
  fi

  if backend_enabled llama; then
    if [[ "$PROFILE" == max ]]; then
      llama_sweep "$m" "${GGUF[$m]}"
    else
      warn "Air24: skipping 19-25GB Q4 llama.cpp matrix for $m; MLX 3-bit is the fit-first path."
    fi
  fi
done
summarize
say "Finished. CSV: $CSV"
say "Summary: $SUMMARY"
say "Logs: $LOGS"
