#!/usr/bin/env bash
# Universal CPU llama.cpp tuning/ablation harness; KNL source changes are isolated in knl/.
set -u
PKG="$(cd "$(dirname "$0")" && pwd)"
ROOT="${LAB_ROOT:-$PWD/.cpu-llama-lab}"
SRC="$ROOT/llama.cpp"; LOG="$ROOT/logs"; mkdir -p "$ROOT" "$LOG" "$ROOT/cache" "$ROOT/tools/bin"
export LAB_ROOT="$ROOT"
# Keep all suite-managed caches/tooling below LAB_ROOT even before `install` has generated env.sh.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/cache/xdg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/cache/uv}"
export CCACHE_DIR="${CCACHE_DIR:-$ROOT/cache/ccache}"
export CCACHE_BASEDIR="${CCACHE_BASEDIR:-$ROOT}"
export CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-content}"
[ -f "$ROOT/env.sh" ] && . "$ROOT/env.sh"
say(){ printf '%s\n' "$*"; }
have(){ command -v "$1" >/dev/null 2>&1; }
detect(){ python3 "$PKG/lib/detect.py" > "$ROOT/hardware.json"; python3 "$PKG/lib/plan.py" "$ROOT/hardware.json" -o "$ROOT/plan.json"; }
fetch(){
  if [ -n "${LLAMA_SRC:-}" ] && [ -d "$LLAMA_SRC/.git" ] && [ ! -e "$SRC" ]; then ln -s "$(cd "$LLAMA_SRC" && pwd)" "$SRC"; fi
  if [ -d "$SRC/.git" ]; then
    say "reuse source: $SRC"
    if [ -n "${LLAMA_REF:-}" ]; then
      cur="$(git -C "$SRC" rev-parse HEAD)" || return 1
      tgt="$(git -C "$SRC" rev-parse "${LLAMA_REF}^{commit}" 2>/dev/null || true)"
      if [ -z "$tgt" ]; then
        git -C "$SRC" fetch origin "$LLAMA_REF" || return 1
        tgt="$(git -C "$SRC" rev-parse FETCH_HEAD^{commit})" || return 1
      fi
      if [ "$cur" != "$tgt" ]; then
        if ! git -C "$SRC" diff --quiet || ! git -C "$SRC" diff --cached --quiet; then
          say "ERROR: LLAMA_REF=$LLAMA_REF resolves to $tgt but existing llama.cpp is dirty at $cur; refusing to repin." >&2
          say "Use a clean checkout or remove $SRC after preserving anything you need." >&2
          return 1
        fi
        say "repin llama.cpp: $cur -> $tgt (LLAMA_REF=$LLAMA_REF)"
        git -C "$SRC" checkout --detach "$tgt" || return 1
      fi
    fi
    git -C "$SRC" rev-parse HEAD > "$ROOT/llama-commit.txt" || return 1
    return 0
  fi
  git clone "${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}" "$SRC" || return 1
  (cd "$SRC" && git checkout "${LLAMA_REF:-master}" && git rev-parse HEAD > "$ROOT/llama-commit.txt")
}
manifest(){ python3 "$PKG/scripts/manifest.py"; }
preflight(){ detect; python3 "$PKG/scripts/preflight.py"; }
clock_check(){ detect; python3 "$PKG/scripts/clock_check.py"; }
abi_audit(){ python3 "$PKG/scripts/abi_audit.py"; }
install(){ detect; HW_JSON="$ROOT/hardware.json" "$PKG/scripts/install-deps.sh" || return 1; [ -f "$ROOT/env.sh" ] && . "$ROOT/env.sh"; detect; manifest; }
plan(){ detect; cat "$ROOT/hardware.json"; echo '--- PLAN ---'; cat "$ROOT/plan.json"; }
patch_check(){
  fetch || return 1
  python3 "$PKG/knl/patches/apply_knl_patch.py" "$SRC" --dry-run
}
build(){ detect; fetch && python3 "$PKG/scripts/build.py"; }
models(){ detect; python3 "$PKG/scripts/models.py" ensure; }
models_plan(){ detect; python3 "$PKG/scripts/models.py" plan; }
backend_tests(){
  rc=0
  for x in "$ROOT"/builds/*/bin/test-backend-ops; do [ -x "$x" ] || continue; n="$(basename "$(dirname "$(dirname "$x")")")"; say "test-backend-ops $n"; "$x" test >"$LOG/test-backend-ops-$n.log" 2>&1 || { rc=1; tail -80 "$LOG/test-backend-ops-$n.log"; }; done
  return "$rc"
}
codegen(){
  mkdir -p "$ROOT/codegen"
  for d in "$ROOT"/builds/*; do
    [ -d "$d" ] || continue; n="$(basename "$d")"
    [ -f "$d/compile_commands.json" ] && cp "$d/compile_commands.json" "$ROOT/codegen/$n-compile_commands.json"
    [ -f "$d/lab-build.json" ] && cp "$d/lab-build.json" "$ROOT/codegen/$n-build-metadata.json"
    so="$(find "$d" -type f \( -name 'libggml-cpu*.so' -o -name llama-bench \) | head -1)"
    [ -n "$so" ] && objdump -d "$so" > "$ROOT/codegen/$n.objdump" 2>/dev/null || true
  done
}
selftest(){
  python3 - "$PKG" <<'PY38' || return 1
import ast,sys
from pathlib import Path
root=Path(sys.argv[1])
for p in sorted(root.rglob('*.py')):
    if any(x in p.parts for x in ('__pycache__','.cpu-llama-lab')): continue
    ast.parse(p.read_text(),filename=str(p),feature_version=8)
print('PASS Python 3.8 grammar parse for every .py file')
PY38
  python3 -m py_compile "$PKG/lib/detect.py" "$PKG/lib/plan.py" "$PKG/lib/models.py" "$PKG/scripts/models.py" "$PKG/scripts/build.py" "$PKG/scripts/sweep.py" "$PKG/scripts/platform_shortlist.py" "$PKG/scripts/calibration.py" "$PKG/scripts/estimate.py" "$PKG/scripts/report.py" "$PKG/scripts/quality.py" "$PKG/scripts/agent_quality.py" "$PKG/scripts/judge.py" "$PKG/scripts/preflight.py" "$PKG/scripts/manifest.py" "$PKG/scripts/abi_audit.py" "$PKG/scripts/clock_check.py" "$PKG/scripts/batch_report.py" "$PKG/lib/clock.py" "$PKG/knl/patches/apply_knl_patch.py" || return 1
  bash -n "$0" "$PKG/scripts/install-deps.sh" || return 1
  python3 "$PKG/tests/test_plan.py" || return 1
  python3 "$PKG/tests/test_hardening.py" || return 1
  python3 "$PKG/tests/test_clock.py" || return 1
  python3 "$PKG/tests/test_models.py" || return 1
  python3 "$PKG/tests/test_sweep.py" || return 1
  python3 "$PKG/tests/test_agent_quality.py" || return 1
  python3 "$PKG/tests/test_platforms.py" || return 1
  python3 "$PKG/knl/test_patcher.py" || return 1
  "$PKG/knl/verify_codegen.sh" "$ROOT/selftest-codegen" || return 1
  say "PASS: suite self-tests"
}
agent_quality(){ preflight || return 1; python3 "$PKG/scripts/agent_quality.py"; }
quality(){ preflight || return 1; python3 "$PKG/scripts/quality.py"; }
judge(){ detect; python3 "$PKG/scripts/judge.py"; }
export_knl_patch(){
  detect; cls="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["cpu"]["class"])' "$ROOT/hardware.json")"
  [ "$cls" = knl ] || say "Exporting KNL patch on a non-KNL host is allowed; source will be patched for diff only."
  fetch || return 1
  python3 "$PKG/knl/patches/apply_knl_patch.py" "$SRC" --apply || return 1
  mkdir -p "$ROOT/upstream"
  (cd "$SRC" && git diff -- ggml/CMakeLists.txt ggml/src/CMakeLists.txt ggml/src/ggml-cpu/CMakeLists.txt ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp ggml/src/ggml-cpu/arch/x86/quants.c) > "$ROOT/upstream/knl-backend.patch"
  (cd "$SRC" && git status --short) > "$ROOT/upstream/git-status.txt"
  echo "$ROOT/upstream/knl-backend.patch"
}
debug_bundle(){
  d="$ROOT/debug-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$d"
  cp -a "$ROOT/hardware.json" "$ROOT/plan.json" "$ROOT/llama-commit.txt" "$ROOT/manifest.json" "$ROOT/preflight.json" "$ROOT/preflight.md" "$ROOT/build-environment.json" "$ROOT/perf-status.txt" "$ROOT/models/model-plan.json" "$ROOT/models/registry.json" "$ROOT/models/models.tsv" "$d" 2>/dev/null || true
  cp -a "$ROOT/logs" "$ROOT/codegen" "$ROOT/results" "$ROOT/abi" "$ROOT/quality" "$d" 2>/dev/null || true
  lscpu > "$d/lscpu.txt" 2>&1 || true; numactl -H > "$d/numactl-H.txt" 2>&1 || true; uname -a > "$d/uname.txt"
  env | LC_ALL=C sort > "$d/environment.txt"
  tar czf "$d.tar.gz" -C "$ROOT" "$(basename "$d")"; echo "$d.tar.gz"
}
run_sweep(){ mode="$1"; preflight || return 1; python3 "$PKG/scripts/sweep.py" "$mode"; }
missing_planned_builds(){
  python3 - "$ROOT/plan.json" "$ROOT/builds" <<'PYBUILD'
import json,sys
from pathlib import Path
plan=Path(sys.argv[1]); root=Path(sys.argv[2])
if not plan.exists(): raise SystemExit(0)
p=json.load(open(plan)); missing=[]
for b in p.get('builds',[]):
    if b.get('optional'): continue
    d=root/str(b['name'])/'bin'
    if not (d/'llama-bench').exists() or not (d/'llama-server').exists(): missing.append(str(b['name']))
print(' '.join(missing))
PYBUILD
}
ensure_planned_builds(){
  detect || return 1
  miss="$(missing_planned_builds)"
  if [ -n "$miss" ]; then
    say "Missing required planned build(s): $miss"
    say "Running incremental build so this release's ablation matrix is actually present."
    build || return 1
    detect || return 1
    miss="$(missing_planned_builds)"
    [ -z "$miss" ] || { say "ERROR: required build(s) still missing after build: $miss" >&2; return 1; }
  fi
}
full_sweep(){
  preflight || return 1
  ensure_planned_builds || return 1
  ensure_platform_shortlist || return 1
  env LAB_PLATFORM_SHORTLIST_FILE="$ROOT/calibration/platform-shortlist.json" python3 "$PKG/scripts/sweep.py" prefix-smoke || return 1
  env LAB_PLATFORM_SHORTLIST_FILE="$ROOT/calibration/platform-shortlist.json" python3 "$PKG/scripts/sweep.py" screen || return 1
  python3 "$PKG/scripts/agent_quality.py" || return 1
  env LAB_PLATFORM_SHORTLIST_FILE="$ROOT/calibration/platform-shortlist.json" python3 "$PKG/scripts/sweep.py" deep || return 1
}
overnight_mark(){
  python3 - "$ROOT/overnight-state.json" "$1" <<'PY'
import json,sys,time
p,stage=sys.argv[1:]
try:
    o=json.load(open(p))
except Exception:
    o={'stages':[]}
o['current_stage']=stage; o['updated_at']=time.time()
o.setdefault('stages',[]).append({'stage':stage,'timestamp':o['updated_at']})
open(p,'w').write(json.dumps(o,indent=2)+'\n')
PY
}
fast_build_names(){
  python3 - "$ROOT/plan.json" "$ROOT/hardware.json" <<'PYFASTBUILD'
import json,sys
p=json.load(open(sys.argv[1])); h=json.load(open(sys.argv[2])); cls=(h.get('cpu',{}) or {}).get('class')
avail={str(x.get('name')) for x in p.get('builds',[])}
want=['knl-base-norepack','knl-generic-avx2-norepack','knl-combo-norepack'] if cls=='knl' else ['native-norepack','native','isa-avx2']
print(' '.join(x for x in want if x in avail))
PYFASTBUILD
}
missing_fast_builds(){
  wanted="$(fast_build_names)"
  for b in $wanted; do
    [ -x "$ROOT/builds/$b/bin/llama-bench" ] && [ -x "$ROOT/builds/$b/bin/llama-server" ] || printf '%s ' "$b"
  done
}
ensure_fast_builds(){
  detect || return 1
  miss="$(missing_fast_builds)"
  [ -z "$miss" ] && { say "Fast calibration builds already present: $(fast_build_names)"; return 0; }
  say "SETUP ABOUT TO RUN: missing minimal build(s): $miss"
  say "  targets per build: llama-bench + llama-server only"
  say "  cold-build ETA: roughly 1-3 minutes per missing build; ccache/reused objects can make this much faster"
  csv="$(printf '%s' "$miss" | tr ' ' ',' | sed 's/,$//')"
  LAB_BUILD_NAMES="$csv" LAB_FAST_BUILD=1 build || return 1
  detect || return 1
  miss="$(missing_fast_builds)"
  [ -z "$miss" ] || { say "ERROR: fast calibration build(s) still missing: $miss" >&2; return 1; }
}
ballpark_build_names(){
  python3 - "$ROOT/plan.json" "$ROOT/hardware.json" <<'PYBALLBUILD'
import json,sys
p=json.load(open(sys.argv[1])); h=json.load(open(sys.argv[2])); cls=(h.get('cpu',{}) or {}).get('class')
avail={str(x.get('name')) for x in p.get('builds',[])}
want=['knl-base-norepack','knl-combo-norepack','knl-combo'] if cls=='knl' else ['native-norepack','native','isa-avx2']
print(' '.join(x for x in want if x in avail))
PYBALLBUILD
}
missing_ballpark_builds(){
  wanted="$(ballpark_build_names)"
  for b in $wanted; do
    [ -x "$ROOT/builds/$b/bin/llama-bench" ] || printf '%s ' "$b"
  done
}
ensure_ballpark_builds(){
  detect || return 1
  miss="$(missing_ballpark_builds)"
  [ -z "$miss" ] && { say "Ballpark builds already present: $(ballpark_build_names)"; return 0; }
  say "SETUP ABOUT TO RUN: missing ballpark build(s): $miss"
  say "  target per build: llama-bench ONLY"
  say "  purpose: compare memory-tier / combo-kernel / repack toggles on the real large models"
  csv="$(printf '%s' "$miss" | tr ' ' ',' | sed 's/,$//')"
  LAB_BUILD_NAMES="$csv" LAB_BENCH_ONLY_BUILD=1 build || return 1
  detect || return 1
  miss="$(missing_ballpark_builds)"
  [ -z "$miss" ] || { say "ERROR: ballpark build(s) still missing: $miss" >&2; return 1; }
}
batch_report(){
  detect || return 1
  python3 "$PKG/scripts/batch_report.py"
}
ballpark(){
  target="${1:-all}"
  case "$target" in
    all) specs='qwen3.6-35b|UD-Q3_K_S nemotron-3.5|MXFP4_MOE qwen3.8-27b|Q3_K_M' ; n=3 ;;
    qwen|qwen3.6|qwen3.6-35b) specs='qwen3.6-35b|UD-Q3_K_S' ; n=1 ;;
    nemotron|nemotron-3.5) specs='nemotron-3.5|MXFP4_MOE' ; n=1 ;;
    qwen38|qwen3.8|qwen3.8-27b) specs='qwen3.8-27b|Q3_K_M' ; n=1 ;;
    *) say "Usage: ./cpu-llama-lab.sh ballpark [all|qwen|nemotron|qwen38]" >&2; return 2 ;;
  esac
  total="${LAB_BALLPARK_BUDGET_S:-300}"
  per="$(python3 - "$total" "$n" <<'PYBPBUD'
import sys
print(max(60,int(float(sys.argv[1])/int(sys.argv[2]))))
PYBPBUD
)"
  say "PRIMARY MODEL BALLPARK: go straight to the real target models before any deep sweep."
  say "  total benchmark budget ~${total}s after builds/models are present; per-model budget ~${per}s"
  say "  exact tests: Qwen3.6/Nemotron use PP128+TG32 @ depth 0/512; dense Qwen3.8 uses PP32+TG8 @ depth 0/128; F16 KV; one repetition"
  say "  KNL toggles: HBM/spill OFF->ON, base->combo kernel, repack OFF->ON (last toggle pruned first if budget is tight)"
  say "  NO agent quality, batch factorial, 2K/8K/16K sweep, or extra quant unless the ballpark result earns it."
  preflight || return 1
  ensure_ballpark_builds || return 1
  for spec in $specs; do
    alias="${spec%%|*}"; quant="${spec#*|}"
    say ""
    say "MODEL ABOUT TO RUN: $alias quant=$quant"
    bp_pp=128; bp_tg=32; bp_depth=512; min_ctx=512
    if [ "$alias" = qwen3.6-35b ]; then
      say "  rationale: Q3_K_S is the conventional ~15.4 GB quant that can fit the KNL 16 GiB MCDRAM tier."
    elif [ "$alias" = qwen3.8-27b ]; then
      bp_pp=32; bp_tg=8; bp_depth=128; min_ctx=128
      say "  rationale: dense 27B reality check; Q3_K_M is ~13.8 GB (~12.85 GiB) and MUST fit wholly in KNL MCDRAM."
      say "  dense probe is intentionally tiny: PP32 + TG8 at depth 0/128; no deeper work unless it surprises us."
    else
      say "  rationale: MXFP4_MOE is the model-specific MoE quant; it is ~23.2 GB and therefore explicitly tests KNL spill/tiering."
    fi
    say "  download/build setup time is separate; models.py will print exact additional GiB before transfer."
    env LAB_MODELS="$alias" LAB_WEIGHT_QUANTS="$quant" LAB_MODEL_MIN_CTX="$min_ctx" LAB_REFRESH_MODELS=1 \
        python3 "$PKG/scripts/models.py" ensure || return 1
    env LAB_MODELS="$alias" LAB_WEIGHT_QUANTS="$quant" LAB_MODEL_MIN_CTX="$min_ctx" LAB_REFRESH_MODELS=0 \
        LAB_BALLPARK_BUDGET_S="$per" LAB_BALLPARK_PP="$bp_pp" LAB_BALLPARK_TG="$bp_tg" LAB_BALLPARK_DEPTH="$bp_depth" \
        python3 "$PKG/scripts/sweep.py" primary-probe || return 1
  done
  say "PRIMARY BALLPARK COMPLETE: per-model JSON lives under $ROOT/calibration/primary-ballpark-*.json; all rows also remain in results JSONL."
  say "Run ./cpu-llama-lab.sh batch-report to salvage/rank historical batch-tune evidence."
}

calibration_session_start(){
  mkdir -p "$ROOT/calibration"
  python3 - "$ROOT/calibration/session.json" "$PKG/VERSION" "${LAB_CAL_MODE:-fast}" <<'PYCALSTART'
import json,sys,time
p,v,mode=sys.argv[1:]
ver=open(v).read().strip()
open(p,'w').write(json.dumps({'lab_version':ver,'mode':mode,'started_at':time.time()},indent=2)+'\n')
PYCALSTART
}
calibrate_fast(){
  # Default contract: answer most platform questions in about five benchmark
  # minutes. Builds/downloads are setup and are announced separately. No batch
  # factorial, no 0.6B/1B ladder, no 8K/16K work unless separately promoted.
  budget="${LAB_CAL_BUDGET_S:-300}"
  export LAB_FAST_MODE=1 LAB_CAL_BUDGET_S="$budget"
  say "FAST CALIBRATION: benchmark budget ~${budget}s AFTER required builds/models are present. Cold model download/build time is additional and printed before it starts."
  preflight || return 1
  ensure_fast_builds || return 1
  say "MODEL SET ABOUT TO RUN: Granite 4.0 350M Q4_K_M (platform/threads/KV) + Granite 4.1 3B Q4_K_M (representative 512->2K promise check)."
  say "  No Qwen-0.6B, Granite-1B, batch factorial, 8K, 16K, or agent diagnostic in vanilla calibration."
  env LAB_MODELS='granite4-350m,granite4.1-3b' LAB_MODEL_MIN_CTX=2048 LAB_REFRESH_MODELS=1 \
      python3 "$PKG/scripts/models.py" ensure || return 1
  start_epoch="$(python3 -c 'import time; print(time.time())')"
  deadline="$(python3 - "$start_epoch" "$budget" <<'PYDEAD'
import sys
print(float(sys.argv[1])+float(sys.argv[2]))
PYDEAD
)"
  export LAB_CAL_DEADLINE_EPOCH="$deadline"
  calibration_session_start || return 1
  env LAB_MODELS='granite4-350m,granite4.1-3b' LAB_MODEL_MIN_CTX=2048 LAB_REFRESH_MODELS=0 \
      LAB_FAST_MODE=1 LAB_CAL_BUDGET_S="$budget" LAB_CAL_DEADLINE_EPOCH="$deadline" \
      python3 "$PKG/scripts/sweep.py" fast-plan || return 1
  say "STAGE ABOUT TO RUN: load+decode bootstrap; model=Granite350M; PP32 + TG16@512; F16 KV; ETA ~10-20s"
  env LAB_MODELS='granite4-350m' LAB_MODEL_MIN_CTX=2048 LAB_REFRESH_MODELS=0 TG_TOKENS=32 \
      python3 "$PKG/scripts/sweep.py" bootstrap || return 1
  say "STAGE ABOUT TO RUN: one cheap-model thread/SMT probe; exact thread counts from plan; PP512 + TG32@2K; F16 KV; ETA ~20-45s"
  env LAB_MODELS='granite4-350m' LAB_MODEL_MIN_CTX=2048 LAB_REFRESH_MODELS=0 TG_TOKENS=32 \
      python3 "$PKG/scripts/sweep.py" thread-probe || return 1
  say "STAGE ABOUT TO RUN: KV capability; model=Granite350M; F16/Q8/Q4; TG32@2K; ETA ~15-30s"
  env LAB_MODELS='granite4-350m' LAB_MODEL_MIN_CTX=2048 LAB_REFRESH_MODELS=0 TG_TOKENS=32 \
      python3 "$PKG/scripts/sweep.py" kv-probe || return 1
  say "STAGE ABOUT TO RUN: prefix-reuse proof; model=Granite350M; 512->2K; F16 KV; ETA ~15-30s"
  env LAB_MODELS='granite4-350m' LAB_MODEL_MIN_CTX=2048 LAB_REFRESH_MODELS=0 TG_TOKENS=32 \
      python3 "$PKG/scripts/sweep.py" prefix-smoke || return 1
  say "STAGE ABOUT TO RUN: lean causal platform map; Granite350M only; PP512 + TG32@2K; F16 KV; exact configs printed next"
  env LAB_MODELS='granite4-350m' LAB_MODEL_MIN_CTX=2048 LAB_REFRESH_MODELS=0 TG_TOKENS=32 \
      python3 "$PKG/scripts/sweep.py" quick-lite || return 1
  env LAB_PLATFORM_SHORTLIST_SIZE=5 python3 "$PKG/scripts/platform_shortlist.py" granite4-350m || return 1
  say "STAGE ABOUT TO RUN: representative 3B promise check; ONE best platform config; depths 512->2K; F16 preferred; fixed batch/ubatch; ETA printed from live evidence"
  env LAB_MODELS='granite4.1-3b' LAB_MODEL_MIN_CTX=2048 LAB_REFRESH_MODELS=0 TG_TOKENS=32 \
      LAB_PLATFORM_SHORTLIST_FILE="$ROOT/calibration/platform-shortlist.json" \
      python3 "$PKG/scripts/sweep.py" fast-scale || return 1
  python3 "$PKG/scripts/calibration.py" || return 1
  say "FAST CALIBRATION COMPLETE. Promise decision: $ROOT/calibration/promise.json"
  say "Legacy exhaustive calibration remains available as: ./cpu-llama-lab.sh calibrate --full"
}
calibrate_full(){
  export LAB_CAL_MODE=full
  # Stage A maps build/NUMA/repack/load once at the physical-core default and
  # separately probes SMT/thread scaling on the 350M floor. Stage B reuses the best+mandatory controls for
  # the 0.6B/2B/3B scaling ladder, proving cached-prefix behavior through 16K.
  preflight || return 1
  ensure_planned_builds || return 1
  mkdir -p "$ROOT/calibration"
  python3 - "$ROOT/calibration/session.json" "$PKG/VERSION" <<'PYCALSTART'
import json,sys,time
p,v=sys.argv[1:]
ver=open(v).read().strip()
open(p,'w').write(json.dumps({'lab_version':ver,'started_at':time.time()},indent=2)+'\n')
PYCALSTART
  say "Agent/coding calibration ladder: Granite 4.0 350M Q4_K_M -> Qwen3 0.6B Q4_0 -> Granite 4.0 1B Q4_K_M -> Granite 4.1 3B Q4_K_M"
  env LAB_MODELS='granite4-350m,qwen3-0.6b,granite4-1b,granite4.1-3b' LAB_MODEL_MIN_CTX=8192 LAB_REFRESH_MODELS=1 \
      python3 "$PKG/scripts/models.py" ensure || return 1
  say "Load-path bootstrap (find one model/build/policy that definitely executes before any sweep)"
  env LAB_MODELS='granite4-350m,qwen3-0.6b,granite4-1b,granite4.1-3b' LAB_MODEL_MIN_CTX=8192 LAB_REFRESH_MODELS=0 \
      python3 "$PKG/scripts/sweep.py" bootstrap || return 1
  say "One-model thread/SMT probe (informational; main sweeps use physical-core count; F16 KV baseline)"
  env LAB_MODELS='granite4-350m,qwen3-0.6b,granite4-1b,granite4.1-3b' LAB_MODEL_MIN_CTX=8192 LAB_REFRESH_MODELS=0 \
      python3 "$PKG/scripts/sweep.py" thread-probe || return 1
  say "Cheap KV decode capability probe (F16 baseline; Q8/Q4 are ISA-sensitive ablations)"
  env LAB_MODELS='granite4-350m,qwen3-0.6b,granite4-1b,granite4.1-3b' LAB_MODEL_MIN_CTX=8192 LAB_REFRESH_MODELS=0 \
      python3 "$PKG/scripts/sweep.py" kv-probe || return 1
  say "Prefix-cache invariant smoke test"
  env LAB_MODELS='granite4-350m,qwen3-0.6b,granite4-1b,granite4.1-3b' LAB_MODEL_MIN_CTX=8192 LAB_REFRESH_MODELS=0 \
      python3 "$PKG/scripts/sweep.py" prefix-smoke || return 1
  say "Full platform behavior map on Granite 350M"
  env LAB_MODELS='granite4-350m' LAB_MODEL_MIN_CTX=8192 LAB_REFRESH_MODELS=0 \
      python3 "$PKG/scripts/sweep.py" quick || return 1
  python3 "$PKG/scripts/platform_shortlist.py" granite4-350m || return 1
  say "Shortlisted scaling ladder through 8K + one 16K continuation"
  env LAB_MODELS='granite4-350m,qwen3-0.6b,granite4-1b,granite4.1-3b' LAB_MODEL_MIN_CTX=8192 LAB_REFRESH_MODELS=0 \
      LAB_PLATFORM_SHORTLIST_FILE="$ROOT/calibration/platform-shortlist.json" \
      LAB_SCREEN_CAP=8192 LAB_SCREEN_DEPTHS='512,2048,8192' LAB_DEEP_CAP=16384 LAB_DEEP_DEPTHS=16384 \
      python3 "$PKG/scripts/sweep.py" overnight || return 1
  mkdir -p "$ROOT/calibration"
  env LAB_AGENT_DIAGNOSTIC=1 LAB_AGENT_DIAGNOSTIC_OUTPUT="$ROOT/calibration/agent-diagnostic.json" \
      LAB_AGENT_MIN_PP_TPS=0 LAB_AGENT_MIN_TG_TPS=0 LAB_AGENT_MAX_CANDIDATES_PER_FAMILY=1 \
      LAB_AGENT_MAX_TURNS=6 LAB_AGENT_TASK_TIMEOUT_S=90 LAB_AGENT_CLAUDE_SMOKE=0 \
      python3 "$PKG/scripts/agent_quality.py" || return 1
  python3 "$PKG/scripts/calibration.py"
}
calibrate(){
  case "${1:-}" in
    --full) calibrate_full ;;
    "") calibrate_fast ;;
    *) say "Usage: ./cpu-llama-lab.sh calibrate [--full]" >&2; return 2 ;;
  esac
}
estimate(){ detect; python3 "$PKG/scripts/estimate.py"; }
platform_shortlist_valid(){
  python3 - "$ROOT/calibration/platform-shortlist.json" "$PKG/VERSION" <<'PYSHORT'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); vp=Path(sys.argv[2])
try:o=json.load(open(p))
except Exception:raise SystemExit(1)
want=vp.read_text().strip()
raise SystemExit(0 if int(o.get('version',0))>=2 and o.get('lab_version')==want and o.get('configs') else 1)
PYSHORT
}
ensure_platform_shortlist(){
  if platform_shortlist_valid; then return 0; fi
  if [ "${LAB_AUTO_CALIBRATE:-1}" = 0 ]; then
    say "No current platform shortlist; continuing full factorial because LAB_AUTO_CALIBRATE=0."
    return 0
  fi
  say "No current v$(cat "$PKG/VERSION") platform calibration shortlist; running calibrate first to avoid a huge primary factorial sweep."
  calibrate
}

overnight(){
  # Unattended benchmark/ablation pipeline for an already configured machine.
  # Deliberately does NOT run package installation or any sudo-requiring action.
  overnight_mark preflight
  preflight || return 1
  overnight_mark build-check
  ensure_planned_builds || return 1
  overnight_mark calibration-shortlist
  ensure_platform_shortlist || return 1
  overnight_mark validate-builds
  abi_audit || return 1
  backend_tests || true
  codegen
  overnight_mark models
  models || return 1
  overnight_mark prefix-smoke
  env LAB_PLATFORM_SHORTLIST_FILE="$ROOT/calibration/platform-shortlist.json" python3 "$PKG/scripts/sweep.py" prefix-smoke || return 1
  overnight_mark performance-screen
  env LAB_PLATFORM_SHORTLIST_FILE="$ROOT/calibration/platform-shortlist.json" python3 "$PKG/scripts/sweep.py" screen || return 1
  overnight_mark agent-quality
  python3 "$PKG/scripts/agent_quality.py" || return 1
  overnight_mark deep-winners
  env LAB_PLATFORM_SHORTLIST_FILE="$ROOT/calibration/platform-shortlist.json" python3 "$PKG/scripts/sweep.py" deep || return 1
  overnight_mark manifest-report
  manifest
  python3 "$PKG/scripts/report.py"
  overnight_mark complete
}
full(){
  install || return 1
  selftest || return 1
  fetch || return 1
  manifest
  preflight || return 1
  build || return 1
  abi_audit || return 1
  backend_tests || true
  codegen
  models || return 1
  full_sweep || return 1
  quality || true
  manifest
  python3 "$PKG/scripts/report.py"
}
usage(){ cat <<EOF2
CPU llama.cpp Lab v$(cat "$PKG/VERSION")
Usage: $0 COMMAND

  plan             detect CPU/NUMA/ISA/memory and print generated experiment plan
  install          install distro-matched deps; KNL gets a proven GCC <=14; tools/caches stay local
  preflight        hygiene checks + loaded-clock throttle probe before benchmarks
  clock-check      short all-core loaded-frequency test; fails when authoritative base-clock gate is violated
  manifest         record exact OS/toolchain/library/model metadata
  fetch            clone/reuse upstream llama.cpp
  patch-check      validate KNL patch against the current llama.cpp tree without modifying it
  build            build CPU/ISA/repack/LTO/BLAS ablations; applies KNL patch only on KNL
  models-plan      metadata-only HF inventory/RAM-fit plan; downloads no GGUF payloads
  models           live GGUF discovery; KNL/MCDRAM auto-shortlists full-HBM/edge-fit/spill-control quants
  bootstrap        find one load-safe model/build/policy before any benchmark matrix
  prefix-smoke     prove persistent-server prefix reuse before expensive sweeps
  thread-probe     one cheap-model 1/2x,1x,2x (+4x on KNL) physical-core SMT sanity check (F16 KV)
  kv-probe         cheap F16/Q8/Q4 decode capability probe on the known-good baseline
  calibrate        ~5-minute lean decision pass; prints exact stages/configs/depths/ETA; no deep factorials
  ballpark         ~5-minute big-model-first smoke: Qwen3.6-35B + Nemotron3.5 + HBM-fitting Qwen3.8-27B; tiny prompts + causal toggles
  batch-report     salvage/rank all historical batch-tune rows using llama-bench raw avg_ts (including legacy rows)
  calibrate --full legacy exhaustive platform + model/KV/batch/depth sweep
  estimate         combine calibration timings with current models-plan to estimate best-case overnight wall time
  abi-audit        ldd/readelf every built binary; fail on unresolved libraries; record GLIBC/GLIBCXX requirements
  backend-tests    run upstream test-backend-ops on every built variant
  codegen-audit    save compile commands, build metadata and disassembly
  export-knl-patch export the isolated KNL llama.cpp diff for upstream review
  quick            preflight + broad PP512/TG<=2K pruning matrix
  batch-tune       preflight + PP1024 batch/ubatch tuning
  agent-quality    hard agentic coding + responsiveness gate; failed quants promote to higher-fidelity fallbacks
  full-sweep       <=8K performance screen -> agent-quality gate -> one <=64K deep winner per accepted family
  overnight        unattended: auto-models -> <=8K ablations -> agent quality/latency gate -> one deep winner/family -> report
  node-workers     preflight + independent per-NUMA-node replica throughput control
  quality          PPL + coding/API review + pinned EvalPlus in a local uv venv
  judge            blind-by-default TUI for human response scoring
  report           produce Markdown report
  selftest         package/clock/KNL patch/codegen/plan/hardening unit tests; no model needed
  debug-bundle     archive machine/build/ABI/preflight/test/results evidence
  full             install -> tests -> fetch -> manifest/preflight -> build/ABI/tests -> sweeps/quality/report

Strict publication hygiene: LAB_STRICT=1 (fails preflight on high-severity conditions).
Inherit custom CFLAGS/LDFLAGS/etc only deliberately: LAB_INHERIT_BUILD_ENV=1.
Hash large model files in manifest only when desired: LAB_HASH_MODELS=1.
Four machines: LAB_WORLD=4 and LAB_RANK=0..3. Results are deterministically sharded.
Optional disruptive NUMA first-touch validation: LAB_DROP_CACHES=1 (uses sudo).
All suite-managed files/caches/tools remain beneath: $ROOT
EOF2
}
case "${1:-}" in
 plan) plan;; install) install;; preflight) preflight;; clock-check) clock_check;; manifest) detect; manifest;; fetch) fetch;; patch-check) patch_check;; build) build;; abi-audit) abi_audit;; backend-tests) backend_tests;; codegen-audit) codegen;; export-knl-patch) export_knl_patch;;
 models-plan) models_plan;; models) models;; ballpark) shift; ballpark "$@";; batch-report) batch_report;; bootstrap) detect; run_sweep bootstrap;; prefix-smoke) detect; run_sweep prefix-smoke;; thread-probe) detect; run_sweep thread-probe;; kv-probe) detect; run_sweep kv-probe;; calibrate) shift; calibrate "$@";; estimate) estimate;; quick) detect; run_sweep quick;; batch-tune) detect; run_sweep batch;; agent-quality) agent_quality;; full-sweep) detect; full_sweep;; node-workers) detect; run_sweep node-workers;; overnight) overnight;;
 quality) quality;; judge) judge;; report) detect; python3 "$PKG/scripts/report.py";; selftest) selftest;; debug-bundle) detect; debug_bundle;; full) full;; *) usage;; esac
