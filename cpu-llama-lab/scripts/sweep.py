#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, math, os, re, shlex, shutil, signal, socket, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path
from typing import Any

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from lib.models import ensure_models, registered_records
from lib.clock import ClockMonitor, monitor_call

ROOT = Path(os.environ.get('LAB_ROOT', str(Path.cwd() / '.cpu-llama-lab')))
PLAN = json.load(open(ROOT / 'plan.json'))
HW = json.load(open(ROOT / 'hardware.json'))
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)
WORLD = int(os.environ.get('LAB_WORLD', '1'))
RANK = int(os.environ.get('LAB_RANK', '0'))
TG = int(os.environ.get('TG_TOKENS', '128'))
QUICK_DEPTH = min(8192, int(os.environ.get('QUICK_DEPTH', '2048')))
QUICK_KV = os.environ.get('LAB_QUICK_KV', 'f16').strip().lower() or 'f16'
THREAD_PROBE_KV = os.environ.get('LAB_THREAD_PROBE_KV', QUICK_KV).strip().lower() or QUICK_KV
BATCHES = [(128,128),(256,128),(512,256),(1024,512),(2048,512),(2048,1024),(2048,2048)]
RUNTIME_ENV_VARS = ('OMP_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES','OPENBLAS_NUM_THREADS','GOTO_NUM_THREADS','MKL_NUM_THREADS','KMP_AFFINITY')

# Unattended-run safety limits. They are intentionally generous for KNL but finite.
QUICK_TIMEOUT = int(os.environ.get('LAB_QUICK_TIMEOUT_S', '180'))
BATCH_TIMEOUT = int(os.environ.get('LAB_BATCH_TIMEOUT_S', '300'))
SERVER_START_TIMEOUT = int(os.environ.get('LAB_SERVER_START_TIMEOUT_S', '300'))
PREFIX_SMOKE_SERVER_START_TIMEOUT = int(os.environ.get('LAB_PREFIX_SMOKE_START_TIMEOUT_S', '90'))
PP_INTERVAL_TIMEOUT = int(os.environ.get('LAB_PP_INTERVAL_TIMEOUT_S', '1200'))
TG_ENDPOINT_TIMEOUT = int(os.environ.get('LAB_TG_ENDPOINT_TIMEOUT_S', '180'))
MIN_QUICK_PP = float(os.environ.get('LAB_MIN_QUICK_PP_TPS', '10'))
MIN_QUICK_TG = float(os.environ.get('LAB_MIN_QUICK_TG_TPS', '1'))
CACHE_TOL = int(os.environ.get('LAB_CACHE_REUSE_TOLERANCE', '8'))

SEARCH = PLAN.get('search', {})
def _depth_list(env_name: str, fallback: list[int]) -> list[int]:
    raw = os.environ.get(env_name, '').strip()
    xs = [int(x) for x in re.split(r'[,\s]+', raw) if x] if raw else [int(x) for x in fallback]
    return sorted(set(x for x in xs if x > 0))

SCREEN_CAP = min(8192, int(os.environ.get('LAB_SCREEN_CAP', str(SEARCH.get('screen_cap', 8192)))))
DEEP_CAP = int(os.environ.get('LAB_DEEP_CAP', str(SEARCH.get('deep_cap', 65536))))
SCREEN_DEPTHS = [d for d in _depth_list('LAB_SCREEN_DEPTHS', SEARCH.get('screen_depths', [512,2048,8192])) if d <= SCREEN_CAP]
if SCREEN_CAP not in SCREEN_DEPTHS:
    SCREEN_DEPTHS.append(SCREEN_CAP); SCREEN_DEPTHS.sort()
DEEP_DEPTHS = [d for d in _depth_list('LAB_DEEP_DEPTHS', SEARCH.get('deep_depths', [16384,32768,65536])) if SCREEN_CAP < d <= DEEP_CAP]
CANDIDATE_FINALISTS = max(1, int(os.environ.get('LAB_CANDIDATE_FINALISTS', str(SEARCH.get('candidate_finalists', 2)))))
WINNER_TG_WEIGHT = float(os.environ.get('LAB_WINNER_TG_WEIGHT', '0.65'))
WINNER_TG_WEIGHT = min(1.0, max(0.0, WINNER_TG_WEIGHT))

FAST_MODE = os.environ.get('LAB_FAST_MODE', '0') == '1'
FAST_BUDGET_S = max(60, int(os.environ.get('LAB_CAL_BUDGET_S', '300')))
USABLE_MIN_PP = float(os.environ.get('LAB_USABLE_MIN_PP_TPS', '20'))
USABLE_MIN_TG = float(os.environ.get('LAB_USABLE_MIN_TG_TPS', '5'))
TARGET_TG = float(os.environ.get('LAB_TARGET_TG_TPS', '10'))
PROGRESS = os.environ.get('LAB_PROGRESS', '1') != '0'
PROCESS_STARTED = time.time()
try:
    FAST_DEADLINE_EPOCH = float(os.environ.get('LAB_CAL_DEADLINE_EPOCH','0') or 0)
except Exception:
    FAST_DEADLINE_EPOCH = 0.0

def _remaining_budget() -> float:
    if FAST_DEADLINE_EPOCH > 0: return max(0.0, FAST_DEADLINE_EPOCH-time.time())
    return max(0.0, FAST_BUDGET_S-(time.time()-PROCESS_STARTED))

def _fmt_s(x: float | int | None) -> str:
    if x is None: return 'unknown'
    x=max(0.0,float(x))
    if x < 90: return f'{x:.0f}s'
    return f'{x/60.0:.1f}m'

def _past_wall(meta: dict[str, Any]) -> float | None:
    # Prefer same model/kind/depth from any prior run on this exact hardware;
    # fall back to same model/kind. Historical evidence is only an ETA hint.
    vals=[]; vals2=[]
    for r in RESULT_INDEX.values():
        w=r.get('wall_s')
        if not isinstance(w,(int,float)) or w <= 0 or r.get('status') not in ('ok','clock-throttled'): continue
        if r.get('model')==meta.get('model') and r.get('kind')==meta.get('kind'):
            vals2.append(float(w))
            if int(r.get('depth') or 0)==int(meta.get('depth') or 0): vals.append(float(w))
    xs=vals or vals2
    if not xs: return None
    xs=sorted(xs); return xs[len(xs)//2]

def _estimate_one(meta: dict[str, Any], timeout_s: int) -> float:
    old=_past_wall(meta)
    if old is not None: return min(float(timeout_s), max(1.0,old))
    kind=str(meta.get('kind',''))
    depth=max(1,int(meta.get('depth') or 1))
    if kind in ('pp','pp_interval'): return min(float(timeout_s), 3.0 + depth/35.0)
    if kind=='tg': return min(float(timeout_s), 3.0 + max(16,TG)/8.0)
    return min(float(timeout_s),10.0)

def _detail(meta: dict[str, Any]) -> str:
    bits=[f"stage={meta.get('phase','?')}",f"pass={meta.get('kind','?')}"]
    for k,label in [('model','model'),('build','build'),('policy_name','policy'),('threads','threads'),('depth','depth'),('kv','kv'),('batch','batch'),('ubatch','ubatch')]:
        v=meta.get(k)
        if v not in (None,''): bits.append(f'{label}={v}')
    return ' '.join(bits)

def _about(meta: dict[str, Any], timeout_s: int, note: str='') -> float:
    eta=_estimate_one(meta,timeout_s)
    if PROGRESS:
        print(f"ABOUT TO RUN: {_detail(meta)} ETA~{_fmt_s(eta)} timeout={_fmt_s(timeout_s)}{(' '+note) if note else ''}",flush=True)
    return eta

def _done(rec: dict[str, Any]) -> None:
    if not PROGRESS: return
    clk=(rec.get('clock') or {}).get('loaded_mhz')
    if clk is None:
        cs=rec.get('clock_samples') or []
        vals=[x.get('loaded_mhz') for x in cs if isinstance(x,dict) and isinstance(x.get('loaded_mhz'),(int,float))]
        clk=(sum(vals)/len(vals)) if vals else None
    c=f' clk={float(clk):.0f}MHz' if isinstance(clk,(int,float)) else ''
    print(f"DONE: {_detail(rec)} status={rec.get('status')} tps={speed(rec):.2f} wall={_fmt_s(rec.get('wall_s',0))}{c}",flush=True)

def _stage(title: str, *, models_: list[str] | None=None, configs: int | None=None, depths: list[int] | None=None,
           kvs: list[str] | None=None, passes: str='', eta_s: float | None=None, budget: bool=False) -> None:
    if not PROGRESS: return
    print('\n'+'='*78,flush=True)
    print('STAGE ABOUT TO RUN: '+title,flush=True)
    if models_: print('  models/quants: '+', '.join(models_),flush=True)
    if configs is not None: print(f'  configurations: {configs}',flush=True)
    if depths is not None: print('  depths: '+', '.join(str(x) for x in depths),flush=True)
    if kvs: print('  KV: '+', '.join(kvs),flush=True)
    if passes: print('  passes: '+passes,flush=True)
    if eta_s is not None: print('  estimated stage time: ~'+_fmt_s(eta_s),flush=True)
    if budget: print(f'  fast-calibration wall budget: {_fmt_s(FAST_BUDGET_S)} (setup/build/download time reported separately)',flush=True)
    print('='*78,flush=True)

def _experiment_tag() -> str:
    ver=(PKG/'VERSION').read_text().strip() if (PKG/'VERSION').exists() else 'unknown'
    commit=(ROOT/'llama-commit.txt').read_text().strip() if (ROOT/'llama-commit.txt').exists() else 'unknown'
    plan_blob=json.dumps(PLAN,sort_keys=True,separators=(',',':'))
    hw_key={k:HW.get(k) for k in ('cpu','numa','features')}
    hw_blob=json.dumps(hw_key,sort_keys=True,separators=(',',':'))
    shortlist_path=os.environ.get('LAB_PLATFORM_SHORTLIST_FILE','')
    shortlist_hash=''
    if shortlist_path and Path(shortlist_path).exists():
        shortlist_hash=hashlib.sha256(Path(shortlist_path).read_bytes()).hexdigest()[:12]
    run_policy={'quick_depth':QUICK_DEPTH,'screen_cap':SCREEN_CAP,'screen_depths':SCREEN_DEPTHS,'deep_cap':DEEP_CAP,'deep_depths':DEEP_DEPTHS,
                'candidate_finalists':CANDIDATE_FINALISTS,'winner_tg_weight':WINNER_TG_WEIGHT,'quick_kv':QUICK_KV,'thread_probe_kv':THREAD_PROBE_KV,'model_min_ctx':os.environ.get('LAB_MODEL_MIN_CTX','8192'),
                'model_scope':os.environ.get('LAB_MODELS','all'),'include_calibration':os.environ.get('LAB_INCLUDE_CALIBRATION','0'),
                'platform_shortlist_hash':shortlist_hash,'clock_sample_ms':os.environ.get('LAB_CLOCK_SAMPLE_MS','100'),'clock_max_cpus':os.environ.get('LAB_CLOCK_MAX_CPUS','32'),
                'clock_min_ratio':os.environ.get('LAB_CLOCK_MIN_RATIO','0.90'),'clock_min_mhz':os.environ.get('LAB_CLOCK_MIN_MHZ',''),
                'clock_enforce':os.environ.get('LAB_CLOCK_ENFORCE','1')}
    policy_blob=json.dumps(run_policy,sort_keys=True,separators=(',',':'))
    return hashlib.sha256((ver+'|'+commit+'|'+plan_blob+'|'+hw_blob+'|'+policy_blob).encode()).hexdigest()[:16]

RUN_TAG = _experiment_tag()
RUNFILE = OUT / f'bench-rank{RANK}.jsonl'
RESULT_INDEX: dict[str, dict[str, Any]] = {}

def rk(key: str) -> str:
    return f'{RUN_TAG}|{key}'


def _load_result_index() -> None:
    for p in OUT.glob('bench-rank*.jsonl'):
        for line in p.read_text(errors='ignore').splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = r.get('run_key')
            if isinstance(k, str) and k:
                RESULT_INDEX[k] = r


_load_result_index()


def _is_current_record(r: dict[str, Any]) -> bool:
    k=r.get('run_key')
    return isinstance(k,str) and k.startswith(RUN_TAG+'|')


def current_records() -> list[dict[str, Any]]:
    return [r for r in RESULT_INDEX.values() if _is_current_record(r)]


def _bootstrap_path() -> Path:
    return ROOT/'calibration'/'load-bootstrap.json'


def _bootstrap_fingerprint() -> str:
    ver=(PKG/'VERSION').read_text().strip() if (PKG/'VERSION').exists() else 'unknown'
    commit=(ROOT/'llama-commit.txt').read_text().strip() if (ROOT/'llama-commit.txt').exists() else 'unknown'
    hw={k:HW.get(k) for k in ('cpu','numa','features')}
    return hashlib.sha256((ver+'|'+commit+'|'+json.dumps(hw,sort_keys=True,separators=(',',':'))).encode()).hexdigest()[:20]


def _read_bootstrap() -> dict[str, Any] | None:
    p=_bootstrap_path()
    try: obj=json.load(open(p))
    except Exception: return None
    if not isinstance(obj,dict) or obj.get('fingerprint')!=_bootstrap_fingerprint(): return None
    bn=str(obj.get('build','')); pn=str(obj.get('policy_name','')); mn=str(obj.get('model',''))
    if not bn or not pn or not mn: return None
    if not (ROOT/'builds'/bn/'bin'/'llama-bench').exists(): return None
    return obj


def shard(k: str) -> bool:
    return int.from_bytes(hashlib.sha256(k.encode()).digest()[:8], 'little') % WORLD == RANK


def models() -> list[tuple[str, Path]]:
    return ensure_models(refresh=os.environ.get('LAB_REFRESH_MODELS', '0') == '1')


def model_record_map() -> dict[str, dict[str, Any]]:
    return {str(r.get('name')): r for r in registered_records() if r.get('name')}


def help_text(exe: Path) -> str:
    try:
        return subprocess.check_output([str(exe), '--help'], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return ''


def supported_policy(exe: Path, p: dict[str, Any]) -> bool:
    if not p.get('requires_cli'):
        return True
    h = help_text(exe)
    return '--numa-mirror' in h and re.search(r'mirror', h, re.I) is not None


def _platform_shortlist() -> set[tuple[str,str,int]] | None:
    p=os.environ.get('LAB_PLATFORM_SHORTLIST_FILE','').strip()
    if not p or not Path(p).exists(): return None
    try: obj=json.load(open(p))
    except Exception: return None
    xs=set()
    for x in obj.get('configs',[]) if isinstance(obj,dict) else []:
        try: xs.add((str(x['build']),str(x['policy_name']),int(x['threads'])))
        except Exception: pass
    return xs or None


PLATFORM_SHORTLIST=_platform_shortlist()

def config_allowed(build: str, policy_name: str, threads: int) -> bool:
    return PLATFORM_SHORTLIST is None or (str(build),str(policy_name),int(threads)) in PLATFORM_SHORTLIST

def policy_applicable(model_name: str, p: dict[str,Any]) -> bool:
    if not p.get('requires_mcdram_fit'): return True
    r=model_record_map().get(model_name,{})
    return r.get('mcdram_fit_class') in ('comfortable','nominal')


def maybe_drop_cache(p: dict[str, Any]) -> None:
    if not p.get('cold_pagecache') or os.environ.get('LAB_DROP_CACHES', '0') != '1':
        return
    subprocess.call(['sync'])
    cmd = ['sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches']
    if os.geteuid() != 0:
        cmd = ['sudo'] + cmd
    subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _append_record(rec: dict[str, Any]) -> dict[str, Any]:
    with RUNFILE.open('a') as f:
        f.write(json.dumps(rec, separators=(',', ':')) + '\n')
    k = rec.get('run_key')
    if isinstance(k, str) and k:
        RESULT_INDEX[k] = rec
    return rec


def _runtime_env() -> tuple[dict[str, str], dict[str, str], bool]:
    env = os.environ.copy()
    env['GGML_KNL_TRACE'] = env.get('GGML_KNL_TRACE', '0')
    runtime_seen = {k: env.get(k) for k in RUNTIME_ENV_VARS if env.get(k)}
    inherited = os.environ.get('LAB_INHERIT_RUNTIME_ENV', '0') == '1'
    if not inherited:
        for k in RUNTIME_ENV_VARS:
            env.pop(k, None)
    return env, runtime_seen, inherited


def _nominal_base_mhz() -> float | None:
    try:
        v=(HW.get('cpu',{}) or {}).get('nominal_base_mhz')
        return float(v) if v else None
    except Exception:
        return None


def _clock_adjust_status(status: str, clock: dict[str, Any]) -> str:
    if status == 'ok' and os.environ.get('LAB_CLOCK_ENFORCE','1') != '0' and clock.get('status') == 'throttled':
        return 'clock-throttled'
    return status


def _run_process(full: list[str], fo, fe, env: dict[str, str], timeout_s: int) -> tuple[int, bool, dict[str, Any]]:
    proc = subprocess.Popen(full, stdout=fo, stderr=fe, env=env, start_new_session=True)
    mon = ClockMonitor(pid=proc.pid, nominal_base_mhz=_nominal_base_mhz()).start()
    timed_out=False
    try:
        try:
            rc=proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            rc=124
    finally:
        clock=mon.stop()
    return rc, timed_out, clock


def run_one(cmd: list[str], meta: dict[str, Any], tmp: Path, run_key: str, timeout_s: int) -> dict[str, Any]:
    if run_key in RESULT_INDEX:
        print(f'RESUME {run_key}', flush=True)
        return RESULT_INDEX[run_key]
    maybe_drop_cache(meta.get('policy', {}))
    _about(meta, timeout_s)
    t0 = time.time()
    env, runtime_seen, inherited = _runtime_env()
    timefile = str(tmp) + '.time'
    stderr = str(tmp) + '.stderr'
    perffile = str(tmp) + '.perf'
    child = list(cmd)
    perf_used = False
    perf_status = (ROOT / 'perf-status.txt').read_text().strip() if (ROOT / 'perf-status.txt').exists() else ''
    if os.environ.get('LAB_PERF', '0') == '1' and perf_status == 'ok' and shutil.which('perf'):
        events = os.environ.get('LAB_PERF_EVENTS', 'cycles,instructions,cache-misses,branches,branch-misses,context-switches,cpu-migrations,page-faults')
        child = ['perf','stat','-x,','-o',perffile,'-e',events,'--'] + child
        perf_used = True
    full = ['/usr/bin/time','-v','-o',timefile] + child if Path('/usr/bin/time').exists() else child
    with open(tmp, 'w') as fo, open(stderr, 'w') as fe:
        rc, timed_out, clock = _run_process(full, fo, fe, env, timeout_s)
    raw: Any = []
    try:
        raw = json.load(open(tmp))
    except Exception:
        pass
    rss = None
    if Path(timefile).exists():
        m = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', Path(timefile).read_text(errors='ignore'))
        if m:
            rss = int(m.group(1)) * 1024
    perf_text = Path(perffile).read_text(errors='ignore') if perf_used and Path(perffile).exists() else ''
    status = 'timeout' if timed_out else ('ok' if rc == 0 else 'failed')
    status = _clock_adjust_status(status, clock)
    stderr_text = Path(stderr).read_text(errors='ignore') if Path(stderr).exists() else ''
    rec = {**meta, 'run_key': run_key, 'status': status, 'timeout_s': timeout_s, 'rc': rc, 'wall_s': time.time()-t0,
           'max_rss_bytes': rss, 'raw': raw, 'command': cmd, 'timestamp': time.time(), 'runtime_env_seen': runtime_seen,
           'runtime_env_inherited': inherited, 'perf_used': perf_used, 'perf_stat': perf_text, 'clock': clock, 'clock_status': clock.get('status'),
           'stderr_tail': '\n'.join(stderr_text.splitlines()[-80:])[-8000:], 'exit_signal': _exit_signal(rc)}
    rec=_append_record(rec)
    _done(rec)
    return rec


def record_skip(run_key: str, meta: dict[str, Any], reason: str) -> dict[str, Any]:
    if run_key in RESULT_INDEX:
        return RESULT_INDEX[run_key]
    rec = {**meta, 'run_key': run_key, 'status': 'skipped', 'skip_reason': reason, 'rc': 125, 'wall_s': 0.0,
           'max_rss_bytes': None, 'raw': [], 'command': [], 'timestamp': time.time(), 'perf_used': False}
    return _append_record(rec)


def _exit_signal(rc: Any) -> int | None:
    # subprocess normally returns -SIGNAL when directly signalled, but wrappers
    # such as /usr/bin/time commonly surface the shell convention 128+SIGNAL.
    if not isinstance(rc, int):
        return None
    if rc < 0:
        return -rc
    if 128 < rc < 192:
        return rc - 128
    return None


def speed(rec: dict[str, Any]) -> float:
    if isinstance(rec.get('tps'), (int, float)):
        return float(rec['tps'])
    raw = rec.get('raw', [])
    raw = [raw] if isinstance(raw, dict) else raw
    vals: list[float] = []
    for r in raw if isinstance(raw, list) else []:
        if isinstance(r, dict):
            for k in ('avg_ts','tokens_per_second','t_s'):
                if isinstance(r.get(k), (int,float)):
                    vals.append(float(r[k])); break
    return max(vals) if vals else 0.0


def build_dirs():
    for b in PLAN['builds']:
        d = ROOT / 'builds' / b['name']
        bench = d / 'bin/llama-bench'
        if bench.exists():
            yield b['name'], bench


def cmd_for(bench: Path, model: Path, threads: int, policy: dict[str, Any], args: list[str]) -> list[str]:
    # llama-bench has one math-thread control (-t). Unlike llama-cli/server it
    # does not expose -tb/--threads-batch. Passing -tb made every quick/batch
    # benchmark fail before inference on current llama.cpp.
    return list(policy.get('prefix', [])) + [str(bench), '-m', str(model), '-t', str(threads), '-r', '1', '-o', 'json'] + list(policy.get('llama', [])) + args


def quick() -> None:
    """Broad factorial screen. No request exceeds QUICK_DEPTH (default 2K)."""
    ms = models()
    if not ms:
        raise SystemExit('automatic model planner produced no usable models')
    q: list[list[Any]] = []
    for mn, model in ms:
        for bn, bench in build_dirs():
            for p in PLAN['numa_policies']:
                if not supported_policy(bench, p) or not policy_applicable(mn,p):
                    continue
                for th in [int(p.get('default_threads', PLAN['threads'][0]))]:
                    if not config_allowed(bn,p['name'],th):
                        continue
                    base = f'{mn}|{bn}|{p["name"]}|{th}'
                    if not shard('quick|' + base):
                        continue
                    tmp = OUT / f'.quick-{RANK}.json'
                    kpp = rk('quick|pp|' + base)
                    recp = run_one(cmd_for(bench, model, th, p, ['-p','512','-n','0']),
                                   {'phase':'quick','kind':'pp','model':mn,'build':bn,'policy_name':p['name'],'threads':th,'depth':512,'kv':None,'policy':p},
                                   tmp, kpp, QUICK_TIMEOUT)
                    if recp.get('status') != 'ok':
                        rect = record_skip(rk('quick|tg|' + base),
                                           {'phase':'quick','kind':'tg','model':mn,'build':bn,'policy_name':p['name'],'threads':th,'depth':QUICK_DEPTH,'kv':QUICK_KV,'policy':p},
                                           'quick-pp-failed-or-timeout')
                    else:
                        ktg = rk('quick|tg|' + base)
                        rect = run_one(cmd_for(bench, model, th, p, ['-p','0','-n',str(TG),'-d',str(QUICK_DEPTH),'-ctk',QUICK_KV,'-ctv',QUICK_KV]),
                                       {'phase':'quick','kind':'tg','model':mn,'build':bn,'policy_name':p['name'],'threads':th,'depth':QUICK_DEPTH,'kv':QUICK_KV,'policy':p},
                                       tmp, ktg, QUICK_TIMEOUT)
                    q.append([mn,bn,p['name'],th,speed(recp),speed(rect),recp.get('status'),rect.get('status'),
                              (recp.get('clock') or {}).get('loaded_mhz'),(rect.get('clock') or {}).get('loaded_mhz'),
                              recp.get('clock_status'),rect.get('clock_status')])
    qpath=OUT / f'quick-summary-{RUN_TAG}-rank{RANK}.json'
    with open(qpath, 'w') as f:
        json.dump(q, f, indent=2)
    ptr=OUT/f'latest-quick-rank{RANK}.json'
    ptr.write_text(json.dumps({'run_tag':RUN_TAG,'path':str(qpath),'package_version':(PKG/'VERSION').read_text().strip(),'timestamp':time.time()},indent=2)+'\n')



def _fast_platform_pairs() -> list[tuple[str,str]]:
    """Small causal platform set for the <=5 minute default calibration.

    This deliberately answers one question per comparison instead of crossing
    every build with every NUMA policy.  `calibrate --full` retains the legacy
    factorial map.
    """
    cls=str((HW.get('cpu',{}) or {}).get('class',''))
    bnames={x['name'] for x in PLAN.get('builds',[])}
    pnames={x['name'] for x in PLAN.get('numa_policies',[])}
    pairs: list[tuple[str,str]]=[]
    def add(b: str,p: str) -> None:
        if b in bnames and p in pnames and (b,p) not in pairs: pairs.append((b,p))
    if cls=='knl':
        # Same KNL backend DDR->HBM isolates memory-tier gain. Generic HBM->KNL
        # base isolates ISA/backend gain. Base->combo isolates our kernel patch.
        # Preferred memory is the one explicit spill/tiering control.
        add('knl-base-norepack','knl-ddr-strict')
        add('knl-base-norepack','knl-mcdram-strict')
        add('knl-generic-avx2-norepack','knl-mcdram-strict')
        add('knl-combo-norepack','knl-mcdram-strict')
        add('knl-combo-norepack','knl-mcdram-preferred')
    else:
        # Repack, explicit ISA, then one whole-machine NUMA policy. Keep this
        # intentionally small on X399/EPYC/Xeon.
        add('native-norepack','disabled-no-mmap')
        add('native','disabled-no-mmap')
        add('isa-avx2','disabled-no-mmap')
        if 'interleave' in pnames: add('native-norepack','interleave')
        elif 'distribute-no-mmap' in pnames: add('native-norepack','distribute-no-mmap')
    return pairs


def fast_plan() -> None:
    pairs=_fast_platform_pairs()
    cls=str((HW.get('cpu',{}) or {}).get('class','cpu'))
    builds=[]
    for b,_ in pairs:
        if b not in builds: builds.append(b)
    counts=[int(x) for x in (PLAN.get('thread_probe') or {}).get('counts',[])]
    # These are intentionally conservative first-run estimates. Per-pass ETAs
    # become data-driven as soon as this machine has completed rows.
    stage_est=[('clock/preflight',15),('bootstrap',15),('thread probe',max(15,8*len(counts))),('KV capability',20),
               ('prefix reuse',20),('lean platform map',max(30,12*len(pairs))),('3B 512->2K promise check',90)]
    total=sum(x[1] for x in stage_est)
    _stage('FAST CALIBRATION OVERVIEW',models_=['granite4-350m--Q4_K_M','granite4.1-3b--Q4_K_M'],configs=len(pairs),
           depths=[512,2048],kvs=['F16 baseline; Q8/Q4 capability only'],passes='PP + TG; one repetition; no batch factorial; no agent diagnostic',eta_s=min(total,FAST_BUDGET_S),budget=True)
    print('  minimal builds: '+', '.join(builds),flush=True)
    print('  platform comparisons:',flush=True)
    for i,(b,p) in enumerate(pairs,1): print(f'    {i}. build={b} policy={p}',flush=True)
    print('  thread probe: '+(', '.join(map(str,counts)) if counts else 'physical-core default only'),flush=True)
    print(f'  promotion gate at 2K: PP>={USABLE_MIN_PP:g} tok/s AND TG>={USABLE_MIN_TG:g} tok/s; responsive target TG>={TARGET_TG:g}',flush=True)
    print('  deeper 8K/16K+ work is NOT part of the default five-minute pass; it is recommended only for a promoted candidate.',flush=True)
    print('  legacy exhaustive behavior: ./cpu-llama-lab.sh calibrate --full',flush=True)


def quick_lite() -> None:
    """Tiny causal platform map used by default calibration."""
    ms=models()
    if not ms: raise SystemExit('automatic model planner produced no usable models')
    mn,model=ms[0]
    bds=dict(build_dirs()); polmap={p['name']:p for p in PLAN.get('numa_policies',[])}
    pairs=[]
    for bn,pn in _fast_platform_pairs():
        bench=bds.get(bn); pol=polmap.get(pn)
        if not bench or not pol or not supported_policy(bench,pol) or not policy_applicable(mn,pol): continue
        th=int(pol.get('default_threads',PLAN.get('hardware_summary',{}).get('phys_cores') or 1))
        pairs.append((bn,pn,th,bench,pol))
    if not pairs: raise SystemExit('lean platform map found no runnable build/policy pairs')
    _stage('LEAN PLATFORM MAP',models_=[mn],configs=len(pairs),depths=[512,QUICK_DEPTH],kvs=[QUICK_KV],
           passes='PP512 + TG%d at context %d for each causal config' % (TG,QUICK_DEPTH),eta_s=sum(2*_estimate_one({'model':mn,'kind':'tg','depth':QUICK_DEPTH},QUICK_TIMEOUT) for _ in pairs))
    for i,(bn,pn,th,_,_) in enumerate(pairs,1):
        print(f'  PLAN {i}/{len(pairs)}: model={mn} build={bn} policy={pn} threads={th} PP512 then TG{TG}@{QUICK_DEPTH} KV={QUICK_KV}',flush=True)
    q=[]; tmp=OUT/f'.quick-lite-{RANK}.json'
    for idx,(bn,pn,th,bench,pol) in enumerate(pairs,1):
        pair_eta=2*_estimate_one({'model':mn,'kind':'tg','depth':QUICK_DEPTH},QUICK_TIMEOUT)
        reserve=float(os.environ.get('LAB_FAST_RESERVE_SCALE_S','120'))
        if FAST_MODE and idx>4 and _remaining_budget() < reserve+pair_eta:
            print(f'BUDGET PRUNE: skipping optional platform config build={bn} policy={pn}; remaining={_fmt_s(_remaining_budget())}, reserving ~{_fmt_s(reserve)} for 3B promise check.',flush=True)
            continue
        base=f'{mn}|{bn}|{pn}|{th}'
        recp=run_one(cmd_for(bench,model,th,pol,['-p','512','-n','0']),
                     {'phase':'quick','kind':'pp','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':512,'kv':None,'policy':pol},tmp,rk('quick|pp|'+base),QUICK_TIMEOUT)
        if recp.get('status')=='ok':
            rect=run_one(cmd_for(bench,model,th,pol,['-p','0','-n',str(TG),'-d',str(QUICK_DEPTH),'-ctk',QUICK_KV,'-ctv',QUICK_KV]),
                         {'phase':'quick','kind':'tg','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':QUICK_DEPTH,'kv':QUICK_KV,'policy':pol},tmp,rk('quick|tg|'+base),QUICK_TIMEOUT)
        else:
            rect=record_skip(rk('quick|tg|'+base),{'phase':'quick','kind':'tg','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':QUICK_DEPTH,'kv':QUICK_KV,'policy':pol},'quick-pp-failed-or-timeout')
        q.append([mn,bn,pn,th,speed(recp),speed(rect),recp.get('status'),rect.get('status'),
                  (recp.get('clock') or {}).get('loaded_mhz'),(rect.get('clock') or {}).get('loaded_mhz'),recp.get('clock_status'),rect.get('clock_status')])
    qpath=OUT/f'quick-summary-{RUN_TAG}-rank{RANK}.json'; qpath.write_text(json.dumps(q,indent=2)+'\n')
    (OUT/f'latest-quick-rank{RANK}.json').write_text(json.dumps({'run_tag':RUN_TAG,'path':str(qpath),'package_version':(PKG/'VERSION').read_text().strip(),'timestamp':time.time(),'mode':'lean'},indent=2)+'\n')


def _fast_scale_config(mn: str) -> tuple[str,str,int,dict[str,Any]]:
    p=ROOT/'calibration'/'platform-shortlist.json'
    try: cfgs=json.load(open(p)).get('configs',[])
    except Exception: cfgs=[]
    recmap=model_record_map(); mr=recmap.get(mn,{})
    polmap={x['name']:x for x in PLAN.get('numa_policies',[])}
    runnable=[]
    for x in cfgs:
        bn=str(x.get('build','')); pn=str(x.get('policy_name','')); pol=polmap.get(pn)
        if not pol or not (ROOT/'builds'/bn/'bin'/'llama-server').exists(): continue
        if pol.get('requires_mcdram_fit') and mr.get('mcdram_fit_class') not in ('comfortable','nominal'): continue
        runnable.append(x)
    if not runnable: raise SystemExit('fast-scale found no runnable shortlisted platform config')
    runnable.sort(key=lambda x:(float(x.get('score') or 0),float(x.get('tg_tps') or 0),float(x.get('pp_tps') or 0)),reverse=True)
    x=runnable[0]; pn=str(x['policy_name']); return str(x['build']),pn,int(x['threads']),polmap[pn]


def fast_scale() -> None:
    """One representative 3B cached-prefix check through 2K; no factorials."""
    ms=models()
    if not ms: raise SystemExit('fast-scale requires the representative model')
    mn,model=next(((n,p) for n,p in ms if str(n).startswith('granite4.1-3b--')),ms[-1])
    bn,pn,th,pol=_fast_scale_config(mn); bdir=ROOT/'builds'/bn
    kv='f16'
    try:
        kp=json.load(open(ROOT/'calibration'/'kv-probe.json'))
        ok=[str(x.get('kv')) for x in kp.get('rows',[]) if x.get('status')=='ok']
        if 'f16' not in ok and ok: kv=ok[0]
    except Exception: pass
    depths=[512,2048]; b=int(os.environ.get('LAB_FAST_BATCH','1024')); ub=int(os.environ.get('LAB_FAST_UBATCH','512'))
    # Estimate from any prior 3B PP/TG evidence and print everything before start.
    pp_eta=_estimate_one({'phase':'fast-scale','kind':'pp_interval','model':mn,'depth':2048},PP_INTERVAL_TIMEOUT)
    tg_eta=_estimate_one({'phase':'fast-scale','kind':'tg','model':mn,'depth':2048},TG_ENDPOINT_TIMEOUT)
    eta=min(pp_eta+2*tg_eta+20,_remaining_budget() or pp_eta+2*tg_eta+20)
    _stage('REPRESENTATIVE 3B PROMISE CHECK',models_=[mn],configs=1,depths=depths,kvs=[kv],
           passes=f'persistent-server cached-prefix PP + TG{TG}; fixed batch={b}/{ub}; NO batch tuning',eta_s=eta,budget=True)
    print(f'  PLAN: build={bn} policy={pn} threads={th} depths={depths} kv={kv} batch={b} ubatch={ub}',flush=True)
    remaining=_remaining_budget()
    print(f'  remaining fast-calibration budget before stage: {_fmt_s(remaining)}',flush=True)
    required=max(45.0,min(pp_eta+2*tg_eta+20,180.0))
    if FAST_MODE and FAST_DEADLINE_EPOCH>0 and remaining < required:
        obj={'version':1,'model':mn,'build':bn,'policy_name':pn,'threads':th,'kv':kv,'depth':2048,'promoted':False,'skipped_for_budget':True,'remaining_budget_s':remaining,'estimated_required_s':required,'timestamp':time.time()}
        q=ROOT/'calibration'/'promise.json'; q.parent.mkdir(parents=True,exist_ok=True); q.write_text(json.dumps(obj,indent=2)+'\n')
        print(f'BUDGET STOP: representative 3B 2K check projected ~{_fmt_s(required)} but only {_fmt_s(remaining)} remains. No deeper work launched.',flush=True)
        print('  evidence: '+str(q),flush=True)
        return
    _progressive_branch('fast-scale',mn,model,bn,bdir,pn,pol,th,b,ub,kv,depths,reps=1)
    pp=RESULT_INDEX.get(_phase_key('fast-scale','pp',mn,bn,pn,th,kv,2048),{})
    tg=RESULT_INDEX.get(_phase_key('fast-scale','tg',mn,bn,pn,th,kv,2048),{})
    ppt=speed(pp); tgt=speed(tg); promoted=pp.get('status')=='ok' and tg.get('status')=='ok' and ppt>=USABLE_MIN_PP and tgt>=USABLE_MIN_TG
    responsive=promoted and tgt>=TARGET_TG
    est8=(8192/max(ppt,1e-9))+(TG/max(tgt,1e-9))+20 if promoted else None
    est16=(16384/max(ppt,1e-9))+(TG/max(tgt,1e-9))+20 if promoted else None
    obj={'version':1,'model':mn,'build':bn,'policy_name':pn,'threads':th,'kv':kv,'batch':b,'ubatch':ub,'depth':2048,
         'pp_tps':ppt,'tg_tps':tgt,'usable_min_pp_tps':USABLE_MIN_PP,'usable_min_tg_tps':USABLE_MIN_TG,'responsive_target_tg_tps':TARGET_TG,
         'promoted':promoted,'responsive_at_2k':responsive,'estimated_8k_s':est8,'estimated_16k_s':est16,'timestamp':time.time()}
    q=ROOT/'calibration'/'promise.json'; q.parent.mkdir(parents=True,exist_ok=True); q.write_text(json.dumps(obj,indent=2)+'\n')
    print('\nPROMISE DECISION:',flush=True)
    print(f'  model={mn} PP@2K={ppt:.2f} TG@2K={tgt:.2f} promoted={promoted} responsive_target_met={responsive}',flush=True)
    if promoted:
        print(f'  projected standalone 8K characterization: ~{_fmt_s(est8)}; 16K: ~{_fmt_s(est16)}',flush=True)
        print('  RESULT: worth deeper 8K work. Default calibration stops here by design.',flush=True)
    else:
        print('  RESULT: below usability floor; do not spend 8K/16K time on this calibration branch.',flush=True)
    print('  evidence: '+str(q),flush=True)

def thread_probe() -> None:
    """Cheap one-model SMT/thread scaling probe; never crossed with the full matrix."""
    ms=models(); bds=dict(build_dirs()); pols=PLAN.get('numa_policies',[]); boot=bootstrap_probe()
    if not ms or not bds: raise SystemExit('thread-probe requires one configured model and built llama-bench')
    mlookup=dict(ms); mn=str(boot.get('model','')); model=mlookup.get(mn)
    if model is None:
        recmap=model_record_map(); mn,model=min(ms,key=lambda x:int(recmap.get(x[0],{}).get('size_bytes') or x[1].stat().st_size))
    bn=str(boot.get('build','')); bn=bn if bn in bds else next(iter(bds)); bench=bds[bn]
    pol=next((p for p in pols if p.get('name')==boot.get('policy_name') and supported_policy(bench,p) and policy_applicable(mn,p)),None)
    if pol is None: pol=next((p for p in pols if supported_policy(bench,p) and policy_applicable(mn,p)),None)
    if pol is None: raise SystemExit('thread-probe found no applicable policy')
    counts=[int(x) for x in (PLAN.get('thread_probe') or {}).get('counts',PLAN.get('threads',[1]))]
    default=int((PLAN.get('thread_probe') or {}).get('default') or PLAN.get('hardware_summary',{}).get('phys_cores') or 1)
    rows=[]; tmp=OUT/f'.thread-probe-{RANK}.json'
    for th in counts:
        base=f'{mn}|{bn}|{pol["name"]}|{th}'
        pp=run_one(cmd_for(bench,model,th,pol,['-p','512','-n','0']),
                   {'phase':'thread-probe','kind':'pp','model':mn,'build':bn,'policy_name':pol['name'],'threads':th,'depth':512,'kv':None,'policy':pol},tmp,rk('thread-probe|pp|'+base),QUICK_TIMEOUT)
        if pp.get('status')=='ok':
            tg=run_one(cmd_for(bench,model,th,pol,['-p','0','-n',str(TG),'-d',str(QUICK_DEPTH),'-ctk',THREAD_PROBE_KV,'-ctv',THREAD_PROBE_KV]),
                       {'phase':'thread-probe','kind':'tg','model':mn,'build':bn,'policy_name':pol['name'],'threads':th,'depth':QUICK_DEPTH,'kv':THREAD_PROBE_KV,'policy':pol},tmp,rk('thread-probe|tg|'+base),QUICK_TIMEOUT)
        else:
            tg=record_skip(rk('thread-probe|tg|'+base),{'phase':'thread-probe','kind':'tg','model':mn,'build':bn,'policy_name':pol['name'],'threads':th,'depth':QUICK_DEPTH,'kv':THREAD_PROBE_KV,'policy':pol},'thread-probe-pp-failed')
        rows.append({'threads':th,'pp_tps':speed(pp),'tg_tps':speed(tg),'pp_status':pp.get('status'),'tg_status':tg.get('status'),
                     'pp_clock_mhz':(pp.get('clock') or {}).get('loaded_mhz'),'tg_clock_mhz':(tg.get('clock') or {}).get('loaded_mhz'),
                     'pp_clock_status':pp.get('clock_status'),'tg_clock_status':tg.get('clock_status')})
    good=[x for x in rows if x['pp_status']=='ok' and x['tg_status']=='ok' and x['pp_tps']>0 and x['tg_tps']>0]
    d=next((x for x in good if x['threads']==default),None)
    best=None
    if good:
        if d:
            for x in good:
                x['relative_score']=(x['pp_tps']/d['pp_tps'])**0.35*(x['tg_tps']/d['tg_tps'])**0.65
            best=max(good,key=lambda x:x['relative_score'])
        else:
            maxpp=max(x['pp_tps'] for x in good);maxtg=max(x['tg_tps'] for x in good)
            for x in good:x['relative_score']=(x['pp_tps']/maxpp)**0.35*(x['tg_tps']/maxtg)**0.65
            best=max(good,key=lambda x:x['relative_score'])
    threshold=float(os.environ.get('LAB_THREAD_SURPRISE_THRESHOLD','0.10'))
    surprise=bool(best and d and best['threads']!=default and best.get('relative_score',0)>=1.0+threshold)
    obj={'version':1,'model':mn,'build':bn,'policy_name':pol['name'],'physical_core_default':default,'counts':counts,'rows':rows,
         'best_threads':best['threads'] if best else default,'best_relative_score':best.get('relative_score') if best else None,'surprise':surprise,
         'note':'Informational one-model thread/SMT probe only; main sweeps still use all physical cores available to each NUMA locality.'}
    dpath=ROOT/'calibration'/'thread-probe.json';dpath.parent.mkdir(parents=True,exist_ok=True);dpath.write_text(json.dumps(obj,indent=2)+'\n')
    print(dpath)
    for x in rows:
        pc=x.get('pp_clock_mhz'); tc=x.get('tg_clock_mhz')
        clk=f" clk={pc:.0f}/{tc:.0f}MHz" if isinstance(pc,(int,float)) and isinstance(tc,(int,float)) else ''
        print(f"  t={x['threads']:4d} pp={x['pp_tps']:8.2f} tg={x['tg_tps']:8.2f} rel={x.get('relative_score',0):6.3f} status={x['pp_status']}/{x['tg_status']}{clk}")
    if surprise: print(f"THREAD SURPRISE: {best['threads']} threads beat physical-core default {default} by >= {threshold*100:.0f}% composite; recorded only, not propagated automatically.")



def kv_probe() -> None:
    """Cheap decode-KV capability probe on the known-good bootstrap path.

    This is deliberately separate from thread/platform scaling.  Quantized KV is
    ISA/kernel sensitive (notably on KNL), so a SIGILL must not make a healthy
    CPU/build look slow.  F16 is the neutral baseline; q8_0/q4_0 are diagnostics
    and remain eligible later only when they actually execute.
    """
    ms=models(); bds=dict(build_dirs()); pols=PLAN.get('numa_policies',[]); boot=bootstrap_probe()
    if not ms or not bds: raise SystemExit('kv-probe requires configured model and built llama-bench')
    mlookup=dict(ms); mn=str(boot.get('model','')); model=mlookup.get(mn)
    if model is None:
        recmap=model_record_map(); mn,model=min(ms,key=lambda x:int(recmap.get(x[0],{}).get('size_bytes') or x[1].stat().st_size))
    bn=str(boot.get('build','')); bn=bn if bn in bds else next(iter(bds)); bench=bds[bn]
    pol=next((p for p in pols if p.get('name')==boot.get('policy_name') and supported_policy(bench,p) and policy_applicable(mn,p)),None)
    if pol is None: pol=next((p for p in pols if supported_policy(bench,p) and policy_applicable(mn,p)),None)
    if pol is None: raise SystemExit('kv-probe found no applicable policy')
    th=int(pol.get('default_threads', PLAN.get('hardware_summary',{}).get('phys_cores') or 1))
    kvs=[]
    for x in ['f16','q8_0','q4_0'] + [str(x) for x in PLAN.get('kv_types',[])]:
        if x not in kvs: kvs.append(x)
    rows=[]; tmp=OUT/f'.kv-probe-{RANK}.json'
    for kv in kvs:
        base=f'{mn}|{bn}|{pol["name"]}|{th}|{kv}'
        rec=run_one(cmd_for(bench,model,th,pol,['-p','0','-n',str(min(TG,64)),'-d',str(min(QUICK_DEPTH,2048)),'-ctk',kv,'-ctv',kv]),
                    {'phase':'kv-probe','kind':'tg','model':mn,'build':bn,'policy_name':pol['name'],'threads':th,'depth':min(QUICK_DEPTH,2048),'kv':kv,'policy':pol},
                    tmp,rk('kv-probe|tg|'+base),QUICK_TIMEOUT)
        rows.append({'kv':kv,'status':rec.get('status'),'rc':rec.get('rc'),'signal':rec.get('exit_signal'),'tg_tps':speed(rec),
                     'clock_mhz':(rec.get('clock') or {}).get('loaded_mhz'),'clock_status':rec.get('clock_status'),
                     'command':rec.get('command',[]),'stderr_tail':rec.get('stderr_tail','')})
    obj={'version':1,'model':mn,'build':bn,'policy_name':pol['name'],'threads':th,'rows':rows,'timestamp':time.time(),
         'note':'Diagnostic only. F16 is the neutral quick/thread baseline; quantized KV remains a separate finalist ablation when executable.'}
    dpath=ROOT/'calibration'/'kv-probe.json'; dpath.parent.mkdir(parents=True,exist_ok=True); dpath.write_text(json.dumps(obj,indent=2)+'\n')
    print(dpath)
    for x in rows:
        sig=f" SIG{x['signal']}" if x.get('signal') else ''
        cm=x.get('clock_mhz'); clk=f" clk={cm:.0f}MHz" if isinstance(cm,(int,float)) else ''
        print(f"  kv={x['kv']:5s} tg={x['tg_tps']:8.2f} status={x['status']} rc={x['rc']}{sig}{clk}")

def finalists() -> dict[str, list[list[Any]]]:
    """Top shallow settings per *weight-quant candidate*, before KV/8K screening."""
    rows: list[list[Any]] = []
    for p in OUT.glob(f'quick-summary-{RUN_TAG}-rank*.json'):
        try:
            rows += json.load(open(p))
        except Exception:
            pass
    by: dict[str, list[list[Any]]] = {}
    for r in rows:
        if len(r) < 6:
            continue
        mn,bn,pol,th,pp,tg = r[:6]
        if len(r) >= 8 and (r[6] != 'ok' or r[7] != 'ok'):
            continue
        if float(pp) < MIN_QUICK_PP or float(tg) < MIN_QUICK_TG:
            continue
        by.setdefault(mn, []).append(r)
    out: dict[str, list[list[Any]]] = {}
    for mn, rs in by.items():
        maxpp = max([float(x[4]) for x in rs] or [1.0])
        maxtg = max([float(x[5]) for x in rs] or [1.0])
        rs = sorted(rs, key=lambda x:(float(x[4])/maxpp if maxpp else 0)+(float(x[5])/maxtg if maxtg else 0), reverse=True)
        out[mn] = rs[:CANDIDATE_FINALISTS]
    json.dump(out, open(OUT / f'finalists-{RUN_TAG}.json','w'), indent=2)
    return out


def tune_batches() -> dict[str, list[int]]:
    fs = finalists()
    ms = dict(models())
    bds = dict(build_dirs())
    pols = {p['name']:p for p in PLAN['numa_policies']}
    best: dict[str, list[int]] = {}
    for mn, arr in fs.items():
        model = ms.get(mn)
        if not model:
            continue
        for r in arr:
            _,bn,pn,th,_,_ = r[:6]
            bench = bds.get(bn); pol = pols[pn]
            if not bench or not supported_policy(bench, pol) or not policy_applicable(mn,pol) or not config_allowed(bn,pn,th):
                continue
            vals: list[tuple[float,int,int]] = []
            for b,ub in BATCHES:
                key = rk(f'batch|{mn}|{bn}|{pn}|{th}|{b}|{ub}')
                tmp = OUT / f'.batch-{RANK}.json'
                rec = run_one(cmd_for(bench,model,th,pol,['-p','1024','-n','0','-b',str(b),'-ub',str(ub)]),
                              {'phase':'batch-tune','kind':'pp','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':1024,'batch':b,'ubatch':ub,'kv':None,'policy':pol},
                              tmp, key, BATCH_TIMEOUT)
                if rec.get('status') == 'ok' and speed(rec) > 0:
                    vals.append((speed(rec),b,ub))
            vals.sort(reverse=True)
            best[f'{mn}|{bn}|{pn}|{th}'] = list(vals[0][1:]) if vals else [2048,512]
    json.dump(best, open(OUT / f'batches-{RUN_TAG}-rank{RANK}.json','w'), indent=2)
    return best


def load_batches() -> dict[str, list[int]]:
    shared = os.environ.get('LAB_BATCH_FILE')
    if shared and Path(shared).exists():
        return json.load(open(shared))
    p = OUT / f'batches-{RUN_TAG}-rank{RANK}.json'
    return json.load(open(p)) if p.exists() else tune_batches()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, separators=(',', ':')).encode()
    req = urllib.request.Request(url, data=data, headers={'content-type':'application/json'} if data is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _server_cmd(server: Path, model: Path, threads: int, policy: dict[str, Any], b: int, ub: int, kv: str, ctx: int, port: int) -> list[str]:
    args = [str(server), '-m', str(model), '-c', str(ctx), '-t', str(threads), '-tb', str(threads), '-b', str(b), '-ub', str(ub),
            '-ctk', kv, '-ctv', kv, '--parallel', '1', '--cache-prompt', '--no-context-shift', '--no-webui', '--host', '127.0.0.1', '--port', str(port)]
    return list(policy.get('prefix', [])) + args + list(policy.get('llama', []))


def _start_server(server: Path, model: Path, threads: int, policy: dict[str, Any], b: int, ub: int, kv: str, max_depth: int, tag: str):
    port = _free_port()
    ctx = max_depth + TG + 32
    cmd = _server_cmd(server, model, threads, policy, b, ub, kv, ctx, port)
    logp = OUT / f'.server-{re.sub(r"[^A-Za-z0-9_.-]+","_",tag)}.log'
    log = open(logp, 'a')
    env, _, _ = _runtime_env()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    timeout_s = PREFIX_SMOKE_SERVER_START_TIMEOUT if str(tag).startswith('prefix-smoke-') else SERVER_START_TIMEOUT
    deadline = time.time() + timeout_s
    ok = False; timed_out=False
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            _http_json(f'http://127.0.0.1:{port}/health', timeout=2)
            ok = True; break
        except Exception:
            time.sleep(0.5)
    if not ok:
        if proc.poll() is None:
            timed_out=True
            try: os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()
        rc=proc.returncode
        log.flush(); log.close()
        tail=''
        try: tail='\n'.join(logp.read_text(errors='ignore').splitlines()[-120:])[-12000:]
        except Exception: pass
        diag={'server_rc':rc,'server_signal':(-rc if isinstance(rc,int) and rc<0 else None),'startup_timeout':timed_out,
              'startup_timeout_s':timeout_s,'server_log':str(logp),'server_log_tail':tail,'server_command':cmd}
        return None, None, cmd, logp, diag
    return proc, log, cmd, port, {'server_rc':None,'server_signal':None,'startup_timeout':False,'server_log':str(logp),'server_command':cmd}


def _stop_server(proc, log) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait()
    if log:
        log.close()


def _completion_metrics(obj: dict[str, Any], requested_prompt_n: int | None = None) -> dict[str, Any]:
    """Normalize server timing/cache fields across llama-server revisions.

    Recent servers expose timings.cache_n/prompt_n, while the completion schema also
    exposes top-level tokens_cached/tokens_evaluated and prompt_progress cache data.
    Cache proof must not fail merely because one response path omits the timings copy.
    """
    t=obj.get('timings',{}) if isinstance(obj,dict) else {}
    t=t if isinstance(t,dict) else {}
    prog=obj.get('prompt_progress',{}) if isinstance(obj,dict) else {}
    prog=prog if isinstance(prog,dict) else {}
    def num(*xs, default=None):
        for x in xs:
            if isinstance(x,(int,float)) and not isinstance(x,bool): return x
        return default
    cache_n=num(t.get('cache_n'),obj.get('tokens_cached') if isinstance(obj,dict) else None,prog.get('cache'),default=-1)
    prompt_n=num(t.get('prompt_n'),default=None)
    if prompt_n is None:
        processed=num(prog.get('processed'),default=None)
        pcache=num(prog.get('cache'),default=None)
        if processed is not None and pcache is not None: prompt_n=max(0,int(processed)-int(pcache))
        elif requested_prompt_n is not None and cache_n is not None and int(cache_n)>=0: prompt_n=max(0,int(requested_prompt_n)-int(cache_n))
        else: prompt_n=0
    return {'cache_n':int(cache_n if cache_n is not None else -1),'prompt_n':int(prompt_n),
            'prompt_ms':float(num(t.get('prompt_ms'),default=0.0) or 0.0),
            'prompt_per_second':float(num(t.get('prompt_per_second'),default=0.0) or 0.0),
            'predicted_n':int(num(t.get('predicted_n'),obj.get('tokens_predicted') if isinstance(obj,dict) else None,default=0) or 0),
            'predicted_per_second':float(num(t.get('predicted_per_second'),default=0.0) or 0.0)}


def cache_reuse_ok(previous_depth: int, timings: dict[str, Any], tolerance: int = CACHE_TOL) -> bool:
    if previous_depth <= 0:
        return True
    try: cache_n=int(timings.get('cache_n',-1))
    except Exception: return False
    return cache_n >= max(0, previous_depth - tolerance)


def _seed_tokens(port: int) -> list[int]:
    obj = _http_json(f'http://127.0.0.1:{port}/tokenize', {'content':'The quick brown fox jumps over the lazy dog. ','add_special':False,'parse_special':False}, timeout=30)
    xs = obj.get('tokens', [])
    out = [int(x['id'] if isinstance(x,dict) else x) for x in xs]
    if not out:
        raise RuntimeError('server tokenizer returned no seed tokens')
    return out


def _completion(port: int, tokens: list[int], n_predict: int, timeout: int) -> dict[str, Any]:
    payload = {'prompt':tokens, 'n_predict':n_predict, 'cache_prompt':True, 'temperature':0.0, 'seed':1,
               'ignore_eos':True, 'n_keep':-1, 'stream':False, 'timings_per_token':False}
    return _http_json(f'http://127.0.0.1:{port}/completion', payload, timeout=timeout)


def _completion_measured(proc, port: int, tokens: list[int], n_predict: int, timeout: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return monitor_call(lambda: _completion(port,tokens,n_predict,timeout), pid=proc.pid if proc is not None else None, nominal_base_mhz=_nominal_base_mhz())


def _phase_key(phase: str, kind: str, mn: str, bn: str, pn: str, th: int, kv: str, depth: int) -> str:
    return rk(f'{phase}|{kind}|{mn}|{bn}|{pn}|{th}|{kv}|{depth}')


def _progressive_branch(phase: str, mn: str, model: Path, bn: str, build_dir: Path, pn: str, pol: dict[str, Any], th: int,
                        b: int, ub: int, kv: str, depths: list[int], *, initial_prefix: int = 0, reps: int = 1, allow_context_fallback: bool = True) -> None:
    """One persistent-server depth sweep. Every measured PP depth reuses the prior prefix."""
    server = build_dir / 'bin/llama-server'
    if not server.exists() or not supported_policy(server, pol):
        return
    depths = sorted(set(int(d) for d in depths if int(d) > initial_prefix)) if initial_prefix else sorted(set(int(d) for d in depths if int(d) > 0))
    if not depths:
        return
    proc = log = None; port = None; start_cmd: list[str] = []
    chosen_depths = list(depths)
    if PROGRESS:
        print(f"ABOUT TO START SERVER: stage={phase} model={mn} build={bn} policy={pn} threads={th} kv={kv} batch={b} ubatch={ub} max_depth={chosen_depths[-1]} ETA~10-30s",flush=True)
    while chosen_depths:
        maxd = chosen_depths[-1]
        tag = f'{phase}-{mn}-{bn}-{pn}-{th}-{kv}-{maxd}'
        proc, log, start_cmd, port_or_log, server_diag = _start_server(server, model, th, pol, b, ub, kv, maxd, tag)
        if proc is not None:
            port = int(port_or_log); break
        meta={'phase':phase,'kind':'server-context-fit','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':maxd,'kv':kv,'policy':pol,'batch':b,'ubatch':ub,**server_diag}
        rec=record_skip(_phase_key(phase,'server-fit',mn,bn,pn,th,kv,maxd), meta,
                    'server-start/context-allocation-failed')
        if not allow_context_fallback:
            chosen_depths=[]; break
        chosen_depths = chosen_depths[:-1]
    if proc is None or port is None or not chosen_depths:
        for d in depths:
            record_skip(_phase_key(phase,'pp',mn,bn,pn,th,kv,d), {'phase':phase,'kind':'pp_interval','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':d,'kv':kv,'policy':pol,'batch':b,'ubatch':ub}, 'no-context-tier-fit')
            record_skip(_phase_key(phase,'tg',mn,bn,pn,th,kv,d), {'phase':phase,'kind':'tg','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':d,'kv':kv,'policy':pol,'batch':b,'ubatch':ub}, 'no-context-tier-fit')
        return
    max_fit = chosen_depths[-1]
    for d in depths:
        if d > max_fit:
            record_skip(_phase_key(phase,'pp',mn,bn,pn,th,kv,d), {'phase':phase,'kind':'pp_interval','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':d,'kv':kv,'policy':pol,'batch':b,'ubatch':ub}, f'context>{max_fit}-did-not-fit')
            record_skip(_phase_key(phase,'tg',mn,bn,pn,th,kv,d), {'phase':phase,'kind':'tg','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':d,'kv':kv,'policy':pol,'batch':b,'ubatch':ub}, f'context>{max_fit}-did-not-fit')
    try:
        seed = _seed_tokens(port)
        full_prompt = (seed * ((max_fit + len(seed) - 1)//len(seed)))[:max_fit]
        completed_pp = [d for d in chosen_depths if _phase_key(phase,'pp',mn,bn,pn,th,kv,d) in RESULT_INDEX and RESULT_INDEX[_phase_key(phase,'pp',mn,bn,pn,th,kv,d)].get('status') == 'ok']
        prev = max([initial_prefix] + completed_pp)
        resume_ceiling = prev
        if prev:
            restored = _completion(port, full_prompt[:prev], 0, PP_INTERVAL_TIMEOUT)
            rt = _completion_metrics(restored, prev)
            # A cold server legitimately has cache_n=0 while reconstructing the
            # prefix. We care that the request succeeds; measured reuse is proved
            # on the next suffix request.
        tg_blocked = False
        for d in chosen_depths:
            ppkey = _phase_key(phase,'pp',mn,bn,pn,th,kv,d)
            tgkey = _phase_key(phase,'tg',mn,bn,pn,th,kv,d)
            if d <= prev and ppkey in RESULT_INDEX:
                pass
            elif ppkey not in RESULT_INDEX:
                ppmeta={'phase':phase,'kind':'pp_interval','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':d,'kv':kv,'batch':b,'ubatch':ub}
                _about(ppmeta, PP_INTERVAL_TIMEOUT, note=f'cached-prefix start={prev} suffix~{max(0,d-prev)} tokens')
                t0 = time.time()
                try:
                    obj, clock = _completion_measured(proc, port, full_prompt[:d], 0, PP_INTERVAL_TIMEOUT)
                    timings = _completion_metrics(obj, d)
                    ok = cache_reuse_ok(prev, timings)
                    pn_eval = int(timings.get('prompt_n', 0) or 0)
                    pms = float(timings.get('prompt_ms', 0.0) or 0.0)
                    tps = float(timings.get('prompt_per_second', 0.0) or (pn_eval*1000.0/pms if pms>0 else 0.0))
                    status = 'ok' if ok and pn_eval > 0 and tps > 0 else 'cache-reuse-failed'
                    status = _clock_adjust_status(status, clock)
                    rec = {'phase':phase,'kind':'pp_interval','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':d,'interval_start':prev,
                           'interval_tokens':pn_eval,'kv':kv,'policy':pol,'batch':b,'ubatch':ub,'run_key':ppkey,'status':status,'rc':0 if status=='ok' else 126,
                           'wall_s':time.time()-t0,'tps':tps,'prompt_ms':pms,'cache_n':timings.get('cache_n'),'prompt_n':pn_eval,'raw':obj,'command':start_cmd,'timestamp':time.time(),
                           'clock':clock,'clock_status':clock.get('status'),
                           'measurement':'persistent-server cached-prefix suffix PP'}
                    _append_record(rec)
                    _done(rec)
                    if status != 'ok':
                        for dd in chosen_depths:
                            if dd > d:
                                record_skip(_phase_key(phase,'pp',mn,bn,pn,th,kv,dd), {'phase':phase,'kind':'pp_interval','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':dd,'kv':kv,'policy':pol,'batch':b,'ubatch':ub}, 'cache-reuse-proof-failed')
                                record_skip(_phase_key(phase,'tg',mn,bn,pn,th,kv,dd), {'phase':phase,'kind':'tg','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':dd,'kv':kv,'policy':pol,'batch':b,'ubatch':ub}, 'cache-reuse-proof-failed')
                        break
                    prev = d
                except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
                    _append_record({'phase':phase,'kind':'pp_interval','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':d,'interval_start':prev,'kv':kv,'policy':pol,'batch':b,'ubatch':ub,
                                    'run_key':ppkey,'status':'timeout','rc':124,'wall_s':time.time()-t0,'tps':0.0,'raw':[],'command':start_cmd,'timestamp':time.time(),'error':repr(e)})
                    for dd in chosen_depths:
                        if dd >= d:
                            record_skip(_phase_key(phase,'tg',mn,bn,pn,th,kv,dd), {'phase':phase,'kind':'tg','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':dd,'kv':kv,'policy':pol,'batch':b,'ubatch':ub}, 'pp-interval-timeout')
                            if dd > d:
                                record_skip(_phase_key(phase,'pp',mn,bn,pn,th,kv,dd), {'phase':phase,'kind':'pp_interval','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':dd,'kv':kv,'policy':pol,'batch':b,'ubatch':ub}, 'lower-pp-interval-timeout')
                    break
            if tgkey in RESULT_INDEX:
                continue
            if tg_blocked:
                record_skip(tgkey, {'phase':phase,'kind':'tg','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':d,'kv':kv,'policy':pol,'batch':b,'ubatch':ub}, 'lower-depth-tg-timeout')
                continue
            tgmeta={'phase':phase,'kind':'tg','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':d,'kv':kv,'batch':b,'ubatch':ub}
            _about(tgmeta, TG_ENDPOINT_TIMEOUT, note=f'{TG} generated tokens; cached endpoint')
            samples: list[float] = []; pred_ns: list[int] = []; tg_objs: list[Any] = []; clocks: list[dict[str,Any]] = []
            t0 = time.time(); timed_out = False
            try:
                for _ in range(max(1,reps)):
                    obj, one_clock = _completion_measured(proc, port, full_prompt[:d], TG, TG_ENDPOINT_TIMEOUT)
                    clocks.append(one_clock)
                    timings = _completion_metrics(obj, d)
                    ps = float(timings.get('predicted_per_second', 0.0) or 0.0)
                    pn_gen = int(timings.get('predicted_n', 0) or 0)
                    if ps > 0: samples.append(ps)
                    pred_ns.append(pn_gen); tg_objs.append(obj)
            except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
                timed_out = True; tg_objs.append({'error':repr(e)})
            status = 'timeout' if timed_out else ('ok' if samples else 'failed')
            throttled_clock=next((x for x in clocks if x.get('status')=='throttled'),None)
            if throttled_clock is not None: status=_clock_adjust_status(status,throttled_clock)
            rec = {'phase':phase,'kind':'tg','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':d,'kv':kv,'policy':pol,'batch':b,'ubatch':ub,
                   'run_key':tgkey,'status':status,'rc':124 if timed_out else (0 if samples else 1),'wall_s':time.time()-t0,'tps':sum(samples)/len(samples) if samples else 0.0,
                   'tps_samples':samples,'predicted_n_samples':pred_ns,'raw':tg_objs,'command':start_cmd,'timestamp':time.time(),'measurement':'TG at cached-prefix endpoint',
                   'clock_samples':clocks,'clock_status':('throttled' if throttled_clock is not None else (clocks[-1].get('status') if clocks else 'unavailable'))}
            _append_record(rec)
            _done(rec)
            if timed_out:
                tg_blocked = True
            if resume_ceiling and d < resume_ceiling:
                try:
                    restored = _completion(port, full_prompt[:resume_ceiling], 0, PP_INTERVAL_TIMEOUT)
                    rt = _completion_metrics(restored, resume_ceiling)
                    if not cache_reuse_ok(d, rt):
                        raise RuntimeError(f'cache restore proof failed after TG backfill at {d}: {rt}')
                except Exception as e:
                    record_skip(rk(f'{phase}|resume-cache-restore|{mn}|{bn}|{pn}|{th}|{kv}|{resume_ceiling}'),
                                {'phase':phase,'kind':'resume-cache-restore','model':mn,'build':bn,'policy_name':pn,'threads':th,'depth':resume_ceiling,'kv':kv,'policy':pol,'batch':b,'ubatch':ub},
                                f'restore-failed:{type(e).__name__}:{e}')
                    break
    finally:
        _stop_server(proc, log)


def shallow() -> None:
    """Cached-prefix screening through 8K only; no candidate can exceed SCREEN_CAP."""
    fs = finalists(); batches = load_batches(); ms = dict(models()); pols = {p['name']:p for p in PLAN['numa_policies']}
    recmap = model_record_map()
    reps = int(SEARCH.get('screen_reps', 1))
    for mn, arr in fs.items():
        model = ms.get(mn)
        if not model:
            continue
        cap = min(SCREEN_CAP, int(recmap.get(mn, {}).get('max_planned_ctx') or SCREEN_CAP))
        depths = [d for d in SCREEN_DEPTHS if d <= cap]
        if SCREEN_CAP not in depths:
            continue
        for r in arr:
            _,bn,pn,th,_,_ = r[:6]
            build_dir = ROOT / 'builds' / bn; pol = pols[pn]
            if not policy_applicable(mn,pol) or not config_allowed(bn,pn,th):
                continue
            b,ub = batches.get(f'{mn}|{bn}|{pn}|{th}', [2048,512])
            for kv in PLAN['kv_types']:
                key = rk(f'shallowbranch|{mn}|{bn}|{pn}|{th}|{kv}')
                if not shard(key):
                    continue
                _progressive_branch('shallow', mn, model, bn, build_dir, pn, pol, int(th), int(b), int(ub), str(kv), depths, reps=reps)


def _candidate_family(mn: str, recmap: dict[str,dict[str,Any]]) -> tuple[str,str]:
    r = recmap.get(mn, {})
    return str(r.get('alias') or mn.split('--',1)[0]), str(r.get('quant') or (mn.split('--',1)[1] if '--' in mn else 'manual'))


def family_candidates() -> dict[str, list[dict[str, Any]]]:
    """Rank quant+runtime candidates per model family from cached-prefix 8K PP/TG.

    Multiple runtime branches of the same GGUF/quant collapse to their fastest
    branch.  Agent quality then decides whether that quant is acceptable; a
    failed fast quant can be promoted to a higher-fidelity fallback without
    repeating the factorial performance search.
    """
    recmap = model_record_map()
    branches: dict[tuple[Any,...], dict[str,dict[str,Any]]] = {}
    for r in RESULT_INDEX.values():
        if r.get('phase') != 'shallow' or int(r.get('depth') or 0) != SCREEN_CAP or r.get('status') != 'ok':
            continue
        if r.get('kind') not in ('pp_interval','tg'):
            continue
        key=(r.get('model'),r.get('build'),r.get('policy_name'),int(r.get('threads') or 0),r.get('kv'),int(r.get('batch') or 0),int(r.get('ubatch') or 0))
        branches.setdefault(key,{})[str(r.get('kind'))]=r
    byfam: dict[str,list[dict[str,Any]]] = {}
    for key,kinds in branches.items():
        if 'pp_interval' not in kinds or 'tg' not in kinds:
            continue
        mn,bn,pn,th,kv,b,ub=key
        pp=speed(kinds['pp_interval']); tg=speed(kinds['tg'])
        if pp <= 0 or tg <= 0:
            continue
        fam,quant=_candidate_family(str(mn),recmap); mr=recmap.get(str(mn),{})
        byfam.setdefault(fam,[]).append({'family':fam,'model':mn,'quant':quant,'build':bn,'policy_name':pn,'threads':th,'kv':kv,'batch':b,'ubatch':ub,
                                         'screen_depth':SCREEN_CAP,'pp_tps':pp,'tg_tps':tg,
                                         'pp_clock':kinds['pp_interval'].get('clock'),'tg_clock_samples':kinds['tg'].get('clock_samples',[]),
                                         'mcdram_fit_class':mr.get('mcdram_fit_class'),'mcdram_headroom_bytes':mr.get('mcdram_headroom_bytes',0),
                                         'selection_reason':mr.get('selection_reason','')})
    out: dict[str,list[dict[str,Any]]] = {}
    for fam,rows in sorted(byfam.items()):
        maxpp=max(float(x['pp_tps']) for x in rows); maxtg=max(float(x['tg_tps']) for x in rows)
        for x in rows:
            pn=max(float(x['pp_tps'])/maxpp,1e-12); tn=max(float(x['tg_tps'])/maxtg,1e-12)
            x['score']=(pn ** (1.0-WINNER_TG_WEIGHT)) * (tn ** WINNER_TG_WEIGHT)
            x['tg_weight']=WINNER_TG_WEIGHT
        # Quality is a property of the model/quant, not of thread placement. Keep
        # only the best runtime branch for each GGUF candidate.
        best_by_model: dict[str,dict[str,Any]]={}
        for x in rows:
            k=str(x['model'])
            if k not in best_by_model or (float(x['score']),float(x['tg_tps']),float(x['pp_tps'])) > (float(best_by_model[k]['score']),float(best_by_model[k]['tg_tps']),float(best_by_model[k]['pp_tps'])):
                best_by_model[k]=x
        out[fam]=sorted(best_by_model.values(),key=lambda x:(float(x['score']),float(x['tg_tps']),float(x['pp_tps'])),reverse=True)
    obj={'run_tag':RUN_TAG,'screen_cap':SCREEN_CAP,'deep_cap':DEEP_CAP,'winner_tg_weight':WINNER_TG_WEIGHT,'families':out,'timestamp':time.time()}
    json.dump(obj, open(OUT / f'family-candidates-{RUN_TAG}.json','w'), indent=2)
    mdir=ROOT/'models';mdir.mkdir(parents=True,exist_ok=True)
    json.dump(obj, open(mdir/'family-candidates.json','w'), indent=2)
    return out


def family_winners() -> list[dict[str, Any]]:
    """Use quality-approved winners when available; otherwise speed winners.

    Calibration and standalone sweep.py runs intentionally retain the speed-only
    fallback.  The shell `overnight`/`full-sweep` workflow inserts the agent
    quality gate between screen and deep stages, so production winners are
    quality-constrained.
    """
    qp=ROOT/'models'/'agent-winners.json'
    if qp.exists():
        try:
            q=json.load(open(qp))
            if q.get('run_tag')==RUN_TAG and isinstance(q.get('winners'),list):
                return [x for x in q['winners'] if isinstance(x,dict)]
        except Exception:
            pass
    fams=family_candidates(); winners=[]
    for fam,rows in sorted(fams.items()):
        if rows:winners.append(rows[0])
    out={'run_tag':RUN_TAG,'screen_cap':SCREEN_CAP,'deep_cap':DEEP_CAP,'winner_tg_weight':WINNER_TG_WEIGHT,'selection':'speed-only-no-agent-gate','winners':winners,'timestamp':time.time()}
    json.dump(out, open(OUT / f'family-winners-{RUN_TAG}.json','w'), indent=2)
    mdir=ROOT/'models'; mdir.mkdir(parents=True,exist_ok=True)
    json.dump(out, open(mdir/'winners.json','w'), indent=2)
    return winners

def deep() -> None:
    """Exactly one >8K cached-prefix sweep per model family, capped at 64K by default."""
    ws=family_winners(); ms=dict(models()); pols={p['name']:p for p in PLAN['numa_policies']}; recmap=model_record_map()
    json.dump({'run_tag':RUN_TAG,'selection':'quality-constrained' if (ROOT/'models'/'agent-winners.json').exists() else 'speed-only','winners':ws,'timestamp':time.time()}, open(ROOT/'models'/'winners.json','w'), indent=2)
    for w in ws:
        mn=str(w['model']); model=ms.get(mn)
        if not model:
            continue
        cap=min(DEEP_CAP,int(recmap.get(mn,{}).get('max_planned_ctx') or DEEP_CAP))
        depths=[d for d in DEEP_DEPTHS if d<=cap]
        if not depths:
            record_skip(rk(f'deep|no-depth|{w["family"]}|{mn}'), {'phase':'deep','kind':'no-deeper-context','model':mn,'build':w['build'],'policy_name':w['policy_name'],'threads':w['threads'],'depth':SCREEN_CAP,'kv':w['kv']}, f'winner max context <= {SCREEN_CAP}')
            continue
        key=rk(f'deepbranch|{w["family"]}|{mn}|{w["build"]}|{w["policy_name"]}|{w["threads"]}|{w["kv"]}')
        if not shard(key):
            continue
        _progressive_branch('deep', mn, model, str(w['build']), ROOT/'builds'/str(w['build']), str(w['policy_name']), pols[str(w['policy_name'])], int(w['threads']),
                            int(w['batch']), int(w['ubatch']), str(w['kv']), depths, initial_prefix=SCREEN_CAP, reps=int(SEARCH.get('final_reps',3)))


def _failure_summary() -> dict[str,Any]:
    counts={}; reasons={}
    for r in current_records():
        if r.get('phase') not in ('bootstrap','thread-probe','quick','batch-tune','shallow','prefix-smoke'): continue
        k=f"{r.get('phase')}:{r.get('kind')}:{r.get('status')}"; counts[k]=counts.get(k,0)+1
        if r.get('skip_reason'):
            q=str(r.get('skip_reason')); reasons[q]=reasons.get(q,0)+1
    examples=[]
    for r in current_records():
        if r.get('phase') not in ('bootstrap','thread-probe','quick','batch-tune','shallow','prefix-smoke') or r.get('status') not in ('failed','timeout','clock-throttled'):
            continue
        examples.append({'phase':r.get('phase'),'kind':r.get('kind'),'model':r.get('model'),'build':r.get('build'),'policy_name':r.get('policy_name'),
                         'threads':r.get('threads'),'rc':r.get('rc'),'command':r.get('command'),'stderr_tail':r.get('stderr_tail',''),
                         'server_rc':r.get('server_rc'),'server_signal':r.get('server_signal'),'startup_timeout':r.get('startup_timeout'),'server_log_tail':r.get('server_log_tail',''),
                         'clock':r.get('clock'),'clock_status':r.get('clock_status')})
        if len(examples)>=8: break
    obj={'run_tag':RUN_TAG,'counts':counts,'skip_reasons':reasons,'failure_examples':examples,'timestamp':time.time()}
    (OUT/f'failure-summary-{RUN_TAG}.json').write_text(json.dumps(obj,indent=2)+'\n')
    return obj


def bootstrap_probe() -> dict[str, Any]:
    """Find one known PP+decode-safe model/build/policy before any matrix.

    This intentionally prefers no-repack KNL paths.  PP-only success is not
    enough: KNL can reach a different GEMV/KV kernel during decode and SIGILL
    there while prefill succeeds.  The bootstrap therefore proves PP32 plus a
    tiny F16-KV decode before becoming the baseline.
    """
    cached=_read_bootstrap()
    if cached:
        pcm=(cached.get('pp_clock') or {}).get('loaded_mhz'); tcm=(cached.get('tg_clock') or {}).get('loaded_mhz'); clk=f" clocks={pcm:.0f}/{tcm:.0f}MHz" if isinstance(pcm,(int,float)) and isinstance(tcm,(int,float)) else ''
        print(f"REUSE load+decode bootstrap: model={cached['model']} build={cached['build']} policy={cached['policy_name']} pp={cached.get('pp_tps',0):.2f} tg={cached.get('tg_tps',0):.2f}{clk}")
        return cached
    ms=models(); recmap=model_record_map(); bds=dict(build_dirs()); pols=PLAN.get('numa_policies',[])
    if not ms or not bds: raise SystemExit('bootstrap requires configured models and built llama-bench binaries')
    cls=(HW.get('cpu',{}) or {}).get('class')
    build_order=(['knl-base-norepack','knl-generic-avx2-norepack','knl-base','knl-combo-norepack','knl-combo'] if cls=='knl'
                 else ['native-norepack','native','native-lto','isa-avx2'])
    build_order=[x for x in build_order if x in bds] + [x for x in bds if x not in build_order]
    policy_order=(['disabled-mmap','disabled-no-mmap','knl-ddr-strict','knl-mcdram-strict'] if cls=='knl'
                  else ['disabled-mmap','disabled-no-mmap','distribute-mmap','distribute-no-mmap'])
    models_sorted=sorted(ms,key=lambda x:int(recmap.get(x[0],{}).get('size_bytes') or x[1].stat().st_size))
    attempts=[]; tmp=OUT/f'.bootstrap-{RANK}.json'
    for mn,model in models_sorted:
        for bn in build_order:
            bench=bds[bn]
            candidates=[]
            for name in policy_order:
                p=next((x for x in pols if x.get('name')==name and supported_policy(bench,x) and policy_applicable(mn,x)),None)
                if p is not None and p not in candidates:candidates.append(p)
            for p in pols:
                if supported_policy(bench,p) and policy_applicable(mn,p) and p not in candidates:candidates.append(p)
            for p in candidates:
                th=int(p.get('default_threads',PLAN['threads'][0]))
                key=rk(f'bootstrap|{mn}|{bn}|{p["name"]}|{th}')
                r=run_one(cmd_for(bench,model,th,p,['-p','32','-n','0']),
                          {'phase':'bootstrap','kind':'pp','model':mn,'build':bn,'policy_name':p['name'],'threads':th,'depth':32,'kv':None,'policy':p},tmp,key,min(QUICK_TIMEOUT,120))
                attempts.append(r)
                if r.get('status')=='ok' and speed(r)>0:
                    tgkey=rk(f'bootstrap-decode|{mn}|{bn}|{p["name"]}|{th}|f16')
                    rtg=run_one(cmd_for(bench,model,th,p,['-p','0','-n','16','-d','512','-ctk','f16','-ctv','f16']),
                                {'phase':'bootstrap','kind':'tg','model':mn,'build':bn,'policy_name':p['name'],'threads':th,'depth':512,'kv':'f16','policy':p},
                                tmp,tgkey,min(QUICK_TIMEOUT,120))
                    attempts.append(rtg)
                    if rtg.get('status')!='ok' or speed(rtg)<=0:
                        continue
                    obj={'version':2,'package_version':(PKG/'VERSION').read_text().strip(),'fingerprint':_bootstrap_fingerprint(),'model':mn,'model_path':str(model),'build':bn,
                         'policy_name':p['name'],'threads':th,'pp_tps':speed(r),'tg_tps':speed(rtg),'baseline_kv':'f16','run_tag':RUN_TAG,'timestamp':time.time(),
                         'pp_clock':r.get('clock'),'tg_clock':rtg.get('clock'),
                         'rationale':'first PP+decode-safe cheap baseline; no-repack is intentionally preferred on KNL'}
                    bp=_bootstrap_path();bp.parent.mkdir(parents=True,exist_ok=True);bp.write_text(json.dumps(obj,indent=2)+'\n')
                    pcm=(r.get('clock') or {}).get('loaded_mhz'); tcm=(rtg.get('clock') or {}).get('loaded_mhz'); clk=f" clocks={pcm:.0f}/{tcm:.0f}MHz" if isinstance(pcm,(int,float)) and isinstance(tcm,(int,float)) else ''
                    print(f"PASS load+decode bootstrap: model={mn} build={bn} policy={p['name']} threads={th} pp32={speed(r):.2f} tg512-f16={speed(rtg):.2f}{clk}")
                    return obj
    obj=_failure_summary()
    raise SystemExit(f'LOAD BOOTSTRAP FAILED: no calibration model/build/policy completed both PP32 and F16-KV decode. diagnostic={OUT/("failure-summary-"+RUN_TAG+".json")} counts={obj["counts"]} examples={json.dumps(obj.get("failure_examples",[]),default=str)[:12000]}')


def prefix_smoke() -> None:
    """Prove persistent-server cache reuse on a load-safe baseline first."""
    ms=models(); bds=dict(build_dirs()); recmap=model_record_map(); boot=bootstrap_probe()
    if not ms or not bds: raise SystemExit('prefix-smoke requires configured models and built binaries')
    mlookup=dict(ms)
    ordered_models=[]
    if boot.get('model') in mlookup: ordered_models.append((str(boot['model']),mlookup[str(boot['model'])]))
    for x in sorted(ms,key=lambda q:int(recmap.get(q[0],{}).get('size_bytes') or q[1].stat().st_size)):
        if x not in ordered_models: ordered_models.append(x)
    cls=(HW.get('cpu',{}) or {}).get('class')
    build_order=[str(boot.get('build',''))]
    extras=(['knl-base-norepack','knl-generic-avx2-norepack','knl-base','knl-combo-norepack'] if cls=='knl' else ['native-norepack','native'])
    for x in extras:
        if x in bds and x not in build_order: build_order.append(x)
    build_order=[x for x in build_order if x in bds]
    pols=PLAN['numa_policies']; smoke_kv=os.environ.get('LAB_PREFIX_SMOKE_KV','f16').strip() or 'f16'
    max_attempts=max(1,int(os.environ.get('LAB_PREFIX_SMOKE_MAX_ATTEMPTS','8'))); ntry=0; attempts=[]
    preferred=[str(boot.get('policy_name','')),'disabled-mmap','disabled-no-mmap']
    if cls=='knl': preferred += ['knl-ddr-strict','knl-mcdram-strict']
    for mn,model in ordered_models:
      for bn in build_order:
        bench=bds[bn]; server=ROOT/'builds'/bn/'bin'/'llama-server'
        if not server.exists(): continue
        candidates=[]
        for name in preferred:
            p=next((x for x in pols if x.get('name')==name and policy_applicable(mn,x) and supported_policy(server,x)),None)
            if p is not None and p not in candidates:candidates.append(p)
        for pol in candidates:
            if ntry>=max_attempts: break
            ntry+=1; th=int(pol.get('default_threads',PLAN['threads'][0]))
            _progressive_branch('prefix-smoke',mn,model,bn,ROOT/'builds'/bn,pol['name'],pol,th,512,256,smoke_kv,[512,2048],reps=1,allow_context_fallback=False)
            pp=[r for r in current_records() if r.get('phase')=='prefix-smoke' and r.get('model')==mn and r.get('build')==bn and r.get('policy_name')==pol['name'] and r.get('threads')==th and r.get('kv')==smoke_kv and r.get('kind')=='pp_interval']
            pp=sorted(pp,key=lambda r:int(r.get('depth') or 0)); good=[r for r in pp if r.get('status')=='ok'];attempts.extend(pp[-2:])
            if len(good)>=2 and int(good[-1].get('depth') or 0)==2048 and int(good[-1].get('cache_n') or -1)>=504:
                print(f'PASS prefix-smoke {mn}: build={bn} policy={pol["name"]} kv={smoke_kv} cache_n={good[-1].get("cache_n")} prompt_n={good[-1].get("prompt_n")}')
                return
        if ntry>=max_attempts: break
      if ntry>=max_attempts: break
    obj=_failure_summary(); logs=sorted(OUT.glob('.server-prefix-smoke-*.log'),key=lambda x:x.stat().st_mtime,reverse=True);tails=[]
    for lp in logs[:6]:tails.append(f'--- {lp.name} ---\n'+'\n'.join(lp.read_text(errors='ignore').splitlines()[-100:]))
    raise SystemExit('PREFIX CACHE SMOKE FAILED: no load-safe server proved 512->2048 suffix reuse. '
                     f'attempts={ntry} kv={smoke_kv} pp_rows={json.dumps(attempts[-12:],default=str)[:8000]} summary={obj} server_tails={(chr(10).join(tails))[-20000:]}')


def full() -> None:
    # Make full-sweep self-contained and resumable: missing earlier stages are
    # filled in, while completed run keys are skipped.
    if not any(OUT.glob(f'quick-summary-{RUN_TAG}-rank*.json')):
        quick()
    if not (OUT / f'batches-{RUN_TAG}-rank{RANK}.json').exists() and not os.environ.get('LAB_BATCH_FILE'):
        tune_batches()
    shallow()
    ws=family_winners()
    if not ws:
        obj=_failure_summary()
        raise SystemExit(f'no model family produced a valid cached-prefix {SCREEN_CAP}-token PP/TG winner; diagnostic={OUT / ("failure-summary-"+RUN_TAG+".json")} counts={obj["counts"]} skip_reasons={obj["skip_reasons"]}')
    deep()



def _raw_pp_tg(rec: dict[str, Any]) -> dict[str, float]:
    """Extract separate llama-bench PP/TG rates from a multi-test JSON record."""
    raw=rec.get('raw',[])
    if isinstance(raw,dict): raw=[raw]
    out: dict[str,float]={}
    for r in raw if isinstance(raw,list) else []:
        if not isinstance(r,dict) or not isinstance(r.get('avg_ts'),(int,float)): continue
        np=int(r.get('n_prompt') or 0); ng=int(r.get('n_gen') or 0); dep=int(r.get('n_depth') or 0)
        if np>0 and ng==0: out[f'pp{np}@{dep}']=float(r['avg_ts'])
        elif ng>0 and np==0: out[f'tg{ng}@{dep}']=float(r['avg_ts'])
    return out


def _primary_probe_pairs(mn: str) -> list[tuple[str,str,str]]:
    """A few causal toggles for big-model ballpark tests.

    KNL isolates memory tier, combo kernel, then repack. Oversize models use
    preferred-MCDRAM spill instead of strict HBM. Generic CPUs isolate repack,
    explicit AVX2, and one NUMA policy.
    """
    cls=str((HW.get('cpu',{}) or {}).get('class',''))
    bnames={x['name'] for x in PLAN.get('builds',[])}
    pnames={x['name'] for x in PLAN.get('numa_policies',[])}
    rec=model_record_map().get(mn,{})
    fit=str(rec.get('mcdram_fit_class') or '')
    pairs: list[tuple[str,str,str]]=[]
    def add(b: str,p: str,why: str) -> None:
        if b in bnames and p in pnames and (b,p,why) not in pairs: pairs.append((b,p,why))
    if cls=='knl':
        tier='knl-mcdram-strict' if fit in ('comfortable','nominal') else 'knl-mcdram-preferred'
        add('knl-base-norepack','knl-ddr-strict','HBM OFF / base KNL kernel / repack OFF')
        add('knl-base-norepack',tier,'HBM or spill tier ON / base KNL kernel / repack OFF')
        add('knl-combo-norepack',tier,'combo kernel ON / repack OFF')
        # Dense Qwen3.8 is only a tiny reality check; do not spend a fourth
        # 27B model load on repack unless explicitly requested.
        rec_alias=str(rec.get('alias') or '').lower()
        if rec_alias!='qwen3.8-27b' or os.environ.get('LAB_BALLPARK_Q38_REPACK','0')=='1':
            add('knl-combo',tier,'combo kernel ON / repack ON')
        if os.environ.get('LAB_BALLPARK_GENERIC_CONTROL','0')=='1':
            add('knl-generic-avx2-norepack',tier,'optional generic AVX2 backend control')
    else:
        base='disabled-no-mmap' if 'disabled-no-mmap' in pnames else next(iter(pnames), '')
        add('native-norepack',base,'native ISA / repack OFF')
        add('native',base,'native ISA / repack ON')
        add('isa-avx2',base,'explicit AVX2 control')
        if 'interleave' in pnames: add('native-norepack','interleave','NUMA interleave control')
    return pairs


def primary_probe() -> None:
    """Very small primary-model smoke: big model first, tiny prompt, causal toggles.

    One llama-bench process per config runs PP128/TG32 at depth 0 and 512 in a
    single model load. This is intentionally a ballpark test, not a quality or
    long-context characterization.
    """
    ms=models()
    if not ms: raise SystemExit('primary probe has no model to run')
    ppn=max(32,int(os.environ.get('LAB_BALLPARK_PP','128')))
    tgn=max(8,int(os.environ.get('LAB_BALLPARK_TG','32')))
    depth=max(0,int(os.environ.get('LAB_BALLPARK_DEPTH','512')))
    budget=max(60,int(os.environ.get('LAB_BALLPARK_BUDGET_S','300')))
    started=time.time()
    bds=dict(build_dirs()); polmap={p['name']:p for p in PLAN.get('numa_policies',[])}
    allout=[]
    for mn,model in ms:
        rec=model_record_map().get(mn,{})
        size_gib=float(rec.get('size_bytes') or 0)/2**30
        # Dense Qwen3.8 is included specifically to test the KNL full-HBM
        # hypothesis. Do not silently turn it into a spill benchmark.
        if str((HW.get('cpu',{}) or {}).get('class',''))=='knl' and str(rec.get('alias') or '').lower()=='qwen3.8-27b':
            fit=str(rec.get('mcdram_fit_class') or '')
            if fit not in ('comfortable','nominal'):
                vis=float(rec.get('mcdram_visible_bytes') or 0)/2**30
                head=float(rec.get('mcdram_headroom_bytes') or 0)/2**30
                raise SystemExit(f'Qwen3.8-27B KNL ballpark requires full MCDRAM residency: quant={rec.get("quant")} size={size_gib:.2f} GiB visible={vis:.2f} GiB headroom={head:.2f} GiB fit={fit}')
        pairs=_primary_probe_pairs(mn)
        runnable=[]
        for bn,pn,why in pairs:
            bench=bds.get(bn); pol=polmap.get(pn)
            if not bench or not pol or not supported_policy(bench,pol) or not policy_applicable(mn,pol): continue
            th=int(pol.get('default_threads',PLAN.get('hardware_summary',{}).get('phys_cores') or 1))
            runnable.append((bn,pn,why,th,bench,pol))
        if not runnable: raise SystemExit(f'no ballpark build/policy pair runnable for {mn}')
        # Cold-read estimate is intentionally conservative. After the first load,
        # page cache usually makes later variants cheaper.
        cold_gibs=max(0.1,float(os.environ.get('LAB_BALLPARK_COLD_GIB_S','0.5')))
        eta_each=max(15.0,size_gib/cold_gibs + ppn/max(5.0,USABLE_MIN_PP) + tgn/max(1.0,USABLE_MIN_TG))
        _stage('PRIMARY MODEL BALLPARK',models_=[mn],configs=len(runnable),depths=[0,depth],kvs=['f16'],
               passes=f'ONE model load/config; PP{ppn} + TG{tgn} at depth 0 and {depth}; one repetition',
               eta_s=min(float(budget),eta_each*len(runnable)))
        print(f'  model_size={size_gib:.2f} GiB; total ballpark budget={_fmt_s(budget)}; cold-read ETA assumes ~{cold_gibs:.2f} GiB/s then benefits from cache if available.',flush=True)
        for i,(bn,pn,why,th,_,_) in enumerate(runnable,1):
            print(f'  PLAN {i}/{len(runnable)}: build={bn} policy={pn} threads={th} :: {why}',flush=True)
        mout=[]
        for idx,(bn,pn,why,th,bench,pol) in enumerate(runnable,1):
            remain=budget-(time.time()-started)
            if remain <= 10:
                print(f'BUDGET EXHAUSTED: stopping {mn} ballpark before build={bn} policy={pn}; remaining={_fmt_s(max(0,remain))}',flush=True)
                break
            if idx>3 and remain < max(20.0,eta_each*0.6):
                print(f'BUDGET PRUNE: skipping optional final toggle build={bn} policy={pn}; remaining={_fmt_s(remain)}',flush=True)
                continue
            meta={'phase':'primary-ballpark','kind':'pp+tg','model':mn,'build':bn,'policy_name':pn,'threads':th,
                  'depth':depth,'kv':'f16','batch':2048,'ubatch':512,'policy':pol,'toggle':why}
            key=rk(f'primary-ballpark|{mn}|{bn}|{pn}|{th}|pp{ppn}|tg{tgn}|d{depth}')
            tmp=OUT/f'.primary-ballpark-{RANK}.json'
            cmd=cmd_for(bench,model,th,pol,['-p',str(ppn),'-n',str(tgn),'-d',f'0,{depth}','-b','2048','-ub','512','-ctk','f16','-ctv','f16'])
            configured=max(15,int(os.environ.get('LAB_BALLPARK_TIMEOUT_S','300')))
            timeout=max(15,min(configured,int(max(15,remain))))
            run=run_one(cmd,meta,tmp,key,timeout)
            metrics=_raw_pp_tg(run)
            x={'model':mn,'build':bn,'policy':pn,'threads':th,'toggle':why,'status':run.get('status'),'wall_s':run.get('wall_s'),
               'clock_mhz':(run.get('clock') or {}).get('loaded_mhz'),'clock_status':run.get('clock_status'),'metrics':metrics}
            mout.append(x); allout.append(x)
            print('  RESULT: '+mn+' | '+bn+' | '+pn+' | '+why,flush=True)
            for k in sorted(metrics): print(f'    {k}={metrics[k]:.2f} t/s',flush=True)
        # Promise is based on tiny-context PP and the more realistic shallow TG.
        def score(x: dict[str,Any]) -> tuple[float,float]:
            m=x.get('metrics') or {}
            pp=float(m.get(f'pp{ppn}@0') or m.get(f'pp{ppn}@{depth}') or 0)
            tg=float(m.get(f'tg{tgn}@{depth}') or m.get(f'tg{tgn}@0') or 0)
            return tg,pp
        ok=[x for x in mout if x.get('status')=='ok' and score(x)[0]>0]
        if ok:
            best=max(ok,key=score); btg,bpp=score(best)
            promote=bpp>=USABLE_MIN_PP and btg>=USABLE_MIN_TG
            print(f'  BALLPARK DECISION {mn}: best PP~{bpp:.2f} TG@{depth}~{btg:.2f} -> promoted={promote} responsive_target={btg>=TARGET_TG}',flush=True)
            if promote: print('  NEXT: this model deserves the 2K/8K + quality path; no deep sweep is run automatically here.',flush=True)
            else: print('  NEXT: below the usability floor; do not spend deep-context time yet.',flush=True)
    out={'version':1,'timestamp':time.time(),'pp_tokens':ppn,'tg_tokens':tgn,'depth':depth,'budget_s':budget,'rows':allout}
    stem=re.sub(r'[^A-Za-z0-9_.-]+','_',str(ms[0][0]).split('--',1)[0]) if ms else 'model'
    q=ROOT/'calibration'/f'primary-ballpark-{stem}.json'; q.parent.mkdir(parents=True,exist_ok=True); q.write_text(json.dumps(out,indent=2)+'\n')
    print(f'WROTE {q}',flush=True)

def node_workers() -> None:
    if HW['numa']['count'] < 2:
        return
    ms = models(); bds = list(build_dirs())
    if not ms or not bds:
        return
    mn,model = ms[0]; bn,bench = bds[0]; procs=[]; files=[]
    for n in HW['numa']['nodes']:
        if not n['cpulist']:
            continue
        o = OUT / f'node-worker-{n["id"]}.json'; files.append(o)
        cmd = ['numactl',f'--cpunodebind={n["id"]}',f'--membind={n["id"]}',str(bench),'-m',str(model),'--numa','numactl','-t',str(HW['cpu']['cores_per_socket']),'-n',str(TG),'-r','3','-o','json']
        procs.append((subprocess.Popen(cmd, stdout=open(o,'w'), stderr=subprocess.DEVNULL),cmd))
    node_clock=ClockMonitor(cpus=sorted(os.sched_getaffinity(0)),nominal_base_mhz=_nominal_base_mhz()).start()
    try:
        for p,_ in procs:
            p.wait()
    finally:
        node_clock_stats=node_clock.stop()
    vals=[]
    for o in files:
        try:
            raw=json.load(open(o)); vals.append(speed({'raw':raw}))
        except Exception:
            pass
    rec={'phase':'numa','kind':'independent_node_workers','model':mn,'build':bn,'aggregate_tg':sum(vals),'per_node_tg':vals,'note':'Aggregate multi-stream throughput; not single-stream weight mirroring.',
         'run_key':rk(f'node-workers|{mn}|{bn}'),'status':_clock_adjust_status('ok',node_clock_stats),'rc':0,'timestamp':time.time(),'raw':[],
         'clock':node_clock_stats,'clock_status':node_clock_stats.get('status')}
    _append_record(rec)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode in ('fast-plan',):
        fast_plan(); return
    if mode in ('quick-lite',):
        quick_lite(); return
    if mode in ('fast-scale',):
        fast_scale(); return
    if mode in ('bootstrap',):
        bootstrap_probe(); return
    if mode in ('prefix-smoke',):
        prefix_smoke(); return
    if mode in ('thread-probe',):
        thread_probe(); return
    if mode in ('kv-probe',):
        kv_probe(); return
    if mode in ('screen',):
        quick(); tune_batches(); shallow(); family_candidates()
        return
    if mode in ('deep',):
        deep()
        return
    if mode in ('quick','all','overnight'):
        quick()
    if mode in ('batch','all','overnight'):
        tune_batches()
    if mode in ('full','all','overnight'):
        full()
    if mode in ('primary-probe',):
        primary_probe(); return
    if mode in ('node-workers','all'):
        node_workers()


if __name__ == '__main__':
    main()
