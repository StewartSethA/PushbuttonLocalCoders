#!/usr/bin/env python3
"""Privacy-preserving performance aggregation used by normal Pushbutton runs.

The important distinction is between *measurement* and *upload*:
- local measurements are always saved so future launches can make better choices;
- network contribution is asked for on first run (default yes) and only sends
  compact performance metadata: model weight load time and PP/TG per context depth;
- the collector stamps each record with its receive time and the sender's IP;
- uploads happen at most once every MIN_UPLOAD_INTERVAL_S (5 minutes);
- prompts, generated text, usernames, hostnames and local paths are never queued.
"""
from __future__ import annotations
import contextlib, gzip, json, os, pathlib, shutil, statistics, time, urllib.request
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCAL_RESULTS = ROOT / "benchmarks/results"
LOCAL_AGGREGATE = ROOT / "benchmarks/aggregate.json"
REMOTE_URL = os.environ.get(
    "PUSHBUTTON_METRICS_URL",
    "https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/benchmarks/aggregate.json",
)
CONFIG = pathlib.Path(os.environ.get("PUSHBUTTON_CONFIG_DIR", pathlib.Path.home()/".config/pushbutton-local"))
CACHE = pathlib.Path(os.environ.get("PUSHBUTTON_CACHE_DIR", pathlib.Path.home()/".cache/pushbutton"))
RUNTIME = pathlib.Path(os.environ.get("PUSHBUTTON_RUNTIME_DIR", pathlib.Path.home()/".local/share/pushbutton/runtime"))
LIVE_RESULTS = RUNTIME / "observations"
QUEUE = CACHE / "telemetry-queue"
CONSENT_VERSION = 2
MIN_UPLOAD_INTERVAL_S = 300
TELEMETRY_SCHEMA = 2
# Set to the deployed ingest relay (telemetry/worker) once it exists; until then
# data stays queued unless the user or $PUSHBUTTON_TELEMETRY_UPLOAD_URL sets one.
DEFAULT_UPLOAD_URL = ""
PUBLIC_MIN_CONTRIBUTORS = 3
CONSENT_TEXT = f"""\
Pushbutton performance telemetry (on by default; answer n, or use --telemetry-off at any time)
  Sent: model, quant, backend, hardware type (CPU/GPU model, sockets/cores, memory size),
    launch strategy, model load time, and prompt/decode tok/s at each context depth.
  Never sent: prompts, outputs, code, file names, paths, usernames or hostnames.
  Stored: the collector adds the receive time and a keyed hash of your IP address. The raw IP
    is used only for rate limiting and is not stored. Records go to a private repository
    readable only by the maintainers.
  Published: only aggregated, anonymized statistics (medians per model/quant/hardware/context
    depth, shown only when at least {PUBLIC_MIN_CONTRIBUTORS} different contributors share a group) on the public
    benchmark page. No IP hashes, timestamps or individual records are ever published.
  Uploads happen at most once every {MIN_UPLOAD_INTERVAL_S // 60} minutes.
  Disclaimer: telemetry is used only for performance analytics; it is never sold or used for
    advertising or tracking. Published benchmarks are community-contributed and provided as-is,
    without warranty. Because records carry no account or install id, past records cannot be
    attributed to you; turning telemetry off stops all further uploads and clears the queue."""
SANITIZED_FIELDS = (
    "schema_version","timestamp_utc","model","backend","artifact","backend_commit",
    "context","gpu_models","gpu_memory_mib","ram_gib","prompt_tokens","completion_tokens",
    "prompt_tokens_estimated","completion_tokens_estimated","pp","tg","ttft","elapsed_s",
    "concurrency","source","method","status_code",
    "event","device","cpu_model","cpu_family","cpu_tier","cpu_sockets","cpu_cores",
    "fast_mem_kind","fast_mem_mode","strategy","threads","threads_batch","memory_tier",
    "load_time_s",
)
STRING_FIELDS = ("model","backend","artifact","device","cpu_model","cpu_family","cpu_tier","strategy",
                 "memory_tier","event","source","method","fast_mem_kind","fast_mem_mode","backend_commit")
DEPTH_FIELDS = ("depth","pp_tps","tg_tps","pp_tps_estimate","tg_tps_estimate")


def _read_json(path: pathlib.Path, default):
    try: return json.loads(path.read_text())
    except Exception: return default


def _write_json(path: pathlib.Path, obj) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2)+"\n")
    tmp.replace(path)
    return path


def telemetry_config() -> dict:
    return _read_json(CONFIG/"telemetry.json", {})


def telemetry_enabled() -> bool:
    """Uploads need consent to the current disclosure (CONSENT_TEXT), not just an old flag."""
    cfg=telemetry_config()
    return bool(cfg.get("enabled")) and int(cfg.get("consent_version",0) or 0) >= CONSENT_VERSION


def telemetry_consent_recorded() -> bool:
    cfg=telemetry_config()
    return int(cfg.get("consent_version",0) or 0) >= CONSENT_VERSION


def set_telemetry(enabled: bool, upload_url: str | None = None) -> pathlib.Path:
    old=telemetry_config()
    obj={**old,"enabled":bool(enabled),"consent_version":CONSENT_VERSION,"consent_timestamp":int(time.time())}
    if upload_url: obj["upload_url"]=upload_url
    if not enabled: shutil.rmtree(QUEUE,ignore_errors=True)
    return _write_json(CONFIG/"telemetry.json",obj)


def save_observation(obs: dict) -> pathlib.Path:
    """Persist one sanitized normal-use observation locally as append-only NDJSON."""
    LIVE_RESULTS.mkdir(parents=True,exist_ok=True)
    day=str(obs.get("timestamp_utc") or time.strftime("%Y-%m-%d",time.gmtime()))[:10]
    p=LIVE_RESULTS/f"{day}.jsonl"
    with p.open("a") as f: f.write(json.dumps(_sanitize_observation(obs),separators=(",",":"))+"\n")
    return p


def _num(x):
    return x if isinstance(x,(int,float)) and not isinstance(x,bool) else None


def _sanitize_depths(rows) -> list[dict]:
    out=[]
    for r in rows if isinstance(rows,list) else []:
        if not isinstance(r,dict):continue
        row={k:_num(r.get(k)) for k in DEPTH_FIELDS}
        row={k:v for k,v in row.items() if v is not None}
        if "depth" in row:out.append(row)
    return out[:32]


def _sanitize_observation(obs: dict) -> dict:
    out={k:obs.get(k) for k in SANITIZED_FIELDS if obs.get(k) is not None}
    for k in STRING_FIELDS:
        if k in out:out[k]=str(out[k])[:160]
    if obs.get("depths") is not None:out["depths"]=_sanitize_depths(obs.get("depths"))
    return out


def _queue_line(obs: dict) -> pathlib.Path | None:
    if not telemetry_enabled(): return None
    QUEUE.mkdir(parents=True,exist_ok=True)
    p=QUEUE/"live.ndjson"
    with p.open("a") as f: f.write(json.dumps(_sanitize_observation(obs),separators=(",",":"))+"\n")
    return p


def queue_observation(obs: dict) -> pathlib.Path | None:
    """Queue one compact datapoint if contribution is enabled."""
    return _queue_line(obs)


def queue_performance(model: str, backend: str, artifact: str | None, *, load_time_s: float | None = None,
                      depths: list[dict] | None = None, context: int | None = None,
                      device: dict | None = None, source: str = "pushbutton") -> pathlib.Path | None:
    """Queue a load-time and/or PP/TG-by-context-depth record.

    ``device`` carries hardware descriptors only (e.g. cpu_platform summary or GPU
    names); it never contains hostnames, paths or prompt text.
    """
    obs={"schema_version":TELEMETRY_SCHEMA,"event":"performance",
         "timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
         "model":model,"backend":backend,"artifact":artifact,"context":context,"source":source,
         "load_time_s":round(float(load_time_s),3) if load_time_s is not None else None,"depths":depths}
    obs.update(device or {})
    return _queue_line(obs)


def device_descriptor(cpu_summary: dict | None = None, strategy: dict | None = None,
                      gpu_models: list[str] | None = None) -> dict:
    d: dict = {}
    if gpu_models:d.update(device="gpu",gpu_models=list(gpu_models))
    if cpu_summary:
        d.setdefault("device","cpu")
        d.update(cpu_model=cpu_summary.get("model_name"),cpu_family=cpu_summary.get("family"),
                 cpu_tier=cpu_summary.get("tier"),cpu_sockets=cpu_summary.get("sockets"),
                 cpu_cores=cpu_summary.get("physical_cores"),fast_mem_kind=cpu_summary.get("fast_mem_kind") or None,
                 fast_mem_mode=cpu_summary.get("fast_mem_mode") or None,
                 ram_gib=round(cpu_summary["ram_total_mib"]/1024,1) if cpu_summary.get("ram_total_mib") else None)
    if strategy:
        d.update(strategy=strategy.get("name"),threads=strategy.get("threads"),
                 threads_batch=strategy.get("threads_batch"),memory_tier=strategy.get("memory_tier"))
    return d


def queue_report(report: dict) -> pathlib.Path | None:
    if not telemetry_enabled():return None
    backend=report.get("backend") or {}
    compact={
        "schema_version":1,"timestamp_utc":report.get("timestamp_utc"),"model":report.get("model"),
        "backend":backend.get("id"),"artifact":backend.get("artifact") or backend.get("quant"),
        "backend_fingerprint":{k:backend.get(k) for k in ("id","commit","version","artifact","quant") if backend.get(k) is not None},
        "context":report.get("context"),"gpus":report.get("gpus"),
        "gpu_models":[g.get("name") for g in (report.get("hardware") or {}).get("gpus",[])],
        "gpu_memory_mib":[g.get("memory_total_mib") for g in (report.get("hardware") or {}).get("gpus",[])],
        "ram_gib":(report.get("hardware") or {}).get("ram_gib"),
        "cases":[{k:c.get(k) for k in ("name","concurrency","median_prompt_tokens","median_prefill_tok_s","median_decode_tok_s","median_ttft_s","aggregate_output_tok_s")} for c in report.get("cases",[])],
    }
    QUEUE.mkdir(parents=True, exist_ok=True)
    name=(compact.get("timestamp_utc") or str(time.time())).replace(":","-")+".json"
    p=QUEUE/name; p.write_text(json.dumps(compact,separators=(",",":"))+"\n"); return p


def _queued_files() -> list[pathlib.Path]:
    if not QUEUE.exists(): return []
    return sorted([*QUEUE.glob("*.json"),*QUEUE.glob("*.ndjson")])


def queued_bytes() -> int:
    return sum(p.stat().st_size for p in _queued_files() if p.exists())


def upload_url() -> str | None:
    return (telemetry_config().get("upload_url") or os.environ.get("PUSHBUTTON_TELEMETRY_UPLOAD_URL")
            or DEFAULT_UPLOAD_URL or None)


@contextlib.contextmanager
def _upload_lock():
    CONFIG.mkdir(parents=True,exist_ok=True)
    with (CONFIG/"telemetry.lock").open("a") as fh:
        if fcntl:fcntl.flock(fh,fcntl.LOCK_EX)
        try:yield
        finally:
            if fcntl:fcntl.flock(fh,fcntl.LOCK_UN)


def seconds_until_upload_allowed(cfg: dict | None = None) -> float:
    cfg=telemetry_config() if cfg is None else cfg
    last=max(float(cfg.get("last_upload_epoch",0) or 0),float(cfg.get("last_attempt_epoch",0) or 0))
    return max(0.0,MIN_UPLOAD_INTERVAL_S-(time.time()-last))


def upload_pending() -> tuple[int,str]:
    """Send the queue to the collector, never more than once per 5 minutes.

    The interval is enforced across processes (broker, proxy, CLIs) with a file
    lock, and the attempt is recorded *before* sending so failures back off too.
    """
    cfg=telemetry_config(); url=upload_url()
    if not telemetry_enabled():return 0,"telemetry disabled or awaiting consent to the current disclosure"
    if not url:return 0,"telemetry enabled; measurements are queued locally because no collector URL is configured"
    with _upload_lock():
        cfg=telemetry_config()
        wait=seconds_until_upload_allowed(cfg)
        if wait>0:return 0,f"queued locally; next upload allowed in {int(wait)+1}s"
        files=_queued_files()
        if not files:return 0,"nothing queued"
        lines=[]
        for p in files:
            lines.extend(x for x in p.read_text().splitlines() if x.strip())
        cfg["last_attempt_epoch"]=int(time.time()); _write_json(CONFIG/"telemetry.json",cfg)
        body=gzip.compress(("\n".join(lines)+"\n").encode(),compresslevel=6)
        req=urllib.request.Request(url,data=body,method="POST",headers={
            "Content-Type":"application/x-ndjson","Content-Encoding":"gzip",
            "User-Agent":"PushbuttonLocalCoders-telemetry/2",
        })
        with urllib.request.urlopen(req,timeout=15) as r:
            if not (200<=r.status<300):raise RuntimeError(f"telemetry HTTP {r.status}")
        for p in files:p.unlink(missing_ok=True)
        cfg["last_upload_epoch"]=int(time.time()); _write_json(CONFIG/"telemetry.json",cfg)
    return len(lines),f"uploaded {len(lines)} compact datapoint(s) in {len(body)} compressed bytes"


def maybe_upload(min_bytes: int = 0, min_interval_s: int = MIN_UPLOAD_INTERVAL_S, force: bool = False) -> tuple[int,str]:
    """Upload queued data when due: never more often than every 5 minutes.

    ``min_bytes`` batches small queues until ``min_interval_s`` (never below the
    5-minute floor) has elapsed; ``force`` skips only that batching. The 5-minute
    floor always applies, whatever the arguments.
    """
    cfg=telemetry_config()
    if not telemetry_enabled(): return 0,"telemetry disabled or awaiting consent to the current disclosure"
    last=max(float(cfg.get("last_upload_epoch",0) or 0),float(cfg.get("last_attempt_epoch",0) or 0))
    age=time.time()-last
    size=queued_bytes()
    if size==0:return 0,"nothing queued"
    if age<MIN_UPLOAD_INTERVAL_S or (not force and size<min_bytes and age<max(MIN_UPLOAD_INTERVAL_S,min_interval_s)):
        return 0,f"queued locally ({size} bytes); next upload allowed in {int(max(0,MIN_UPLOAD_INTERVAL_S-age))+1}s"
    try:return upload_pending()
    except Exception as exc:return 0,f"upload deferred: {exc}"


def _bundled_rows():return (_read_json(LOCAL_AGGREGATE,{}) or {}).get("samples",[])


def _remote_rows(max_age_s: int = 86400) -> list[dict]:
    CACHE.mkdir(parents=True,exist_ok=True); p=CACHE/"aggregate.json"
    if p.exists() and time.time()-p.stat().st_mtime<max_age_s:return (_read_json(p,{}) or {}).get("samples",[]) or _bundled_rows()
    try:
        with urllib.request.urlopen(REMOTE_URL,timeout=3) as r:raw=r.read()
        p.write_bytes(raw); return (json.loads(raw) or {}).get("samples",[]) or _bundled_rows()
    except Exception:return (_read_json(p,{}) or {}).get("samples",[]) or _bundled_rows()


def _benchmark_rows() -> list[dict]:
    rows=[]
    if not LOCAL_RESULTS.exists():return rows
    for p in LOCAL_RESULTS.glob("*.json"):
        x=_read_json(p,{})
        if not x:continue
        gpu_names=[g.get("name") for g in (x.get("hardware") or {}).get("gpus",[])]; backend=(x.get("backend") or {}).get("id"); artifact=(x.get("backend") or {}).get("artifact") or (x.get("backend") or {}).get("quant")
        for c in x.get("cases",[]):rows.append({
            "source":"local-benchmark","model":x.get("model"),"backend":backend,"artifact":artifact,
            "gpu_models":gpu_names,"context":x.get("context"),"case":c.get("name"),
            "prompt_tokens":c.get("median_prompt_tokens"),"pp":c.get("median_prefill_tok_s"),
            "tg":c.get("median_decode_tok_s"),"ttft":c.get("median_ttft_s")})
    return rows


def _live_rows() -> list[dict]:
    rows=[]
    if not LIVE_RESULTS.exists():return rows
    for p in sorted(LIVE_RESULTS.glob("*.jsonl")):
        try:
            for line in p.read_text().splitlines():
                if not line.strip():continue
                x=json.loads(line)
                rows.append({
                    "source":"local-live","model":x.get("model"),"backend":x.get("backend"),
                    "artifact":x.get("artifact"),"gpu_models":x.get("gpu_models") or [],
                    "context":x.get("context"),"case":"live","prompt_tokens":x.get("prompt_tokens"),
                    "pp":x.get("pp"),"tg":x.get("tg"),"ttft":x.get("ttft"),
                })
        except Exception:pass
    return rows


def local_rows() -> list[dict]:
    return _benchmark_rows()+_live_rows()


def observations(model: str, backend: str, artifact: str | None = None, gpu_name: str | None = None, context: int | None = None) -> list[dict]:
    if isinstance(gpu_name,(int,float)) and context is None:context=int(gpu_name); gpu_name=artifact; artifact=None
    if backend=='llama.cpp' and artifact is None:return []
    out=[]
    rows = [dict(r, evidence_origin="local") for r in local_rows()]
    rows += [dict(r, evidence_origin="reference") for r in _remote_rows()]
    for r in rows:
        if r.get("model")!=model or r.get("backend")!=backend:continue
        if artifact:
            ra=r.get("artifact")
            if not ra or (artifact.lower() not in str(ra).lower() and str(ra).lower() not in artifact.lower()):continue
        names=r.get("gpu_models") or ([r.get("gpu")] if r.get("gpu") else [])
        if gpu_name and names and not any(str(gpu_name).lower() in str(n).lower() or str(n).lower() in str(gpu_name).lower() for n in names):continue
        if context and r.get("context"):
            rc=int(r["context"])
            if rc<max(1024,int(context*.25)) or rc>int(context*4):continue
        out.append(r)
    return out


def summarize_observations(rows: list[dict]) -> dict:
    def stats(k):
        v=sorted(float(r[k]) for r in rows if r.get(k) is not None)
        if not v:return None
        q=lambda f:v[min(len(v)-1,max(0,round((len(v)-1)*f)))]
        return {"n":len(v),"median":statistics.median(v),"p10":q(.10),"p90":q(.90)}
    return {"pp":stats("pp"),"tg":stats("tg"),"ttft":stats("ttft")}


def depth_bucket(prompt_tokens: int | float | None) -> str:
    if not prompt_tokens:return "unknown"
    n=int(prompt_tokens)
    if n<4096:return "<4K"
    if n<16384:return "4-16K"
    if n<65536:return "16-64K"
    if n<131072:return "64-128K"
    return ">=128K"


def sla_by_depth(rows: list[dict]) -> dict[str,dict]:
    groups={}
    for r in rows:groups.setdefault(depth_bucket(r.get("prompt_tokens")),[]).append(r)
    return {k:summarize_observations(v) for k,v in groups.items()}
