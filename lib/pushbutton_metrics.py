#!/usr/bin/env python3
"""Small, privacy-preserving benchmark aggregation and optional telemetry client."""
from __future__ import annotations
import gzip, json, os, pathlib, statistics, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCAL_RESULTS = ROOT / "benchmarks/results"
REMOTE_URL = os.environ.get(
    "PUSHBUTTON_METRICS_URL",
    "https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/benchmarks/aggregate.json",
)
CONFIG = pathlib.Path(os.environ.get("PUSHBUTTON_CONFIG_DIR", pathlib.Path.home()/".config/pushbutton-local"))
CACHE = pathlib.Path(os.environ.get("PUSHBUTTON_CACHE_DIR", pathlib.Path.home()/".cache/pushbutton"))
QUEUE = CACHE / "telemetry-queue"


def _read_json(path: pathlib.Path, default):
    try: return json.loads(path.read_text())
    except Exception: return default


def telemetry_config() -> dict:
    return _read_json(CONFIG/"telemetry.json", {})


def set_telemetry(enabled: bool, upload_url: str | None = None) -> pathlib.Path:
    CONFIG.mkdir(parents=True, exist_ok=True)
    obj={"enabled":bool(enabled)}
    if upload_url: obj["upload_url"]=upload_url
    p=CONFIG/"telemetry.json"; p.write_text(json.dumps(obj,indent=2)+"\n"); return p


def queue_report(report: dict) -> pathlib.Path | None:
    cfg=telemetry_config()
    if not cfg.get("enabled"): return None
    # Deliberately exclude prompts, generated text, usernames, paths and hostnames.
    compact={
      "schema_version":1,
      "timestamp_utc":report.get("timestamp_utc"),
      "model":report.get("model"),
      "backend":(report.get("backend") or {}).get("id"),
      "backend_fingerprint":report.get("backend"),
      "context":report.get("context"),
      "gpus":report.get("gpus"),
      "gpu_models":[g.get("name") for g in (report.get("hardware") or {}).get("gpus",[])],
      "gpu_memory_mib":[g.get("memory_total_mib") for g in (report.get("hardware") or {}).get("gpus",[])],
      "ram_gib":(report.get("hardware") or {}).get("ram_gib"),
      "cases":[{k:c.get(k) for k in ("name","concurrency","median_prompt_tokens","median_prefill_tok_s","median_decode_tok_s","median_ttft_s","aggregate_output_tok_s")} for c in report.get("cases",[])],
    }
    QUEUE.mkdir(parents=True, exist_ok=True)
    name=(compact.get("timestamp_utc") or str(time.time())).replace(":","-")+".json"
    p=QUEUE/name; p.write_text(json.dumps(compact,separators=(",",":"))+"\n"); return p


def upload_pending() -> tuple[int,str]:
    cfg=telemetry_config(); url=cfg.get("upload_url") or os.environ.get("PUSHBUTTON_TELEMETRY_UPLOAD_URL")
    if not cfg.get("enabled"): return 0,"telemetry disabled"
    if not url: return 0,"telemetry enabled locally; no upload URL configured"
    files=sorted(QUEUE.glob("*.json")) if QUEUE.exists() else []
    if not files: return 0,"nothing queued"
    payload=("\n".join(p.read_text().strip() for p in files)+"\n").encode()
    body=gzip.compress(payload,compresslevel=6)
    req=urllib.request.Request(url,data=body,method="POST",headers={"Content-Type":"application/x-ndjson","Content-Encoding":"gzip","User-Agent":"PushbuttonLocalCoders-telemetry/1"})
    with urllib.request.urlopen(req,timeout=15) as r:
        if not (200 <= r.status < 300): raise RuntimeError(f"telemetry HTTP {r.status}")
    for p in files: p.unlink(missing_ok=True)
    return len(files),f"uploaded {len(files)} compact datapoint file(s)"


def _remote_rows(max_age_s: int = 86400) -> list[dict]:
    CACHE.mkdir(parents=True,exist_ok=True); p=CACHE/"aggregate.json"
    if p.exists() and time.time()-p.stat().st_mtime < max_age_s:
        return (_read_json(p,{}) or {}).get("samples",[])
    try:
        with urllib.request.urlopen(REMOTE_URL,timeout=3) as r: raw=r.read()
        p.write_bytes(raw); return (json.loads(raw) or {}).get("samples",[])
    except Exception:
        return (_read_json(p,{}) or {}).get("samples",[])


def local_rows() -> list[dict]:
    rows=[]
    if not LOCAL_RESULTS.exists(): return rows
    for p in LOCAL_RESULTS.glob("*.json"):
        x=_read_json(p,{})
        if not x: continue
        gpu_names=[g.get("name") for g in (x.get("hardware") or {}).get("gpus",[])]
        backend=(x.get("backend") or {}).get("id")
        artifact=(x.get("backend") or {}).get("artifact") or (x.get("backend") or {}).get("quant")
        for c in x.get("cases",[]):
            rows.append({"source":"local","model":x.get("model"),"backend":backend,"artifact":artifact,"gpu_models":gpu_names,"context":x.get("context"),"case":c.get("name"),"prompt_tokens":c.get("median_prompt_tokens"),"pp":c.get("median_prefill_tok_s"),"tg":c.get("median_decode_tok_s"),"ttft":c.get("median_ttft_s")})
    return rows


def observations(model: str, backend: str, gpu_name: str | None = None, context: int | None = None) -> list[dict]:
    rows=local_rows()+_remote_rows()
    out=[]
    for r in rows:
        if r.get("model")!=model or r.get("backend")!=backend: continue
        names=r.get("gpu_models") or ([r.get("gpu")] if r.get("gpu") else [])
        if gpu_name and names and not any(gpu_name.lower() in str(n).lower() or str(n).lower() in gpu_name.lower() for n in names): continue
        out.append(r)
    return out


def summarize_observations(rows: list[dict]) -> dict:
    def stats(k):
        v=sorted(float(r[k]) for r in rows if r.get(k) is not None)
        if not v:return None
        q=lambda f:v[min(len(v)-1,max(0,round((len(v)-1)*f)))]
        return {"n":len(v),"median":statistics.median(v),"p10":q(.10),"p90":q(.90)}
    return {"pp":stats("pp"),"tg":stats("tg"),"ttft":stats("ttft")}
