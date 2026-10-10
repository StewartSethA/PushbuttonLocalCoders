#!/usr/bin/env python3
"""Aggregate raw telemetry records into the anonymized public benchmark dataset.

Input: one or more directories of raw ``*.ndjson`` records (the private data
repository written by ``telemetry/worker`` or ``pushbutton-telemetry-collector``).
Output: ``summary.json`` and a static ``index.html`` dashboard for GitHub Pages.

Only aggregate analytics leave this script:

- Records are reduced to whitelisted dimensions (model, quant, backend,
  normalized hardware class, CPU launch strategy, context-depth bucket) and
  metrics (prompt/decode tok/s, model load time). Sender hashes, IP addresses,
  timestamps and every other field are dropped.
- A group is published only when at least ``--min-contributors`` distinct senders
  (default ``pushbutton_metrics.PUBLIC_MIN_CONTRIBUTORS``) contributed to it
  (k-anonymity); smaller groups are counted as suppressed, without detail.
- Free-text dimensions that do not look like model/hardware names are replaced
  by ``other``.
"""
from __future__ import annotations
import argparse, collections, hashlib, hmac, json, pathlib, re, secrets, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
import pushbutton_metrics as metrics  # noqa: E402

TEMPLATE = pathlib.Path(__file__).resolve().parent / "dashboard/index.html"
DEPTH_EDGES = (4096, 16384, 32768, 65536, 131072)
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:+/@()-]{0,95}$")
MAX_TPS = 1e6
MAX_LOAD_S = 24 * 3600


def safe(v) -> str:
    s = re.sub(r"\s+", " ", str(v or "")).strip()
    return s if s and SAFE.match(s) else "other"


def depth_bucket(depth: float) -> str:
    for edge in DEPTH_EDGES:
        if depth <= edge:
            return f"≤{edge // 1024}K"
    return f">{DEPTH_EDGES[-1] // 1024}K"


def bucket_order(label: str) -> int:
    for i, edge in enumerate(DEPTH_EDGES):
        if label == f"≤{edge // 1024}K":
            return i
    return len(DEPTH_EDGES)


def cpu_label(rec: dict) -> str:
    name = str(rec.get("cpu_model") or rec.get("cpu_family") or "CPU")
    name = re.sub(r"\((R|TM|tm|r)\)", "", name)
    name = re.sub(r"\bCPU\b|@\s*[\d.]+\s*GHz|\bProcessor\b", "", name)
    name = re.sub(r"\s+", " ", name).strip() or "CPU"
    sockets = rec.get("cpu_sockets")
    label = f"{int(sockets)}x {name}" if isinstance(sockets, (int, float)) and sockets > 1 else name
    if rec.get("fast_mem_kind"):
        mode = f" {rec['fast_mem_mode']}" if rec.get("fast_mem_mode") else ""
        label += f" ({rec['fast_mem_kind']}{mode})"
    return label


def hardware(rec: dict) -> str:
    gpus = [str(g) for g in rec.get("gpu_models") or [] if g]
    if gpus and rec.get("device") != "cpu":
        c = collections.Counter(gpus)
        return " + ".join(f"{n}x {g}" if n > 1 else g for g, n in sorted(c.items()))
    if rec.get("device") == "cpu" or rec.get("cpu_model") or rec.get("cpu_tier"):
        return cpu_label(rec)
    return "unknown"


def rate(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < MAX_TPS else None


class Pseudonymizer:
    """Legacy records that still carry a raw ``sender_ip`` are hashed in memory only."""

    def __init__(self):
        self.key = secrets.token_bytes(32)

    def sender(self, rec: dict) -> str | None:
        if rec.get("sender_hash"):
            return str(rec["sender_hash"])
        if rec.get("sender_ip"):
            return hmac.new(self.key, str(rec["sender_ip"]).encode(), hashlib.sha256).hexdigest()[:16]
        return None


def read_records(dirs: list[pathlib.Path]):
    for d in dirs:
        if not d.exists():
            continue
        for path in sorted(d.rglob("*.ndjson")):
            for line in path.read_text(errors="replace").splitlines():
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    yield obj


def samples(rec: dict):
    """Yield (depth, pp, tg) measured at a context depth; estimates are ignored."""
    if isinstance(rec.get("depths"), list):
        for d in metrics._sanitize_depths(rec["depths"]):
            pp, tg = rate(d.get("pp_tps")), rate(d.get("tg_tps"))
            if pp is not None or tg is not None:
                yield d["depth"], pp, tg
    elif isinstance(rec.get("prompt_tokens"), (int, float)) and not rec.get("prompt_tokens_estimated"):
        pp, tg = rate(rec.get("pp")), rate(rec.get("tg"))
        if pp is not None or tg is not None:
            yield rec["prompt_tokens"], pp, tg


def stats(values: list[float]) -> dict | None:
    if not values:
        return None
    vals = sorted(values)
    q = statistics.quantiles(vals, n=4) if len(vals) >= 2 else [vals[0], vals[0], vals[0]]
    return {"median": round(statistics.median(vals), 1), "p25": round(q[0], 1), "p75": round(q[2], 1), "n": len(vals)}


def aggregate(records, min_contributors: int = metrics.PUBLIC_MIN_CONTRIBUTORS) -> dict:
    pseudo = Pseudonymizer()
    perf = collections.defaultdict(lambda: {"senders": set(), "pp": [], "tg": []})
    load = collections.defaultdict(lambda: {"senders": set(), "load": []})
    for raw in records:
        sender = pseudo.sender(raw)
        if not sender:
            continue
        rec = metrics._sanitize_observation(raw)
        base = (safe(rec.get("model")), safe(rec.get("artifact")), safe(rec.get("backend")), safe(hardware(rec)),
                safe(rec.get("strategy")) if rec.get("strategy") else "")
        for depth, pp, tg in samples(rec):
            g = perf[base + (depth_bucket(depth),)]
            g["senders"].add(sender)
            if pp is not None:
                g["pp"].append(pp)
            if tg is not None:
                g["tg"].append(tg)
        lt = rec.get("load_time_s")
        if isinstance(lt, (int, float)) and not isinstance(lt, bool) and 0 < lt < MAX_LOAD_S:
            g = load[base]
            g["senders"].add(sender)
            g["load"].append(float(lt))

    names = ("model", "quant", "backend", "hardware", "strategy")
    perf_rows, load_rows, suppressed = [], [], 0
    for key, g in perf.items():
        if len(g["senders"]) < min_contributors:
            suppressed += 1
            continue
        perf_rows.append({**dict(zip(names, key[:5])), "depth": key[5], "contributors": len(g["senders"]),
                          "pp_tps": stats(g["pp"]), "tg_tps": stats(g["tg"])})
    for key, g in load.items():
        if len(g["senders"]) < min_contributors:
            suppressed += 1
            continue
        load_rows.append({**dict(zip(names, key)), "contributors": len(g["senders"]), "load_time_s": stats(g["load"])})
    perf_rows.sort(key=lambda r: (r["model"], r["quant"], r["hardware"], r["strategy"], bucket_order(r["depth"])))
    load_rows.sort(key=lambda r: (r["model"], r["quant"], r["hardware"], r["strategy"]))
    return {
        "schema": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:00Z", time.gmtime()),
        "min_contributors": min_contributors,
        "groups_published": len(perf_rows) + len(load_rows),
        "groups_suppressed": suppressed,
        "performance": perf_rows,
        "load_time": load_rows,
    }


def render_html(summary: dict) -> str:
    data = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return TEMPLATE.read_text().replace("/*__SUMMARY__*/null", data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", nargs="*", type=pathlib.Path, help="directories of raw *.ndjson records")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("_site"))
    ap.add_argument("--min-contributors", type=int, default=metrics.PUBLIC_MIN_CONTRIBUTORS)
    a = ap.parse_args(argv)
    if a.min_contributors < 2:
        ap.error("--min-contributors must be at least 2 to keep the public page anonymous")
    summary = aggregate(read_records(a.records), a.min_contributors)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    (a.out / "index.html").write_text(render_html(summary))
    print(f"published {summary['groups_published']} group(s), suppressed {summary['groups_suppressed']} "
          f"(< {a.min_contributors} contributors) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
