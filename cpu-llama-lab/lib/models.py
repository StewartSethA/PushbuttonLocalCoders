#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get('LAB_ROOT', str(Path.cwd() / '.cpu-llama-lab')))
MODELS_DIR = ROOT / 'models'
INV_DIR = MODELS_DIR / 'inventory'
REGISTRY = MODELS_DIR / 'registry.json'
PLAN_FILE = MODELS_DIR / 'model-plan.json'
TSV_FILE = MODELS_DIR / 'models.tsv'

# Model catalog. Calibration-only families are excluded from ordinary auto mode
# unless explicitly named via LAB_MODELS or LAB_INCLUDE_CALIBRATION=1.
CATALOG_REV = 6
# Calibration is deliberately agent/coding-oriented rather than a generic size
# ladder.  Fixed quants make cross-machine comparisons apples-to-apples.
CALIBRATION_ALIASES = ('granite4-350m', 'qwen3-0.6b', 'granite4-1b', 'granite4.1-3b')
PRIMARY_ALIASES = ('qwen3.6-35b', 'nemotron-3.5', 'qwen3.8-27b', 'qwen3.6-27b')
CATALOG = [
    # Granite 4.0 350M dense: all 28 layers are attention. IBM publishes
    # coding and BFCL results for this instruct model, making it a useful
    # sub-0.5B coding-agent floor without hybrid/recurrent cache semantics.
    dict(alias='granite4-350m', repo='ibm-granite/granite-4.0-350m-GGUF', prefix='granite-4.0-350m', native_ctx=32768,
         model_class='dense', attn_layers=28, kv_heads=4, head_dim=64, total_params_b=0.4, active_params_b=0.4, fixed_quant='Q4_K_M', calibration_only=True,
         calibration_role='sub-0.5B dense coding/tool-use floor'),
    # Qwen3-0.6B is the reasoning/tool-use-oriented <=1B rung and has direct
    # ggml-org llama.cpp GGUF support.
    dict(alias='qwen3-0.6b', repo='ggml-org/Qwen3-0.6B-GGUF', prefix='Qwen3-0.6B', native_ctx=32768,
         model_class='dense', attn_layers=28, kv_heads=8, head_dim=128, total_params_b=0.6, active_params_b=0.6, fixed_quant='Q4_0', calibration_only=True,
         calibration_role='<=1B reasoning + agent/tool-use'),
    # Granite 4.0 1B dense is the stronger small coding-agent rung. All 40
    # layers are attention, keeping cached-prefix measurement semantics simple.
    dict(alias='granite4-1b', repo='ibm-granite/granite-4.0-1b-GGUF', prefix='granite-4.0-1b', native_ctx=131072,
         model_class='dense', attn_layers=40, kv_heads=4, head_dim=128, total_params_b=2.0, active_params_b=2.0, fixed_quant='Q4_K_M', calibration_only=True,
         calibration_role='small dense coding + tool-use agent'),
    dict(alias='granite4.1-3b', repo='ibm-granite/granite-4.1-3b-GGUF', prefix='granite-4.1-3b', native_ctx=131072,
         model_class='dense', attn_layers=40, kv_heads=8, head_dim=64, total_params_b=3.4, active_params_b=3.4, fixed_quant='Q4_K_M', calibration_only=True,
         calibration_role='realistic small Claude-Code/coding-agent anchor'),
    dict(alias='qwen3.6-35b', repo='unsloth/Qwen3.6-35B-A3B-GGUF', prefix='Qwen3.6-35B-A3B', native_ctx=262144,
         model_class='moe', attn_layers=10, kv_heads=2, head_dim=256, total_params_b=35.0, active_params_b=3.0,
         # KNL Flat-mode shortlist: safe conventional full-HBM, comfortable IQ3, edge-fit higher quality, then one Q4 spill/control.
         knl_mcdram_priority=('UD-Q3_K_S','UD-IQ3_S','UD-Q4_K_M')),
    dict(alias='nemotron-3.5', repo='unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF', prefix='NVIDIA-Nemotron-3.5-Lightning-30B-A3B', native_ctx=262144,
         model_class='moe', attn_layers=6, kv_heads=2, head_dim=128, total_params_b=30.0, active_params_b=3.0),
    dict(alias='qwen3.8-27b', repo='unsloth/Qwen3.8-27B-GGUF', prefix='Qwen3.8-27B', native_ctx=262144,
         model_class='dense', attn_layers=16, kv_heads=4, head_dim=256, total_params_b=27.0, active_params_b=27.0,
         # Dense 27B KNL reality-check: Q3_K_M is the highest simple conventional
         # current Unsloth quant with comfortable 16-GiB MCDRAM headroom.
         knl_mcdram_priority=('Q3_K_M','Q3_K_S','UD-Q3_K_XL')),
    dict(alias='qwen3.6-27b', repo='unsloth/Qwen3.6-27B-GGUF', prefix='Qwen3.6-27B', native_ctx=262144,
         model_class='dense', attn_layers=16, kv_heads=4, head_dim=256, total_params_b=27.0, active_params_b=27.0),
]

# Preference within a representative quant family. Primary auto mode no longer
# benchmarks every remote quant; it selects one useful representative per family.
CPU_QUANT_PREFERENCE = [
    'Q6_K', 'UD-Q6_K_XL', 'UD-Q6_K',
    'Q5_K_M', 'UD-Q5_K_M', 'UD-Q5_K_XL', 'Q5_K_S', 'UD-Q5_K_S',
    'Q4_K_M', 'UD-Q4_K_M', 'UD-Q4_K_XL', 'Q4_K_S', 'UD-Q4_K_S', 'Q4_1', 'Q4_0',
    'Q3_K_M', 'UD-Q3_K_M', 'Q3_K_S', 'UD-Q3_K_S', 'UD-Q3_K_XL',
    'UD-IQ3_M', 'IQ3_M', 'UD-IQ3_S', 'IQ3_S', 'UD-IQ3_XS', 'IQ3_XS', 'UD-IQ3_XXS', 'IQ3_XXS',
    'MXFP4_MOE', 'UD-IQ4_NL', 'IQ4_NL', 'UD-IQ4_XS', 'IQ4_XS',
    'Q8_0', 'UD-Q8_K_XL',
]

KV_BYTES_PER_32 = {'f32': 128, 'f16': 64, 'bf16': 64, 'q8_0': 34, 'q5_0': 22, 'q4_0': 18}


def _read_json(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _meminfo() -> tuple[int, int]:
    vals: dict[str, int] = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            m = re.search(r'(\d+)', v)
            if m:
                vals[k] = int(m.group(1)) * 1024
    except Exception:
        pass
    return vals.get('MemTotal', 0), vals.get('MemAvailable', 0)


def _model_scope() -> set[str] | None:
    raw = os.environ.get('LAB_MODELS', 'all').strip()
    if not raw or raw.lower() == 'all':
        return None
    aliases = {x.strip().lower() for x in re.split(r'[,\s]+', raw) if x.strip()}
    return aliases


def _wanted_quant_set() -> set[str] | None:
    raw = os.environ.get('LAB_WEIGHT_QUANTS', 'auto').strip()
    if not raw or raw.lower() in ('auto', 'all'):
        return None
    return {x.strip().lower() for x in re.split(r'[,\s]+', raw) if x.strip()}


def manual_models() -> list[tuple[str, Path]]:
    ans: list[tuple[str, Path]] = []
    for key, name in [('MODEL_Q36', 'qwen36'), ('MODEL_Q38', 'qwen38'), ('MODEL_NEMO', 'nemotron'), ('MODEL', 'model')]:
        p = os.environ.get(key)
        if p and Path(p).is_file():
            ans.append((name, Path(p).resolve()))
    extra = os.environ.get('EXTRA_MODELS_FILE')
    if extra and Path(extra).is_file():
        for line in Path(extra).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            a = line.split('|')
            if len(a) >= 2 and Path(a[1]).is_file():
                ans.append((a[0], Path(a[1]).resolve()))
    seen: set[str] = set(); out: list[tuple[str, Path]] = []
    for name, p in ans:
        if str(p) not in seen:
            seen.add(str(p)); out.append((name, p))
    return out


def registered_records() -> list[dict[str, Any]]:
    obj = _read_json(REGISTRY, {}) or {}
    rows = obj.get('models', []) if isinstance(obj, dict) else []
    return [r for r in rows if isinstance(r, dict) and r.get('path') and Path(r['path']).is_file()]


def configured_models() -> list[tuple[str, Path]]:
    # Manual paths remain supported as additive controls; auto is default unless
    # explicitly disabled.  De-duplicate by resolved file path.
    rows = manual_models()
    if os.environ.get('LAB_AUTO_MODELS', '1') != '0':
        scope = _model_scope()
        for r in registered_records():
            alias = str(r.get('alias') or str(r.get('name','')).split('--',1)[0]).lower()
            if scope is not None and alias not in scope:
                continue
            if scope is None and alias in CALIBRATION_ALIASES and os.environ.get('LAB_INCLUDE_CALIBRATION','0') != '1':
                continue
            rows.append((r['name'], Path(r['path'])))
    seen: set[str] = set(); out: list[tuple[str, Path]] = []
    for name, p in rows:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k); out.append((name, p.resolve()))
    return out


def _hf_url(repo: str) -> str:
    return f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true&expand=false&limit=1000"


def fetch_tree(repo: str, *, opener=urllib.request.urlopen) -> list[dict[str, Any]]:
    """Fetch the HF tree using metadata only. Follows RFC Link rel=next pages."""
    url = _hf_url(repo)
    token = os.environ.get('HF_TOKEN', '')
    items: list[dict[str, Any]] = []
    pages = 0
    while url:
        req = urllib.request.Request(url, headers={'User-Agent': 'cpu-llama-lab/5.4'})
        if token:
            req.add_header('Authorization', f'Bearer {token}')
        with opener(req, timeout=45) as r:
            data = json.load(r)
            link = r.headers.get('Link', '')
        if not isinstance(data, list):
            raise RuntimeError(f'unexpected Hugging Face tree response for {repo}')
        items.extend(data); pages += 1
        nxt = None
        for part in link.split(','):
            if 'rel="next"' in part:
                m = re.search(r'<([^>]+)>', part)
                if m:
                    nxt = urllib.parse.urljoin(url, m.group(1))
        url = nxt
    if not items:
        raise RuntimeError(f'empty Hugging Face tree metadata for {repo}')
    return items


def group_inventory(items: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Group one unsharded GGUF or all shards into a quant candidate."""
    groups: dict[str, dict[str, Any]] = {}
    pre = prefix.lower() + '-'
    for it in items:
        path = str(it.get('path') or it.get('rfilename') or '')
        base = path.rsplit('/', 1)[-1]
        low = base.lower()
        if not low.endswith('.gguf'):
            continue
        if any(x in low for x in ('mmproj', 'imatrix', 'mtp', 'dflash', 'eagle')):
            continue
        stem = base[:-5]
        if not stem.lower().startswith(pre):
            continue
        q = stem[len(prefix) + 1:]
        q = re.sub(r'-\d{5}-of-\d{5}$', '', q, flags=re.I)
        if not q:
            continue
        size = it.get('size')
        if size is None and isinstance(it.get('lfs'), dict):
            size = it['lfs'].get('size')
        try:
            size = int(size or 0)
        except Exception:
            size = 0
        key = q.lower()
        g = groups.setdefault(key, {'quant': q, 'size_bytes': 0, 'parts': 0, 'files': [], 'file_sizes': {}})
        g['size_bytes'] += size
        g['parts'] += 1
        g['files'].append(path)
        g['file_sizes'][path] = size
    for g in groups.values():
        g['files'].sort()
    return sorted(groups.values(), key=lambda x: (-int(x['size_bytes']), str(x['quant']).lower()))


def inventory_for(meta: dict[str, Any], *, refresh: bool = False) -> list[dict[str, Any]]:
    INV_DIR.mkdir(parents=True, exist_ok=True)
    p = INV_DIR / (meta['repo'].replace('/', '__').replace(':', '__') + '.json')
    max_age = int(os.environ.get('LAB_HF_INVENTORY_MAX_AGE', '21600'))
    obj = _read_json(p, {}) or {}
    if not refresh and obj.get('inventory') and time.time() - float(obj.get('timestamp', 0)) <= max_age:
        return obj['inventory']
    items = fetch_tree(meta['repo'])
    inv = group_inventory(items, meta['prefix'])
    if not inv:
        raise RuntimeError(f"no matching model GGUFs for {meta['repo']} prefix {meta['prefix']}")
    p.write_text(json.dumps({'repo': meta['repo'], 'prefix': meta['prefix'], 'timestamp': time.time(), 'inventory': inv}, indent=2) + '\n')
    return inv


def _kv_bytes(meta: dict[str, Any], ctx: int, kv: str) -> int:
    b = KV_BYTES_PER_32.get(kv.lower())
    if b is None:
        raise ValueError(f'unknown KV type {kv}')
    return int(ctx) * int(meta['attn_layers']) * int(meta['kv_heads']) * int(meta['head_dim']) * (b + b) // 32


def _budget() -> dict[str, int]:
    total, avail = _meminfo()
    reserve = max(4 * 2**30, total // 20 if total else 0, int(os.environ.get('LAB_MODEL_RESERVE_MIB', '0')) * 2**20)
    overhead = int(os.environ.get('LAB_MODEL_PREFLIGHT_OVERHEAD_MIB', '0')) * 2**20
    return {'mem_total_bytes': total, 'mem_available_bytes': avail, 'reserve_bytes': reserve, 'overhead_bytes': overhead,
            'usable_bytes': max(0, avail - reserve - overhead)}


def _kv_fit_type(plan: dict[str, Any]) -> str:
    kvs = [str(x).lower() for x in plan.get('kv_types', ['q4_0'])]
    known = [x for x in kvs if x in KV_BYTES_PER_32]
    return min(known, key=lambda x: KV_BYTES_PER_32[x]) if known else 'q4_0'


def _quant_norm(q: str) -> str:
    q=q.upper().replace('UD-','')
    return q


def _quant_family(q: str) -> str:
    q=_quant_norm(q)
    if 'MXFP4' in q: return 'mxfp4'
    if 'IQ4' in q: return 'iq4'
    if 'Q4' in q: return 'q4'
    if 'IQ3' in q: return 'iq3'
    if 'Q3' in q: return 'q3'
    if 'IQ2' in q: return 'iq2'
    if 'Q2' in q: return 'q2'
    if 'Q5' in q: return 'q5'
    if 'Q6' in q: return 'q6'
    if 'Q8' in q: return 'q8'
    if 'BF16' in q or 'F16' in q: return 'f16'
    return 'other'


def _quant_quality_score(q: str) -> float:
    """Heuristic only for search ordering; the agent-quality gate is authoritative."""
    x=_quant_norm(q)
    fam=_quant_family(x)
    base={'f16':16.0,'q8':8.0,'q6':6.0,'q5':5.0,'q4':4.0,'mxfp4':4.05,'iq4':4.15,'q3':3.0,'iq3':3.15,'q2':2.0,'iq2':2.15,'other':1.0}.get(fam,1.0)
    # Within a nominal bit class, favor quality-retaining variants without making
    # the heuristic more important than actual local agent-quality results.
    if '_XL' in x: base += 0.28
    elif '_L' in x: base += 0.20
    elif '_M' in x: base += 0.14
    elif '_S' in x: base += 0.04
    if 'K_M' in x: base += 0.08
    if 'K_S' in x: base += 0.03
    return base


def _mcdram_budget() -> dict[str,int]:
    hw=_read_json(ROOT/'hardware.json',{}) or {}
    m=(hw.get('memory',{}) or {}).get('mcdram',{}) or {}
    visible=int(m.get('visible_bytes') or 0)
    reserve=max(0,int(os.environ.get('LAB_KNL_MCDRAM_RESERVE_MIB','1024')))*2**20
    near=max(0,int(os.environ.get('LAB_KNL_MCDRAM_NEAR_MIB','768')))*2**20
    return {'visible_bytes':visible,'reserve_bytes':reserve,'comfortable_bytes':max(0,visible-reserve),'near_bytes':visible+near}


def _annotate_mcdram(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    hb=_mcdram_budget(); vis=hb['visible_bytes']; comfortable=hb['comfortable_bytes']; near=hb['near_bytes']
    out=[]
    for g in rows:
        r=dict(g); w=int(r.get('size_bytes') or 0)
        if not vis:
            cls='not-applicable'; head=0
        elif w <= comfortable:
            cls='comfortable'; head=vis-w
        elif w <= vis:
            cls='nominal'; head=vis-w
        elif w <= near:
            cls='near'; head=vis-w
        else:
            cls='oversize'; head=vis-w
        r.update({'mcdram_visible_bytes':vis,'mcdram_reserve_bytes':hb['reserve_bytes'],
                  'mcdram_headroom_bytes':head,'mcdram_fit_class':cls})
        out.append(r)
    return out


def _knl_mcdram_shortlist(meta: dict[str,Any], rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Small KNL HBM shortlist: capacity boundary + kernel-family diversity.

    Default policy intentionally avoids same-bit spam. Qwen3.6 gets one
    conventional Q3, one IQ3, one Q4 spill/quality control, then at most one
    genuinely different format. Edge Q3 duplicates such as Q3_K_M are opt-in.
    """
    if not rows: return rows
    hw=_read_json(ROOT/'hardware.json',{}) or {}
    if (hw.get('cpu',{}) or {}).get('class') != 'knl' or _mcdram_budget()['visible_bytes'] <= 0:
        return rows
    if _wanted_quant_set() is not None or meta.get('calibration_only'):
        return rows
    limit=max(1,int(os.environ.get('LAB_KNL_MCDRAM_QUANT_BUDGET','4')))
    allow_edge_dup=os.environ.get('LAB_KNL_EDGE_DUPLICATE_Q3','0')=='1'
    byq={str(x['quant']).lower():x for x in rows}
    picked=[]; seen=set(); used_fams=set()
    def add(x,reason,allow_same=False):
        if not x or id(x) in seen or len(picked)>=limit: return
        fam=_quant_family(str(x['quant']))
        if fam in used_fams and not allow_same: return
        y=dict(x); y['selection_reason']=reason; picked.append(y); seen.add(id(x)); used_fams.add(fam)
    for q in meta.get('knl_mcdram_priority',()):
        add(byq.get(str(q).lower()),f'KNL MCDRAM priority: {q}')
    def best(classes,fams=None):
        xs=[x for x in rows if x.get('mcdram_fit_class') in classes and id(x) not in seen]
        if fams is not None: xs=[x for x in xs if _quant_family(str(x['quant'])) in fams]
        xs=[x for x in xs if _quant_family(str(x['quant'])) not in used_fams]
        return max(xs,key=lambda x:(_quant_quality_score(str(x['quant'])),-int(x['size_bytes'])),default=None)
    # Prefer a genuinely different full/near-HBM format before another Q3 variant.
    add(best({'comfortable','nominal','near'},{'mxfp4','iq4','q4','q2','iq2'}),'different-format HBM-boundary surprise candidate')
    # Ensure one higher-quality spill control if Q4 was not already explicitly selected.
    hb=_mcdram_budget(); max_ratio=float(os.environ.get('LAB_KNL_MCDRAM_CONTROL_MAX_RATIO','1.45'))
    spill=[x for x in rows if x.get('mcdram_fit_class') in ('near','oversize') and int(x['size_bytes']) <= hb['visible_bytes']*max_ratio and id(x) not in seen and _quant_family(str(x['quant'])) in {'q4','iq4','mxfp4'} and _quant_family(str(x['quant'])) not in used_fams]
    add(max(spill,key=lambda x:(_quant_quality_score(str(x['quant'])),-int(x['size_bytes'])),default=None),'higher-quality partial-MCDRAM spill/control')
    # Optional experiment: allow one second conventional Q3 right at the HBM edge.
    if allow_edge_dup and len(picked)<limit:
        edge=[x for x in rows if _quant_family(str(x['quant']))=='q3' and id(x) not in seen and x.get('mcdram_fit_class') in ('nominal','near')]
        x=max(edge,key=lambda x:(_quant_quality_score(str(x['quant'])),-abs(int(x.get('mcdram_headroom_bytes') or 0))),default=None)
        if x:
            y=dict(x);y['selection_reason']='opt-in edge-fit conventional Q3 duplicate';picked.append(y);seen.add(id(x))
    # Fill remaining slots with new low-bit families only; Q6/Q8/F16 are never
    # silently pulled into the KNL short-context search.
    for x in sorted(rows,key=lambda x:(_quant_quality_score(str(x['quant'])),-int(x['size_bytes'])),reverse=True):
        if _quant_family(str(x['quant'])) not in {'q2','iq2','q3','iq3','q4','iq4','mxfp4'}: continue
        add(x,'low-bit representative fallback')
    return picked


def _representative_shortlist(meta: dict[str,Any], rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """One representative per useful quant family on ordinary CPUs.

    Avoid an exhaustive quant cross-product. High-bit Q6/Q8/F16 controls are
    opt-in because the lab's target is an interactive coding agent, not a
    maximum-quality throughput-agnostic benchmark.
    """
    if not rows or meta.get('calibration_only') or _wanted_quant_set() is not None:
        return rows
    hw=_read_json(ROOT/'hardware.json',{}) or {}
    if (hw.get('cpu',{}) or {}).get('class')=='knl' and _mcdram_budget()['visible_bytes']>0:
        return rows
    limit=max(1,int(os.environ.get('LAB_REP_QUANT_BUDGET','4')))
    include_high=os.environ.get('LAB_INCLUDE_HIGH_BIT_CONTROLS','0')=='1'
    pref=['q3','iq3','q4']
    if str(meta.get('model_class','')).lower()=='moe': pref+=['mxfp4','iq4','q5']
    else: pref+=['q5','iq4','mxfp4']
    if include_high: pref+=['q6','q8']
    picked=[]
    for fam in pref:
        xs=[x for x in rows if _quant_family(str(x['quant']))==fam]
        if not xs: continue
        ordered=order_quants(xs)
        x=ordered[0]
        y=dict(x);y['selection_reason']=f'representative {fam} quant family';picked.append(y)
        if len(picked)>=limit: break
    if not picked:
        picked=rows[:limit]
    return picked

def order_quants(inv: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {q.lower(): i for i, q in enumerate(CPU_QUANT_PREFERENCE)}
    return sorted(inv, key=lambda g: (rank.get(str(g['quant']).lower(), 10000), -_quant_quality_score(str(g['quant'])), -int(g['size_bytes']), str(g['quant']).lower()))


def select_inventory(meta: dict[str, Any], inv: list[dict[str, Any]], plan: dict[str, Any], budget: dict[str, int]) -> list[dict[str, Any]]:
    min_ctx = int(os.environ.get('LAB_MODEL_MIN_CTX', '8192'))
    depths = sorted({int(x) for x in plan.get('depths', []) if int(x) > 0})
    if not depths:
        depths = [512, 2048, 8192, 16384, 32768, 65536]
    native = int(meta.get('native_ctx', max(depths)))
    cap = min(native, max(depths))
    fit_kv = _kv_fit_type(plan)
    wanted = _wanted_quant_set()
    fixed = str(meta.get('fixed_quant') or '').lower()
    selected: list[dict[str, Any]] = []
    for g in order_quants(inv):
        q = str(g['quant'])
        if fixed and os.environ.get('LAB_CALIBRATION_ALL_QUANTS','0') != '1' and q.lower() != fixed:
            continue
        if wanted is not None and q.lower() not in wanted:
            continue
        w = int(g.get('size_bytes') or 0)
        if w <= 0:
            continue
        need_min = w + _kv_bytes(meta, min_ctx, fit_kv)
        if need_min > budget['usable_bytes']:
            continue
        max_ctx = 0
        for d in depths:
            if d <= cap and w + _kv_bytes(meta, d, fit_kv) <= budget['usable_bytes']:
                max_ctx = d
        if max_ctx < min_ctx:
            continue
        r = dict(g)
        r.update({'fit_kv': fit_kv, 'min_ctx': min_ctx, 'max_planned_ctx': max_ctx,
                  'estimated_min_bytes': need_min, 'estimated_max_bytes': w + _kv_bytes(meta, max_ctx, fit_kv)})
        selected.append(r)
    selected=_annotate_mcdram(selected)
    selected=_knl_mcdram_shortlist(meta, selected)
    return _representative_shortlist(meta, selected)

def plan_models(*, refresh: bool = False) -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    hw = _read_json(ROOT / 'hardware.json', {}) or {}
    plan = _read_json(ROOT / 'plan.json', {}) or {}
    budget = _budget()
    scope = _model_scope()
    families = []
    for meta0 in CATALOG:
        meta = dict(meta0)
        alias = meta['alias'].lower()
        if scope is not None and alias not in scope:
            continue
        if scope is None and meta.get('calibration_only') and os.environ.get('LAB_INCLUDE_CALIBRATION','0') != '1':
            continue
        try:
            inv = inventory_for(meta, refresh=refresh)
            selected = select_inventory(meta, inv, plan, budget)
            families.append({**meta, 'inventory_count': len(inv), 'selected': selected})
        except Exception as e:
            # One remote family must not destroy an otherwise useful unattended
            # night. Preserve the failure in the plan and continue; if every
            # family fails/no candidate fits, ensure_auto_models still fails.
            families.append({**meta, 'inventory_count': 0, 'selected': [],
                             'planning_error': f'{type(e).__name__}: {e}'})
    obj = {'version': 1, 'catalog_rev': CATALOG_REV, 'scope': sorted(scope) if scope is not None else ('all+calibration' if os.environ.get('LAB_INCLUDE_CALIBRATION','0') == '1' else 'primary'), 'timestamp': time.time(), 'hardware_class': hw.get('cpu', {}).get('class'),
           'budget': budget, 'min_ctx': int(os.environ.get('LAB_MODEL_MIN_CTX', '8192')),
           'weight_quant_spec': os.environ.get('LAB_WEIGHT_QUANTS', 'auto'), 'mcdram_budget': _mcdram_budget(), 'families': families}
    PLAN_FILE.write_text(json.dumps(obj, indent=2) + '\n')
    return obj


def _curl_download(repo: str, remote: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Reuse an already complete file.  Exact remote size is checked by caller.
    part = dest.with_name(dest.name + '.part')
    url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(remote, safe='/')}?download=true"
    cmd = ['curl', '-L', '--fail', '--retry', '5', '--retry-delay', '2', '--connect-timeout', '30',
           '--speed-limit', '1024', '--speed-time', '120', '-C', '-', '-o', str(part)]
    token = os.environ.get('HF_TOKEN', '')
    if token:
        cmd += ['-H', f'Authorization: Bearer {token}']
    cmd += [url]
    shown = [x if not x.startswith('Authorization: Bearer ') else 'Authorization: Bearer <redacted>' for x in cmd[:-1]]
    print('+', ' '.join(shlex_quote(x) for x in shown), '<HF_URL>')
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f'curl failed ({rc}) downloading {repo}/{remote}')
    part.replace(dest)


def shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


def _download_group(meta: dict[str, Any], g: dict[str, Any]) -> Path:
    qdir = MODELS_DIR / meta['alias'] / re.sub(r'[^A-Za-z0-9_.-]+', '_', str(g['quant']))
    qdir.mkdir(parents=True, exist_ok=True)
    files = list(g['files'])
    if not files:
        raise RuntimeError(f"empty file list for {meta['alias']} {g['quant']}")
    # For sharded GGUFs the first shard is the llama.cpp entry point.
    first: Path | None = None
    remaining_total = int(g['size_bytes'])
    sizes = g.get('file_sizes', {})
    for remote in files:
        dest = qdir / Path(remote).name
        expected = int(sizes.get(remote) or 0)
        if dest.exists() and expected > 0 and dest.stat().st_size != expected:
            dest.unlink()
        if not dest.exists():
            _curl_download(meta['repo'], remote, dest)
        if expected > 0 and dest.stat().st_size != expected:
            raise RuntimeError(f'download size mismatch for {meta["alias"]}:{g["quant"]} file {remote}')
        if first is None:
            first = dest
    actual = sum((qdir / Path(x).name).stat().st_size for x in files if (qdir / Path(x).name).exists())
    if remaining_total > 0 and actual != remaining_total:
        # A stale/truncated final file is unsafe; remove only this quant's files so
        # a rerun can resume/re-download them cleanly.
        for remote in files:
            p = qdir / Path(remote).name
            if p.exists():
                p.unlink()
        raise RuntimeError(f"download size mismatch for {meta['alias']}:{g['quant']}: got {actual}, expected {remaining_total}")
    assert first is not None
    return first


def _required_download_bytes(obj: dict[str, Any]) -> int:
    need = 0
    for fam in obj.get('families', []):
        for g in fam.get('selected', []):
            qdir = MODELS_DIR / fam['alias'] / re.sub(r'[^A-Za-z0-9_.-]+', '_', str(g['quant']))
            have = sum((qdir / Path(x).name).stat().st_size for x in g['files'] if (qdir / Path(x).name).exists())
            need += max(0, int(g['size_bytes']) - have)
    return need


def ensure_auto_models(*, refresh: bool = False) -> list[tuple[str, Path]]:
    if os.environ.get('LAB_AUTO_MODELS', '1') == '0':
        ms = manual_models()
        if not ms:
            raise RuntimeError('LAB_AUTO_MODELS=0 and no manual MODEL_* / EXTRA_MODELS_FILE entries are configured')
        return ms
    obj = plan_models(refresh=refresh)
    count = sum(len(f.get('selected', [])) for f in obj['families'])
    if count == 0:
        b = obj['budget']
        raise RuntimeError(f"automatic model planner found no GGUF quant that fits >= {obj['min_ctx']} tokens; MemAvailable={b['mem_available_bytes']/2**30:.1f} GiB usable={b['usable_bytes']/2**30:.1f} GiB")
    need = _required_download_bytes(obj)
    free = shutil.disk_usage(MODELS_DIR).free
    if need > free:
        raise RuntimeError(f'automatic GGUF set needs {need/2**30:.1f} GiB additional disk but only {free/2**30:.1f} GiB is free below LAB_ROOT')
    print(f"Auto model plan: {count} fitting model/quant candidate(s); additional download={need/2**30:.1f} GiB; model root={MODELS_DIR}")
    rows: list[dict[str, Any]] = []
    for fam in obj['families']:
        if not fam['selected']:
            why = fam.get('planning_error') or f"no quant satisfies the >= {obj['min_ctx']} fit floor"
            print(f"SKIP model family {fam['alias']}: {why}")
            continue
        for g in fam['selected']:
            name = f"{fam['alias']}--{g['quant']}"
            print(f"MODEL {name}: {int(g['size_bytes'])/2**30:.2f} GiB, estimated max sweep ctx={g['max_planned_ctx']}, fit KV={g['fit_kv']}")
            try:
                path = _download_group(fam, g)
            except Exception as e:
                print(f"SKIP MODEL {name}: download failed after retries: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            rows.append({'name': name, 'alias': fam['alias'], 'quant': g['quant'], 'repo': fam['repo'], 'prefix': fam['prefix'],
                         'path': str(path.resolve()), 'files': g['files'], 'size_bytes': g['size_bytes'], 'parts': g['parts'],
                         'min_ctx': g['min_ctx'], 'max_planned_ctx': g['max_planned_ctx'], 'fit_kv': g['fit_kv'],
                         'mcdram_fit_class': g.get('mcdram_fit_class'), 'mcdram_visible_bytes': g.get('mcdram_visible_bytes',0),
                         'mcdram_headroom_bytes': g.get('mcdram_headroom_bytes',0), 'selection_reason': g.get('selection_reason','')})
    if not rows:
        raise RuntimeError('automatic model downloads produced no usable GGUF after retries')
    reg = {'version': 1, 'catalog_rev': CATALOG_REV, 'scope': obj.get('scope'), 'timestamp': time.time(), 'auto': True, 'models': rows, 'plan_file': str(PLAN_FILE)}
    REGISTRY.write_text(json.dumps(reg, indent=2) + '\n')
    with TSV_FILE.open('w') as f:
        f.write('name\talias\tquant\trepo\tsize_bytes\tmin_ctx\tmax_planned_ctx\tmcdram_fit\tmcdram_headroom_bytes\tselection_reason\tpath\n')
        for r in rows:
            f.write(f"{r['name']}\t{r['alias']}\t{r['quant']}\t{r['repo']}\t{r['size_bytes']}\t{r['min_ctx']}\t{r['max_planned_ctx']}\t{r.get('mcdram_fit_class','')}\t{r.get('mcdram_headroom_bytes',0)}\t{r.get('selection_reason','')}\t{r['path']}\n")
    return configured_models()


def ensure_models(*, refresh: bool = False) -> list[tuple[str, Path]]:
    # Existing complete registry avoids network calls on every phase. Refresh can
    # be explicitly requested or occurs when there is no usable registry.
    manual = manual_models()
    if os.environ.get('LAB_AUTO_MODELS', '1') == '0':
        if not manual:
            raise RuntimeError('no models configured')
        return manual
    current = registered_records()
    if current and not refresh:
        desired_min = int(os.environ.get('LAB_MODEL_MIN_CTX', '8192'))
        scope = _model_scope()
        desired_scope = sorted(scope) if scope is not None else ('all+calibration' if os.environ.get('LAB_INCLUDE_CALIBRATION','0') == '1' else 'primary')
        robj = _read_json(REGISTRY, {}) or {}
        if int(robj.get('catalog_rev') or 0) == CATALOG_REV and robj.get('scope') == desired_scope and all(int(r.get('min_ctx') or 0) == desired_min for r in current):
            rows = configured_models()
            if rows:
                return rows
    return ensure_auto_models(refresh=refresh)
