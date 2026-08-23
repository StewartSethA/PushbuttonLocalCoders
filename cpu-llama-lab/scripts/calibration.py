#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, statistics, time
from pathlib import Path

ROOT=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab')))
OUT=ROOT/'results'; MODELS=ROOT/'models'
ALIASES=('granite4-350m','qwen3-0.6b','granite4-1b','granite4.1-3b')
# User's prior Ollama/KNL Granite 4.1 3B observations. These are anchors, not
# claims that current llama.cpp will reproduce Ollama exactly.
OLLAMA_GRANITE=[
    # Same KNL / Granite 4.1 3B A/B. PP uses first repetition only because later
    # Ollama repetitions reused prompt state. TG repetitions were stable.
    {'ctx':635,'pp_tps':27.59,'tg_ddr':3.139,'tg_mcdram':9.190},
    {'ctx':1090,'pp_tps':28.54,'tg_ddr':3.063,'tg_mcdram':8.583},
    {'ctx':2037,'pp_tps':27.19,'tg_ddr':2.941,'tg_mcdram':7.495},
    {'ctx':2958,'pp_tps':26.31,'tg_ddr':2.772,'tg_mcdram':6.651},
]

def load_rows():
    rows=[]; started=0.0
    try: started=float(json.loads((ROOT/'calibration'/'session.json').read_text()).get('started_at',0))
    except Exception: pass
    for p in OUT.glob('bench-rank*.jsonl'):
        for line in p.read_text(errors='ignore').splitlines():
            try:r=json.loads(line)
            except Exception:continue
            if started and float(r.get('timestamp',0) or 0) < started: continue
            m=str(r.get('model',''))
            if any(m.startswith(a+'--') or m==a for a in ALIASES): rows.append(r)
    return rows

def tps(r):
    x=r.get('tps')
    if isinstance(x,(int,float)): return float(x)
    raw=r.get('raw',[]); raw=[raw] if isinstance(raw,dict) else raw
    vals=[]
    for q in raw if isinstance(raw,list) else []:
        if isinstance(q,dict):
            for k in ('avg_ts','tokens_per_second','t_s'):
                if isinstance(q.get(k),(int,float)): vals.append(float(q[k]));break
    return max(vals) if vals else 0.0

def fit_latency(field:str, ctx:int=8192):
    # Linear token latency (seconds/token) versus occupied context is a local
    # extrapolator only; the current harness measures the real 8K point.
    xs=[x['ctx'] for x in OLLAMA_GRANITE]; ys=[1/x[field] for x in OLLAMA_GRANITE]
    n=len(xs); sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    b=(n*sxy-sx*sy)/(n*sxx-sx*sx); a=(sy-b*sx)/n
    return 1/(a+b*ctx),a,b

def historical_anchor():
    ddr8,ad,bd=fit_latency('tg_ddr'); hbm8,ah,bh=fit_latency('tg_mcdram')
    return statistics.mean(x['pp_tps'] for x in OLLAMA_GRANITE),ddr8,hbm8,ad,bd,ah,bh

def main():
    rows=load_rows()
    if not rows: raise SystemExit('no calibration benchmark rows found; run ./cpu-llama-lab.sh calibrate first')
    try: session=json.loads((ROOT/'calibration'/'session.json').read_text())
    except Exception: session={}
    mode=str(session.get('mode') or 'full')
    fast_mode=(mode=='fast')
    display_aliases=('granite4-350m','granite4.1-3b') if fast_mode else ALIASES
    report_depth=2048 if fast_mode else 8192
    ok=[r for r in rows if r.get('status')=='ok']
    starts=[float(r.get('timestamp',0))-float(r.get('wall_s',0) or 0) for r in rows if r.get('timestamp')]
    ends=[float(r.get('timestamp',0)) for r in rows if r.get('timestamp')]
    elapsed=max(ends)-min(starts) if starts and ends else sum(float(r.get('wall_s',0) or 0) for r in rows)
    pp0,tg8_ddr,tg8_hbm,ad,bd,ah,bh=historical_anchor()
    hw=json.loads((ROOT/'hardware.json').read_text()) if (ROOT/'hardware.json').exists() else {}
    cpu=hw.get('cpu',{}) if isinstance(hw,dict) else {}; platform_label=f"{cpu.get('class','cpu')} / {cpu.get('generation','unknown')} / {cpu.get('name','unknown')}"
    try: preflight_clock=json.loads((ROOT/'preflight.json').read_text()).get('loaded_clock_probe',{})
    except Exception: preflight_clock={}
    lines=['# CPU agent/coding calibration report','',f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}',f'Platform: **{platform_label}**','',
           '## Historical KNL/Ollama Granite 4.1 3B anchors','',
           '| occupied context | first-run PP tok/s | DDR TG tok/s | MCDRAM TG tok/s |','|---:|---:|---:|---:|']
    if preflight_clock:
        lm=preflight_clock.get('loaded_mhz'); rm=preflight_clock.get('reference_mhz'); ratio=preflight_clock.get('loaded_to_reference_ratio')
        msg='Preflight loaded clock: **%.0f MHz**; reference: **%.0f MHz** (%s); ratio: **%.3f**; status: **%s**.' % (float(lm or 0),float(rm or 0),preflight_clock.get('reference_kind',''),float(ratio or 0),preflight_clock.get('status',''))
        lines[5:5]=['## Loaded-clock validation','',msg,'']
    for x in OLLAMA_GRANITE: lines.append(f"| {x['ctx']} | {x['pp_tps']:.2f} | {x['tg_ddr']:.3f} | {x['tg_mcdram']:.3f} |")
    lines += ['',f'Historical mean PP: **{pp0:.2f} tok/s**.',
              f'Historical latency-fit extrapolation to 8192: DDR **{tg8_ddr:.2f} tok/s**, MCDRAM **{tg8_hbm:.2f} tok/s** (heuristics, not guarantees).',
              f'Historical fitted fixed TG latency: DDR **{ad*1000:.1f} ms/token**, MCDRAM **{ah*1000:.1f} ms/token**; context slopes are {bd*1000:.5f} vs {bh*1000:.5f} ms/context-token.','',
              ('## Current llama.cpp fast decision pass' if fast_mode else '## Current llama.cpp calibration ladder'),'',
              (f'| family | best PP @ {report_depth//1024}K | best TG @ {report_depth//1024}K | best TG @ 16K | benchmark rows |'),'|---|---:|---:|---:|---:|']
    family_best={}
    for fam in display_aliases:
        rr=[r for r in ok if str(r.get('model','')).startswith(fam+'--') or str(r.get('model',''))==fam]
        pp=max([tps(r) for r in rr if r.get('kind')=='pp_interval' and int(r.get('depth') or 0)==report_depth] or [0])
        tg=max([tps(r) for r in rr if r.get('kind')=='tg' and int(r.get('depth') or 0)==report_depth] or [0])
        tg16=max([tps(r) for r in rr if r.get('kind')=='tg' and int(r.get('depth') or 0)==16384] or [0])
        family_best[fam]=(pp,tg,tg16,len(rr))
        lines.append(f'| {fam} | {pp:.2f} | {tg:.2f} | {tg16:.2f} | {len(rr)} |')
    gp,gt,_,_=family_best['granite4.1-3b']
    lines += ['',f'Observed calibration elapsed wall time: **{elapsed/60:.1f} minutes**.']
    if gp>0:
        if fast_mode: lines.append(f'Granite 3B PP at the 2K decision depth: **{gp:.2f} tok/s**.')
        else: lines.append(f'Granite PP speedup versus the prior Ollama mean at short context: **{gp/pp0:.2f}x** (8K vs historical ~1–3.5K, so this is deliberately conservative).')
    if gt>0:
        if fast_mode: lines.append(f'Granite 3B TG at the 2K decision depth: **{gt:.2f} tok/s**.')
        else: lines.append(f'Granite TG at 8K versus the historical MCDRAM 8K extrapolator: **{gt/tg8_hbm:.2f}x**.')
    try: tp=json.loads((ROOT/'calibration'/'thread-probe.json').read_text())
    except Exception: tp={}
    if isinstance(tp,dict) and tp.get('rows'):
        lines += ['', '## One-model thread/SMT sanity probe', '',
                  'This is intentionally the only thread-count sweep. Main experiments use one worker per physical core available to the selected NUMA locality; a surprising SMT result is recorded but not multiplied across every model/quant/build.', '',
                  '| threads | PP512 | TG2K | PP clock MHz | TG clock MHz | relative composite |','|---:|---:|---:|---:|---:|---:|']
        for x in tp.get('rows',[]): lines.append(f"| {x.get('threads','')} | {float(x.get('pp_tps',0)):.2f} | {float(x.get('tg_tps',0)):.2f} | {float(x.get('pp_clock_mhz') or 0):.0f} | {float(x.get('tg_clock_mhz') or 0):.0f} | {float(x.get('relative_score',0)):.3f} |")
        lines.append(f"Physical-core default: **{tp.get('physical_core_default')}** threads; best observed: **{tp.get('best_threads')}**; material surprise: **{bool(tp.get('surprise'))}**.")
    try: kvp=json.loads((ROOT/'calibration'/'kv-probe.json').read_text())
    except Exception: kvp={}
    if isinstance(kvp,dict) and kvp.get('rows'):
        lines += ['', '## Decode/KV capability probe', '',
                  'F16 is the neutral thread/platform baseline. Q8/Q4 KV are separate ISA-sensitive optimizations; a SIGILL is recorded as unsupported rather than misreported as zero TG.', '',
                  '| KV | TG @ <=2K | clock MHz | clock status | status | rc | signal |','|---|---:|---:|---|---|---:|---:|']
        for x in kvp.get('rows',[]): lines.append(f"| {x.get('kv','')} | {float(x.get('tg_tps',0)):.2f} | {float(x.get('clock_mhz') or 0):.0f} | {x.get('clock_status','')} | {x.get('status','')} | {x.get('rc','')} | {x.get('signal','') if x.get('signal') is not None else ''} |")
    # Calibration-only agent diagnostic is deliberately non-blocking: tiny models
    # can fail every repair task and the timing calibration still completes.
    diag_path=ROOT/'calibration'/'agent-diagnostic.json'
    diag={}
    try: diag=json.loads(diag_path.read_text())
    except Exception: pass
    attempts=diag.get('attempts',[]) if isinstance(diag,dict) else []
    if attempts:
        lines += ['', '## Agent/repository-repair diagnostic', '',
                  'This is diagnostic-only during `calibrate`; failures do not abort calibration. Production `overnight`/`full` use the hard quality + responsiveness gate.', '',
                  '| family | quant | repo tasks passed | median task s | accepted by normal gate | reason |','|---|---|---:|---:|---|---|']
        for a in attempts:
            c=a.get('candidate',{}) if isinstance(a,dict) else {}; fam=str(a.get('family') or c.get('family') or '')
            if fam not in display_aliases: continue
            tasks=a.get('tasks',[]) if isinstance(a.get('tasks'),list) else []; passed=sum(1 for t in tasks if t.get('pass'))
            med=a.get('median_task_s'); medtxt=f'{float(med):.1f}' if isinstance(med,(int,float)) else ''
            lines.append(f"| {fam} | {c.get('quant','')} | {passed}/{len(tasks)} | {medtxt} | {bool(a.get('accepted'))} | {a.get('reason','')} |")
    shortlist={}
    try: shortlist=json.loads((ROOT/'calibration'/'platform-shortlist.json').read_text())
    except Exception: pass
    cfgs=shortlist.get('configs',[]) if isinstance(shortlist,dict) else []
    if cfgs:
        lines += ['', '## Platform ablation shortlist', '',
                  'The 350M rung maps the full platform once; later calibration/primary models reuse this shortlist plus mandatory scientific controls.', '',
                  '| build | memory/NUMA policy | threads | PP512 | TG2K | PP/TG clock MHz | rationale |','|---|---|---:|---:|---:|---:|---|']
        for c in cfgs:
            pc=float(c.get('pp_clock_mhz') or 0);tc=float(c.get('tg_clock_mhz') or 0); lines.append(f"| {c.get('build','')} | {c.get('policy_name','')} | {c.get('threads','')} | {float(c.get('pp_tps',0)):.2f} | {float(c.get('tg_tps',0)):.2f} | {pc:.0f}/{tc:.0f} | {c.get('reason','')} |")
    if fast_mode:
        promise={}
        try: promise=json.loads((ROOT/'calibration'/'promise.json').read_text())
        except Exception: pass
        if promise:
            lines += ['', '## Promotion decision', '',
                      f"Representative model: **{promise.get('model','')}** at 2K; PP **{float(promise.get('pp_tps') or 0):.2f} tok/s**; TG **{float(promise.get('tg_tps') or 0):.2f} tok/s**.",
                      f"Promoted for deeper work: **{bool(promise.get('promoted'))}**; >=10 tok/s responsive target at 2K: **{bool(promise.get('responsive_at_2k'))}**.",
                      f"Projected standalone 8K characterization: **{float(promise.get('estimated_8k_s') or 0)/60:.1f} min**; 16K: **{float(promise.get('estimated_16k_s') or 0)/60:.1f} min**."]
        lines += ['', '## How to use this', '',
                  'Default calibration is deliberately a decision pass, not a depth study: Granite 350M maps the platform causally and Granite 4.1 3B checks whether scaling remains usable through 2K.',
                  'Only promoted primary candidates should spend time at 8K/16K/32K/64K. Use `calibrate --full` only when the legacy factorial evidence is explicitly wanted.',
                  '', 'Run `./cpu-llama-lab.sh models-plan` next for the expensive families.']
    else:
        lines += ['', '## How to use this', '',
                  'The ladder is intentionally agent/coding-oriented: Granite 350M is the dense tiny coding/tool-use floor, Qwen3 0.6B tests sub-1B reasoning/tool use, Granite 1B is a stronger small dense coding-agent rung, and Granite 4.1 3B is the realistic small-agent anchor tied to the prior KNL/Ollama data.',
                  'Exactly the same fixed GGUF quants and context/sweep semantics are used on KNL, Xeon, EPYC and Threadripper; only the CPU-appropriate build/ISA/NUMA ablations differ.',
                  '', 'Run `./cpu-llama-lab.sh models-plan` next for the expensive families, then `./cpu-llama-lab.sh estimate` to combine the live candidate count with this calibration.']
    p=ROOT/'calibration'/'CALIBRATION.md'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('\n'.join(lines)+'\n')
    j={'platform':platform_label,'mode':mode,'decision_depth':report_depth,'elapsed_s':elapsed,'loaded_clock_probe':preflight_clock,'ollama_granite':OLLAMA_GRANITE,'ollama_pp_mean':pp0,'ollama_tg8_ddr_estimate':tg8_ddr,'ollama_tg8_mcdram_estimate':tg8_hbm,'families':{k:{'pp8':v[0],'tg8':v[1],'tg16':v[2],'rows':v[3]} for k,v in family_best.items()}}
    (p.parent/'calibration.json').write_text(json.dumps(j,indent=2)+'\n')
    print(p)

if __name__=='__main__': main()
