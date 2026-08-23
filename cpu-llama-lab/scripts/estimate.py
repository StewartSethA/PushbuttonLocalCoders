#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, statistics
from pathlib import Path
ROOT=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab')))
CAL=ROOT/'calibration'/'calibration.json'; MP=ROOT/'models'/'model-plan.json'; PLAN=ROOT/'plan.json'

def med(xs,default):
    xs=[float(x) for x in xs if float(x)>0]
    return statistics.median(xs) if xs else default

def rows():
    out=[]
    for p in (ROOT/'results').glob('bench-rank*.jsonl'):
        for line in p.read_text(errors='ignore').splitlines():
            try:r=json.loads(line)
            except Exception:continue
            if str(r.get('model','')).startswith('granite4.1-3b--') and r.get('status')=='ok': out.append(r)
    return out

def main():
    if not CAL.exists(): raise SystemExit('missing calibration.json; run ./cpu-llama-lab.sh calibrate first')
    if not MP.exists(): raise SystemExit('missing model-plan.json; run ./cpu-llama-lab.sh models-plan after calibration')
    cal=json.load(open(CAL)); mp=json.load(open(MP)); plan=json.load(open(PLAN)); rr=rows()
    # Per-operation Granite anchors from the exact same harness. Fallbacks come
    # from the user's historical Ollama Granite observations.
    quick={}
    for r in rr:
        if r.get('phase')=='quick':
            k=(r.get('model'),r.get('build'),r.get('policy_name'),r.get('threads')); quick[k]=quick.get(k,0)+float(r.get('wall_s',0) or 0)
    shallow={}
    for r in rr:
        if r.get('phase')=='shallow':
            k=(r.get('model'),r.get('build'),r.get('policy_name'),r.get('threads'),r.get('kv')); shallow[k]=shallow.get(k,0)+float(r.get('wall_s',0) or 0)
    deep={}
    for r in rr:
        if r.get('phase')=='deep':
            k=(r.get('model'),r.get('build'),r.get('policy_name'),r.get('threads'),r.get('kv')); deep[k]=deep.get(k,0)+float(r.get('wall_s',0) or 0)
    qsec=med(quick.values(), 512/26.7 + 128/4.0)
    bsec=med([r.get('wall_s',0) for r in rr if r.get('phase')=='batch-tune'],1024/26.7)
    ssec=med(shallow.values(), 420.0)
    dsec=med(deep.values(), 600.0)
    builds=len([b for b in plan.get('builds',[])]) or 1; pols=len(plan.get('numa_policies',[])) or 1; threads=len(plan.get('threads',[])) or 1; kvs=len(plan.get('kv_types',[])) or 1
    finalists=max(1,int(plan.get('search',{}).get('candidate_finalists',2)))
    world=max(1,int(os.environ.get('LAB_WORLD','1')))
    anchor_bytes=2.1*2**30
    totals=[]
    lines=['# Best-case runtime estimate','',f'Granite harness anchors: quick={qsec:.1f}s/config, batch={bsec:.1f}s/trial, shallow8K={ssec:.1f}s/branch, deep winner={dsec:.1f}s/branch.','',
           '| family | quant | GGUF GiB | active-scale | estimated hours |','|---|---|---:|---:|---:|']
    for fam in mp.get('families',[]):
        if fam.get('calibration_only'): continue
        total=float(fam.get('total_params_b') or 1); active=float(fam.get('active_params_b') or total)
        for g in fam.get('selected',[]):
            size=float(g.get('size_bytes') or 0); eff=size*(active/total)
            # Memory-streaming work should scale roughly with active weight bytes;
            # retain a 20% fixed-overhead floor so tiny extrapolations do not vanish.
            scale=max(0.20,(eff/anchor_bytes)**0.90) if eff>0 else 1.0
            quick_n=math.ceil(builds*pols*threads/world)
            batch_n=finalists*7/world
            shallow_n=finalists*kvs/world
            sec=scale*(quick_n*qsec + batch_n*bsec + shallow_n*ssec)
            # One deep branch is per family, not per quant. Charge it only to the
            # first selected quant in the family for total accounting.
            if g is fam.get('selected',[None])[0]: sec += scale*dsec/world
            totals.append(sec)
            lines.append(f"| {fam.get('alias')} | {g.get('quant')} | {size/2**30:.2f} | {scale:.2f}x | {sec/3600:.2f} |")
    total=sum(totals)
    lines += ['',f'**Estimated best-case aggregate benchmark wall: {total/3600:.1f} hours** on LAB_WORLD={world}.',
              '', 'This is a lower-bound planning estimate, not a completion promise. It scales the measured Granite harness cost by active-weight bytes; dense 27B models can be much slower, while 3B-active MoE models may track Granite more closely. Timeouts/pruning can reduce actual elapsed time.']
    p=ROOT/'calibration'/'ESTIMATE.md'; p.write_text('\n'.join(lines)+'\n'); print('\n'.join(lines)); print(f'\nWrote {p}')
if __name__=='__main__': main()
