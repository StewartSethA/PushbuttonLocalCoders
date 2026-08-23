#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path

ROOT=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab')))
OUT=ROOT/'results'; alias=(sys.argv[1] if len(sys.argv)>1 else 'granite4-350m').lower()
limit=max(4,int(os.environ.get('LAB_PLATFORM_SHORTLIST_SIZE','8')))
plan=json.load(open(ROOT/'plan.json')); hw=json.load(open(ROOT/'hardware.json'))
rows=[]
summary_files=[]
for ptr in sorted(OUT.glob('latest-quick-rank*.json')):
    try:
        o=json.load(open(ptr)); q=Path(str(o.get('path','')))
        if q.exists(): summary_files.append(q)
    except Exception: pass
if not summary_files:
    qs=sorted(OUT.glob('quick-summary-*-rank*.json'),key=lambda x:x.stat().st_mtime,reverse=True)
    if qs: summary_files=[qs[0]]
for p in summary_files:
    try: xs=json.load(open(p))
    except Exception: continue
    for r in xs:
        if len(r)<8: continue
        mn,bn,pn,th,pp,tg,ps,ts=r[:8]
        if not (str(mn).lower()==alias or str(mn).lower().startswith(alias+'--')): continue
        if ps!='ok' or ts!='ok' or float(pp)<=0 or float(tg)<=0: continue
        rows.append({'model':mn,'build':bn,'policy_name':pn,'threads':int(th),'pp_tps':float(pp),'tg_tps':float(tg),
                     'pp_clock_mhz':(r[8] if len(r)>8 else None),'tg_clock_mhz':(r[9] if len(r)>9 else None),
                     'pp_clock_status':(r[10] if len(r)>10 else None),'tg_clock_status':(r[11] if len(r)>11 else None)})
if not rows: raise SystemExit(f'no successful quick rows for platform calibration model {alias}')
maxpp=max(x['pp_tps'] for x in rows); maxtg=max(x['tg_tps'] for x in rows)
for x in rows:
    x['score']=(max(x['pp_tps']/maxpp,1e-12)**0.35)*(max(x['tg_tps']/maxtg,1e-12)**0.65)
rows.sort(key=lambda x:(x['score'],x['tg_tps'],x['pp_tps']),reverse=True)
bykey={(x['build'],x['policy_name'],x['threads']):x for x in rows}
phys=int(plan.get('hardware_summary',{}).get('phys_cores') or 1)
policy_threads={str(x.get('name')):int(x.get('default_threads') or phys) for x in plan.get('numa_policies',[])}
mandatory=[]
def add_key(b,p,t=None,reason='mandatory scientific control'):
    if t is None: t=policy_threads.get(str(p),phys)
    k=(b,p,int(t)); x=bykey.get(k)
    # A mandatory *comparison* is only useful downstream if it actually ran.
    # Failed controls remain explicit in the raw results; do not spend larger
    # models retrying a load/crash path already falsified by the 350M map.
    if x is None: return
    y=dict(x);y['reason']=reason; mandatory.append(y)
cls=(hw.get('cpu',{}) or {}).get('class')
polnames={x['name'] for x in plan.get('numa_policies',[])}; bnames={x['name'] for x in plan.get('builds',[])}
if cls=='knl':
    for b,p,r in [
        ('knl-generic-avx2-norepack','knl-ddr-strict','generic AVX2/no-repack load-safe DDR control'),
        ('knl-base','knl-ddr-strict','stock-kernel DDR control'),
        ('knl-base','knl-mcdram-strict','stock-kernel strict-MCDRAM control'),
        ('knl-base-norepack','knl-mcdram-strict','repack-off MCDRAM control'),
        ('knl-combo','knl-mcdram-strict','optimized-kernel MCDRAM hypothesis'),
        ('knl-combo-norepack','knl-mcdram-strict','optimized-kernel repack interaction'),
        ('knl-combo','knl-mcdram-preferred','HBM-preferred spill/tiering control')]:
        if b in bnames and p in polnames: add_key(b,p,reason=r)
else:
    for b,p,r in [('native','disabled-mmap','native+repack+mmap control'),
                  ('native-norepack','disabled-mmap','native no-repack control'),
                  ('native','disabled-no-mmap','native load-mode=none control')]:
        if b in bnames and p in polnames: add_key(b,p,reason=r)
    sock=next((p for p in sorted(polnames) if p.startswith('socket0-local')),None)
    if sock and 'native' in bnames: add_key('native',sock,reason='whole-socket locality control')
chosen=[]; seen=set()
for x in mandatory+rows:
    k=(x['build'],x['policy_name'],int(x['threads']))
    if k in seen: continue
    y=dict(x); y.setdefault('reason','top 350M PP/TG platform score'); chosen.append(y); seen.add(k)
    if len(chosen)>=limit: break
labver=(Path(__file__).resolve().parents[1]/'VERSION').read_text().strip()
obj={'version':2,'lab_version':labver,'source_model':alias,'hardware_class':cls,'generated_at':time.time(),'score':'0.35*PP + 0.65*TG geometric normalized','configs':chosen}
p=ROOT/'calibration'/'platform-shortlist.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')
print(p)
for x in chosen:
    pc=x.get('pp_clock_mhz');tc=x.get('tg_clock_mhz');clk=f" clk={pc:.0f}/{tc:.0f}MHz" if isinstance(pc,(int,float)) and isinstance(tc,(int,float)) else ''
    print(f"  {x['build']:24s} {x['policy_name']:24s} t={x['threads']:4d} pp={x.get('pp_tps',0):8.2f} tg={x.get('tg_tps',0):8.2f}{clk} {x.get('reason','')}")
