#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys
from pathlib import Path
PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG))
from lib.models import configured_models, ensure_models, plan_models, PLAN_FILE, REGISTRY

def main():
    cmd=sys.argv[1] if len(sys.argv)>1 else 'ensure'
    if cmd=='plan':
        obj=plan_models(refresh=os.environ.get('LAB_REFRESH_MODELS','0')=='1')
        print(PLAN_FILE)
        total_count=0; total_bytes=0
        for fam in obj['families']:
            print(f"{fam['alias']}: discovered={fam['inventory_count']} fitting={len(fam['selected'])}")
            for q in fam['selected']:
                total_count += 1; total_bytes += int(q['size_bytes'])
                head=float(q.get('mcdram_headroom_bytes',0))/2**30
                hbm=q.get('mcdram_fit_class','')
                why=q.get('selection_reason','')
                extra=f' HBM={hbm} headroom={head:+.2f} GiB' if hbm and hbm!='not-applicable' else ''
                print(f"  {q['quant']:20s} {q['size_bytes']/2**30:7.2f} GiB max_ctx={q['max_planned_ctx']}{extra} {why}".rstrip())
        print(f"TOTAL fitting={total_count} payload={total_bytes/2**30:.2f} GiB (before reuse)")
        return 0
    if cmd=='list':
        for n,p in configured_models(): print(f'{n}\t{p}')
        return 0
    if cmd=='ensure':
        ms=ensure_models(refresh=os.environ.get('LAB_REFRESH_MODELS','0')=='1')
        print(REGISTRY)
        print(f'{len(ms)} configured model/quant file(s)')
        return 0
    raise SystemExit('usage: models.py [plan|ensure|list]')
if __name__=='__main__': raise SystemExit(main())
