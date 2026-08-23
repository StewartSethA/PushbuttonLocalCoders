#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys
from pathlib import Path
PKG=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PKG))
from lib.clock import synthetic_load_probe
ROOT=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab')))
HW=json.load(open(ROOT/'hardware.json'))
c=HW.get('cpu',{})
phys=max(1,int(c.get('sockets',1))*int(c.get('cores_per_socket',1)))
obj=synthetic_load_probe(phys,c.get('nominal_base_mhz'))
ROOT.mkdir(parents=True,exist_ok=True)
(ROOT/'clock-check.json').write_text(json.dumps(obj,indent=2)+'\n')
ref=obj.get('reference_mhz');ratio=obj.get('loaded_to_reference_ratio');
print('loaded_clock_mhz=%s reference_mhz=%s reference_kind=%s ratio=%s status=%s samples=%s' % (
      ('%.0f'%obj['loaded_mhz']) if isinstance(obj.get('loaded_mhz'),(int,float)) else 'n/a',
      ('%.0f'%ref) if isinstance(ref,(int,float)) else 'n/a',obj.get('reference_kind','n/a'),
      ('%.3f'%ratio) if isinstance(ratio,(int,float)) else 'n/a',obj.get('status','unavailable'),obj.get('sample_ticks',0)))
raise SystemExit(2 if obj.get('status')=='throttled' and os.environ.get('LAB_CLOCK_ENFORCE','1')!='0' else 0)
