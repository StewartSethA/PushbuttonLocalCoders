#!/usr/bin/env python3
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
PKG=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PKG))
import lib.models as lm
from lib.models import group_inventory, select_inventory, order_quants, CATALOG, CALIBRATION_ALIASES

def assert_(x,msg):
    if not x: raise AssertionError(msg)

def main():
    items=[
      {'path':'Qwen3.6-35B-A3B-Q4_K_M.gguf','size':18*2**30},
      {'path':'Qwen3.6-35B-A3B-Q8_0-00001-of-00002.gguf','size':18*2**30},
      {'path':'Qwen3.6-35B-A3B-Q8_0-00002-of-00002.gguf','size':19*2**30},
      {'path':'mmproj-BF16.gguf','size':2**30},
      {'path':'imatrix_unsloth.gguf','size':10_000_000},
      {'path':'README.md','size':1000},
    ]
    inv=group_inventory(items,'Qwen3.6-35B-A3B')
    d={x['quant']:x for x in inv}
    assert_(set(d)=={'Q4_K_M','Q8_0'},f'bad inventory {d}')
    assert_(d['Q8_0']['parts']==2 and d['Q8_0']['size_bytes']==37*2**30,'shard aggregation failed')
    assert_(len(d['Q8_0']['file_sizes'])==2,'per-file size metadata missing')

    meta=dict(alias='qwen3.6-35b',native_ctx=262144,attn_layers=10,kv_heads=2,head_dim=256)
    plan={'depths':[512,2048,8192,32768,65536,131072,262144],'kv_types':['q4_0','q8_0','f16']}
    old=os.environ.get('LAB_MODEL_MIN_CTX'); os.environ['LAB_MODEL_MIN_CTX']='65536'
    try:
      # 30 GiB usable accepts Q4 but rejects Q8 at the 64K hard floor.
      sel=select_inventory(meta,inv,plan,{'usable_bytes':30*2**30})
      assert_([x['quant'] for x in sel]==['Q4_K_M'],f'fit gate failed: {sel}')
      assert_(sel[0]['max_planned_ctx']>=65536,'must record a usable max context')
      # 64 GiB usable could fit both, but ordinary primary search intentionally
      # omits high-bit Q8 unless explicitly requested.
      sel2=select_inventory(meta,inv,plan,{'usable_bytes':64*2**30})
      assert_([x['quant'] for x in sel2]==['Q4_K_M'],f'representative high-bit pruning failed: {sel2}')
    finally:
      if old is None: os.environ.pop('LAB_MODEL_MIN_CTX',None)
      else: os.environ['LAB_MODEL_MIN_CTX']=old
    # KNL Flat/MCDRAM: collapse same-bit spam into a four-candidate, capacity-aware
    # shortlist. Q3_K_S and IQ3_S are comfortable distinct 3-bit/kernel families;
    # Q4_K_M is the partial-HBM quality control. Same-family Q3 edge duplicates
    # are opt-in rather than consuming the default budget.
    with tempfile.TemporaryDirectory() as td:
      old_root=lm.ROOT; old_wq=os.environ.pop('LAB_WEIGHT_QUANTS',None); old_min=os.environ.get('LAB_MODEL_MIN_CTX')
      try:
        lm.ROOT=Path(td); os.environ['LAB_MODEL_MIN_CTX']='8192'
        (lm.ROOT/'hardware.json').write_text(__import__('json').dumps({'cpu':{'class':'knl'},'memory':{'mcdram':{'visible_bytes':16*2**30,'visible_nodes':[1]}}}))
        qitems=[]
        for q,gib in [('UD-IQ3_XXS',12.3),('UD-IQ3_S',12.76),('UD-Q3_K_S',14.34),('UD-Q3_K_M',15.46),('UD-Q3_K_XL',15.65),('UD-Q4_K_M',20.58),('UD-Q5_K_M',25.0)]:
          qitems.append({'path':f'Qwen3.6-35B-A3B-{q}.gguf','size':int(gib*2**30)})
        qinv=group_inventory(qitems,'Qwen3.6-35B-A3B')
        qmeta=dict(next(x for x in CATALOG if x['alias']=='qwen3.6-35b'))
        qplan={'depths':[512,2048,8192,16384,32768,65536],'kv_types':['q4_0','q8_0','f16']}
        qsel=select_inventory(qmeta,qinv,qplan,{'usable_bytes':64*2**30})
        qs=[x['quant'] for x in qsel]
        assert_(qs==['UD-Q3_K_S','UD-IQ3_S','UD-Q4_K_M'],f'bad KNL HBM shortlist: {[(x["quant"],x.get("mcdram_fit_class")) for x in qsel]}')
        cls={x['quant']:x['mcdram_fit_class'] for x in qsel}
        assert_(cls['UD-Q3_K_S']=='comfortable' and cls['UD-IQ3_S']=='comfortable','comfortable HBM fits misclassified')
        assert_(cls['UD-Q4_K_M']=='oversize','spill control misclassified')
        assert_(all(x.get('selection_reason') for x in qsel),'KNL shortlist rationale must be recorded')
        # Generic CPUs use one representative per useful quant family rather than
        # crossing every remote GGUF. Q6/high-bit controls are opt-in.
        (lm.ROOT/'hardware.json').write_text(__import__('json').dumps({'cpu':{'class':'epyc'},'memory':{'mcdram':{'visible_bytes':0,'visible_nodes':[]}}}))

        # Qwen3.8-27B KNL ballpark must prefer a simple conventional quant that
        # actually fits inside 16 GiB MCDRAM with useful headroom. Current
        # Unsloth Q3_K_M is ~13.8 GB decimal (~12.85 GiB).
        q38meta=dict(next(x for x in CATALOG if x['alias']=='qwen3.8-27b'))
        q38items=[
          {'path':'Qwen3.8-27B-Q3_K_S.gguf','size':12_600_000_000},
          {'path':'Qwen3.8-27B-Q3_K_M.gguf','size':13_800_000_000},
          {'path':'Qwen3.8-27B-Q4_0.gguf','size':16_100_000_000},
        ]
        q38inv=group_inventory(q38items,'Qwen3.8-27B')
        (lm.ROOT/'hardware.json').write_text(__import__('json').dumps({'cpu':{'class':'knl'},'memory':{'mcdram':{'visible_bytes':16*2**30,'visible_nodes':[1]}}}))
        q38sel=select_inventory(q38meta,q38inv,qplan,{'usable_bytes':64*2**30})
        q38={x['quant']:x for x in q38sel}
        assert_('Q3_K_M' in q38 and q38['Q3_K_M']['mcdram_fit_class']=='comfortable',f'Qwen3.8 Q3_K_M must be a comfortable KNL HBM fit: {q38sel}')
        assert_(q38['Q3_K_M']['mcdram_headroom_bytes']>2*2**30,'Qwen3.8 Q3_K_M should leave >2 GiB raw MCDRAM headroom')

        generic=select_inventory(qmeta,qinv,qplan,{'usable_bytes':64*2**30})
        gf=[lm._quant_family(x['quant']) for x in generic]
        assert_(len(generic)<=4 and len(gf)==len(set(gf)),f'generic representative pruning failed: {[(x["quant"],lm._quant_family(x["quant"])) for x in generic]}')
        assert_('q6' not in gf,'high-bit Q6 should be opt-in for ordinary primary search')
      finally:
        lm.ROOT=old_root
        if old_wq is not None: os.environ['LAB_WEIGHT_QUANTS']=old_wq
        else: os.environ.pop('LAB_WEIGHT_QUANTS',None)
        if old_min is None: os.environ.pop('LAB_MODEL_MIN_CTX',None)
        else: os.environ['LAB_MODEL_MIN_CTX']=old_min

    c={x['alias']:x for x in CATALOG}
    assert_(tuple(CALIBRATION_ALIASES)==('granite4-350m','qwen3-0.6b','granite4-1b','granite4.1-3b'),'agent calibration ladder drifted')
    assert_(c['granite4-350m']['fixed_quant']=='Q4_K_M' and c['granite4-350m']['attn_layers']==28 and c['granite4-350m']['model_class']=='dense','bad Granite 350M calibration metadata')
    assert_(c['qwen3-0.6b']['fixed_quant']=='Q4_0' and c['qwen3-0.6b']['kv_heads']==8 and c['qwen3-0.6b']['head_dim']==128,'bad Qwen3 0.6B calibration metadata')
    assert_(c['granite4-1b']['fixed_quant']=='Q4_K_M' and c['granite4-1b']['attn_layers']==40 and c['granite4-1b']['model_class']=='dense','bad Granite 1B calibration metadata')
    print('PASS test_models')
if __name__=='__main__': main()
