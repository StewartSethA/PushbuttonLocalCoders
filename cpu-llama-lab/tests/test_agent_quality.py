#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, os, tempfile
from pathlib import Path

PKG=Path(__file__).resolve().parents[1]

def assert_(x,msg):
    if not x: raise AssertionError(msg)

def main():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); (root/'models').mkdir(); (root/'quality').mkdir()
        (root/'plan.json').write_text(json.dumps({
            'numa_policies':[{'name':'disabled','prefix':[],'llama':[]}]
        }))
        (root/'hardware.json').write_text(json.dumps({'cpu':{'class':'knl'}}))
        old=os.environ.get('LAB_ROOT'); os.environ['LAB_ROOT']=str(root)
        try:
            spec=importlib.util.spec_from_file_location('agent_quality_test',PKG/'scripts/agent_quality.py')
            m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

            assert_(m.MIN_PASS_RATE==1.0,'agent correctness gate should default to 100% on tiny deterministic tasks')
            assert_(m.MIN_PP_TPS==20.0 and m.MIN_TG_TPS==3.0,'responsiveness floors changed unexpectedly')
            assert_(m.sampling_for('qwen3-0.6b--Q4_0').get('temperature')==0.6 and m.sampling_for('qwen3-0.6b--Q4_0').get('top_p')==0.95,'Qwen agent gate must not use greedy decoding')
            assert_(m.sampling_for('granite4-1b--Q4_K_M').get('temperature')==0.0,'non-Qwen deterministic sampling changed unexpectedly')

            rows=[
                {'model':'fam--Q4_K_M','quant':'Q4_K_M','score':1.00,'tg_tps':10,'pp_tps':100},
                {'model':'fam--Q5_K_M','quant':'Q5_K_M','score':0.90,'tg_tps':9,'pp_tps':95},
                {'model':'fam--Q6_K','quant':'Q6_K','score':0.70,'tg_tps':7,'pp_tps':90},
                {'model':'fam--Q3_K_M','quant':'Q3_K_M','score':0.95,'tg_tps':9.5,'pp_tps':98},
            ]
            order=m.fallback_order(rows)
            assert_(order[0]['quant']=='Q4_K_M','fastest candidate must be tested first')
            assert_([x['quant'] for x in order[1:]]==['Q6_K','Q5_K_M','Q3_K_M'],f'higher-fidelity fallback order wrong: {order}')

            # Every fixture starts genuinely broken, and source tools cannot mutate tests.
            with tempfile.TemporaryDirectory() as td2:
                base=Path(td2)
                for name,files in m.TASKS:
                    d=m.make_task(base,name,files)
                    ok,_=m.run_tests(d)
                    assert_(not ok,f'fixture {name} unexpectedly passes before repair')
                    test=(d/'tests'/next((d/'tests').glob('test_*.py')).name)
                    before=test.read_text()
                    r=m.exec_tool(d,'write_file',{'path':str(test.relative_to(d)),'content':'tampered'})
                    assert_(r.startswith('DENIED') and test.read_text()==before,'test immutability guard failed')

            # Claude-Code absence is allowed only when not explicitly required; if present
            # and the smoke test fails, production gating must reject it (checked structurally).
            src=(PKG/'scripts/agent_quality.py').read_text()
            assert_("if CLAUDE_SMOKE and cs.get('available')" in src,'installed Claude Code smoke must be a hard integration gate')
            assert_("pp>=MIN_PP_TPS and tg>=MIN_TG_TPS" in src,'PP/TG responsiveness must be part of hard gate')
        finally:
            if old is None: os.environ.pop('LAB_ROOT',None)
            else: os.environ['LAB_ROOT']=old
    print('PASS test_agent_quality')

if __name__=='__main__': main()
