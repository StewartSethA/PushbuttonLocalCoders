#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, os, tempfile, textwrap
from pathlib import Path

PKG=Path(__file__).resolve().parents[1]

def assert_(x,msg):
    if not x: raise AssertionError(msg)

def fake_server_text():
    return textwrap.dedent("""\
        #!/usr/bin/env python3
        import json,sys,signal,ctypes
        try: ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        except Exception: pass
        from http.server import BaseHTTPRequestHandler,HTTPServer
        port=int(sys.argv[sys.argv.index('--port')+1]); cache=[]
        class H(BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def sendj(self,o):
                b=json.dumps(o).encode(); self.send_response(200); self.send_header('content-type','application/json'); self.send_header('content-length',str(len(b))); self.end_headers(); self.wfile.write(b)
            def do_GET(self): self.sendj({'status':'ok'})
            def do_POST(self):
                global cache
                n=int(self.headers.get('content-length','0')); o=json.loads(self.rfile.read(n) or b'{}')
                if self.path=='/tokenize': return self.sendj({'tokens':[11,12,13,14]})
                if self.path=='/completion':
                    prompt=[int(x) for x in o.get('prompt',[])]; common=0
                    for a,b in zip(cache,prompt):
                        if a!=b: break
                        common+=1
                    pn=len(prompt)-common; ng=int(o.get('n_predict',0)); cache=prompt+([99]*ng)
                    return self.sendj({'timings':{'cache_n':common,'prompt_n':pn,'prompt_ms':pn/2 if pn else 0,'prompt_per_second':2000 if pn else 0,'predicted_n':ng,'predicted_ms':ng*10,'predicted_per_second':100 if ng else 0}})
                self.sendj({})
        HTTPServer(('127.0.0.1',port),H).serve_forever()
    """)

def main():
    with tempfile.TemporaryDirectory(dir=os.environ.get('LAB_ROOT') or None) as td:
        root=Path(td); (root/'results').mkdir()
        (root/'plan.json').write_text(json.dumps({
            'builds':[{'name':'knl-base'},{'name':'knl-base-norepack'},{'name':'knl-generic-avx2-norepack'},{'name':'knl-combo-norepack'},{'name':'knl-combo'}], 'threads':[68], 'thread_probe':{'model':'granite4-350m','counts':[34,68,136,272],'default':68}, 'numa_policies':[{'name':'disabled','llama':[],'prefix':[],'default_threads':68},{'name':'knl-ddr-strict','llama':[],'prefix':[],'default_threads':68},{'name':'knl-mcdram-strict','llama':[],'prefix':[],'default_threads':68,'requires_mcdram_fit':True},{'name':'knl-mcdram-preferred','llama':[],'prefix':[],'default_threads':68}],
            'depths':[512,2048,8192,16384], 'kv_types':['q4_0'],
            'search':{'quick_depths':[512,2048],'screen_depths':[512,2048,8192],'deep_depths':[16384],
                      'screen_cap':8192,'deep_cap':16384,'prune_fraction':0.5,'candidate_finalists':1,'screen_reps':1,'final_reps':2}
        }))
        (root/'hardware.json').write_text(json.dumps({'cpu':{'class':'knl','cores_per_socket':68,'sockets':1},'numa':{'count':1,'nodes':[{'id':0,'cpulist':'0-67'}]}}))
        old=os.environ.get('LAB_ROOT'); os.environ['LAB_ROOT']=str(root)
        for k in ('LAB_SERVER_START_TIMEOUT_S','LAB_PP_INTERVAL_TIMEOUT_S','LAB_TG_ENDPOINT_TIMEOUT_S','LAB_QUICK_TIMEOUT_S','LAB_BATCH_TIMEOUT_S'): os.environ[k]='5'
        try:
            spec=importlib.util.spec_from_file_location('sweep_test',PKG/'scripts/sweep.py')
            m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            assert_(m.SCREEN_CAP==8192,'screen cap must default to 8K')
            assert_(m.DEEP_CAP==16384,'test plan deep cap not honored')
            assert_(max(m.SCREEN_DEPTHS)==8192 and min(m.DEEP_DEPTHS)>8192,'screen/deep boundary broken')
            assert_(m.cache_reuse_ok(8192,{'cache_n':8192}),'exact cache reuse should pass')
            assert_(not m.cache_reuse_ok(8192,{'cache_n':4000},8),'large cache loss must fail closed')
            mt=m._completion_metrics({'tokens_cached':512,'prompt_progress':{'processed':2048,'cache':512},'timings':{'prompt_ms':750.0,'prompt_per_second':2048.0}},2048)
            assert_(mt['cache_n']==512 and mt['prompt_n']==1536,'top-level tokens_cached/prompt_progress compatibility fallback failed')

            bd=root/'builds'/'knl-base'/'bin'; bd.mkdir(parents=True)
            fake=bd/'llama-server'; fake.write_text(fake_server_text()); fake.chmod(0o755)
            bench=bd/'llama-bench'; bench.write_text("#!/usr/bin/env python3\nimport json,sys\nif '-tb' in sys.argv: raise SystemExit(2)\nif '-ctk' in sys.argv and sys.argv[sys.argv.index('-ctk')+1]=='q4_0': raise SystemExit(132)\nprint(json.dumps([{'avg_ts':123.0}]))\n"); bench.chmod(0o755)
            model=root/'fake.gguf'; model.write_bytes(b'x')
            probe_cmd=m.cmd_for(bench,model,68,{'prefix':[],'llama':[]},['-p','512','-n','0'])
            assert_('-tb' not in probe_cmd,'llama-bench must never receive unsupported -tb/--threads-batch')
            assert_(m.QUICK_KV=='f16' and m.THREAD_PROBE_KV=='f16','neutral quick/thread baseline must default to F16 KV')
            assert_(m._fast_platform_pairs()==[('knl-base-norepack','knl-ddr-strict'),('knl-base-norepack','knl-mcdram-strict'),('knl-generic-avx2-norepack','knl-mcdram-strict'),('knl-combo-norepack','knl-mcdram-strict'),('knl-combo-norepack','knl-mcdram-preferred')],'fast KNL platform map must remain the five causal controls')
            bp=m._raw_pp_tg({'raw':[{'n_prompt':128,'n_gen':0,'n_depth':0,'avg_ts':40.0},{'n_prompt':0,'n_gen':32,'n_depth':512,'avg_ts':7.0}]})
            assert_(bp.get('pp128@0')==40.0 and bp.get('tg32@512')==7.0,'primary ballpark must preserve separate PP/TG/depth metrics from one llama-bench load')
            m.model_record_map=lambda:{'qwen3.8-27b--Q3_K_M':{'alias':'qwen3.8-27b','quant':'Q3_K_M','size_bytes':13_800_000_000,'mcdram_fit_class':'comfortable'}}
            q38pairs=m._primary_probe_pairs('qwen3.8-27b--Q3_K_M')
            assert_([x[:2] for x in q38pairs]==[('knl-base-norepack','knl-ddr-strict'),('knl-base-norepack','knl-mcdram-strict'),('knl-combo-norepack','knl-mcdram-strict')],'Qwen3.8 dense ballpark must be exactly DDR -> strict-HBM -> combo/no-repack by default')
            assert_('-tb' not in __import__('inspect').getsource(m.node_workers),'node-workers must not pass unsupported -tb to llama-bench')
            # Old-run failures must not contaminate current failure summaries.
            m.RESULT_INDEX['oldrelease|quick|pp|stale']={'run_key':'oldrelease|quick|pp|stale','phase':'quick','kind':'pp','status':'failed','rc':1}
            fs=m._failure_summary(); assert_(not fs['counts'].get('quick:pp:failed'),'failure summary leaked stale run rows')
            m.models=lambda:[('fake--Q4',model)]
            m.model_record_map=lambda:{'fake--Q4':{'alias':'fake','quant':'Q4','size_bytes':1,'mcdram_fit_class':'comfortable'}}
            boot=m.bootstrap_probe(); assert_(boot.get('pp_tps')==123.0 and boot.get('tg_tps')==123.0 and boot.get('baseline_kv')=='f16','bootstrap must prove both PP and F16 decode')
            m.prefix_smoke()
            m.thread_probe()
            m.kv_probe()
            kp=json.load(open(root/'calibration'/'kv-probe.json')); krows={x['kv']:x for x in kp['rows']}
            assert_(krows['f16']['status']=='ok' and krows['q8_0']['status']=='ok','F16/Q8 fake KV controls should execute')
            assert_(krows['q4_0']['status']=='failed' and krows['q4_0']['signal']==4,'Q4 fake SIGILL must be recorded as signal 4')
            tp=json.load(open(root/'calibration'/'thread-probe.json'))
            assert_(tp['counts']==[34,68,136,272] and tp['physical_core_default']==68,'KNL thread probe must sample 1/2x,1x,2x,4x only')
            assert_(len(tp['rows'])==4,'thread probe did not run exactly four KNL SMT points')
            assert_(all(x['tg_status']=='ok' and x['tg_tps']>0 for x in tp['rows']),'thread probe must not depend on Q4 KV support')
            smoke=[json.loads(x) for x in (root/'results'/'bench-rank0.jsonl').read_text().splitlines() if 'prefix-smoke' in x]
            spp=[x for x in smoke if x.get('kind')=='pp_interval']
            assert_(len(spp)>=2 and spp[-1].get('cache_n')==512 and spp[-1].get('prompt_n')==1536,'prefix smoke did not prove 512->2048 reuse')

            # Persistent shallow screen must process only suffixes: 0->512,
            # 512->2048, 2048->8192. No independent 0->N refills.
            m._progressive_branch('shallow','fake--Q4',model,'knl-base',root/'builds'/'knl-base','disabled',{'name':'disabled','llama':[],'prefix':[]},68,512,256,'q4_0',[512,2048,8192],reps=1)
            rows=[json.loads(x) for x in (root/'results'/'bench-rank0.jsonl').read_text().splitlines()]
            pp=[x for x in rows if x.get('phase')=='shallow' and x.get('model')=='fake--Q4' and x.get('kind')=='pp_interval']
            assert_([(x['interval_start'],x['depth'],x['prompt_n']) for x in pp]==[(0,512,512),(512,2048,1536),(2048,8192,6144)],f'bad suffix PP: {pp}')
            assert_(not any(int(x.get('depth') or 0)>8192 for x in rows if x.get('phase')=='shallow'),'shallow screen exceeded 8K')

            # A deep continuation reconstructs the 8K winner prefix once, then
            # measures only 8K->16K. It does not repeat the 0->8K measurement.
            m._progressive_branch('deep','fake--Q4',model,'knl-base',root/'builds'/'knl-base','disabled',{'name':'disabled','llama':[],'prefix':[]},68,512,256,'q4_0',[16384],initial_prefix=8192,reps=1)
            rows=[json.loads(x) for x in (root/'results'/'bench-rank0.jsonl').read_text().splitlines()]
            dpp=[x for x in rows if x.get('phase')=='deep' and x.get('model')=='fake--Q4' and x.get('kind')=='pp_interval']
            assert_(len(dpp)==1 and dpp[0]['interval_start']==8192 and dpp[0]['prompt_n']==8192,'deep PP must be 8K->16K suffix only')

        finally:
            if old is None: os.environ.pop('LAB_ROOT',None)
            else: os.environ['LAB_ROOT']=old

    # Fresh temp root for orchestration/winner selection, avoiding pre-seeded rows.
    with tempfile.TemporaryDirectory(dir=os.environ.get('LAB_ROOT') or None) as td:
        root=Path(td); (root/'results').mkdir()
        (root/'plan.json').write_text(json.dumps({
            'builds':[{'name':'knl-base'}], 'threads':[68], 'numa_policies':[{'name':'disabled','llama':[],'prefix':[]}],
            'depths':[512,2048,8192,16384], 'kv_types':['q4_0'],
            'search':{'quick_depths':[512,2048],'screen_depths':[512,2048,8192],'deep_depths':[16384],
                      'screen_cap':8192,'deep_cap':16384,'prune_fraction':0.5,'candidate_finalists':1,'screen_reps':1,'final_reps':1}
        }))
        (root/'hardware.json').write_text(json.dumps({'cpu':{'class':'knl','cores_per_socket':68,'sockets':1},'numa':{'count':1,'nodes':[{'id':0,'cpulist':'0-67'}]}}))
        old=os.environ.get('LAB_ROOT'); os.environ['LAB_ROOT']=str(root)
        for k in ('LAB_SERVER_START_TIMEOUT_S','LAB_PP_INTERVAL_TIMEOUT_S','LAB_TG_ENDPOINT_TIMEOUT_S','LAB_QUICK_TIMEOUT_S','LAB_BATCH_TIMEOUT_S'): os.environ[k]='5'
        try:
            spec=importlib.util.spec_from_file_location('sweep_e2e',PKG/'scripts/sweep.py')
            m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            bd=root/'builds'/'knl-base'/'bin'; bd.mkdir(parents=True)
            fake=bd/'llama-server'; fake.write_text(fake_server_text()); fake.chmod(0o755)
            bench=bd/'llama-bench'; bench.write_text("#!/usr/bin/env python3\nimport json,sys\nif '-tb' in sys.argv: raise SystemExit(2)\nif '-ctk' in sys.argv and sys.argv[sys.argv.index('-ctk')+1]=='q4_0': raise SystemExit(132)\nprint(json.dumps([{'avg_ts':123.0}]))\n"); bench.chmod(0o755)
            q4=root/'q4.gguf'; q6=root/'q6.gguf'; q4.write_bytes(b'x'); q6.write_bytes(b'x')
            m.models=lambda:[('family--Q4',q4),('family--Q6',q6)]
            m.model_record_map=lambda:{
                'family--Q4':{'alias':'family','quant':'Q4','max_planned_ctx':16384},
                'family--Q6':{'alias':'family','quant':'Q6','max_planned_ctx':16384},
            }
            m.quick(); m.tune_batches(); m.shallow()
            ws=m.family_winners()
            assert_(len(ws)==1 and ws[0]['family']=='family','exactly one winner per family required')
            m.deep()
            rows=[json.loads(x) for x in (root/'results'/'bench-rank0.jsonl').read_text().splitlines()]
            deep_models={x.get('model') for x in rows if x.get('phase')=='deep' and x.get('kind') in ('pp_interval','tg') and x.get('status')=='ok'}
            assert_(len(deep_models)==1,'more than one quant/config got a deep sweep')
            assert_(not any(int(x.get('depth') or 0)>8192 for x in rows if x.get('phase') in ('quick','batch-tune','shallow')),'pre-deep work exceeded 8K')
            assert_(any(int(x.get('depth') or 0)==16384 for x in rows if x.get('phase')=='deep'),'deep winner did not continue beyond 8K')
        finally:
            if old is None: os.environ.pop('LAB_ROOT',None)
            else: os.environ['LAB_ROOT']=old
    print('PASS test_sweep')

if __name__=='__main__': main()
