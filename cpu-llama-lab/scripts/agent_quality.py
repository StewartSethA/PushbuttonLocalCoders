#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, os, re, shutil, signal, socket, statistics, subprocess, sys, tempfile, time, urllib.error, urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.clock import ClockMonitor, monitor_call

PKG=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab')))
PLAN=json.load(open(ROOT/'plan.json'))
HW=json.load(open(ROOT/'hardware.json'))
MODELS=ROOT/'models'; OUT=ROOT/'quality'/'agent'; OUT.mkdir(parents=True,exist_ok=True); TMP=ROOT/'tmp'; TMP.mkdir(parents=True,exist_ok=True)
CTX=int(os.environ.get('LAB_AGENT_CTX','8192'))
MAX_TURNS=int(os.environ.get('LAB_AGENT_MAX_TURNS','8'))
TASK_TIMEOUT=int(os.environ.get('LAB_AGENT_TASK_TIMEOUT_S','180'))
MIN_PASS_RATE=float(os.environ.get('LAB_AGENT_MIN_PASS_RATE','1.0'))
MAX_MEDIAN=float(os.environ.get('LAB_AGENT_MAX_MEDIAN_TASK_S','90'))
MIN_PP_TPS=float(os.environ.get('LAB_AGENT_MIN_PP_TPS','20'))
MIN_TG_TPS=float(os.environ.get('LAB_AGENT_MIN_TG_TPS','3'))
MAX_CANDIDATES=max(1,int(os.environ.get('LAB_AGENT_MAX_CANDIDATES_PER_FAMILY','4')))
REQUIRE_CLAUDE=os.environ.get('LAB_REQUIRE_CLAUDE_CODE','0')=='1'
CLAUDE_SMOKE=os.environ.get('LAB_AGENT_CLAUDE_SMOKE','1')!='0'
DIAGNOSTIC=os.environ.get('LAB_AGENT_DIAGNOSTIC','0')=='1'
DIAGNOSTIC_OUTPUT=os.environ.get('LAB_AGENT_DIAGNOSTIC_OUTPUT','')


def jload(p:Path,default=None):
 try:return json.load(open(p))
 except Exception:return default

def registered()->dict[str,dict[str,Any]]:
 o=jload(MODELS/'registry.json',{}) or {}; return {str(x.get('name')):x for x in o.get('models',[]) if isinstance(x,dict) and x.get('name')}

def candidates_obj()->dict[str,Any]:
 p=MODELS/'family-candidates.json'; o=jload(p,{}) or {}
 if not o.get('families'): raise SystemExit('No family candidates. Run the <=8K screen first.')
 return o

def quant_quality(q:str)->float:
 s=q.upper().replace('UD-','')
 for pat,val in [(r'Q8',8.0),(r'Q6',6.0),(r'Q5',5.0),(r'(?:IQ4|Q4|MXFP4)',4.0),(r'(?:IQ3|Q3)',3.0),(r'(?:IQ2|Q2)',2.0),(r'(?:IQ1|Q1)',1.0)]:
  if re.search(pat,s): return val
 return 4.0

def fallback_order(rows:list[dict[str,Any]])->list[dict[str,Any]]:
 if not rows:return []
 speed=sorted(rows,key=lambda x:(float(x.get('score',0)),float(x.get('tg_tps',0)),float(x.get('pp_tps',0))),reverse=True)
 first=speed[0]; rest=[x for x in speed[1:] if x.get('model')!=first.get('model')]
 rest.sort(key=lambda x:(quant_quality(str(x.get('quant',''))),float(x.get('score',0))),reverse=True)
 return [first]+rest

def free_port()->int:
 with socket.socket() as s:s.bind(('127.0.0.1',0));return int(s.getsockname()[1])

def policy(name:str)->dict[str,Any]:
 for p in PLAN.get('numa_policies',[]):
  if p.get('name')==name:return p
 raise KeyError(name)

def server_cmd(c:dict[str,Any],model:Path,port:int)->list[str]:
 d=ROOT/'builds'/str(c['build']); exe=d/'bin/llama-server'
 if not exe.exists():raise FileNotFoundError(exe)
 p=policy(str(c['policy_name'])); th=int(c['threads']); b=int(c.get('batch') or 2048); ub=int(c.get('ubatch') or 512); kv=str(c.get('kv') or 'q4_0')
 args=[str(exe),'-m',str(model),'--alias','local-model','-c',str(CTX),'-t',str(th),'-tb',str(th),'-b',str(b),'-ub',str(ub),'-ctk',kv,'-ctv',kv,
       '--parallel','1','--cache-prompt','--no-context-shift','--jinja','--no-webui','--host','127.0.0.1','--port',str(port)]
 return list(p.get('prefix',[]))+args+list(p.get('llama',[]))

def start(c,model):
 port=free_port(); cmd=server_cmd(c,model,port); lp=OUT/f"server-{re.sub(r'[^A-Za-z0-9_.-]+','_',str(c['model']))}.log"; log=open(lp,'a')
 env=os.environ.copy()
 for k in ('OMP_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES','OPENBLAS_NUM_THREADS','GOTO_NUM_THREADS','MKL_NUM_THREADS','KMP_AFFINITY'):env.pop(k,None)
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 deadline=time.time()+int(os.environ.get('LAB_SERVER_START_TIMEOUT_S','300'))
 while time.time()<deadline:
  if proc.poll() is not None:break
  try:
   urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2).read();return proc,log,port,cmd
  except Exception:time.sleep(.5)
 stop(proc,log);return None,None,None,cmd

def stop(proc,log):
 if proc:
  try:os.killpg(proc.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:proc.wait(10)
  except subprocess.TimeoutExpired:
   try:os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   proc.wait()
 if log:log.close()

def post(port:int,path:str,obj:dict[str,Any],timeout:int)->dict[str,Any]:
 req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'content-type':'application/json','x-api-key':'local-key','anthropic-version':'2023-06-01'})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)

TASKS=[
 ('range-chunks', {'app/ranges.py':'''def chunk_ranges(n, size):\n    """Return half-open (start,end) chunks covering range(n)."""\n    if size <= 0:\n        raise ValueError("size")\n    return [(s, min(s + size, n - 1)) for s in range(0, n, size)]\n''',
                   'tests/test_ranges.py':'''import unittest\nfrom app.ranges import chunk_ranges\nclass T(unittest.TestCase):\n def test_basic(self): self.assertEqual(chunk_ranges(10,4),[(0,4),(4,8),(8,10)])\n def test_small(self): self.assertEqual(chunk_ranges(1,4),[(0,1)])\n def test_zero(self): self.assertEqual(chunk_ranges(0,4),[])\nif __name__=="__main__": unittest.main()\n'''}),
 ('cache-invalidation', {'app/store.py':'''class Store:\n def __init__(self): self.data={}; self.cache={}\n def set(self,key,value): self.data[key]=value\n def get(self,key):\n  if key not in self.cache: self.cache[key]=self.data.get(key)\n  return self.cache[key]\n''',
                        'tests/test_store.py':'''import unittest\nfrom app.store import Store\nclass T(unittest.TestCase):\n def test_update_visible(self):\n  s=Store(); s.set("x",1); self.assertEqual(s.get("x"),1); s.set("x",2); self.assertEqual(s.get("x"),2)\n def test_missing_then_set(self):\n  s=Store(); self.assertIsNone(s.get("x")); s.set("x",7); self.assertEqual(s.get("x"),7)\nif __name__=="__main__": unittest.main()\n'''}),
 ('nonmutating-merge', {'app/config.py':'''def merged(defaults, override):\n out=defaults\n for k,v in override.items():\n  if isinstance(v,dict) and isinstance(out.get(k),dict): out[k].update(v)\n  else: out[k]=v\n return out\n''',
                       'tests/test_config.py':'''import unittest\nfrom app.config import merged\nclass T(unittest.TestCase):\n def test_nested_no_mutation(self):\n  d={"db":{"host":"a","port":1},"debug":False}; r=merged(d,{"db":{"port":2}})\n  self.assertEqual(r,{"db":{"host":"a","port":2},"debug":False}); self.assertEqual(d,{"db":{"host":"a","port":1},"debug":False})\n def test_independent_nested(self):\n  d={"x":{"a":1}}; r=merged(d,{}); r["x"]["a"]=9; self.assertEqual(d["x"]["a"],1)\nif __name__=="__main__": unittest.main()\n'''}),
 ('env-parser', {'app/envparse.py':'''def parse_line(line):\n line=line.strip()\n if not line or line.startswith("#"): return None\n key,value=line.split("=")\n return key.strip(),value.strip()\n''',
                 'tests/test_envparse.py':'''import unittest\nfrom app.envparse import parse_line\nclass T(unittest.TestCase):\n def test_value_equals(self): self.assertEqual(parse_line("URL=https://x?a=1&b=2"),("URL","https://x?a=1&b=2"))\n def test_blank_comment(self): self.assertIsNone(parse_line(" # hello")); self.assertIsNone(parse_line("  "))\n def test_spaces(self): self.assertEqual(parse_line(" A = b "),("A","b"))\nif __name__=="__main__": unittest.main()\n'''}),
]

def make_task(base:Path,name:str,files:dict[str,str])->Path:
 d=base/name; d.mkdir(parents=True); (d/'app').mkdir();(d/'tests').mkdir();(d/'app/__init__.py').write_text('');(d/'tests/__init__.py').write_text('')
 for rel,txt in files.items():p=d/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(txt)
 subprocess.run(['git','init','-q'],cwd=d,check=False);subprocess.run(['git','add','.'],cwd=d,check=False);subprocess.run(['git','-c','user.email=lab@example.invalid','-c','user.name=CPU Lab','commit','-qm','fixture'],cwd=d,check=False)
 return d

def tests_hash(d:Path)->str:
 h=hashlib.sha256()
 for p in sorted((d/'tests').rglob('*')):
  if p.is_file():h.update(p.relative_to(d).as_posix().encode()+b'\0'+p.read_bytes())
 return h.hexdigest()

def run_tests(d:Path)->tuple[bool,str]:
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=d,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
 return p.returncode==0,p.stdout[-8000:]

def safe_path(root:Path,raw:str)->Path:
 p=(root/raw).resolve(); rr=root.resolve()
 if p!=rr and rr not in p.parents:raise ValueError('path escapes repository')
 return p

def exec_tool(root:Path,name:str,inp:dict[str,Any])->str:
 try:
  if name=='list_files':return '\n'.join(str(p.relative_to(root)) for p in sorted(root.rglob('*')) if p.is_file() and '.git' not in p.parts)
  if name=='read_file':
   p=safe_path(root,str(inp.get('path','')));return p.read_text()[:20000]
  if name=='write_file':
   p=safe_path(root,str(inp.get('path','')))
   if 'tests' in p.relative_to(root).parts:return 'DENIED: tests are immutable'
   p.parent.mkdir(parents=True,exist_ok=True);p.write_text(str(inp.get('content','')));return 'OK'
  if name=='run_tests':
   ok,out=run_tests(root);return ('PASS\n' if ok else 'FAIL\n')+out
  return 'ERROR unknown tool'
 except Exception as e:return f'ERROR {type(e).__name__}: {e}'

TOOLS=[
 {'name':'list_files','description':'List repository files.','input_schema':{'type':'object','properties':{},'additionalProperties':False}},
 {'name':'read_file','description':'Read a UTF-8 file in the repository.','input_schema':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}},
 {'name':'write_file','description':'Replace a source file. Test files cannot be modified.','input_schema':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}},'required':['path','content'],'additionalProperties':False}},
 {'name':'run_tests','description':'Run the repository unittest suite.','input_schema':{'type':'object','properties':{},'additionalProperties':False}},
]

def sampling_for(model_name:str)->dict[str,Any]:
 # Qwen3 explicitly warns against greedy decoding; use its recommended thinking
 # sampling family so the agent gate does not falsely reject a capable Qwen
 # quant because of a benchmark-harness decoding mistake. Granite/Nemotron
 # remain deterministic by default.
 n=model_name.lower()
 if 'qwen' in n:
  return {'temperature':0.6,'top_p':0.95,'top_k':20}
 return {'temperature':0.0}

def builtin_task(port:int,d:Path,model_name:str="")->dict[str,Any]:
 before=tests_hash(d); t0=time.time(); used=0; turns=0
 system='You are a coding agent. Fix the repository with the smallest correct change. Inspect files, use tools, never modify tests, run tests, and finish only when tests pass.'
 msgs=[{'role':'user','content':'Fix this repository so all tests pass. Do not change tests. Inspect the code, make the minimal correct source change, and verify it.'}]
 err=''
 try:
  for turns in range(1,MAX_TURNS+1):
   payload={'model':'local-model','max_tokens':512,'system':system,'messages':msgs,'tools':TOOLS,'tool_choice':{'type':'auto'}}; payload.update(sampling_for(model_name))
   o=post(port,'/v1/messages',payload,TASK_TIMEOUT)
   blocks=o.get('content',[]) if isinstance(o,dict) else []
   tool_blocks=[b for b in blocks if isinstance(b,dict) and b.get('type')=='tool_use']
   msgs.append({'role':'assistant','content':blocks})
   if not tool_blocks:break
   results=[]
   for b in tool_blocks:
    used+=1; results.append({'type':'tool_result','tool_use_id':b.get('id'),'content':exec_tool(d,str(b.get('name','')),b.get('input') or {})})
   msgs.append({'role':'user','content':results})
   ok,_=run_tests(d)
   if ok:break
 except Exception as e:err=f'{type(e).__name__}: {e}'
 try:ok,out=run_tests(d)
 except Exception as e:ok=False;out=repr(e)
 intact=tests_hash(d)==before
 return {'pass':bool(ok and intact),'tests_pass':ok,'tests_intact':intact,'tool_calls':used,'turns':turns,'wall_s':time.time()-t0,'error':err,'test_output':out[-2000:]}

def claude_smoke(port:int,fixture:tuple[str,dict[str,str]],server_pid:int|None=None)->dict[str,Any]:
 exe=shutil.which('claude')
 if not exe:return {'available':False,'pass':None,'note':'claude CLI not installed'}
 mon=ClockMonitor(pid=server_pid,nominal_base_mhz=(HW.get('cpu',{}) or {}).get('nominal_base_mhz')).start() if server_pid else None
 result=None
 try:
  with tempfile.TemporaryDirectory(prefix='cpu-lab-claude-',dir=str(TMP)) as td:
   d=make_task(Path(td),fixture[0],fixture[1]); before=tests_hash(d); t0=time.time()
   env=os.environ.copy();env.update({'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{port}','ANTHROPIC_API_KEY':'local-key','ANTHROPIC_AUTH_TOKEN':'local-key','CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1'})
   cmd=[exe,'-p','Fix this repository so all tests pass. Do not modify tests. Inspect source and tests, make the smallest correct edit.','--model','local-model','--output-format','json','--max-turns','8','--allowedTools','Read,Edit,Write,Glob,Grep','--disallowedTools','Bash']
   try:p=subprocess.run(cmd,cwd=d,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=int(os.environ.get('LAB_CLAUDE_TIMEOUT_S','300'))); err=''
   except subprocess.TimeoutExpired as e:p=None;err='timeout';output=(e.stdout or '') if isinstance(e.stdout,str) else ''
   if p is not None:output=p.stdout;rc=p.returncode
   else:rc=124
   ok,tout=run_tests(d); intact=tests_hash(d)==before
   result={'available':True,'pass':bool(rc==0 and ok and intact),'rc':rc,'tests_pass':ok,'tests_intact':intact,'wall_s':time.time()-t0,'error':err,'output':output[-4000:],'test_output':tout[-1500:]}
 finally:
  if mon is not None:
   clk=mon.stop()
   if result is None: result={'available':True,'pass':False,'error':'claude-smoke-aborted'}
   result['clock']=clk
   if clk.get('status')=='throttled': result['pass']=False; result['error']=(str(result.get('error',''))+' clock-throttled').strip()
 return result or {'available':True,'pass':False,'error':'claude-smoke-no-result'}

def evaluate_candidate(c:dict[str,Any],rec:dict[str,dict[str,Any]])->dict[str,Any]:
 r=rec.get(str(c['model']),{}); model=Path(str(r.get('path','')))
 pp=float(c.get('pp_tps') or 0); tg=float(c.get('tg_tps') or 0)
 out={'candidate':c,'model_path':str(model),'timestamp':time.time(),'tasks':[],'accepted':False,
      'performance_gate':{'pp_tps':pp,'tg_tps':tg,'min_pp_tps':MIN_PP_TPS,'min_tg_tps':MIN_TG_TPS}}
 # Do not spend agent-quality wall time on a branch that already fails the
 # human-responsiveness floor at the 8K screen.
 if pp < MIN_PP_TPS or tg < MIN_TG_TPS:
  out['reason']='failed:performance-precheck';out['core_quality_pass']=None;out['responsiveness_pass']=False;return out
 if not model.is_file():out['reason']='model-file-missing';return out
 proc=log=port=None
 try:
  proc,log,port,cmd=start(c,model);out['server_command']=cmd
  if proc is None or port is None:out['reason']='server-start-failed';return out
  with tempfile.TemporaryDirectory(prefix='cpu-lab-agent-',dir=str(TMP)) as td:
   base=Path(td)
   for name,files in TASKS:
    d=make_task(base,name,files); res,clk=monitor_call(lambda: builtin_task(port,d,str(c.get('model',''))),pid=proc.pid,nominal_base_mhz=(HW.get('cpu',{}) or {}).get('nominal_base_mhz'));res['task']=name;res['sampling']=sampling_for(str(c.get('model','')));res['clock']=clk;out['tasks'].append(res)
  passes=sum(1 for x in out['tasks'] if x['pass']); rate=passes/len(out['tasks']); walls=[float(x['wall_s']) for x in out['tasks']]; med=statistics.median(walls) if walls else 1e9
  out['pass_rate']=rate;out['median_task_s']=med;out['core_quality_pass']=rate>=MIN_PASS_RATE
  out['clock_pass']=not any((x.get('clock') or {}).get('status')=='throttled' for x in out['tasks'])
  out['responsiveness_pass']=bool(med<=MAX_MEDIAN and pp>=MIN_PP_TPS and tg>=MIN_TG_TPS and out['clock_pass'])
  if CLAUDE_SMOKE:out['claude_code_smoke']=claude_smoke(port,TASKS[1],proc.pid)
  cs=out.get('claude_code_smoke',{})
  # If Claude Code is installed and we asked for a smoke test, integration failure
  # is a real failure.  Absence is tolerated unless explicitly required.
  if CLAUDE_SMOKE and cs.get('available'):
   claude_ok=cs.get('pass') is True
  else:
   claude_ok=not REQUIRE_CLAUDE
  out['accepted']=bool(out['core_quality_pass'] and out['responsiveness_pass'] and claude_ok)
  if not out['accepted']:
   why=[]
   if not out['core_quality_pass']:why.append('agent-correctness')
   if not out['responsiveness_pass']:why.append('agent-latency')
   if not out.get('clock_pass',True):why.append('clock-throttle')
   if not claude_ok:why.append('claude-code-smoke')
   out['reason']='failed:'+','.join(why)
  return out
 finally:stop(proc,log)

def main():
 obj=candidates_obj(); rec=registered(); run_tag=str(obj.get('run_tag','')); accepted=[]; attempts=[]; rejected=[]
 for fam,rows in sorted(obj.get('families',{}).items()):
  order=fallback_order([x for x in rows if isinstance(x,dict)])[:MAX_CANDIDATES]
  winner=None
  for c in order:
   print(f"AGENT GATE {fam}: {c.get('quant')} {c.get('build')} {c.get('policy_name')} t={c.get('threads')} KV={c.get('kv')}",flush=True)
   e=evaluate_candidate(c,rec);e['family']=fam;attempts.append(e)
   (OUT/'attempts.jsonl').open('a').write(json.dumps(e)+'\n')
   if e.get('accepted'):
    winner=dict(c);winner['agent_pass_rate']=e.get('pass_rate');winner['agent_median_task_s']=e.get('median_task_s');winner['selection']='quality-constrained';winner['quality_attempt_index']=len(attempts)-1
    winner['claude_code_smoke']=e.get('claude_code_smoke',{});accepted.append(winner);winner=winner;break
   rejected.append({'family':fam,'model':c.get('model'),'quant':c.get('quant'),'reason':e.get('reason'),'pass_rate':e.get('pass_rate'),'median_task_s':e.get('median_task_s')})
  if winner is None:print(f'REJECT FAMILY {fam}: no tested quant met agent quality + latency gates',file=sys.stderr)
 out={'version':1,'run_tag':run_tag,'agent_ctx':CTX,'min_pass_rate':MIN_PASS_RATE,'max_median_task_s':MAX_MEDIAN,'require_claude_code':REQUIRE_CLAUDE,'diagnostic':DIAGNOSTIC,'winners':accepted,'rejected':rejected,'attempts':attempts,'attempts_count':len(attempts),'timestamp':time.time()}
 if DIAGNOSTIC:
  dp=Path(DIAGNOSTIC_OUTPUT) if DIAGNOSTIC_OUTPUT else OUT/f'diagnostic-{run_tag}.json'
  dp.parent.mkdir(parents=True,exist_ok=True);dp.write_text(json.dumps(out,indent=2)+'\n');print(dp)
  return
 (MODELS/'agent-winners.json').write_text(json.dumps(out,indent=2)+'\n')
 print(MODELS/'agent-winners.json')
 if not accepted:raise SystemExit('No model family met the agent quality/responsiveness gate')

if __name__=='__main__':main()
