#!/usr/bin/env python3
from __future__ import annotations
import gzip,hashlib,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
PKG=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PKG))
from lib.models import ensure_models
from lib.clock import ClockMonitor
ROOT=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab'))); Q=ROOT/'quality'; Q.mkdir(parents=True,exist_ok=True)
HW=json.load(open(ROOT/'hardware.json'))

def record_clock(kind,n,clock):
 with open(Q/'clock-tests.jsonl','a') as f:f.write(json.dumps({'kind':kind,'model':n,'clock':clock,'timestamp':time.time()})+'\n')

def nominal_base():
 try:return (HW.get('cpu',{}) or {}).get('nominal_base_mhz')
 except:return None

def models():
 mods=ensure_models(); by=dict(mods); wp=ROOT/'models/winners.json'
 if wp.exists():
  try:
   w=json.load(open(wp)).get('winners',[])
   picked=[(str(x.get('model')),by[str(x.get('model'))]) for x in w if str(x.get('model')) in by]
   if picked:return picked
  except Exception:pass
 return mods

def build():
 names=['knl-combo-norepack','knl-base-norepack','knl-combo','native','isa-avx512','isa-avx2'] if HW['cpu']['class']=='knl' else ['native','isa-avx512','isa-avx2']
 for n in names:
  d=ROOT/'builds'/n
  if (d/'bin/llama-server').exists():return d
 raise SystemExit('No usable build. Run build first.')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()

def corpus():
 p=Q/'wikitext2-valid.txt'
 if p.exists():return p
 url=os.environ.get('PPL_CORPUS_URL','https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt')
 try:urllib.request.urlretrieve(url,p)
 except Exception as e:(Q/'flags.txt').open('a').write(f'PPL_CORPUS_DOWNLOAD_FAILED {e}\n')
 return p

def ppl(b,mods):
 c=corpus(); exe=b/'bin/llama-perplexity'
 if not c.exists() or not exe.exists():return
 with open(Q/'perplexity.tsv','w') as o:
  o.write('model\tsha256\tresult\n')
  for n,m in mods:
   log=Q/f'ppl-{n}.log'; cmd=[str(exe),'-m',str(m),'-f',str(c),'-t',str(HW['cpu']['sockets']*HW['cpu']['cores_per_socket'])]
   with open(log,'w') as f:
    p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);mon=ClockMonitor(pid=p.pid,nominal_base_mhz=nominal_base()).start();p.wait();clk=mon.stop();record_clock('perplexity',n,clk)
   lines=[x for x in log.read_text(errors='ignore').splitlines() if 'ppl' in x.lower() or 'perplex' in x.lower()]
   o.write(f'{n}\t{sha(m)}\t{lines[-1] if lines else "NO_RESULT"}\n')

def prompts():
 return [
  ('c-oob','Review this C function for correctness and undefined behavior. Give the smallest safe fix.\n\nint sum(const int *a, size_t n) { int s=0; for (int i=0; i<=n; ++i) s += a[i]; return s; }'),
  ('python-race','A Python service occasionally loses increments: `counter += 1` is called by many worker threads. Diagnose precisely and give a robust fix.'),
  ('shell','Review `for f in $(find . -name "*.log"); do rm $f; done` and give a safe replacement handling spaces, newlines, leading dashes and no matches.'),
  ('algorithm','Design an O(n) algorithm for the shortest contiguous subarray with sum at least K when values may be negative. Explain the invariant and provide concise Python.'),
  ('cpp','Explain double-checked locking around a non-atomic pointer in modern C++, and give a correct C++20 implementation.'),
  ('repo','You inherit a medium C++ repository whose CI started failing after a dependency update. Give an exact investigation and proof sequence before/after your patch.'),
 ]

def wait(port):
 for _ in range(180):
  try:
   urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1);return True
  except:time.sleep(1)
 return False

def request(port,prompt,max_tokens=1000):
 body={'model':'local-model','temperature':0,'max_tokens':max_tokens,'messages':[{'role':'user','content':prompt}]}
 req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'content-type':'application/json','authorization':'Bearer local-key'})
 with urllib.request.urlopen(req,timeout=3600) as r:o=json.load(r)
 return o['choices'][0]['message'].get('content','')

def start_server(b,n,m,port):
 exe=b/'bin/llama-server'; log=open(Q/f'server-{n}.log','w')
 cmd=[str(exe),'-m',str(m),'--alias','local-model','-c',os.environ.get('QUALITY_CTX','32768'),'-t',str(HW['cpu']['sockets']*HW['cpu']['cores_per_socket']),'-tb',str(HW['cpu']['sockets']*HW['cpu']['cores_per_socket']),'-ctk',os.environ.get('LAB_QUALITY_KV','f16'),'-ctv',os.environ.get('LAB_QUALITY_KV','f16'),'--jinja','--no-webui','--host','127.0.0.1','--port',str(port)]
 return subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log

def reviews(b,mods):
 out=Q/'human-responses.jsonl'
 for idx,(n,m) in enumerate(mods):
  port=19080+idx; p,lg=start_server(b,n,m,port)
  if not wait(port):
   (Q/'flags.txt').open('a').write(f'{n} SERVER_FAILED\n');p.kill();lg.close();continue
  sh=sha(m); key=f'{n}|{m.name}|{sh[:12]}'
  for pid,pr in prompts():
   mon=ClockMonitor(pid=p.pid,nominal_base_mhz=nominal_base()).start()
   try:r=request(port,pr)
   except Exception as e:r=f'[REQUEST_ERROR] {e}'
   finally:clk=mon.stop();record_clock('review:'+pid,n,clk)
   rec={'response_id':hashlib.sha256((pid+sh).encode()).hexdigest()[:20],'prompt_id':pid,'prompt':pr,'response':r,'candidate_key':key,'model':n,'file':m.name,'sha256':sh}
   with open(out,'a') as f:f.write(json.dumps(rec)+'\n')
  p.terminate();
  try:p.wait(30)
  except: p.kill()
  lg.close()

def ensure_evalplus():
 root_uv=ROOT/'tools/bin/uv'
 uv=str(root_uv) if root_uv.exists() else (shutil.which('uv') if os.environ.get('LAB_USE_SYSTEM_UV','0')=='1' else None)
 if not uv:return None
 v=ROOT/'venvs/evalplus'; py=v/'bin/python'; exe=v/'bin/evalplus.codegen'
 env=os.environ.copy(); env['UV_CACHE_DIR']=str(ROOT/'cache/uv'); env['XDG_CACHE_HOME']=str(ROOT/'cache/xdg')
 spec=os.environ.get('EVALPLUS_SPEC','evalplus==0.3.1')
 if not exe.exists():
  subprocess.call([uv,'venv','--python',sys.executable,str(v)],env=env)
  subprocess.call([uv,'pip','install','--python',str(py),spec],env=env)
 if exe.exists():
  with open(Q/'evalplus-freeze.txt','w') as f:subprocess.call([uv,'pip','freeze','--python',str(py)],env=env,stdout=f,stderr=subprocess.STDOUT)
  (Q/'evalplus-spec.txt').write_text(spec+'\n')
 return exe if exe.exists() else None

def evalplus(b,mods):
 # Generation per model. Execution is done only in Docker if available.
 exe=ensure_evalplus()
 if not exe:return
 for idx,(n,m) in enumerate(mods):
  port=20080+idx; p,lg=start_server(b,n,m,port)
  if not wait(port): p.kill();lg.close();continue
  qd=Q/f'evalplus-{n}';qd.mkdir(exist_ok=True)
  env=os.environ.copy();env['OPENAI_API_KEY']='local-key'
  for ds in (['humaneval','mbpp'] if os.environ.get('QUALITY_LEVEL','overnight')=='weekend' else ['humaneval']):
   log=qd/f'codegen-{ds}.log'
   with open(log,'w') as f:
    mon=ClockMonitor(pid=p.pid,nominal_base_mhz=nominal_base()).start();rc=subprocess.call([str(exe),'--model','local-model','--dataset',ds,'--backend','openai','--base-url',f'http://127.0.0.1:{port}/v1','--greedy','--root',str(qd)],env=env,stdout=f,stderr=subprocess.STDOUT);clk=mon.stop();record_clock('evalplus:'+ds,n,clk)
   if rc:(Q/'flags.txt').open('a').write(f'{n} {ds} CODEGEN_FAILED\n')
  p.terminate();
  try:p.wait(30)
  except:p.kill()
  lg.close()

def main():
 mods=models();
 if not mods:raise SystemExit('No models configured')
 b=build(); (Q/'flags.txt').write_text('')
 ppl(b,mods); reviews(b,mods); evalplus(b,mods)
 print('Quality artifacts:',Q);print('Run: ./cpu-llama-lab.sh judge')
if __name__=='__main__':main()
