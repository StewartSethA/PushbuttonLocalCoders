#!/usr/bin/env python3
from __future__ import annotations
import json,os,re,shutil,subprocess
from pathlib import Path
ROOT=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab'))); OUT=ROOT/'abi'; OUT.mkdir(parents=True,exist_ok=True)

def run(cmd):
 try:return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
 except subprocess.CalledProcessError as e:return e.output or ''
 except:return ''
def vt(s):return tuple(int(x) for x in s.split('.'))
def vmax(vals):
 vals=list(vals); return max(vals,key=vt) if vals else None

def audit(p):
 f=run(['file',str(p)]) if shutil.which('file') else ''
 if 'ELF' not in f:return None
 ldd=run(['ldd',str(p)]) if shutil.which('ldd') else ''
 dyn=run(['readelf','-d',str(p)]) if shutil.which('readelf') else ''
 ver=run(['readelf','--version-info',str(p)]) if shutil.which('readelf') else ''
 glibc=set(re.findall(r'GLIBC_(\d+(?:\.\d+)+)',ver)); glibcxx=set(re.findall(r'GLIBCXX_(\d+(?:\.\d+)+)',ver))
 missing=[]
 for line in ldd.splitlines():
  if 'not found' in line:missing.append(line.strip())
 rpaths=[x.strip() for x in dyn.splitlines() if '(RPATH)' in x or '(RUNPATH)' in x]
 return {'path':str(p),'file':f,'ldd':ldd,'missing':missing,'rpath_runpath':rpaths,'max_glibc':vmax(glibc),'max_glibcxx':vmax(glibcxx)}

def main():
 rows=[]
 for b in sorted((ROOT/'builds').glob('*')):
  if not b.is_dir():continue
  cand=[]
  for n in ('llama-bench','llama-cli','llama-server','llama-perplexity','test-backend-ops'):
   p=b/'bin'/n
   if p.exists():cand.append(p)
  cand += list(b.glob('**/libggml*.so'))
  seen=set()
  for p in cand:
   if str(p) in seen:continue
   seen.add(str(p)); x=audit(p)
   if x:x['build']=b.name;rows.append(x)
 bad=[x for x in rows if x['missing']]
 obj={'host_glibc':run(['getconf','GNU_LIBC_VERSION']),'rows':rows,'unresolved_count':len(bad)}
 (OUT/'abi-audit.json').write_text(json.dumps(obj,indent=2)+'\n')
 L=['# ABI/linkage audit','',f"Host: `{obj['host_glibc']}`",'', '| build | file | max GLIBC | max GLIBCXX | unresolved | RPATH/RUNPATH |','|---|---|---|---|---:|---|']
 for x in rows:L.append(f"| {x['build']} | {Path(x['path']).name} | {x['max_glibc'] or ''} | {x['max_glibcxx'] or ''} | {len(x['missing'])} | {'; '.join(x['rpath_runpath']).replace('|','/')} |")
 if bad:
  L+=['','## ERROR: unresolved libraries']
  for x in bad:L += [f"### {x['path']}",'```',*x['missing'],'```']
 (OUT/'ABI.md').write_text('\n'.join(L)+'\n')
 print(OUT/'ABI.md')
 return 1 if bad else 0
if __name__=='__main__':raise SystemExit(main())
