#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,platform,shutil,subprocess,sys,time
from pathlib import Path
PKG=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PKG))
from lib.models import configured_models
ROOT=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab'))); ROOT.mkdir(parents=True,exist_ok=True)

def run(cmd):
 try:return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
 except:return ''
def ver(c,args=('--version',)):
 p=shutil.which(c)
 return {'path':p,'version':run([p,*args]).splitlines()[0] if p else ''}
def file_sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()
def models():
 return configured_models()

def main():
 tools={x:ver(x) for x in ('gcc','g++','gcc-14','g++-14','clang','clang++','cmake','ninja','ccache','ld','ldd','readelf','objdump','numactl','perf','python3','uv','pkg-config')}
 packages=''
 if shutil.which('dpkg-query'):
  names=['libc6','libstdc++6','gcc','gcc-14','g++-14','binutils','cmake','libnuma1','libnuma-dev','libopenblas0','libopenblas-dev','libmemkind0','libmemkind-dev','linux-tools-common','linux-tools-generic']
  lines=[]
  for n in names:
   x=run(['dpkg-query','-W','-f=${Package}\t${Version}\n',n])
   if x:lines.append(x)
  packages='\n'.join(lines)
 elif shutil.which('rpm'):
  packages=run(['rpm','-qa','--qf','%{NAME}\t%{VERSION}-%{RELEASE}\n'])
 mods=[]; dohash=os.environ.get('LAB_HASH_MODELS','0')=='1'
 for n,p in models():
  x={'name':n,'path':str(p),'size_bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
  if dohash:x['sha256']=file_sha(p)
  mods.append(x)
 envkeys=['CC','CXX','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','LD_LIBRARY_PATH','PKG_CONFIG_PATH','CMAKE_PREFIX_PATH','LAB_INHERIT_BUILD_ENV','LAB_STRICT','LAB_DROP_CACHES','LAB_WORLD','LAB_RANK','LAB_CLOCK_SAMPLE_MS','LAB_CLOCK_MAX_CPUS','LAB_CLOCK_MIN_RATIO','LAB_CLOCK_MIN_MHZ','LAB_CLOCK_REFERENCE_MHZ','LAB_CLOCK_ENFORCE']
 vuln={}
 for p in Path('/sys/devices/system/cpu/vulnerabilities').glob('*'):
  try:vuln[p.name]=p.read_text().strip()
  except:pass
 microcode=''
 try:
  for l in Path('/proc/cpuinfo').read_text(errors='ignore').splitlines():
   if l.lower().startswith('microcode'):
    microcode=l.split(':',1)[1].strip();break
 except:pass
 obj={'timestamp':time.time(),'host':platform.node(),'uname':run(['uname','-a']),'os_release':Path('/etc/os-release').read_text(errors='ignore') if Path('/etc/os-release').exists() else '',
      'kernel_cmdline':Path('/proc/cmdline').read_text().strip() if Path('/proc/cmdline').exists() else '',
      'glibc':run(['getconf','GNU_LIBC_VERSION']),'microcode':microcode,'cpu_vulnerability_mitigations':vuln,
      'clocksource':(Path('/sys/devices/system/clocksource/clocksource0/current_clocksource').read_text().strip() if Path('/sys/devices/system/clocksource/clocksource0/current_clocksource').exists() else ''),
      'tools':tools,'packages':packages,'environment':{k:os.environ.get(k) for k in envkeys if os.environ.get(k) is not None},'models':mods,
      'llama_commit':(ROOT/'llama-commit.txt').read_text().strip() if (ROOT/'llama-commit.txt').exists() else ''}
 if (ROOT/'models/registry.json').exists():
  try: obj['model_registry']=json.load(open(ROOT/'models/registry.json'))
  except Exception: pass
 if (ROOT/'models/model-plan.json').exists():
  try: obj['model_plan']=json.load(open(ROOT/'models/model-plan.json'))
  except Exception: pass
 if (ROOT/'knl-cc.txt').exists():obj['knl_compiler']=(ROOT/'knl-cc.txt').read_text().strip()
 if (ROOT/'clock-check.json').exists():
  try:obj['clock_check']=json.load(open(ROOT/'clock-check.json'))
  except Exception:pass
 if (ROOT/'preflight.json').exists():
  try:obj['loaded_clock_probe']=json.load(open(ROOT/'preflight.json')).get('loaded_clock_probe')
  except Exception:pass
 (ROOT/'manifest.json').write_text(json.dumps(obj,indent=2)+'\n')
 print(ROOT/'manifest.json')
if __name__=='__main__':main()
