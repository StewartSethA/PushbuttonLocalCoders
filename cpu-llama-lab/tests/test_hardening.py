#!/usr/bin/env python3
import importlib.util,json,os,subprocess,sys
from pathlib import Path
PKG=Path(__file__).resolve().parent.parent
ROOT=Path(os.environ.get('LAB_ROOT',str(PKG/'.selftest-root')))/'hardening-test'
ROOT.mkdir(parents=True,exist_ok=True)

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

d=load('detect',PKG/'lib/detect.py')
assert d.mcdram_info('knl',[{'id':0,'cpulist':'0-67','memory_bytes':64<<30},{'id':1,'cpulist':'','memory_bytes':16<<30}])['mode_hint']=='flat'
assert d.mcdram_info('knl',[{'id':0,'cpulist':'0-67','memory_bytes':64<<30}])['mode_hint']=='cache-or-hidden'

b=(PKG/'scripts/build.py').read_text()
for x in ('LAB_INHERIT_BUILD_ENV','CCACHE_DIR','PKG_CONFIG_PATH','pkg_exists','memkind'):
 assert x in b,x
inst=(PKG/'scripts/install-deps.sh').read_text()
for x in ('linux-tools-$k','UV_UNMANAGED_INSTALL','apt-get check','libmemkind-dev','perf stat','g++-$v'):
 assert x in inst,x

# Exercise preflight/manifest/ABI writers under LAB_ROOT; no system mutation.
hw=d.run(['true'])
subprocess.run([sys.executable,str(PKG/'lib/detect.py')],check=True,stdout=open(ROOT/'hardware.json','w'))
env=os.environ.copy();env['LAB_ROOT']=str(ROOT);env['LAB_STRICT']='0'
subprocess.run([sys.executable,str(PKG/'scripts/preflight.py')],env=env,check=True,stdout=subprocess.DEVNULL)
subprocess.run([sys.executable,str(PKG/'scripts/manifest.py')],env=env,check=True,stdout=subprocess.DEVNULL)
(ROOT/'builds').mkdir(exist_ok=True)
subprocess.run([sys.executable,str(PKG/'scripts/abi_audit.py')],env=env,check=True,stdout=subprocess.DEVNULL)
for p in ('preflight.json','manifest.json','abi/abi-audit.json'):
 assert (ROOT/p).exists(),p
print('PASS test_hardening')
