#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
from pathlib import Path

PKG=Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
planm=load('plan_platforms',PKG/'lib/plan.py'); det=load('detect_platforms',PKG/'lib/detect.py')

def A(x,msg):
    if not x: raise AssertionError(msg)

def hw(cls,gen,name,sockets,cores,nodes,features,packages):
    f={'avx2':True,'avx512f':False,'avx512bw':False,'avx512dq':False,'avx512vl':False,'avx512vnni':False,'avx512bf16':False,'avxvnni':False,'amxint8':False,'amxbf16':False};f.update(features)
    ns=[]
    for i,pkg in enumerate(packages):ns.append({'id':i,'cpulist':str(i),'memory_bytes':64<<30,'packages':[pkg]})
    return {'cpu':{'class':cls,'generation':gen,'name':name,'sockets':sockets,'cores_per_socket':cores,'logical_cpus':sockets*cores*2},'features':f,'numa':{'count':nodes,'nodes':ns},'memory':{'platform_channels_per_socket_hint':det.channels_hint(cls,gen,name)}}

def names(p):return {x['name'] for x in p['builds']}
def pols(p):return {x['name'] for x in p['numa_policies']}
def common(p):
    A({'disabled-mmap','disabled-no-mmap'} <= pols(p),'explicit mmap/no-mmap controls missing')
    A(p['search']['screen_cap']==8192 and p['search']['deep_cap']==65536,'context contract differs by CPU')
    A(p['search']['screen_depths']==[512,2048,8192],'screen depths differ by CPU')
    A(p['search']['deep_depths']==[16384,32768,65536],'deep depths differ by CPU')

# KNL Flat: isolated patched build family plus explicit DDR/HBM/repack controls.
kh=hw('knl','knights-landing','Intel Xeon Phi 7250',1,68,1,{'avx512f':True},[0])
kh['cpu']['logical_cpus']=272
kh['numa']={'count':2,'nodes':[{'id':0,'cpulist':'0-271','memory_bytes':64<<30,'packages':[0]}, {'id':1,'cpulist':'','memory_bytes':16<<30,'packages':[]}]}
kh['memory']['mcdram']={'mode_hint':'flat','visible_nodes':[1],'visible_bytes':16<<30}
p=planm.plan(kh);common(p)
A(all(x.get('knl') or x.get('generic_knl_control') for x in p['builds']),'KNL plan contains an unexpected generic build')
A({'knl-base','knl-base-norepack','knl-generic-avx2-norepack','knl-combo','knl-combo-norepack'} <= names(p),'KNL repack/kernel/load-safe controls missing')
A({'knl-ddr-strict','knl-mcdram-strict','knl-mcdram-preferred','disabled-mmap','disabled-no-mmap'} <= pols(p),'KNL Flat memory/load policies missing')
hbmp=next(x for x in p['numa_policies'] if x['name']=='knl-mcdram-strict')
A(hbmp.get('requires_mcdram_fit') is True,'strict MCDRAM policy must fail closed for spill candidates')
A(p['threads']==[68] and p['thread_probe']['counts']==[34,68,136,272],'KNL main/thread-probe contract wrong')
A(hbmp.get('default_threads')==68,'KNL strict-HBM policy must use all 68 physical cores by default')

# Older surplus Xeon E5 v4: AVX2-only planning, four channels/socket.
p=planm.plan(hw('xeon','broadwell-ep','Intel Xeon E5-2699 v4',2,22,2,{},[0,1]));common(p)
A({'native','native-norepack','native-lto','isa-avx2'} <= names(p),'Broadwell-EP build matrix incomplete')
A('isa-avx512' not in names(p),'Broadwell-EP must not invent AVX-512')
A({'socket0-local','socket1-local','interleave'} <= pols(p),'Broadwell-EP NUMA/socket controls incomplete')
A(det.channels_hint('xeon','broadwell-ep','Intel Xeon E5-2699 v4')==4,'Broadwell-EP channel hint wrong')

# Single-socket Xeon uses the same script/build semantics without requiring NUMA.
p=planm.plan(hw('xeon','skylake-sp','Intel Xeon Gold 6130',1,16,1,{'avx512f':True,'avx512bw':True,'avx512dq':True,'avx512vl':True},[0]));common(p)
A({'native','native-norepack','isa-avx2','isa-avx512'} <= names(p),'single-socket Xeon build matrix incomplete')
A(p['threads']==[16] and p['thread_probe']['counts']==[8,16,32],'single-socket Xeon thread contract wrong')

# 2x Platinum 8168 / Skylake-SP: native + AVX2 + AVX-512 and whole-socket controls.
p=planm.plan(hw('xeon','skylake-sp','Intel Xeon Platinum 8168',2,24,2,{'avx512f':True,'avx512bw':True,'avx512dq':True,'avx512vl':True},[0,1]));common(p)
A({'native','native-norepack','native-lto','isa-avx2','isa-avx512'} <= names(p),'8168 generic ISA/build ablations incomplete')
A({'socket0-local','socket1-local','interleave','distribute-mmap','distribute-no-mmap'} <= pols(p),'8168 NUMA/socket controls incomplete')
A(not any(x.get('knl') for x in p['builds']),'KNL source variants must never be selected on Xeon')
A(det.channels_hint('xeon','skylake-sp','Intel Xeon Platinum 8168')==6,'8168 channel hint wrong')
A(p['threads']==[48] and p['thread_probe']['counts']==[24,48,96],'8168 thread contract wrong')
A(next(x for x in p['numa_policies'] if x['name']=='socket0-local')['default_threads']==24,'8168 socket-local must use 24 physical cores, not oversubscribe 48')

# Naples EPYC 7551P: first-generation SP3, eight channels, often four NUMA
# domains in one package. This is a key cheap-surplus comparison target.
A(det.cpu_class('AMD EPYC 7551P 32-Core Processor','AuthenticAMD','23','1',set())==('epyc','zen1'),'7551P generation detection wrong')
p=planm.plan(hw('epyc','zen1','AMD EPYC 7551P 32-Core Processor',1,32,4,{},[0,0,0,0]));common(p)
A({'native','native-norepack','native-lto','isa-avx2'} <= names(p),'Naples build matrix incomplete')
A('socket0-local' in pols(p),'Naples multi-NUMA whole-socket policy missing')
A(det.channels_hint('epyc','zen1','AMD EPYC 7551P 32-Core Processor')==8,'Naples channel hint wrong')
A(p['threads']==[32] and p['thread_probe']['counts']==[16,32,64],'7551P thread contract wrong')

# Rome keeps the same eight-channel SP3 memory contract while scaling NUMA.
A(det.cpu_class('AMD EPYC 7742 64-Core Processor','AuthenticAMD','23','49',set())==('epyc','zen2'),'7742 generation detection wrong')
p=planm.plan(hw('epyc','zen2','AMD EPYC 7742 64-Core Processor',1,64,4,{},[0,0,0,0]));common(p)
A('socket0-local' in pols(p) and det.channels_hint('epyc','zen2','AMD EPYC 7742 64-Core Processor')==8,'Rome topology/channel planning wrong')

# Milan SP3 NPS4: one physical socket with four NUMA domains must expose a whole-socket policy.
p=planm.plan(hw('epyc','zen3','AMD EPYC 7543P',1,32,4,{},[0,0,0,0]));common(p)
A({'native','native-norepack','native-lto','isa-avx2'} <= names(p),'Milan build matrix incomplete')
A('socket0-local' in pols(p),'Milan NPS4 whole-socket locality missing')
A(det.channels_hint('epyc','zen3','AMD EPYC 7543P')==8,'Milan channel hint wrong')

# Genoa SP5 NPS4: preserve full 12-channel socket and explicit AVX-512/VNNI ablation.
p=planm.plan(hw('epyc','zen4','AMD EPYC 9354P',1,32,4,{'avx512f':True,'avx512bw':True,'avx512dq':True,'avx512vl':True,'avx512vnni':True,'avx512bf16':True},[0,0,0,0]));common(p)
A({'native','native-norepack','native-lto','isa-avx2','isa-avx512'} <= names(p),'Genoa build matrix incomplete')
A('socket0-local' in pols(p),'Genoa NPS4 whole-socket locality missing')
A(det.channels_hint('epyc','zen4','AMD EPYC 9354P')==12,'Genoa channel hint wrong')


# X399 first-generation Threadripper: four-channel Zen1 AVX2 platform, no KNL source path.
p=planm.plan(hw('threadripper','zen1','AMD Ryzen Threadripper 1950X 16-Core Processor',1,16,2,{},[0,0]));common(p)
A({'native','native-norepack','native-lto','isa-avx2'} <= names(p),'1950X build matrix incomplete')
A('socket0-local' in pols(p),'1950X two-node whole-socket policy missing')
A(not any(x.get('knl') for x in p['builds']),'KNL variants must never be selected on X399')
A(det.channels_hint('threadripper','zen1','AMD Ryzen Threadripper 1950X 16-Core Processor')==4,'1950X channel hint wrong')
A(p['threads']==[16] and p['thread_probe']['counts']==[8,16,32],'1950X thread contract wrong')
A(det.cpu_class('AMD Ryzen Threadripper 1950X 16-Core Processor','AuthenticAMD','23','1',set())==('threadripper','zen1'),'1950X generation detection wrong')
A(det.cpu_class('AMD Ryzen Threadripper 1920X 12-Core Processor','AuthenticAMD','23','1',set())==('threadripper','zen1'),'1920X generation detection wrong')
A(det.cpu_class('AMD Ryzen Threadripper 2990WX 32-Core Processor','AuthenticAMD','23','8',set())==('threadripper','zen+'),'2990WX generation detection wrong')
A(det.nominal_base_mhz('AMD Ryzen Threadripper 1950X 16-Core Processor')==3400.0,'1950X nominal base clock wrong')
A(det.nominal_base_mhz('AMD Ryzen Threadripper 1920X 12-Core Processor')==3500.0,'1920X nominal base clock wrong')
A(det.nominal_base_mhz('Intel(R) Xeon Phi(TM) CPU 7250 @ 1.40GHz')==1400.0,'7250 nominal base clock wrong')

# WRX80 Threadripper Pro: same generic source/build semantics, eight-channel hint.
p=planm.plan(hw('threadripper','zen3','AMD Ryzen Threadripper PRO 5995WX',1,64,1,{},[0]));common(p)
A({'native','native-norepack','native-lto','isa-avx2'} <= names(p),'Threadripper Pro build matrix incomplete')
A(not any(x.get('knl') for x in p['builds']),'KNL variants must never be selected on Threadripper')
A(det.channels_hint('threadripper','zen3','AMD Ryzen Threadripper PRO 5995WX')==8,'TR Pro channel hint wrong')

print('PASS test_platforms')
