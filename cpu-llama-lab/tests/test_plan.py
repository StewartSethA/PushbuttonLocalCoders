#!/usr/bin/env python3
import importlib.util,json,tempfile,sys
from pathlib import Path
P=Path(__file__).resolve().parent.parent/'lib/plan.py'; s=importlib.util.spec_from_file_location('plan',P); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def hw(cls,gen,nodes,flags=None,cores=24,sockets=2,tpc=2):
 f={'avx2':True,'avx512f':False,'avx512bw':False,'avx512dq':False,'avx512vl':False,'avx512vnni':False,'avx512bf16':False,'amxint8':False,'amxbf16':False}; f.update(flags or {})
 return {'cpu':{'class':cls,'generation':gen,'sockets':sockets,'cores_per_socket':cores,'logical_cpus':sockets*cores*tpc},'features':f,'numa':{'count':nodes,'nodes':[{'id':i,'cpulist':str(i),'memory_bytes':1,'packages':[min(i,max(0,sockets-1))] if nodes<=sockets else [min(i//max(1,nodes//sockets),sockets-1)]} for i in range(nodes)]}}
p=m.plan(hw('knl','knights-landing',1,cores=68,sockets=1,tpc=4)); assert all(x.get('knl') or x.get('generic_knl_control') for x in p['builds']); assert p['threads']==[68]; assert p['thread_probe']['counts']==[34,68,136,272]; assert p['search']['screen_cap']==8192; assert p['search']['deep_cap']==65536; assert max(p['search']['screen_depths'])==8192; assert max(p['search']['deep_depths'])==65536
p=m.plan(hw('xeon','skylake-sp',2,{'avx512f':True,'avx512bw':True,'avx512dq':True,'avx512vl':True})); assert any(x['name']=='isa-avx512' for x in p['builds']); assert any(x['name']=='interleave' for x in p['numa_policies']); assert any(x['name']=='socket0-local' for x in p['numa_policies'])
p=m.plan(hw('epyc','zen3',2)); assert not any(x.get('knl') for x in p['builds']); assert any(x['name']=='isa-avx2' for x in p['builds'])
p=m.plan(hw('epyc','zen4',2,{'avx512f':True,'avx512bw':True,'avx512dq':True,'avx512vl':True,'avx512vnni':True})); assert any(x['name']=='isa-avx512' for x in p['builds'])
# Multiple NUMA domains per socket must still expose whole-socket policies.
h=hw('epyc','zen4',4,{'avx512f':True,'avx512bw':True,'avx512dq':True,'avx512vl':True},cores=32,sockets=2); h['numa']['nodes'][0]['packages']=[0]; h['numa']['nodes'][1]['packages']=[0]; h['numa']['nodes'][2]['packages']=[1]; h['numa']['nodes'][3]['packages']=[1]; p=m.plan(h); sp={x['name']:x for x in p['numa_policies']}; assert sp['socket0-local']['nodes']==[0,1] and sp['socket1-local']['nodes']==[2,3]
print('PASS test_plan')
