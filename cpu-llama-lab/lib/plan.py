#!/usr/bin/env python3
import argparse,json

def uniq(xs):
    out=[]
    for x in xs:
        if x not in out: out.append(x)
    return out

def plan(h):
    c=h['cpu']; f=h['features']; nn=h['numa']['count']; phys=c['sockets']*c['cores_per_socket']; logical=c['logical_cpus']
    builds=[{'name':'native','cmake':['-DGGML_NATIVE=ON','-DGGML_CPU_REPACK=ON'],'repack':'ON'},
            {'name':'native-norepack','cmake':['-DGGML_NATIVE=ON','-DGGML_CPU_REPACK=OFF'],'repack':'OFF'},
            {'name':'native-lto','cmake':['-DGGML_NATIVE=ON','-DGGML_CPU_REPACK=ON','-DGGML_LTO=ON'],'repack':'ON'},
            {'name':'native-openblas','cmake':['-DGGML_NATIVE=ON','-DGGML_CPU_REPACK=ON','-DGGML_BLAS=ON','-DGGML_BLAS_VENDOR=OpenBLAS'],'optional':'openblas','repack':'ON'}]
    if f['avx2']:
        builds.append({'name':'isa-avx2','cmake':['-DGGML_NATIVE=OFF','-DGGML_SSE42=ON','-DGGML_AVX=ON','-DGGML_F16C=ON','-DGGML_FMA=ON','-DGGML_AVX2=ON','-DGGML_BMI2=ON','-DGGML_AVX512=OFF','-DGGML_CPU_REPACK=ON'],'repack':'ON'})
    if f.get('avxvnni'):
        builds.append({'name':'isa-avx2-vnni','cmake':['-DGGML_NATIVE=OFF','-DGGML_SSE42=ON','-DGGML_AVX=ON','-DGGML_F16C=ON','-DGGML_FMA=ON','-DGGML_AVX2=ON','-DGGML_BMI2=ON','-DGGML_AVX_VNNI=ON','-DGGML_AVX512=OFF','-DGGML_CPU_REPACK=ON'],'repack':'ON'})
    if f['avx512f'] and f.get('avx512bw') and f.get('avx512dq') and f.get('avx512vl') and c['class']!='knl':
        x=['-DGGML_NATIVE=OFF','-DGGML_SSE42=ON','-DGGML_AVX=ON','-DGGML_F16C=ON','-DGGML_FMA=ON','-DGGML_AVX2=ON','-DGGML_BMI2=ON','-DGGML_AVX512=ON','-DGGML_CPU_REPACK=ON']
        if f['avx512vnni']: x+=['-DGGML_AVX512_VNNI=ON']
        if f['avx512bf16']: x+=['-DGGML_AVX512_BF16=ON']
        builds.append({'name':'isa-avx512','cmake':x,'repack':'ON'})
        if f.get('amxint8'):
            a=x+['-DGGML_AMX_TILE=ON','-DGGML_AMX_INT8=ON']
            if f.get('amxbf16'): a+=['-DGGML_AMX_BF16=ON']
            builds.append({'name':'isa-amx','cmake':a,'repack':'ON'})
    if c['class']=='knl':
        # Two no-repack controls are enough to measure the layout interaction
        # without doubling every prefetch/unroll build.
        builds=[{'name':'knl-base','knl':True,'pf':'OFF','dist':2,'hint':0,'unroll':'OFF','instrument':'OFF','repack':'ON'},
                {'name':'knl-base-norepack','knl':True,'pf':'OFF','dist':2,'hint':0,'unroll':'OFF','instrument':'OFF','repack':'OFF'},
                # Conservative generic AVX2/no-repack load-safe control. The KNL
                # source patch remains present in the tree, but this build does not
                # select the KNL AVX-512 backend or CPU repacker. It is both a
                # bootstrap fallback and a useful upstream-ish performance control.
                {'name':'knl-generic-avx2-norepack','cmake':['-DGGML_NATIVE=OFF','-DGGML_SSE42=ON','-DGGML_AVX=ON','-DGGML_F16C=ON','-DGGML_FMA=ON','-DGGML_AVX2=ON','-DGGML_BMI2=OFF','-DGGML_AVX512=OFF','-DGGML_CPU_REPACK=OFF'],'repack':'OFF','generic_knl_control':True},
                {'name':'knl-pf-d1','knl':True,'pf':'ON','dist':1,'hint':0,'unroll':'OFF','instrument':'OFF','repack':'ON'},
                {'name':'knl-pf-d2','knl':True,'pf':'ON','dist':2,'hint':0,'unroll':'OFF','instrument':'OFF','repack':'ON'},
                {'name':'knl-pf-d4','knl':True,'pf':'ON','dist':4,'hint':0,'unroll':'OFF','instrument':'OFF','repack':'ON'},
                {'name':'knl-unroll','knl':True,'pf':'OFF','dist':2,'hint':0,'unroll':'ON','instrument':'OFF','repack':'ON'},
                {'name':'knl-combo','knl':True,'pf':'ON','dist':2,'hint':0,'unroll':'ON','instrument':'OFF','repack':'ON'},
                {'name':'knl-combo-norepack','knl':True,'pf':'ON','dist':2,'hint':0,'unroll':'ON','instrument':'OFF','repack':'OFF'},
                {'name':'knl-combo-hbm','knl':True,'pf':'ON','dist':2,'hint':0,'unroll':'ON','instrument':'OFF','repack':'ON','hbm':True,'optional':'memkind'},
                {'name':'knl-instrument','knl':True,'pf':'ON','dist':2,'hint':0,'unroll':'ON','instrument':'ON','repack':'ON'}]
    # Main experiments use one physical worker per physical core. Thread scaling
    # is intentionally isolated to one cheap calibration-model probe rather than
    # multiplied across every build/model/quant.
    threads=[phys]
    thread_probe_counts=uniq([max(1,phys//2),phys,min(logical,phys*2)])
    if c['class']=='knl':
        # KNL exposes 4-way SMT: explicitly sample 1/2, 1, 2 and 4 threads/core.
        thread_probe_counts=uniq([max(1,phys//2),phys,min(logical,phys*2),logical])
    # Explicit mmap/no-mmap controls exist even on one-node systems. This makes
    # repack x load behavior comparable on KNL, Xeon, EPYC and Threadripper.
    numa=[{'name':'disabled-mmap','llama':['--load-mode','mmap'],'prefix':[],'load_mode':'mmap'},
          {'name':'disabled-no-mmap','llama':['--load-mode','none'],'prefix':[],'load_mode':'none'}]
    if nn>1:
        numa += [
          {'name':'distribute-mmap','llama':['--numa','distribute','--load-mode','mmap'],'prefix':[],'cold_pagecache':True,'load_mode':'mmap'},
          {'name':'distribute-no-mmap','llama':['--numa','distribute','--load-mode','none'],'prefix':[],'load_mode':'none'},
          {'name':'interleave','llama':['--numa','numactl','--load-mode','none'],'prefix':['numactl','--interleave=all'],'load_mode':'none'},
        ]
        for n in h['numa']['nodes']:
            if n['cpulist']:
                numa.append({'name':f'node{n["id"]}-local','llama':['--numa','numactl','--load-mode','none'],'prefix':['numactl',f'--cpunodebind={n["id"]}',f'--membind={n["id"]}'],'load_mode':'none','memory_tier':'local'})
        package_nodes={}
        for n in h['numa']['nodes']:
            if not n.get('cpulist'): continue
            for pkg in n.get('packages',[]): package_nodes.setdefault(int(pkg),[]).append(int(n['id']))
        if c.get('sockets',1)>1 or any(len(v)>1 for v in package_nodes.values()):
            for pkg,nids in sorted(package_nodes.items()):
                ns=','.join(str(x) for x in sorted(set(nids)))
                numa.append({'name':f'socket{pkg}-local','llama':['--numa','numactl','--load-mode','none'],'prefix':['numactl',f'--cpunodebind={ns}',f'--membind={ns}'],'package':pkg,'nodes':sorted(set(nids)),'load_mode':'none','memory_tier':'socket-local'})
        numa.append({'name':'mirror-if-supported','llama':['--numa','mirror','--numa-mirror','weights','--load-mode','none'],'prefix':[],'requires_cli':'--numa mirror','load_mode':'none'})
    # KNL Flat mode: CPU nodes are DDR-backed; memory-only nodes are MCDRAM.
    # Strict HBM is for candidates whose raw weights fit; preferred HBM is the
    # spill/tiering control for Q4 or edge-fit candidates.
    mc=(h.get('memory',{}) or {}).get('mcdram',{}) or {}
    hnodes=[int(x) for x in mc.get('visible_nodes',[])]; cpu_nodes=[int(n['id']) for n in h.get('numa',{}).get('nodes',[]) if n.get('cpulist')]
    if c['class']=='knl' and hnodes and cpu_nodes:
        cpus=','.join(map(str,cpu_nodes)); hbs=','.join(map(str,hnodes))
        numa += [
          {'name':'knl-ddr-strict','llama':['--numa','numactl','--load-mode','none'],'prefix':['numactl',f'--cpunodebind={cpus}',f'--membind={cpus}'],'load_mode':'none','memory_tier':'ddr'},
          {'name':'knl-mcdram-strict','llama':['--numa','numactl','--load-mode','none'],'prefix':['numactl',f'--cpunodebind={cpus}',f'--membind={hbs}'],'load_mode':'none','memory_tier':'mcdram-strict','requires_mcdram_fit':True},
        ]
        if len(hnodes)==1:
            numa.append({'name':'knl-mcdram-preferred','llama':['--numa','numactl','--load-mode','none'],'prefix':['numactl',f'--cpunodebind={cpus}',f'--preferred={hnodes[0]}'],'load_mode':'none','memory_tier':'mcdram-preferred'})
    # Attach the physical-core count actually available inside each locality.
    # This avoids e.g. running 48 llama threads on a 24-core socket-local policy
    # on a dual-8168 machine while still using all 48 cores for whole-machine policies.
    package_nodes={}
    node_by_id={int(n['id']):n for n in h.get('numa',{}).get('nodes',[])}
    for n in node_by_id.values():
        if not n.get('cpulist'): continue
        for pkg in n.get('packages',[]): package_nodes.setdefault(int(pkg),[]).append(int(n['id']))
    def locality_cores(pol):
        name=str(pol.get('name',''))
        if name.startswith('socket') and name.endswith('-local'):
            return max(1,int(c['cores_per_socket']))
        if name.startswith('node') and name.endswith('-local'):
            try:nid=int(name[4:].split('-',1)[0])
            except Exception:return phys
            n=node_by_id.get(nid,{})
            pkgs=[int(x) for x in n.get('packages',[])]
            if pkgs:
                total=0.0
                for pkg in pkgs:
                    total += float(c['cores_per_socket'])/max(1,len(set(package_nodes.get(pkg,[nid]))))
                return max(1,int(round(total)))
        return phys
    for pol in numa: pol['default_threads']=locality_cores(pol)

    return {'hardware_summary':{'class':c['class'],'generation':c['generation'],'phys_cores':phys,'logical':logical,'numa_nodes':nn,
                                'mcdram_visible_bytes':int(mc.get('visible_bytes') or 0)},
            'builds':builds,'threads':threads,'thread_probe':{'model':'granite4-350m','counts':thread_probe_counts,'default':phys},'numa_policies':numa,
            'depths':[512,2048,8192,16384,32768,65536], 'kv_types':['q4_0','q8_0','f16'],
            'search':{'quick_depths':[512,2048],
                      'screen_depths':[512,2048,8192],
                      'deep_depths':[16384,32768,65536],
                      'screen_cap':8192,'deep_cap':65536,
                      'quick_reps':1,'screen_reps':1,'final_reps':3,
                      'prune_fraction':0.5,'candidate_finalists':2}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('hardware'); ap.add_argument('-o','--output'); a=ap.parse_args()
    h=json.load(open(a.hardware)); p=plan(h); s=json.dumps(p,indent=2)
    if a.output: open(a.output,'w').write(s+'\n')
    else: print(s)
if __name__=='__main__': main()
