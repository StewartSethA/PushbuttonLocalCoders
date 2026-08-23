#!/usr/bin/env python3
import json, os, platform, re, shutil, subprocess, sys
from pathlib import Path

def run(cmd):
    try: return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return ''

def lscpu_pairs():
    out={}
    txt=run(['lscpu'])
    for line in txt.splitlines():
        if ':' in line:
            k,v=line.split(':',1); out[k.strip()]=v.strip()
    return out

def parse_nodes():
    nodes=[]
    base=Path('/sys/devices/system/node')
    for p in sorted(base.glob('node[0-9]*')):
        n=int(p.name[4:]); cpus=''
        if (p/'cpulist').exists(): cpus=(p/'cpulist').read_text().strip()
        mem=''
        meminfo=p/'meminfo'
        if meminfo.exists():
            m=re.search(r'MemTotal:\s+(\d+) kB',meminfo.read_text())
            if m: mem=int(m.group(1))*1024
        packages=set()
        if cpus:
            for part in cpus.split(','):
                if '-' in part:
                    a,b=map(int,part.split('-',1)); ids=range(a,b+1)
                else:
                    try: ids=[int(part)]
                    except ValueError: ids=[]
                for cpu in ids:
                    q=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id')
                    try: packages.add(int(q.read_text().strip()))
                    except Exception: pass
        nodes.append({'id':n,'cpulist':cpus,'memory_bytes':mem,'packages':sorted(packages)})
    return nodes

def compiler_info():
    ans=[]
    for c in ('gcc-14','g++-14','gcc','g++','clang','clang++','icc','icpc','icx','icpx'):
        if shutil.which(c): ans.append({'name':c,'path':shutil.which(c),'version':run([c,'--version']).split('\n')[0]})
    return ans

def cpu_class(name,vendor,family,model,flags):
    n=name.lower(); f=int(family or 0); m=int(model or -1)
    if vendor=='GenuineIntel' and f==6 and m==87: return ('knl','knights-landing')
    if 'xeon phi' in n: return ('knl','knights-landing')
    if vendor=='GenuineIntel':
        if re.search(r'e5-.*v4',n): return ('xeon','broadwell-ep')
        if any(x in n for x in ['platinum 81','gold 61','silver 41']): return ('xeon','skylake-sp')
        if any(x in n for x in ['platinum 82','gold 62']): return ('xeon','cascade-lake')
        if any(x in n for x in ['platinum 83','gold 63']): return ('xeon','ice-lake')
        if any(x in n for x in ['platinum 84','gold 64']): return ('xeon','sapphire-rapids')
        if any(x in n for x in ['platinum 85','gold 65']): return ('xeon','emerald-rapids')
        if any(x in n for x in ['xeon 6','granite rapids']): return ('xeon','granite-rapids')
        return ('xeon','intel-x86')
    if vendor=='AuthenticAMD':
        if 'epyc' in n:
            # EPYC generation is encoded in the final digit of the four-character
            # SKU family (including F parts such as 7F52/75F3). Keep this broad so
            # Naples/Rome/Milan/Genoa/Turin surplus CPUs all plan correctly.
            if re.search(r'\b7[0-9a-f]{2}1p?\b',n): return ('epyc','zen1')
            if re.search(r'\b7[0-9a-f]{2}2p?\b',n): return ('epyc','zen2')
            if re.search(r'\b7[0-9a-f]{2}3p?\b',n): return ('epyc','zen3')
            if re.search(r'\b9[0-9a-f]{2}4p?\b',n): return ('epyc','zen4')
            if re.search(r'\b9[0-9a-f]{2}5p?\b',n): return ('epyc','zen5')
            return ('epyc','amd-epyc')
        if 'threadripper' in n:
            # X399 first/second generation: keep these explicit because they are
            # useful low-cost four-channel CPU-inference comparison platforms.
            if re.search(r'\b(?:1920x|1950x)\b', n): return ('threadripper','zen1')
            if re.search(r'\b(?:2920x|2950x|2970wx|2990wx)\b', n): return ('threadripper','zen+')
            if any(x in n for x in ['39','pro 39']): return ('threadripper','zen2')
            if any(x in n for x in ['59','pro 59']): return ('threadripper','zen3')
            if any(x in n for x in ['79','pro 79']): return ('threadripper','zen4')
            if any(x in n for x in ['99','pro 99']): return ('threadripper','zen5')
            return ('threadripper','amd-threadripper')
        return ('amd','amd-x86')
    return ('x86','unknown')


def nominal_base_mhz(name):
    n=name.lower()
    # Stable published base clocks for platforms explicitly targeted by this lab.
    # sysfs base_frequency, when available, remains the preferred runtime reference.
    known=[
        (r'\bxeon phi.*\b7250\b',1400.0),
        (r'\bthreadripper 1920x\b',3500.0),
        (r'\bthreadripper 1950x\b',3400.0),
    ]
    for pat,mhz in known:
        if re.search(pat,n): return mhz
    return None

def channels_hint(cls,gen,name):
    n=name.lower()
    if cls=='knl': return 6
    if gen=='broadwell-ep': return 4
    if gen in ('skylake-sp','cascade-lake','ice-lake'): return 6
    if gen in ('sapphire-rapids','emerald-rapids'): return 8
    if gen=='granite-rapids':
        # Xeon 6 server parts expose up to 12 channels; some Xeon 600 workstation
        # derivatives expose 8, so keep the hint conservative for known X-series.
        if re.search(r'\b(?:698x|676x|674x)\b', n): return 8
        return 12
    if cls=='epyc': return 12 if gen in ('zen4','zen5') else 8
    if cls=='threadripper':
        # Do not test bare 'pro': every CPU name ends in 'Processor'.
        if re.search(r'\bthreadripper\s+pro\b', n): return 8
        return 4
    return None

def mcdram_info(cls,nodes):
    if cls != 'knl':
        return {'mode_hint':'not-applicable','visible_nodes':[],'visible_bytes':0}
    memonly=[n for n in nodes if not (n.get('cpulist') or '').strip() and int(n.get('memory_bytes') or 0) > 512*1024**2]
    total=sum(int(n.get('memory_bytes') or 0) for n in memonly)
    if total >= 12*1024**3: mode='flat'
    elif total >= 1*1024**3: mode='hybrid-flat-visible'
    else: mode='cache-or-hidden'
    return {'mode_hint':mode,'visible_nodes':[n['id'] for n in memonly],'visible_bytes':total}

def dimms():
    txt=run(['dmidecode','-t','memory']) if shutil.which('dmidecode') else ''
    vals=[]
    cur={}
    for line in txt.splitlines():
        s=line.strip()
        if s=='Memory Device':
            if cur: vals.append(cur)
            cur={}
        elif ':' in s:
            k,v=s.split(':',1); cur[k.strip()]=v.strip()
    if cur: vals.append(cur)
    vals=[x for x in vals if x.get('Size','').lower() not in ('','no module installed','not installed','unknown')]
    return vals

def main():
    L=lscpu_pairs(); name=L.get('Model name',''); vendor=L.get('Vendor ID',''); flags=set(L.get('Flags','').split())
    cls,gen=cpu_class(name,vendor,L.get('CPU family'),L.get('Model'),flags)
    nodes=parse_nodes(); d=dimms()
    out={
      'hostname':platform.node(),'kernel':platform.release(),'os':platform.platform(),
      'cpu':{'name':name,'vendor':vendor,'family':L.get('CPU family'),'model':L.get('Model'),'class':cls,'generation':gen,
             'sockets':int(L.get('Socket(s)','1') or 1),'cores_per_socket':int(L.get('Core(s) per socket','1') or 1),
             'threads_per_core':int(L.get('Thread(s) per core','1') or 1),'logical_cpus':os.cpu_count() or 1,'flags':sorted(flags),
             'nominal_base_mhz':nominal_base_mhz(name)},
      'numa':{'count':len(nodes) or 1,'nodes':nodes},
      'memory':{'populated_dimms':len(d) if d else None,'dimms':d,'platform_channels_per_socket_hint':channels_hint(cls,gen,name),'mcdram':mcdram_info(cls,nodes)},
      'toolchain':compiler_info(),
      'features':{
        'avx2':'avx2' in flags,'avx512f':'avx512f' in flags,'avx512bw':'avx512bw' in flags,'avx512dq':'avx512dq' in flags,
        'avx512vl':'avx512vl' in flags,'avx512vnni':('avx512_vnni' in flags or 'avx512vnni' in flags),
        'avxvnni':'avx_vnni' in flags,'avx512bf16':'avx512_bf16' in flags,'amxint8':'amx_int8' in flags,'amxbf16':'amx_bf16' in flags,
        'knl_pf':'avx512pf' in flags,'knl_er':'avx512er' in flags,'prefetchwt1':'prefetchwt1' in flags,
      }
    }
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
