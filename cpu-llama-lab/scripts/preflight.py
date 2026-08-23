#!/usr/bin/env python3
from __future__ import annotations
import json, os, platform, shutil, subprocess, sys
from pathlib import Path
PKG=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PKG))
from lib.models import configured_models
from lib.clock import synthetic_load_probe

ROOT=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab')))
HW=json.load(open(ROOT/'hardware.json')) if (ROOT/'hardware.json').exists() else {}

def run(cmd):
    try:return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
    except Exception as e:return ''

def read(p):
    try:return Path(p).read_text().strip()
    except:return ''

def add(items,severity,code,message,evidence=None):
    d={'severity':severity,'code':code,'message':message}
    if evidence is not None:d['evidence']=evidence
    items.append(d)

def model_paths():
    return configured_models()

def cpufreq_state():
    gs=[]
    for p in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
        gs.append({'policy':p.name,'governor':read(p/'scaling_governor'),'driver':read(p/'scaling_driver'),'min_khz':read(p/'scaling_min_freq'),'max_khz':read(p/'scaling_max_freq')})
    return gs

def main():
    items=[]; phys=1
    c=HW.get('cpu',{})
    phys=max(1,int(c.get('sockets',1))*int(c.get('cores_per_socket',1)))
    numa=HW.get('numa',{}).get('count',1)

    load=os.getloadavg() if hasattr(os,'getloadavg') else (0,0,0)
    if load[0] > max(2,phys*0.25):add(items,'high','system-load',f'1-minute load average {load[0]:.2f} is high for a clean benchmark on {phys} physical cores.',load)

    swaps=[]
    try:
        lines=Path('/proc/swaps').read_text().splitlines()[1:]
        for l in lines:
            a=l.split(); swaps.append({'path':a[0],'type':a[1],'size_kib':int(a[2]),'used_kib':int(a[3]),'priority':int(a[4])})
    except:pass
    used=sum(x['used_kib'] for x in swaps)
    if used:add(items,'high','swap-used',f'{used/1024:.1f} MiB swap is currently in use; results may have large latency variance.',swaps)
    elif swaps:add(items,'info','swap-enabled','Swap is enabled but unused.',swaps)

    governors=cpufreq_state(); bad=[x for x in governors if x['governor'] and x['governor']!='performance']
    if bad:add(items,'medium','cpu-governor','One or more CPU frequency policies are not set to performance; record this or set performance for low-variance publication runs.',bad)

    # Frequency is a first-class benchmark invariant. A short all-core load probe
    # catches PSU/power/thermal/governor caps before an expensive sweep starts;
    # every actual performance row is independently sampled again by sweep.py.
    try:
        clock_probe=synthetic_load_probe(phys, c.get('nominal_base_mhz'))
    except Exception as e:
        clock_probe={'status':'unavailable','error':repr(e)}
    if clock_probe.get('status')=='throttled':
        add(items,'high','sustained-frequency-throttle',
            f"Loaded CPU clock {clock_probe.get('loaded_mhz',0):.0f} MHz is below the allowed reference ratio; benchmark results would be rejected.",clock_probe)
    elif clock_probe.get('status') in ('unavailable','insufficient-samples'):
        add(items,'medium','clock-validation-incomplete','Loaded clock could not be validated reliably; per-test clock sampling will still be recorded.',clock_probe)
    else:
        print(f"INFO   loaded-clock: {clock_probe.get('loaded_mhz',0):.0f} MHz ({clock_probe.get('status')}, reference={clock_probe.get('reference_mhz') or 'n/a'} MHz)")

    no_turbo=read('/sys/devices/system/cpu/intel_pstate/no_turbo')
    boost=read('/sys/devices/system/cpu/cpufreq/boost') or read('/sys/devices/system/cpu/amd_pstate/cpb_boost')
    if no_turbo=='1' or boost=='0':add(items,'medium','turbo-disabled','CPU turbo/boost appears disabled. This is valid if intentional, but must be stated in comparisons.',{'intel_no_turbo':no_turbo,'boost':boost})

    nb=read('/proc/sys/kernel/numa_balancing')
    if numa>1 and nb not in ('','0'):add(items,'medium','auto-numa-balancing','Kernel automatic NUMA balancing is enabled. Explicit mbind/node-local policies are still tested, but publication runs should record this state.',nb)

    thp=read('/sys/kernel/mm/transparent_hugepage/enabled'); thpd=read('/sys/kernel/mm/transparent_hugepage/defrag')
    if thp:add(items,'info','transparent-hugepages','Transparent Huge Page policy recorded; do not change it between compared runs.',{'enabled':thp,'defrag':thpd})

    perf=(ROOT/'perf-status.txt').read_text().strip() if (ROOT/'perf-status.txt').exists() else ('present-unprobed' if shutil.which('perf') else 'missing')
    paranoid=read('/proc/sys/kernel/perf_event_paranoid')
    if perf not in ('ok',):add(items,'medium','perf-unavailable',f'perf is not fully usable ({perf}); performance counters will be omitted.',{'perf_status':perf,'perf_event_paranoid':paranoid})

    allowed=''
    try:
        for l in Path('/proc/self/status').read_text().splitlines():
            if l.startswith('Cpus_allowed_list:'):allowed=l.split(':',1)[1].strip();break
    except:pass
    if allowed and c.get('logical_cpus'):
        # Cpus_allowed_list may contain possible/offline CPU IDs beyond the online
        # topology (common on some kernels). Compare the effective affinity count,
        # not the textual range, to avoid false 'restricted' warnings.
        try:
            allowed_count=len(os.sched_getaffinity(0))
        except Exception:
            allowed_count=None
        if allowed_count is not None and allowed_count < int(c['logical_cpus']):
            add(items,'medium','cpuset-restricted',f'Process CPU allowance is restricted to {allowed} ({allowed_count}/{c["logical_cpus"]} logical CPUs); results represent this cpuset, not necessarily the whole machine.',allowed)

    contam={}
    for k in ('CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','LD_LIBRARY_PATH','PKG_CONFIG_PATH','CMAKE_PREFIX_PATH'):
        if os.environ.get(k):contam[k]=os.environ[k]
    if contam and os.environ.get('LAB_INHERIT_BUILD_ENV','0')!='1':add(items,'info','build-env-sanitized','Build-affecting environment variables are set, but v5 sanitizes them unless LAB_INHERIT_BUILD_ENV=1.',contam)
    elif contam:add(items,'high','build-env-inherited','LAB_INHERIT_BUILD_ENV=1: external build/link paths can affect ABI and reproducibility.',contam)

    # RAM and disk sanity relative to model files.
    mem_avail=None
    try:
        for l in Path('/proc/meminfo').read_text().splitlines():
            if l.startswith('MemAvailable:'):mem_avail=int(l.split()[1])*1024;break
    except:pass
    mods=[]
    for k,p in model_paths():
        st=p.stat(); fs=run(['findmnt','-T',str(p),'-no','FSTYPE,SOURCE,TARGET']) if shutil.which('findmnt') else ''
        mods.append({'name':k,'path':str(p),'size_bytes':st.st_size,'filesystem':fs})
        if fs and any(x in fs.lower() for x in ('nfs','cifs','fuse.sshfs','lustre','ceph')):
            add(items,'medium','network-model-filesystem',f'Model {p.name} appears to reside on a network/distributed filesystem; model load and first-touch behavior may differ.',fs)
    if mem_avail and mods:
        largest=max(x['size_bytes'] for x in mods)
        if mem_avail < largest*1.15:add(items,'high','low-free-memory',f'Available RAM ({mem_avail/2**30:.1f} GiB) is close to/below the largest model ({largest/2**30:.1f} GiB).',None)
    try:
        du=shutil.disk_usage(ROOT)
        if du.free < 8*2**30:add(items,'high','low-disk-space',f'Only {du.free/2**30:.1f} GiB free below LAB_ROOT; multiple builds/logs may fail.')
        elif du.free < 25*2**30:add(items,'medium','limited-disk-space',f'{du.free/2**30:.1f} GiB free below LAB_ROOT; a full build matrix can consume substantial space.')
    except:pass

    # KNL-specific MCDRAM sanity from detector.
    mc=HW.get('memory',{}).get('mcdram',{})
    if c.get('class')=='knl':
        mode=mc.get('mode_hint','unknown')
        if mode not in ('flat','hybrid-flat-visible'):
            add(items,'medium','knl-mcdram-not-flat',f'MCDRAM is not detected as a separate memory-only NUMA node ({mode}). Explicit Flat-mode HBM-vs-DDR placement tests will be unavailable; Cache mode remains a valid separate ablation.',mc)
        if HW.get('memory',{}).get('populated_dimms')==1:add(items,'info','knl-one-dimm','One populated DIMM detected: this is a useful deliberate one-stick configuration, but tag the physical channel configuration explicitly.')

    obj={
      'time':run(['date','--iso-8601=seconds']) or '', 'host':platform.node(), 'kernel':platform.release(),
      'loadavg':load,'physical_cores':phys,'numa_nodes':numa,'cpufreq':governors,'loaded_clock_probe':clock_probe,
      'numa_balancing':nb,'cpus_allowed_list':allowed,'thp':{'enabled':thp,'defrag':thpd},'swap':swaps,
      'perf':{'status':perf,'perf_event_paranoid':paranoid},'models':mods,'mem_available_bytes':mem_avail,
      'issues':items,
    }
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'preflight.json').write_text(json.dumps(obj,indent=2)+'\n')
    lines=['# Benchmark preflight','',f"Host: `{obj['host']}`  Kernel: `{obj['kernel']}`",'', '| severity | code | message |','|---|---|---|']
    for x in items:lines.append(f"| {x['severity']} | {x['code']} | {x['message'].replace('|','/')} |")
    if not items:lines.append('| ok | none | No noteworthy conditions detected. |')
    (ROOT/'preflight.md').write_text('\n'.join(lines)+'\n')
    for x in items:print(f"{x['severity'].upper():6s} {x['code']}: {x['message']}")
    strict=os.environ.get('LAB_STRICT','0')=='1' or os.environ.get('LAB_BENCH_STRICT','0')=='1'
    if strict and any(x['severity'] in ('high','error') for x in items):return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
