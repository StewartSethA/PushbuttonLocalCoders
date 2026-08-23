#!/usr/bin/env python3
from __future__ import annotations
import json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab')))
src=root/'llama.cpp'; plan=json.load(open(root/'plan.json')); hw=json.load(open(root/'hardware.json')); builds=root/'builds'; logs=root/'logs'; builds.mkdir(parents=True,exist_ok=True); logs.mkdir(exist_ok=True)
pkg=Path(__file__).resolve().parent.parent

BUILD_ENV_VARS=('CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','LD_LIBRARY_PATH','PKG_CONFIG_PATH','CMAKE_PREFIX_PATH')

def run(cmd,env=None,log=None):
    print('+',' '.join(map(str,cmd)),flush=True)
    with open(log,'w') if log else open(os.devnull,'w') as f:
        return subprocess.call(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)

def cmdout(cmd):
    try:return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
    except:return ''

def compiler():
    if hw['cpu']['class']=='knl':
        p=root/'knl-cc.txt'; cc=p.read_text().strip() if p.exists() else None
        if not cc:
            probe=root/'.compiler-probe.c'; probe.write_text('int x;\n')
            for x in ('gcc-14','gcc-13','gcc-12','gcc-11','gcc'):
                cxx=x.replace('gcc','g++')
                if shutil.which(x) and shutil.which(cxx) and subprocess.call([x,'-march=knl','-c',str(probe),'-o',str(root/'.compiler-probe.o')],stderr=subprocess.DEVNULL)==0:
                    cc=x; break
            for z in (root/'.compiler-probe.c',root/'.compiler-probe.o'):
                try:z.unlink()
                except:pass
        if not cc:raise SystemExit('No KNL-capable GCC. Run install first.')
        cxx=cc.replace('gcc','g++')
        if not shutil.which(cxx):raise SystemExit(f'KNL C++ compiler missing: {cxx}')
        return cc,cxx
    return os.environ.get('CC','gcc'),os.environ.get('CXX','g++')

def sanitized_env(cc,cxx):
    env=os.environ.copy(); inherited={k:env.get(k) for k in BUILD_ENV_VARS if env.get(k)}
    if os.environ.get('LAB_INHERIT_BUILD_ENV','0')!='1':
        for k in BUILD_ENV_VARS:env.pop(k,None)
    env['CC']=cc; env['CXX']=cxx
    env['CCACHE_DIR']=str(root/'cache/ccache'); env['CCACHE_BASEDIR']=str(root); env['CCACHE_COMPILERCHECK']='content'
    env['XDG_CACHE_HOME']=str(root/'cache/xdg')
    Path(env['CCACHE_DIR']).mkdir(parents=True,exist_ok=True); Path(env['XDG_CACHE_HOME']).mkdir(parents=True,exist_ok=True)
    return env,inherited

def pkg_exists(name):
    return shutil.which('pkg-config') is not None and subprocess.call(['pkg-config','--exists',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)==0

if not (src/'.git').exists():raise SystemExit(f'Missing source: {src}')
if hw['cpu']['class']=='knl':
    patcher=pkg/'knl/patches/apply_knl_patch.py'
    def run_patcher(mode):
        cmd=[sys.executable,str(patcher),str(src),mode]
        proc=subprocess.run(cmd,text=True,capture_output=True)
        text=(proc.stdout or '')+(proc.stderr or '')
        (logs/f'knl-patch-{mode.lstrip("-")}.log').write_text(text)
        if text: print(text,end='' if text.endswith('\n') else '\n')
        if proc.returncode:
            commit=cmdout(['git','-C',str(src),'rev-parse','HEAD'])
            print(f'ERROR: KNL patch compatibility failed for llama.cpp commit {commit or "unknown"}.',file=sys.stderr)
            print('The patcher is transactional; upstream source was not partially modified.',file=sys.stderr)
            patch_log = logs / ('knl-patch-{}.log'.format(mode.lstrip('-')))
            print('Patch log: {}'.format(patch_log), file=sys.stderr)
            raise SystemExit(proc.returncode)
    run_patcher('--dry-run')
    run_patcher('--apply')
cc,cxx=compiler(); env,inherited=sanitized_env(cc,cxx)
(root/'build-environment.json').write_text(json.dumps({'cc':cc,'cxx':cxx,'cc_version':cmdout([cc,'--version']).splitlines()[:1],'cxx_version':cmdout([cxx,'--version']).splitlines()[:1],'inherit_build_env':os.environ.get('LAB_INHERIT_BUILD_ENV','0')=='1','external_build_env_seen':inherited,'sanitized_vars':[] if os.environ.get('LAB_INHERIT_BUILD_ENV','0')=='1' else list(BUILD_ENV_VARS)},indent=2)+'\n')

wanted={x.strip() for x in os.environ.get('LAB_BUILD_NAMES','').split(',') if x.strip()}
if wanted:
    unknown=sorted(wanted-{str(x.get('name')) for x in plan.get('builds',[])})
    if unknown: raise SystemExit('Unknown LAB_BUILD_NAMES: '+', '.join(unknown))
    print('Selective build set:', ', '.join(sorted(wanted)), flush=True)

for b in plan['builds']:
    if wanted and str(b.get('name')) not in wanted:
        continue
    name=b['name']; d=builds/name; d.mkdir(exist_ok=True)
    if b.get('optional')=='openblas' and not pkg_exists('openblas'):
        print('SKIP',name,'OpenBLAS pkg-config metadata not found');continue
    if b.get('optional')=='memkind' and not pkg_exists('memkind'):
        print('SKIP',name,'memkind pkg-config metadata not found');continue
    args=['cmake','-S',str(src),'-B',str(d),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DLLAMA_BUILD_TESTS=ON','-DGGML_CPU=ON','-DGGML_LLAMAFILE=OFF','-DGGML_CCACHE=ON','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
    if b.get('knl'):
        args += ['-DGGML_NATIVE=OFF','-DGGML_AVX512=OFF','-DGGML_AVX512_KNL=ON',f'-DGGML_CPU_REPACK={b.get("repack","ON")}',f'-DGGML_KNL_PREFETCH={b["pf"]}',f'-DGGML_KNL_PREFETCH_DISTANCE={b["dist"]}',f'-DGGML_KNL_PREFETCH_HINT={b["hint"]}',f'-DGGML_KNL_Q4_0_UNROLL={b["unroll"]}',f'-DGGML_KNL_INSTRUMENT={b["instrument"]}']
        if b.get('hbm'):args += ['-DGGML_CPU_HBM=ON']
    else:args += b['cmake']
    meta={'name':name,'cmake_args':args,'cc':cc,'cxx':cxx,'started':time.time(),'optional':b.get('optional')}
    (d/'lab-build.json').write_text(json.dumps(meta,indent=2)+'\n')
    rc=run(args,env,logs/f'cmake-{name}.log')
    if rc:print('FAIL configure',name);continue
    if os.environ.get('LAB_BENCH_ONLY_BUILD','0')=='1':
        targets=['llama-bench']
        print('Ballpark build target: llama-bench only', flush=True)
    elif os.environ.get('LAB_FAST_BUILD','0')=='1':
        targets=['llama-bench','llama-server']
        print('Fast calibration build targets:', ', '.join(targets), flush=True)
    else:
        targets=['llama-bench','llama-cli','llama-server','llama-perplexity','test-backend-ops']
    rc=run(['cmake','--build',str(d),'-j',os.environ.get('BUILD_JOBS',str(os.cpu_count() or 4)),'--target']+targets,env,logs/f'build-{name}.log')
    meta.update({'finished':time.time(),'build_rc':rc,'ccache_stats':cmdout(['ccache','-s']) if shutil.which('ccache') else ''})
    (d/'lab-build.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(('PASS' if rc==0 else 'FAIL'),'build',name)
