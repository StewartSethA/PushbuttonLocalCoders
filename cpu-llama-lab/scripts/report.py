#!/usr/bin/env python3
import json,os,re,sys
from pathlib import Path
R=Path(os.environ.get('LAB_ROOT',str(Path.cwd()/'.cpu-llama-lab'))); out=R/'report'; out.mkdir(parents=True,exist_ok=True)
h=json.load(open(R/'hardware.json')); p=json.load(open(R/'plan.json')); rows=[]
for f in (R/'results').glob('bench-rank*.jsonl'):
 for l in f.read_text(errors='ignore').splitlines():
  try:rows.append(json.loads(l))
  except:pass

def sp(r):
 if isinstance(r.get('tps'),(int,float)): return float(r['tps'])
 raw=r.get('raw',[]); raw=[raw] if isinstance(raw,dict) else raw; vs=[]
 for x in raw if isinstance(raw,list) else []:
  if isinstance(x,dict):
   for k in ('avg_ts','tokens_per_second','t_s'):
    if isinstance(x.get(k),(int,float)):vs.append(float(x[k]));break
 return max(vs) if vs else None

def jload(p):
 try:return json.load(open(p))
 except:return {}
manifest=jload(R/'manifest.json'); pre=jload(R/'preflight.json'); abi=jload(R/'abi/abi-audit.json')
L=['# CPU llama.cpp Lab report','',f"**Host:** {h['hostname']}",f"**CPU:** {h['cpu']['name']}",f"**Detected profile:** {h['cpu']['class']} / {h['cpu']['generation']}",f"**Sockets / NUMA:** {h['cpu']['sockets']} / {h['numa']['count']}",f"**Kernel:** {h.get('kernel','')}",f"**glibc:** {manifest.get('glibc','')}",'']
if h['cpu']['class']=='knl':
 mc=h.get('memory',{}).get('mcdram',{});L += [f"**KNL MCDRAM mode hint:** {mc.get('mode_hint','unknown')} — visible memory-only nodes {mc.get('visible_nodes',[])}",'']
L+=['## Reproducibility / environment','','The lab sanitizes external build/link flags by default, uses suite-local ccache/uv caches, records exact compiler/package versions, and post-build audits dynamic linkage. Set `LAB_INHERIT_BUILD_ENV=1` only when deliberately testing a custom toolchain/library path.','']
if manifest.get('llama_commit'):L.append(f"**llama.cpp commit:** `{manifest['llama_commit']}`")
be=jload(R/'build-environment.json')
if be:L += [f"**Build compiler:** `{be.get('cc','')}` / `{be.get('cxx','')}`",f"**External build environment inherited:** `{be.get('inherit_build_env',False)}`",'']
issues=pre.get('issues',[])
L+=['## Preflight','','| severity | condition | message |','|---|---|---|']
clkpre=pre.get('loaded_clock_probe',{}) if isinstance(pre,dict) else {}
if clkpre:
 cm=clkpre.get('loaded_mhz');cr=clkpre.get('reference_mhz');rr=clkpre.get('loaded_to_reference_ratio')
 L += [f"**Loaded clock preflight:** {float(cm or 0):.0f} MHz; reference {float(cr or 0):.0f} MHz ({clkpre.get('reference_kind','')}); ratio {float(rr or 0):.3f}; status `{clkpre.get('status','')}`.",'']
if issues:
 for x in issues:L.append(f"| {x.get('severity','')} | {x.get('code','')} | {x.get('message','').replace('|','/')} |")
else:L.append('| ok | none | No noteworthy conditions recorded. |')
L+=['',f"**ABI unresolved libraries:** {abi.get('unresolved_count','not audited')}",'']
L+=['## Applicability','','The KNL source patch is applied only on detected Knights Landing. Other CPUs use unmodified upstream llama.cpp; their tuning is performed entirely by builds/runtime policies in the lab.','']
L+=['## Benchmark results','','The broad screen never exceeds 2K. Finalist depth screening uses a persistent cached prefix through 8K. Exactly one winning quant/runtime branch per model family is then allowed to continue beyond 8K, with the deep sweep capped at 64K by default. PP interval rows report only the newly processed suffix (`interval_start -> depth`); TG rows are endpoint samples on the same recycled prefix. Failed/time-limited branches remain explicit records rather than vanishing.','', '| phase | status | kind | model | build | NUMA policy | threads | KV | interval/depth | tok/s | loaded MHz | clock status | max RSS GiB |','|---|---|---|---|---|---|---:|---|---|---:|---:|---|---:|']
for r in rows:
 s=sp(r)
 rss=r.get('max_rss_bytes');rg=f'{rss/2**30:.2f}' if rss else ''
 dep=f"{r.get('interval_start')}->{r.get('depth')}" if r.get('kind')=='pp_interval' and r.get('interval_start') is not None else str(r.get('depth',''))
 sval=f'{s:.3f}' if s is not None else ''
 clk=r.get('clock') or {}; cm=clk.get('loaded_mhz')
 if cm is None and isinstance(r.get('clock_samples'),list) and r.get('clock_samples'):
  vals=[x.get('loaded_mhz') for x in r.get('clock_samples') if isinstance(x,dict) and isinstance(x.get('loaded_mhz'),(int,float))]; cm=(sum(vals)/len(vals) if vals else None)
 cmt=f'{float(cm):.0f}' if isinstance(cm,(int,float)) else ''
 L.append(f"| {r.get('phase','')} | {r.get('status','')} | {r.get('kind','')} | {r.get('model','')} | {r.get('build','')} | {r.get('policy_name','')} | {r.get('threads','')} | {r.get('kv','')} | {dep} | {sval} | {cmt} | {r.get('clock_status','')} | {rg} |")
agent=jload(R/'models/agent-winners.json')
winners=jload(R/'models/winners.json')
if agent:
 L+=['','## Agentic coding gate','','Deep-sweep eligibility is quality-constrained: the fastest 8K candidate is tested first on deterministic repository-repair/tool-use tasks over the Anthropic Messages API. A failing quant is rejected and a higher-fidelity quant is tried. A family with no acceptable quant gets no deep sweep. Responsiveness is a hard gate alongside correctness.','',f"**Agent context:** {agent.get('agent_ctx','')} tokens; **minimum task pass rate:** {agent.get('min_pass_rate','')}; **minimum PP/TG:** {agent.get('min_pp_tps','')}/{agent.get('min_tg_tps','')} tok/s; **maximum median task time:** {agent.get('max_median_task_s','')} s; **Claude Code required when absent:** {agent.get('require_claude_code',False)}",'']
 if agent.get('rejected'):
  L+=['| rejected family | quant | reason | pass rate | median task s |','|---|---|---|---:|---:|']
  for x in agent['rejected']:
   L.append(f"| {x.get('family','')} | {x.get('quant','')} | {x.get('reason','')} | {x.get('pass_rate','')} | {x.get('median_task_s','')} |")
if winners.get('winners'):
 L+=['','## Per-family deep-sweep winners','','Only quality-accepted branches are permitted to run beyond the 8K screening ceiling in the unattended/full-sweep workflow.','','| family | quant/model | build | NUMA | threads | KV | batch/ubatch | PP@8K | TG@8K | score | agent pass | agent median s |','|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|']
 for w in winners['winners']:
  L.append(f"| {w.get('family','')} | {w.get('quant',w.get('model',''))} | {w.get('build','')} | {w.get('policy_name','')} | {w.get('threads','')} | {w.get('kv','')} | {w.get('batch','')}/{w.get('ubatch','')} | {float(w.get('pp_tps',0)):.3f} | {float(w.get('tg_tps',0)):.3f} | {float(w.get('score',0)):.4f} | {w.get('agent_pass_rate','')} | {w.get('agent_median_task_s','')} |")

L+=['','## NUMA interpretation','','On multi-socket/multi-NUMA systems the lab also tests complete `socketN-local` policies when sysfs can map NUMA nodes to physical packages. This is important because one NUMA node may expose only part of a socket memory controller, while cross-socket placement can lose decode bandwidth to remote traffic.','', '`distribute-mmap` uses upstream first-touch NUMA behavior; for meaningful comparison the harness can cold-drop page cache when `LAB_DROP_CACHES=1`. `interleave` is a different policy, not a mirror substitute. `node-workers` measures independent per-node replicas and aggregate multi-stream throughput. Weights-only mirror is included only if the binary actually exposes it.','',
'## Correctness / upstream readiness','',
'- Every KNL build is paired with `test-backend-ops` and code-generation auditing.','- Every produced ELF is checked with `ldd`/`readelf`; unresolved dependencies fail `abi-audit`.','- Generic Xeon/EPYC/Threadripper builds do not modify llama.cpp source.','- KNL changes are contained in the dedicated patcher and can be reverted cleanly.','- Experimental NUMA mirroring is intentionally not smuggled into the KNL patch.','']
if (R/'abi/ABI.md').exists():L += ['See `abi/ABI.md` for per-binary GLIBC/GLIBCXX and RPATH/RUNPATH details.','']
(out/'REPORT.md').write_text('\n'.join(L)+'\n');print(out/'REPORT.md')
