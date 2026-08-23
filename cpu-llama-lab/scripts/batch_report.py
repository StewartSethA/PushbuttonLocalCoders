#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import os

ROOT = Path(os.environ.get('LAB_ROOT', str(Path.cwd() / '.cpu-llama-lab')))
OUT = ROOT / 'results'
CAL = ROOT / 'calibration'
CAL.mkdir(parents=True, exist_ok=True)


def speed(rec: dict[str, Any]) -> float:
    if isinstance(rec.get('tps'), (int, float)):
        return float(rec['tps'])
    raw = rec.get('raw', [])
    if isinstance(raw, dict):
        raw = [raw]
    vals = []
    if isinstance(raw, list):
        for r in raw:
            if not isinstance(r, dict):
                continue
            for k in ('avg_ts', 'tokens_per_second', 't_s'):
                if isinstance(r.get(k), (int, float)):
                    vals.append(float(r[k]))
                    break
    return max(vals) if vals else 0.0

rows = []
for p in sorted(OUT.glob('bench-rank*.jsonl')):
    for line in p.read_text(errors='replace').splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('phase') != 'batch-tune' or r.get('kind') != 'pp':
            continue
        x = {
            'timestamp': r.get('timestamp'),
            'run_key': r.get('run_key'),
            'model': r.get('model'),
            'build': r.get('build'),
            'policy': r.get('policy_name'),
            'threads': r.get('threads'),
            'batch': r.get('batch'),
            'ubatch': r.get('ubatch'),
            'status': r.get('status'),
            'pp_tps': speed(r),
            'wall_s': r.get('wall_s'),
            'clock_mhz': (r.get('clock') or {}).get('loaded_mhz') if isinstance(r.get('clock'), dict) else None,
            'clock_status': r.get('clock_status'),
        }
        rows.append(x)

if not rows:
    raise SystemExit('No batch-tune rows found under .cpu-llama-lab/results')

groups = defaultdict(list)
for r in rows:
    groups[(r['model'], r['build'], r['policy'], r['threads'])].append(r)

report = {'version': 1, 'rows': rows, 'groups': []}
print(f'BATCH-TUNE SALVAGE: {len(rows)} historical rows found')
print('NOTE: old rows without clock_mhz are useful for heuristic batch ordering but not publication-grade comparisons.')
for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
    ok = [x for x in rs if x['status'] == 'ok' and x['pp_tps'] > 0]
    ok.sort(key=lambda x: x['pp_tps'], reverse=True)
    model, build, policy, threads = key
    best = ok[0]['pp_tps'] if ok else 0.0
    print('\n%s | %s | %s | t=%s' % (model, build, policy, threads))
    if not ok:
        print('  no successful row with recoverable avg_ts')
    for x in ok[:7]:
        rel = x['pp_tps'] / best if best else 0.0
        clk = ('%.0fMHz' % x['clock_mhz']) if isinstance(x['clock_mhz'], (int, float)) else 'clock-unknown'
        print('  b=%4s ub=%4s  pp=%8.2f t/s  rel=%5.3f  wall=%6.1fs  %s' %
              (x['batch'], x['ubatch'], x['pp_tps'], rel, float(x['wall_s'] or 0), clk))
    report['groups'].append({'model': model, 'build': build, 'policy': policy, 'threads': threads,
                             'best': ok[0] if ok else None, 'ranked': ok})

j = CAL / 'batch-salvage.json'
t = CAL / 'batch-salvage.tsv'
j.write_text(json.dumps(report, indent=2) + '\n')
with t.open('w') as f:
    f.write('model\tbuild\tpolicy\tthreads\tbatch\tubatch\tstatus\tpp_tps\trel_to_group_best\twall_s\tclock_mhz\n')
    for g in report['groups']:
        best = float((g.get('best') or {}).get('pp_tps') or 0)
        for x in g['ranked']:
            rel = x['pp_tps'] / best if best else 0.0
            f.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.6f\t%.6f\t%.3f\t%s\n' %
                    (x['model'], x['build'], x['policy'], x['threads'], x['batch'], x['ubatch'], x['status'],
                     x['pp_tps'], rel, float(x['wall_s'] or 0), '' if x['clock_mhz'] is None else x['clock_mhz']))
print('\nWROTE %s' % j)
print('WROTE %s' % t)
