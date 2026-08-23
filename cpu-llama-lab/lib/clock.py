#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

CPU_SYS = Path('/sys/devices/system/cpu')
PROC = Path('/proc')


def _read_text(p: Path) -> str:
    try:
        return p.read_text().strip()
    except Exception:
        return ''


def parse_cpulist(text: str) -> list[int]:
    out = []
    for part in (text or '').split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                a, b = (int(x) for x in part.split('-', 1))
                out.extend(range(a, b + 1))
            except Exception:
                pass
        else:
            try:
                out.append(int(part))
            except Exception:
                pass
    return sorted(set(out))


def _percentile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    xs = sorted(float(x) for x in vals)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * min(1.0, max(0.0, q))
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _cpu_freq_path(cpu: int, name: str, cpu_sys: Path = CPU_SYS) -> Path:
    return cpu_sys / ('cpu%d' % int(cpu)) / 'cpufreq' / name


def _read_khz(cpu: int, names: tuple[str, ...], cpu_sys: Path = CPU_SYS) -> float | None:
    for name in names:
        s = _read_text(_cpu_freq_path(cpu, name, cpu_sys))
        try:
            v = float(s)
            if v > 0:
                return v
        except Exception:
            pass
    return None


def available_clock_reference(cpus: list[int] | None = None, cpu_sys: Path = CPU_SYS, nominal_base_mhz: float | None = None) -> dict[str, Any]:
    cpus = list(cpus if cpus is not None else sorted(os.sched_getaffinity(0)))
    override = os.environ.get('LAB_CLOCK_REFERENCE_MHZ', '').strip()
    if override:
        try:
            mhz = float(override)
            if mhz > 0:
                return {'mhz': mhz, 'kind': 'env-override', 'authoritative': True}
        except Exception:
            pass
    bases = [_read_khz(c, ('base_frequency',), cpu_sys) for c in cpus]
    bases = [x for x in bases if x]
    if bases:
        return {'mhz': statistics.median(bases) / 1000.0, 'kind': 'sysfs-base_frequency', 'authoritative': True}
    if nominal_base_mhz and nominal_base_mhz > 0:
        return {'mhz': float(nominal_base_mhz), 'kind': 'detected-model-base', 'authoritative': True}
    maxes = [_read_khz(c, ('cpuinfo_max_freq', 'scaling_max_freq'), cpu_sys) for c in cpus]
    maxes = [x for x in maxes if x]
    if maxes:
        return {'mhz': statistics.median(maxes) / 1000.0, 'kind': 'sysfs-max-frequency', 'authoritative': False}
    return {'mhz': None, 'kind': 'unavailable', 'authoritative': False}


def _stat_processor(stat_text: str) -> int | None:
    # /proc/PID/stat field 2 is parenthesized and can contain spaces/parens.
    # Field 39 (processor) is item 36 after the closing ')' and field 3.
    try:
        rest = stat_text[stat_text.rfind(')') + 2:].split()
        return int(rest[36]) if len(rest) > 36 else None
    except Exception:
        return None


def _children(pid: int, proc_root: Path = PROC) -> list[int]:
    s = _read_text(proc_root / str(pid) / 'task' / str(pid) / 'children')
    out = []
    for x in s.split():
        try:
            out.append(int(x))
        except Exception:
            pass
    return out


def process_tree_pids(root_pid: int, proc_root: Path = PROC) -> list[int]:
    seen = set(); todo = [int(root_pid)]
    while todo and len(seen) < 4096:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        todo.extend(x for x in _children(p, proc_root) if x not in seen)
    return sorted(seen)


def process_tree_cpus(root_pid: int, proc_root: Path = PROC) -> list[int]:
    cpus = set()
    for pid in process_tree_pids(root_pid, proc_root):
        tdir = proc_root / str(pid) / 'task'
        try:
            tids = list(tdir.iterdir())
        except Exception:
            continue
        for td in tids:
            c = _stat_processor(_read_text(td / 'stat'))
            if c is not None:
                cpus.add(c)
    return sorted(cpus)



def physical_cpu_representatives(allowed: list[int] | None = None, cpu_sys: Path = CPU_SYS) -> list[int]:
    try:
        allowed_set=set(allowed if allowed is not None else os.sched_getaffinity(0))
    except Exception:
        allowed_set=set(allowed or range(os.cpu_count() or 1))
    by_core={}
    for cpu in sorted(allowed_set):
        pkg=_read_text(cpu_sys/('cpu%d'%cpu)/'topology'/'physical_package_id')
        core=_read_text(cpu_sys/('cpu%d'%cpu)/'topology'/'core_id')
        if pkg and core:
            by_core.setdefault((pkg,core),cpu)
    return sorted(by_core.values()) if by_core else sorted(allowed_set)


def select_sample_cpus(cpus: list[int]) -> list[int]:
    xs=sorted(set(int(x) for x in cpus))
    cap=max(1,int(os.environ.get('LAB_CLOCK_MAX_CPUS','32')))
    if len(xs)<=cap:return xs
    # Evenly span the active CPU set instead of sampling only low-numbered cores.
    idx=sorted(set(int(round(i*(len(xs)-1)/(cap-1))) for i in range(cap))) if cap>1 else [0]
    return [xs[i] for i in idx]

def _sample_freqs(cpus: list[int], cpu_sys: Path = CPU_SYS) -> list[float]:
    vals = []
    for c in sorted(set(int(x) for x in cpus)):
        khz = _read_khz(c, ('scaling_cur_freq', 'cpuinfo_cur_freq'), cpu_sys)
        if khz:
            vals.append(khz / 1000.0)
    return vals


def summarize_clock_samples(samples: list[dict[str, Any]], reference: dict[str, Any], min_ratio: float = 0.90, min_mhz: float | None = None) -> dict[str, Any]:
    vals = []
    active_counts = []; sampled_counts=[]
    for s in samples:
        xs = [float(x) for x in s.get('mhz', []) if isinstance(x, (int, float)) and float(x) > 0]
        vals.extend(xs)
        if s.get('active_cpus') is not None:
            active_counts.append(int(s['active_cpus']))
        if s.get('sampled_cpus') is not None:
            sampled_counts.append(int(s['sampled_cpus']))
    ref = reference.get('mhz')
    p90 = _percentile(vals, 0.90)
    p10 = _percentile(vals, 0.10)
    mean = statistics.fmean(vals) if vals else None
    ratio = (p90 / float(ref)) if p90 is not None and isinstance(ref, (int, float)) and float(ref) > 0 else None
    enough = len(samples) >= int(os.environ.get('LAB_CLOCK_MIN_SAMPLES', '2')) and len(vals) >= 2
    throttled = False; reason = ''
    if enough and min_mhz is not None and p90 is not None and p90 < float(min_mhz):
        throttled = True; reason = 'loaded-clock-below-absolute-minimum'
    if enough and reference.get('authoritative') and ratio is not None and ratio < float(min_ratio):
        throttled = True; reason = 'loaded-clock-below-base-ratio'
    if not vals:
        status = 'unavailable'
    elif not enough:
        status = 'insufficient-samples'
    elif throttled:
        status = 'throttled'
    elif reference.get('authoritative'):
        status = 'ok'
    else:
        status = 'observed-no-authoritative-reference'
    return {
        'status': status,
        'method': 'sysfs-cpufreq-active-process',
        'sample_ticks': len(samples),
        'frequency_samples': len(vals),
        'active_cpu_count_min': min(active_counts) if active_counts else None,
        'active_cpu_count_mean': statistics.fmean(active_counts) if active_counts else None,
        'active_cpu_count_max': max(active_counts) if active_counts else None,
        'sampled_cpu_count_mean': statistics.fmean(sampled_counts) if sampled_counts else None,
        'min_mhz': min(vals) if vals else None,
        'p10_mhz': p10,
        'mean_mhz': mean,
        'p90_mhz': p90,
        'max_mhz': max(vals) if vals else None,
        'loaded_mhz': p90,
        'reference_mhz': ref,
        'reference_kind': reference.get('kind'),
        'reference_authoritative': bool(reference.get('authoritative')),
        'loaded_to_reference_ratio': ratio,
        'min_ratio_gate': float(min_ratio),
        'min_mhz_gate': min_mhz,
        'throttled': throttled,
        'throttle_reason': reason,
    }


class ClockMonitor:
    def __init__(self, pid: int | None = None, cpus: list[int] | None = None, interval_s: float | None = None,
                 nominal_base_mhz: float | None = None, cpu_sys: Path = CPU_SYS, proc_root: Path = PROC):
        self.pid = int(pid) if pid is not None else None
        self.cpus = list(cpus) if cpus is not None else None
        self.interval_s = float(interval_s if interval_s is not None else max(0.02, float(os.environ.get('LAB_CLOCK_SAMPLE_MS', '100')) / 1000.0))
        self.nominal_base_mhz = nominal_base_mhz
        self.cpu_sys = cpu_sys; self.proc_root = proc_root
        self.samples = []
        self._stop = threading.Event(); self._thread = None
        try:
            self.affinity = sorted(os.sched_getaffinity(0))
        except Exception:
            self.affinity = list(range(os.cpu_count() or 1))
        self.reference = available_clock_reference(self.affinity, self.cpu_sys, nominal_base_mhz)

    def _active_cpus(self) -> list[int]:
        if self.cpus is not None:
            return self.cpus
        if self.pid is not None:
            xs = process_tree_cpus(self.pid, self.proc_root)
            if xs:
                return xs
            if not (self.proc_root / str(self.pid)).exists():
                return []
        return self.affinity

    def sample(self) -> None:
        cpus = self._active_cpus()
        sample_cpus=select_sample_cpus(cpus)
        vals = _sample_freqs(sample_cpus, self.cpu_sys)
        self.samples.append({'t': time.time(), 'active_cpus': len(cpus), 'sampled_cpus':len(sample_cpus), 'mhz': vals})

    def _loop(self) -> None:
        self.sample()
        while not self._stop.wait(self.interval_s):
            self.sample()

    def start(self) -> 'ClockMonitor':
        self._thread = threading.Thread(target=self._loop, name='cpu-clock-monitor', daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        self.sample()
        min_ratio = float(os.environ.get('LAB_CLOCK_MIN_RATIO', '0.90'))
        raw_min = os.environ.get('LAB_CLOCK_MIN_MHZ', '').strip()
        min_mhz = float(raw_min) if raw_min else None
        return summarize_clock_samples(self.samples, self.reference, min_ratio, min_mhz)


def monitor_call(fn: Callable[[], Any], pid: int | None = None, cpus: list[int] | None = None, nominal_base_mhz: float | None = None) -> tuple[Any, dict[str, Any]]:
    mon = ClockMonitor(pid=pid, cpus=cpus, nominal_base_mhz=nominal_base_mhz).start()
    try:
        result = fn()
    finally:
        stats = mon.stop()
    return result, stats


def synthetic_load_probe(physical_cores: int, nominal_base_mhz: float | None = None, duration_s: float | None = None) -> dict[str, Any]:
    """Short all-core load probe used by preflight.

    Uses child Python processes and writes nothing. The actual benchmark rows still
    carry their own independent clock monitors; this is only an early warning.
    """
    duration = float(duration_s if duration_s is not None else os.environ.get('LAB_CLOCK_PREFLIGHT_S', '1.5'))
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except Exception:
        allowed = list(range(os.cpu_count() or 1))
    reps=physical_cpu_representatives(allowed)
    n = max(1, min(int(physical_cores or 1), len(reps), int(os.environ.get('LAB_CLOCK_PREFLIGHT_MAX_WORKERS', '256'))))
    reps=reps[:n]
    code = ('import time\nend=time.monotonic()+%r\nx=0x12345678\n'
            'while time.monotonic()<end:\n'
            ' x=((x*1664525+1013904223)&0xffffffff)\n'
            'print(x)\n') % duration
    procs = []
    for cpu in reps:
        p=subprocess.Popen([os.environ.get('PYTHON', 'python3'), '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:os.sched_setaffinity(p.pid,{cpu})
        except Exception:pass
        procs.append(p)
    root_pids = [p.pid for p in procs]
    mon = ClockMonitor(cpus=reps, nominal_base_mhz=nominal_base_mhz).start()
    # Prefer sampling CPUs actually occupied by our children while they are alive.
    try:
        while any(p.poll() is None for p in procs):
            active = set()
            for pid in root_pids:
                active.update(process_tree_cpus(pid))
            if active:
                sample_cpus=select_sample_cpus(sorted(active)); vals = _sample_freqs(sample_cpus)
                mon.samples.append({'t': time.time(), 'active_cpus': len(active), 'sampled_cpus':len(sample_cpus), 'mhz': vals})
            time.sleep(min(0.05, mon.interval_s))
    finally:
        for p in procs:
            try:
                p.wait(timeout=1)
            except Exception:
                try: p.kill()
                except Exception: pass
        stats = mon.stop()
    stats['probe'] = 'synthetic-all-core-preflight'
    stats['workers'] = n
    stats['duration_s'] = duration
    return stats
