#!/usr/bin/env python3
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

PKG=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PKG))
from lib.clock import parse_cpulist, summarize_clock_samples, available_clock_reference
from lib.detect import nominal_base_mhz

assert parse_cpulist('0-3,8,10-11') == [0,1,2,3,8,10,11]

healthy=summarize_clock_samples(
    [{'active_cpus':68,'mhz':[1490.0,1500.0]},{'active_cpus':68,'mhz':[1505.0,1498.0]}],
    {'mhz':1400.0,'kind':'detected-model-base','authoritative':True},0.90,None)
assert healthy['status']=='ok' and not healthy['throttled'],healthy
assert healthy['loaded_mhz'] > 1490

throttled=summarize_clock_samples(
    [{'active_cpus':68,'mhz':[990.0,1000.0]},{'active_cpus':68,'mhz':[1005.0,995.0]}],
    {'mhz':1400.0,'kind':'detected-model-base','authoritative':True},0.90,None)
assert throttled['status']=='throttled' and throttled['throttled'],throttled
assert throttled['loaded_to_reference_ratio'] < .90

non_authoritative=summarize_clock_samples(
    [{'active_cpus':16,'mhz':[3200.0,3300.0]},{'active_cpus':16,'mhz':[3400.0,3350.0]}],
    {'mhz':4000.0,'kind':'sysfs-max-frequency','authoritative':False},0.90,None)
assert non_authoritative['status']=='observed-no-authoritative-reference',non_authoritative
assert not non_authoritative['throttled']

assert nominal_base_mhz('Intel(R) Xeon Phi(TM) CPU 7250 @ 1.40GHz') == 1400.0
assert nominal_base_mhz('AMD Ryzen Threadripper 1950X 16-Core Processor') == 3400.0
assert nominal_base_mhz('AMD Ryzen Threadripper 1920X 12-Core Processor') == 3500.0

with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    for c in (0,1):
        d=root/f'cpu{c}'/'cpufreq';d.mkdir(parents=True)
        (d/'base_frequency').write_text('1400000\n')
    old=os.environ.pop('LAB_CLOCK_REFERENCE_MHZ',None)
    try:
        ref=available_clock_reference([0,1],root,None)
        assert ref['authoritative'] and ref['kind']=='sysfs-base_frequency' and ref['mhz']==1400.0,ref
    finally:
        if old is not None:os.environ['LAB_CLOCK_REFERENCE_MHZ']=old

print('PASS test_clock')
