#!/usr/bin/env python3
"""Cross-process state helpers for the Pushbutton runtime.

Uses atomic mkdir locks rather than a platform-specific flock so Linux/macOS and
other Python hosts get the same single-flight behavior. Stale locks whose owner
PID is gone are reclaimed conservatively.
"""
from __future__ import annotations
import contextlib, json, os, pathlib, shutil, time

STATE=pathlib.Path(os.environ.get('PUSHBUTTON_RUNTIME_STATE',pathlib.Path.home()/'.local/share/pushbutton/runtime'))
DEFAULT_REGISTRY=STATE/'instances.json'
LOCKS=STATE/'locks'


def _safe(name: str) -> str:
    return ''.join(c if c.isalnum() or c in '._-' else '-' for c in str(name))[:180]

def _pid_alive(pid: int) -> bool:
    if pid<=0:return False
    try:os.kill(pid,0);return True
    except PermissionError:return True
    except Exception:return False

def _lock_stale(path: pathlib.Path, orphan_grace_s: float = 30.0) -> bool:
    try:
        owner=json.loads((path/'owner.json').read_text());pid=int(owner.get('pid') or 0)
        return not _pid_alive(pid)
    except Exception:
        try:return time.time()-path.stat().st_mtime>orphan_grace_s
        except Exception:return True

@contextlib.contextmanager
def named_lock(name: str, timeout_s: float = 3600.0, poll_s: float = .2):
    LOCKS.mkdir(parents=True,exist_ok=True);p=LOCKS/_safe(name);deadline=time.monotonic()+timeout_s
    while True:
        try:
            p.mkdir();(p/'owner.json').write_text(json.dumps({'pid':os.getpid(),'started':time.time()})+'\n');break
        except FileExistsError:
            if _lock_stale(p):
                try:shutil.rmtree(p);continue
                except Exception:pass
            if time.monotonic()>=deadline:raise TimeoutError(f'timed out waiting for runtime lock {name!r}')
            time.sleep(poll_s)
    try:yield p
    finally:
        try:shutil.rmtree(p)
        except Exception:pass

def load(path: pathlib.Path|None=None) -> dict:
    path=path or DEFAULT_REGISTRY
    try:
        obj=json.loads(path.read_text());return obj if isinstance(obj,dict) else {'instances':[]}
    except Exception:return {'instances':[]}

def _atomic_write(path: pathlib.Path,obj: dict):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def replace(obj: dict,path: pathlib.Path|None=None):
    path=path or DEFAULT_REGISTRY
    with named_lock('registry-write',timeout_s=120):_atomic_write(path,obj)

def update(mutator,path: pathlib.Path|None=None):
    path=path or DEFAULT_REGISTRY
    with named_lock('registry-write',timeout_s=120):
        obj=load(path);result=mutator(obj);_atomic_write(path,obj);return result

def append_instance(inst: dict,path: pathlib.Path|None=None):
    iid=str(inst.get('id') or '')
    def mutate(obj):
        rows=obj.setdefault('instances',[])
        if iid and any(str(x.get('id'))==iid for x in rows):return False
        rows.append(inst);return True
    return update(mutate,path)

def mark_health(instance_id: str,healthy: bool,path: pathlib.Path|None=None):
    def mutate(obj):
        for x in obj.setdefault('instances',[]):
            if str(x.get('id'))==str(instance_id):x['healthy']=bool(healthy);return True
        return False
    return update(mutate,path)
