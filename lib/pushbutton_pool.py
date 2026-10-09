#!/usr/bin/env python3
"""Pure scheduling helpers for the Pushbutton resident inference pool."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import pushbutton_policy as policy


@dataclass
class Instance:
    id: str
    model: str
    backend: str
    endpoint: str
    artifact: str | None = None
    served_model: str | None = None
    aliases: list[str] = field(default_factory=list)
    max_context: int | None = None
    framework_max_concurrency: int | None = None
    measured_envelopes: list[dict[str, Any]] = field(default_factory=list)
    tg: float | None = None
    tg_measured: bool = False
    active: int = 0
    queued: int = 0
    healthy: bool = True

    def accepts_model(self, requested: str) -> bool:
        q=requested.lower()
        return q in {self.model.lower(),self.id.lower(),*(x.lower() for x in self.aliases)}

    def capacity(self, context: int) -> int:
        return policy.proven_concurrency(
            backend=self.backend,
            requested_context=context,
            framework_max=self.framework_max_concurrency,
            measured_envelopes=self.measured_envelopes,
        )

    def context_limit(self) -> int:
        return policy.backend_limits(self.backend,declared_context=self.max_context).max_context

    def speed(self) -> tuple[str,str|None]:
        return policy.speed_label(self.tg,measured=self.tg_measured)


def instance_from_dict(x: dict) -> Instance:
    return Instance(
        id=str(x.get('id') or x.get('model') or 'instance'),
        model=str(x.get('model') or x.get('id') or 'local-model'),
        backend=str(x.get('backend') or 'resident'),
        endpoint=str(x.get('endpoint') or ''),
        artifact=x.get('artifact'),served_model=x.get('served_model'),aliases=list(x.get('aliases') or []),
        max_context=x.get('max_context'),framework_max_concurrency=x.get('framework_max_concurrency'),
        measured_envelopes=list(x.get('measured_envelopes') or []),tg=x.get('tg'),
        tg_measured=bool(x.get('tg_measured')),active=int(x.get('active') or 0),
        queued=int(x.get('queued') or 0),healthy=bool(x.get('healthy',True)),
    )


def compatible(inst: Instance, requested_model: str, context: int) -> bool:
    return inst.healthy and inst.endpoint and inst.accepts_model(requested_model) and context <= inst.context_limit()


def rank_instance(inst: Instance, context: int) -> tuple:
    cap=inst.capacity(context)
    available=max(0,cap-inst.active)
    speed_label,_=inst.speed()
    speed_rank={'OK':0,'UNKNOWN':1,'SLOW':2,'VERY_SLOW':3}.get(speed_label,4)
    # Prefer immediate capacity, then less pressure, then acceptable speed, then TG.
    return (0 if available else 1, (inst.active+inst.queued)/max(1,cap), speed_rank, -(inst.tg or 0.0), inst.id)


def choose_instance(instances: list[Instance], requested_model: str, context: int) -> Instance | None:
    rows=[x for x in instances if compatible(x,requested_model,context)]
    return min(rows,key=lambda x:rank_instance(x,context)) if rows else None


def auto_candidates(instances: list[Instance], context: int, allow_slow: bool=False) -> list[Instance]:
    rows=[x for x in instances if x.healthy and x.endpoint and context<=x.context_limit()]
    if not allow_slow:
        fast=[x for x in rows if x.speed()[0] in {'OK','UNKNOWN'}]
        if fast:rows=fast
    return sorted(rows,key=lambda x:rank_instance(x,context))
