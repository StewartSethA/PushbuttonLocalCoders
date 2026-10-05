"""Bounded real inference warmup and memory checks for a complete layout."""
import argparse
import concurrent.futures
import json
import math
import re
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path

import claude_local_plan as base
from claude_local_resources import host_inventory


def warmup(url, alias, slot, timeout):
    payload = json.dumps({"prompt": f"Local startup check {slot}: say ready.",
                          "n_predict": 1, "cache_prompt": False,
                          "temperature": 0, "stream": False}).encode()
    request = urllib.request.Request(url + "/completion", data=payload,
                                    headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict) or result.get("error") or not isinstance(result.get("content"), str):
        raise ValueError(f"{alias}: invalid real inference response")


def check_memory(plan, processes, *, host=None, gpus=None):
    host = host or host_inventory()
    reserve = plan["host"]["reserve_mib"]
    if host.usable_mib < reserve:
        raise ValueError("RAM headroom fell below startup reserve (swap is excluded)")
    current = {g.index: g for g in (base.inventory() if gpus is None else gpus)}
    for server in plan["servers"]:
        pid = processes[server["id"]][1]
        status = Path(f"/proc/{pid}/status").read_text()
        rss = next(int(line.split()[1]) // 1024 for line in status.splitlines()
                   if line.startswith("VmRSS:"))
        if rss > server["ram_required_mib"] + reserve:
            raise ValueError(f"{server['id']}: resident RAM exceeds calibrated envelope")
        if server["gpus"]:
            group = server["gpus"]
            if any(g["index"] not in current for g in group):
                raise ValueError("GPU disappeared during startup")
            free = sum(current[g["index"]].free_mib for g in group)
            if any(current[g["index"]].free_mib < 512 for g in group):
                raise ValueError("GPU headroom below reserve")
            consumed = sum(g["free_mib"] for g in group) - free
            if consumed > server["required_mib"] + 512 * len(group):
                raise ValueError(f"{server['id']}: GPU consumption exceeds planned envelope")


def admit_layout(plan, *, host=None, gpus=None):
    """Recheck live capacity after provisioning and before each allocation."""
    host = host or host_inventory()
    if plan["ram_required_mib"] + host.reserve_mib > host.usable_mib:
        raise ValueError("live joint RAM budget no longer fits")
    if sum(s["threads"] for s in plan["servers"]) > math.floor(host.cpu_capacity):
        raise ValueError("live CPU quota/affinity no longer fits thread budgets")
    current = {g.index: g for g in (base.inventory() if gpus is None else gpus)}
    for server in plan["servers"]:
        if not server["gpus"]:
            continue
        if any(g["index"] not in current for g in server["gpus"]):
            raise ValueError("planned GPU is no longer available")
        group = [current[g["index"]] for g in server["gpus"]]
        if any(g.free_mib < 512 for g in group) or sum(g.free_mib - 512 for g in group) < server["required_mib"]:
            raise ValueError("live GPU budget no longer fits")
        server["gpus"] = [asdict(g) for g in group]
        server["allocated_mib"] = sum(g.free_mib for g in group)
        server["headroom_mib"] = server["allocated_mib"] - server["required_mib"]
    plan["host"] = asdict(host)


def check_offload(plan, logs):
    for server in plan["servers"]:
        path = logs / f"{server['id']}.log"
        text = path.read_text() if path.is_file() else ""
        reports = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", text)
        mode = server["mode"]
        if mode == "cpu":
            if reports and int(reports[-1][0]) != 0:
                raise ValueError(f"{server['id']}: CPU plan unexpectedly offloaded layers")
            continue
        if not reports:
            raise ValueError(f"{server['id']}: GPU offload not confirmed by llama.cpp load log")
        actual, total = map(int, reports[-1])
        if actual <= 0 or mode == "gpu" and actual != total:
            raise ValueError(f"{server['id']}: full GPU placement was not honored")
        if mode != "gpu":
            raise ValueError(f"{server['id']}: unsupported placement mode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("backends", nargs="?")
    ap.add_argument("--admit", action="store_true")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()
    if not 1 <= args.timeout <= 3600:
        ap.error("warmup timeout must be 1..3600 seconds")
    try:
        plan = json.loads(Path(args.plan).read_text())
        if args.admit:
            admit_layout(plan)
            Path(args.plan).write_text(json.dumps(plan, indent=2))
            return 0
        if not args.backends:
            raise ValueError("backend process inventory required for warmup")
        processes = {}
        for line in Path(args.backends).read_text().splitlines():
            alias, port, _, pid = line.split("\t")
            processes[alias] = (int(port), int(pid))
        count = sum(s["slots"] for s in plan["servers"])
        if count > 64:
            raise ValueError("bounded startup supports at most 64 total slots")
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            futures = [
                pool.submit(warmup, f"http://127.0.0.1:{processes[s['id']][0]}",
                            s["id"], slot, args.timeout)
                for s in plan["servers"] for slot in range(s["slots"])
            ]
            done, pending = concurrent.futures.wait(futures, timeout=args.timeout)
            if pending:
                raise ValueError("complete-layout inference warmup timed out")
            for future in done:
                future.result()
        check_offload(plan, Path(args.plan).parent / "logs")
        check_memory(plan, processes)
        return 0
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        print(f"claude-local startup validation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
