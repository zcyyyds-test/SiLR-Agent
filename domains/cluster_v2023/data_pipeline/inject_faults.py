"""Fault injection for cluster_v2023 recovery scenarios.

Four fault types (ratios per spec §6, Philly ATC'19 distribution):
  node_failure (50%), gpu_spec_mismatch (20%),
  qos_pressure (20%), fragmentation_surge (10%).

All injection functions MUTATE the passed manager in place. They
return a dict of fault metadata (for scenario JSON emission).
"""

from __future__ import annotations

import random
from typing import Any


def _next_jid(manager: Any) -> int:
    """Return next free integer suffix for `j<N>` job IDs.

    Robust to whatever the existing ID scheme is — only looks at
    `j<digits>` patterns. If the trace uses non-j-prefixed IDs, this
    still picks a safe starting point at 0.
    """
    existing = [int(j.lstrip("j")) for j in manager._jobs
                if j.startswith("j") and j.lstrip("j").isdigit()]
    return max(existing, default=-1) + 1


def inject_node_failure(manager: Any, *, n_nodes: int, seed: int) -> dict:
    rng = random.Random(seed)
    ready_nodes = [nid for nid, n in manager._nodes.items()
                   if n["status"] == "Ready"]
    victims = rng.sample(ready_nodes, min(n_nodes, len(ready_nodes)))
    affected_jobs: list[str] = []
    for nid in victims:
        manager._nodes[nid]["status"] = "Down"
        for jid, target in list(manager._assignments.items()):
            if target == nid:
                manager._jobs[jid]["status"] = "Preempted"
                affected_jobs.append(jid)
                manager._assignments.pop(jid, None)
    manager._recompute_node_usage()
    return {"fault_type": "node_failure",
            "fault_nodes": victims,
            "affected_jobs": affected_jobs}


def inject_gpu_spec_mismatch(manager: Any, *, n_jobs: int, seed: int) -> dict:
    """Inject `gpu_spec_required` into running jobs, pointing to a
    DIFFERENT model than their current node.

    Guaranteed to produce real mismatches (not tautologies).
    """
    rng = random.Random(seed)
    models = sorted({n["model"] for n in manager._nodes.values()})
    running = [jid for jid, v in manager._jobs.items() if v["status"] == "Running"]
    rng.shuffle(running)
    affected: list[str] = []
    for jid in running:
        if len(affected) >= n_jobs:
            break
        nid = manager._assignments.get(jid)
        if nid is None:
            continue
        current = manager._nodes[nid]["model"]
        alt = [m for m in models if m != current]
        if not alt:
            continue
        manager._jobs[jid]["gpu_spec_required"] = rng.choice(alt)
        affected.append(jid)
    return {"fault_type": "gpu_spec_mismatch", "affected_jobs": affected}


def inject_qos_pressure(manager: Any, *, n_ls_queued: int, seed: int) -> dict:
    """Force some Running jobs to qos=BE, then queue up new LS jobs.

    Together these trigger PriorityChecker (LS queued + BE running).
    """
    rng = random.Random(seed)
    running = [jid for jid, v in manager._jobs.items() if v["status"] == "Running"]
    for jid in rng.sample(running, min(3, len(running))):
        manager._jobs[jid]["qos"] = "BE"
    ls_queued: list[str] = []
    next_idx = _next_jid(manager)
    for _ in range(n_ls_queued):
        jid = f"j{next_idx}"
        next_idx += 1
        manager._jobs[jid] = {
            "cpu": 8000, "ram_mib": 16384, "gpu": 1,
            "gpu_spec_required": None, "qos": "LS", "status": "Queued",
        }
        ls_queued.append(jid)
    return {"fault_type": "qos_pressure", "ls_queued": ls_queued}


def inject_fragmentation_surge(manager: Any, *, seed: int) -> dict:
    """Scatter 1-GPU BE jobs to leave small remainders, then queue a
    large LS job (4 GPUs) that can't fit anywhere."""
    rng = random.Random(seed)
    ready = [nid for nid, n in manager._nodes.items() if n["status"] == "Ready"]
    rng.shuffle(ready)
    next_idx = _next_jid(manager)
    affected: list[str] = []
    for nid in ready:
        while (manager._nodes[nid]["gpu_total"]
               - manager._nodes[nid]["gpu_used"] > 1):
            jid = f"j{next_idx}"
            next_idx += 1
            manager._jobs[jid] = {"cpu": 500, "ram_mib": 1024, "gpu": 1,
                                  "gpu_spec_required": None, "qos": "BE",
                                  "status": "Running"}
            manager._assignments[jid] = nid
            manager._recompute_node_usage()
            affected.append(jid)
    big = f"j{next_idx}"
    manager._jobs[big] = {"cpu": 8000, "ram_mib": 16384, "gpu": 4,
                          "gpu_spec_required": None, "qos": "LS",
                          "status": "Queued"}
    return {"fault_type": "fragmentation_surge",
            "affected_jobs": affected + [big]}
