"""GPU cluster domain observer for coordinator multi-agent support."""

from __future__ import annotations

import json
from typing import Any

from silr.agent.observation import BaseObserver
from silr.agent.types import Observation
from .checkers import (
    ResourceCapacityChecker,
    AffinityChecker,
    RackSpreadChecker,
    PriorityChecker,
    QueueChecker,
)


class ClusterObserver(BaseObserver):
    """Observer for the GPU cluster scheduling domain.

    Queries system state and all 5 constraint checkers to produce a
    compressed observation suitable for LLM consumption.
    """

    def __init__(self, manager):
        self._manager = manager
        self._checkers = [
            ResourceCapacityChecker(),
            AffinityChecker(),
            RackSpreadChecker(),
            PriorityChecker(),
            QueueChecker(),
        ]

    def observe(self) -> Observation:
        state = self._manager.system_state
        nodes = state["nodes"]
        jobs = state["jobs"]
        assignments = state["assignments"]

        # Run all checkers to detect violations
        violations = []
        checker_summaries = {}
        for checker in self._checkers:
            cr = checker.check(state, self._manager.base_mva)
            checker_summaries[checker.name] = cr.summary
            for v in cr.violations:
                violations.append({
                    "type": v.constraint_type,
                    "device": v.device_id,
                    "detail": v.detail,
                    "severity": v.severity,
                })

        # Build compressed summaries for the LLM
        down_nodes = [
            nid for nid, n in nodes.items()
            if n["status"] == "NotReady"
        ]

        cordoned_nodes = [
            nid for nid, n in nodes.items()
            if n["status"] == "Cordoned"
        ]

        queued_jobs = []
        for jid in sorted(jobs.keys()):
            if jobs[jid]["status"] != "Queued":
                continue
            entry: dict[str, Any] = {
                "job_id": jid,
                "priority": jobs[jid]["priority"],
                "gpu": jobs[jid]["gpu"],
                "cpu": jobs[jid]["cpu"],
                "ram_gb": jobs[jid]["ram_gb"],
            }
            if jobs[jid].get("rack_affinity"):
                entry["rack_affinity"] = jobs[jid]["rack_affinity"]
            queued_jobs.append(entry)

        # Available nodes: all Ready nodes with free GPU capacity
        available_nodes = []
        for nid, n in sorted(nodes.items()):
            if n["status"] != "Ready":
                continue
            gpu_free = n["gpu_total"] - n["gpu_used"]
            if gpu_free > 0:
                available_nodes.append({
                    "node_id": nid,
                    "rack": n["rack"],
                    "gpu_free": gpu_free,
                    "gpu_total": n["gpu_total"],
                    "cpu_free": n["cpu_total"] - n["cpu_used"],
                    "ram_free_gb": n["ram_total_gb"] - n["ram_used_gb"],
                })

        # Busy nodes: 0 free GPUs among Ready nodes
        busy_nodes = []
        for nid, n in sorted(nodes.items()):
            if n["status"] != "Ready":
                continue
            gpu_total = n["gpu_total"]
            if gpu_total <= 0:
                continue
            if n["gpu_used"] >= gpu_total:
                busy_nodes.append({
                    "node_id": nid,
                    "gpu_used": n["gpu_used"],
                    "gpu_total": gpu_total,
                })

        # Preemptible running jobs: targets for preempt_job when capacity is tight
        preemptible_running = []
        for jid in sorted(jobs.keys()):
            if jobs[jid]["status"] != "Running":
                continue
            if jobs[jid]["priority"] != "preemptible":
                continue
            if jid not in assignments:
                continue
            preemptible_running.append({
                "job_id": jid,
                "node_id": assignments[jid],
                "rack": nodes[assignments[jid]]["rack"],
                "gpu": jobs[jid]["gpu"],
            })

        compressed = {
            "down_nodes": sorted(down_nodes),
            "cordoned_nodes": sorted(cordoned_nodes),
            "queued_jobs": queued_jobs,
            "available_nodes": available_nodes,
            "preemptible_running": preemptible_running,
            "busy_nodes": busy_nodes,
            "checkers": checker_summaries,
            "n_violations": len(violations),
        }

        is_stable = len(violations) == 0

        return Observation(
            raw=state,
            compressed_json=json.dumps(compressed, separators=(",", ":")),
            violations=violations,
            is_stable=is_stable,
        )
