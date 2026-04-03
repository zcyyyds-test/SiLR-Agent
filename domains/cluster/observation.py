"""GPU cluster domain observer for coordinator multi-agent support."""

from __future__ import annotations

import json

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

        queued_jobs = [
            {"job_id": jid, "priority": jobs[jid]["priority"], "gpu": jobs[jid]["gpu"]}
            for jid in sorted(jobs.keys())
            if jobs[jid]["status"] == "Queued"
        ]

        # Busy nodes: >70% GPU utilization among Ready nodes
        busy_nodes = []
        for nid, n in sorted(nodes.items()):
            if n["status"] != "Ready":
                continue
            gpu_total = n["gpu_total"]
            if gpu_total <= 0:
                continue
            gpu_util = n["gpu_used"] / gpu_total
            if gpu_util > 0.70:
                busy_nodes.append({
                    "node_id": nid,
                    "gpu_util_pct": round(gpu_util * 100, 1),
                    "gpu_used": n["gpu_used"],
                    "gpu_total": gpu_total,
                })

        compressed = {
            "down_nodes": sorted(down_nodes),
            "cordoned_nodes": sorted(cordoned_nodes),
            "queued_jobs": queued_jobs,
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
