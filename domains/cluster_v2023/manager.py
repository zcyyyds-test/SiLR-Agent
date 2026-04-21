"""ClusterV2023Manager — Alibaba OpenB GPU cluster simulator for SiLR.

State schema (self.system_state dict):
    {
        "nodes": {
            node_id: {
                "model": str,           # V100M32 / G1 / T4 / ...
                "cpu_total": int,       # cpu_milli
                "ram_total_mib": int,
                "gpu_total": int,
                "cpu_used": int,
                "ram_used_mib": int,
                "gpu_used": int,
                "status": "Ready" | "Down",
            },
            ...
        },
        "jobs": {
            job_id: {
                "cpu": int,                    # cpu_milli
                "ram_mib": int,
                "gpu": int,
                "gpu_spec_required": str|None, # None or specific model
                "qos": "LS" | "Burstable" | "BE",
                "status": "Queued" | "Running" | "Preempted",
            },
            ...
        },
        "assignments": {job_id: node_id},  # Running jobs only
        "sim_time": float,
    }
"""

from __future__ import annotations

import copy
from typing import Any

from silr.core.interfaces import BaseSystemManager


class ClusterV2023Manager(BaseSystemManager):
    def __init__(self, *, nodes: dict, jobs: dict,
                 assignments: dict | None = None,
                 sim_time: float = 0.0):
        self._nodes = copy.deepcopy(nodes)
        self._jobs = copy.deepcopy(jobs)
        self._assignments = copy.deepcopy(assignments) if assignments else {}
        self._sim_time = float(sim_time)

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def base_mva(self) -> float:
        return 1.0

    @property
    def system_state(self) -> Any:
        return {
            "nodes": self._nodes,
            "jobs": self._jobs,
            "assignments": self._assignments,
            "sim_time": self._sim_time,
        }

    def solve(self) -> bool:
        """Recompute per-node resource usage from assignments. Always converges."""
        self._recompute_node_usage()
        return True

    def _recompute_node_usage(self) -> None:
        for nid, node in self._nodes.items():
            node["cpu_used"] = 0
            node["ram_used_mib"] = 0
            node["gpu_used"] = 0

        for jid, nid in self._assignments.items():
            job = self._jobs.get(jid)
            if job is None:
                continue
            if job["status"] != "Running":
                continue
            node = self._nodes.get(nid)
            if node is None or node["status"] != "Ready":
                continue
            node["cpu_used"] += int(job["cpu"])
            node["ram_used_mib"] += int(job["ram_mib"])
            node["gpu_used"] += int(job["gpu"])

    # Filled in Task 7
    def create_shadow_copy(self) -> "ClusterV2023Manager":
        raise NotImplementedError
