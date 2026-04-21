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

    # Filled in Task 6 & 7
    def solve(self) -> bool:
        raise NotImplementedError

    def create_shadow_copy(self) -> "ClusterV2023Manager":
        raise NotImplementedError
