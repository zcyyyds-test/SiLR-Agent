"""Observation builder for cluster_v2023.

Returns silr.agent.types.Observation (required by ReActAgent.react_loop,
which accesses obs.is_stable and obs.compressed_json).

is_stable is True iff all 4 non-fragmentation checkers pass (Capacity +
Affinity + Priority + Queue). Fragmentation is episode-level (observer
signal, not a stability gate) per spec §5.1.
"""

from __future__ import annotations

import json
from typing import Any

from silr.agent.types import Observation

from .checkers import (
    AffinityChecker,
    DEFAULT_JOB_SIZE_DIST,
    FragmentationChecker,
    PriorityChecker,
    QueueChecker,
    ResourceCapacityChecker,
)


class ClusterV2023Observer:
    def __init__(self, manager: Any,
                 *, f_threshold: float = 10.0,
                 job_size_dist: dict[int, float] | None = None):
        self.manager = manager
        self._frag = FragmentationChecker(
            f_threshold=f_threshold,
            job_size_dist=dict(job_size_dist or DEFAULT_JOB_SIZE_DIST),
        )
        self._stability_checkers = [
            ResourceCapacityChecker(),
            AffinityChecker(),
            PriorityChecker(),
            QueueChecker(),
        ]

    def observe(self) -> Observation:
        state = self.manager.system_state
        nodes = []
        for nid, n in state["nodes"].items():
            nodes.append({
                "id": nid,
                "model": n["model"],
                "status": n["status"],
                "gpu": f"{n['gpu_used']}/{n['gpu_total']}",
                "cpu_milli": f"{n['cpu_used']}/{n['cpu_total']}",
                "ram_mib": f"{n['ram_used_mib']}/{n['ram_total_mib']}",
            })

        queued = [{"id": j, "qos": v["qos"], "gpu": v["gpu"],
                   "gpu_spec_required": v.get("gpu_spec_required")}
                  for j, v in state["jobs"].items() if v["status"] == "Queued"]

        running = [{"id": j, "node_id": state["assignments"].get(j),
                    "qos": v["qos"], "gpu": v["gpu"]}
                   for j, v in state["jobs"].items() if v["status"] == "Running"]

        all_violations: list[dict] = []
        is_stable = True
        for chk in self._stability_checkers:
            res = chk.check(state, base_mva=1.0)
            if not res.passed:
                is_stable = False
            for v in res.violations:
                all_violations.append({
                    "constraint_type": v.constraint_type,
                    "device_type": v.device_type,
                    "device_id": v.device_id,
                    "metric": v.metric,
                    "value": v.value,
                    "limit": v.limit,
                    "severity": v.severity,
                    "detail": v.detail,
                })

        frag = self._frag.check(state, base_mva=1.0)

        raw = {
            "sim_time": state["sim_time"],
            "nodes": nodes,
            "queued_jobs": queued,
            "running_jobs": running,
            "fragmentation_F": frag.summary["F"],
            "fragmentation_threshold": self._frag.f_threshold,
            "violations": all_violations,
        }
        return Observation(
            raw=raw,
            compressed_json=json.dumps(raw, separators=(",", ":")),
            violations=all_violations,
            is_stable=is_stable,
        )
