"""Rule-based failsafe strategy for GPU cluster scheduling domain."""

from __future__ import annotations

import json
from typing import Optional

from silr.agent.failsafe import BaseFailsafe
from silr.agent.types import Observation


class ClusterFailsafe(BaseFailsafe):
    """Conservative rule-based fallback for cluster scheduling.

    Strategy:
    1. Find the highest-priority queued job.
    2. Try to assign it to a Ready node with sufficient capacity.
    3. If no node has capacity, preempt the lowest-priority running job
       to free resources.
    """

    # Priority ordering: urgent > normal > preemptible
    _PRIORITY_ORDER = {"urgent": 0, "normal": 1, "preemptible": 2}

    def __init__(self, manager):
        self._manager = manager

    def suggest(self, obs: Observation) -> Optional[dict]:
        """Suggest a conservative action based on current observation.

        Returns an action dict {"tool_name": str, "params": dict} or None.
        """
        state = obs.raw
        nodes = state["nodes"]
        jobs = state["jobs"]
        assignments = state["assignments"]

        # Find queued jobs sorted by priority (urgent first), then job ID
        queued = [
            jid for jid, j in jobs.items()
            if j["status"] == "Queued"
        ]
        if not queued:
            return None

        queued.sort(key=lambda jid: (
            self._PRIORITY_ORDER.get(jobs[jid]["priority"], 9),
            jid,
        ))

        target_jid = queued[0]
        target_job = jobs[target_jid]

        # Compute remaining capacity per Ready node
        remaining = {}
        for nid, n in nodes.items():
            if n["status"] != "Ready":
                continue
            remaining[nid] = {
                "gpu": n["gpu_total"] - n["gpu_used"],
                "cpu": n["cpu_total"] - n["cpu_used"],
                "ram_gb": n["ram_total_gb"] - n["ram_used_gb"],
            }

        # Respect rack affinity if set
        candidates = sorted(remaining.keys())
        if target_job.get("rack_affinity"):
            affinity_candidates = [
                nid for nid in candidates
                if nodes[nid]["rack"] == target_job["rack_affinity"]
            ]
            # Fall back to all candidates only if affinity yields nothing
            if affinity_candidates:
                candidates = affinity_candidates

        # Try to find a node with sufficient capacity
        for nid in candidates:
            r = remaining[nid]
            if (r["gpu"] >= target_job["gpu"]
                    and r["cpu"] >= target_job["cpu"]
                    and r["ram_gb"] >= target_job["ram_gb"]):
                return {
                    "tool_name": "assign_job",
                    "params": {"job_id": target_jid, "node_id": nid},
                }

        # No node has capacity -- try preempting a preemptible running job
        preemptible_running = [
            jid for jid, j in jobs.items()
            if j["priority"] == "preemptible" and j["status"] == "Running"
        ]
        if preemptible_running:
            # Pick the preemptible job using the fewest GPUs (least disruptive)
            preemptible_running.sort(key=lambda jid: (jobs[jid]["gpu"], jid))
            victim = preemptible_running[0]
            return {
                "tool_name": "preempt_job",
                "params": {"job_id": victim},
            }

        return None

    def suggest_escalated(
        self, obs: Observation, last_rejected: Optional[dict] = None,
    ) -> Optional[dict]:
        """Escalated fallback -- delegate to the same suggest() logic.

        In a more sophisticated implementation this could try different
        strategies (e.g., drain an overloaded node, or preempt normal-
        priority jobs). For now, reuse the base strategy.
        """
        return self.suggest(obs)
