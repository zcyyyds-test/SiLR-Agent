"""Best-fit GPU scheduler — SFT data source + baseline for cluster_v2023.

Policy per step:
  1. Migrate Running jobs off Down nodes to the most-free Ready node.
  2. Assign Queued jobs by qos order (LS → Burstable → BE), each onto
     the Ready node whose remaining GPU is smallest but ≥ job.gpu
     (tightest Best-fit). Ties broken by seeded RNG (for SFT data
     diversity).
  3. If an LS can't be placed and some BE is Running, preempt the BE
     with the largest GPU footprint.
"""

from __future__ import annotations

import random
from typing import Any

from .tools import create_toolset


class BestFitExpert:
    QOS_ORDER = ("LS", "Burstable", "BE")

    def __init__(self, *, seed: int = 0):
        self._rng = random.Random(seed)

    def plan(self, manager: Any, *, max_steps: int = 20) -> list[dict]:
        """Produce an action list that would recover `manager`.

        Runs on a shadow copy — does NOT mutate `manager`.
        """
        shadow = manager.create_shadow_copy()
        shadow.solve()
        actions: list[dict] = []
        for _ in range(max_steps):
            a = self._next_action(shadow)
            if a is None:
                break
            self.apply(shadow, a)
            actions.append(a)
        return actions

    @staticmethod
    def apply(manager: Any, action: dict) -> None:
        tools = create_toolset(manager)
        tools[action["tool_name"]].execute(**action["params"])

    # --- policy ---

    def _next_action(self, mgr: Any) -> dict | None:
        st = mgr.system_state
        # Step 1: evict Running jobs from Down nodes
        for jid, nid in list(st["assignments"].items()):
            if st["nodes"][nid]["status"] == "Down":
                target = self._most_free(mgr, st["jobs"][jid], exclude=[nid])
                if target:
                    return {"tool_name": "migrate_job",
                            "params": {"job_id": jid, "node_id": target}}
        # Step 2: assign queued by qos priority
        for qos in self.QOS_ORDER:
            for jid, j in st["jobs"].items():
                if j["status"] != "Queued" or j["qos"] != qos:
                    continue
                target = self._best_fit(mgr, j)
                if target:
                    return {"tool_name": "assign_job",
                            "params": {"job_id": jid, "node_id": target}}
                # Step 3: LS can't fit → preempt largest BE running
                if qos == "LS":
                    be_victim = self._largest_be_running(st)
                    if be_victim:
                        return {"tool_name": "preempt_job",
                                "params": {"job_id": be_victim}}
        return None

    def _best_fit(self, mgr, job, exclude=()):
        """Tightest Ready node with enough capacity. RNG-tiebreak on ties."""
        st = mgr.system_state
        candidates: list[tuple[int, str]] = []
        for nid, n in st["nodes"].items():
            if nid in exclude or n["status"] != "Ready":
                continue
            if job.get("gpu_spec_required") and n["model"] != job["gpu_spec_required"]:
                continue
            rem_gpu = n["gpu_total"] - n["gpu_used"]
            if (rem_gpu < job["gpu"]
                    or n["cpu_total"] - n["cpu_used"] < job["cpu"]
                    or n["ram_total_mib"] - n["ram_used_mib"] < job["ram_mib"]):
                continue
            candidates.append((rem_gpu, nid))
        if not candidates:
            return None
        min_rem = min(c[0] for c in candidates)
        tied = [nid for rem, nid in candidates if rem == min_rem]
        return self._rng.choice(tied)

    def _most_free(self, mgr, job, exclude=()):
        """Most-free Ready node with enough capacity. RNG-tiebreak on ties."""
        st = mgr.system_state
        candidates: list[tuple[int, str]] = []
        for nid, n in st["nodes"].items():
            if nid in exclude or n["status"] != "Ready":
                continue
            if job.get("gpu_spec_required") and n["model"] != job["gpu_spec_required"]:
                continue
            rem_gpu = n["gpu_total"] - n["gpu_used"]
            if (rem_gpu < job["gpu"]
                    or n["cpu_total"] - n["cpu_used"] < job["cpu"]
                    or n["ram_total_mib"] - n["ram_used_mib"] < job["ram_mib"]):
                continue
            candidates.append((rem_gpu, nid))
        if not candidates:
            return None
        max_rem = max(c[0] for c in candidates)
        tied = [nid for rem, nid in candidates if rem == max_rem]
        return self._rng.choice(tied)

    @staticmethod
    def _largest_be_running(st):
        candidates = [(j, v["gpu"]) for j, v in st["jobs"].items()
                      if v["status"] == "Running" and v["qos"] == "BE"]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]
