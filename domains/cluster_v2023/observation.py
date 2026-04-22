"""Observation builder for cluster_v2023.

Compact schema (≤ ~500 tokens for 40-node scenarios) so SFT training
fits max_seq_len=4096 per-sample with assistant target intact. The
original dict-per-node / dict-per-job rendering was ~7000 tokens and
pushed QLoRA 14B training past 96 GB VRAM into allocator slow-path
(see decisions-cluster-v2023.md Part 7). Following GridAgent's
observation-compression precedent: summary counts + only the entities
an action can target (free GPU nodes, queued / stranded / preemptable
jobs), with short IDs.

Shape:
  {
    "t": <sim_time>,
    "sum": {total_nodes, ready, down, free_gpu, queued, running},
    "down": [<short_node_id>, ...],
    "free": [[<id>, <model>, <free_gpu>], ...],   # nodes with >=1 free GPU
    "q":    [[<job>, <qos>, <gpu>, <spec>], ...], # Queued jobs
    "strand": [[<job>, <from_node>, <qos>, <gpu>], ...],  # running on Down
    "be_run": [[<job>, <node>, <gpu>], ...],      # preemptable BE jobs, capped
    "F": <fragmentation_F>,
    "F_th": <threshold>,
    "viol": [<constraint_type>, ...]              # names only, agent need not act
  }

IDs are stripped of "openb-node-" / "openb-pod-" prefixes to save 440
chars × 40 nodes = ~200 tokens per observation.
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

_BE_RUN_CAP = 12   # only surface top-N preemptable BE jobs


def _short_node(nid: str) -> str:
    return nid[len("openb-node-"):] if nid.startswith("openb-node-") else nid


def _short_job(jid: str) -> str:
    return jid[len("openb-pod-"):] if jid.startswith("openb-pod-") else jid


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
        nodes = state["nodes"]
        jobs = state["jobs"]
        assignments = state["assignments"]

        total_nodes = len(nodes)
        ready = 0
        down = []                 # short IDs
        free = []                 # [id, model, free_gpu] for nodes with spare GPU
        total_free_gpu = 0
        for nid, n in nodes.items():
            sid = _short_node(nid)
            if n["status"] == "Down":
                down.append(sid)
                continue
            ready += 1
            spare = n["gpu_total"] - n["gpu_used"]
            if spare > 0:
                free.append([sid, n["model"], spare])
                total_free_gpu += spare

        queued = []
        running = []
        stranded = []
        be_run = []
        for jid, j in jobs.items():
            if j["status"] == "Queued":
                queued.append([
                    _short_job(jid),
                    j["qos"],
                    j["gpu"],
                    j.get("gpu_spec_required") or "",
                ])
            elif j["status"] == "Running":
                running.append(jid)
                node_id = assignments.get(jid)
                node_short = _short_node(node_id) if node_id else ""
                if node_id and nodes[node_id]["status"] == "Down":
                    stranded.append([
                        _short_job(jid), node_short, j["qos"], j["gpu"],
                    ])
                elif j["qos"] == "BE" and len(be_run) < _BE_RUN_CAP:
                    be_run.append([
                        _short_job(jid), node_short, j["gpu"],
                    ])

        # Stability + violations (constraint types only, not full detail).
        all_violations: list[dict] = []
        violation_types: list[str] = []
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
                if v.constraint_type not in violation_types:
                    violation_types.append(v.constraint_type)

        frag = self._frag.check(state, base_mva=1.0)

        raw = {
            "t": state["sim_time"],
            "sum": {
                "total_nodes": total_nodes,
                "ready": ready,
                "down": len(down),
                "free_gpu": total_free_gpu,
                "queued": len(queued),
                "running": len(running),
            },
            "down": down,
            "free": free,
            "q": queued,
            "strand": stranded,
            "be_run": be_run,
            "F": round(frag.summary["F"], 3),
            "F_th": self._frag.f_threshold,
            "viol": violation_types,
        }
        return Observation(
            raw=raw,
            compressed_json=json.dumps(raw, separators=(",", ":")),
            violations=all_violations,
            is_stable=is_stable,
        )
