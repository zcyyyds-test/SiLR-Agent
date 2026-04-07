"""ClusterManager: GPU cluster scheduling simulator for SiLR framework.

Topology (15 nodes across 3 racks):

    rack-a: 2 standard + 2 highmem + 1 fat  (5 nodes)
    rack-b: 2 standard + 2 highmem + 1 fat  (5 nodes)
    rack-c: 2 standard + 2 highmem + 1 fat  (5 nodes)

Node types:
    standard : 4 GPU (80 GB), 64 CPU,  256 GB RAM
    highmem  : 4 GPU (80 GB), 64 CPU,  512 GB RAM
    fat      : 8 GPU (80 GB), 128 CPU, 1024 GB RAM

Jobs are scheduled via greedy bin-packing (deterministic, seed=42).
Constraints = GPU/CPU/RAM capacity, node health, rack affinity.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Optional

from silr.core.interfaces import BaseSystemManager

# ---------------------------------------------------------------------------
# Node specs: (gpu_count, gpu_mem_gb, cpu_count, ram_gb)
# ---------------------------------------------------------------------------
_NODE_SPECS = {
    "standard": {"gpu": 4, "gpu_mem_gb": 80, "cpu": 64, "ram_gb": 256},
    "highmem":  {"gpu": 4, "gpu_mem_gb": 80, "cpu": 64, "ram_gb": 512},
    "fat":      {"gpu": 8, "gpu_mem_gb": 80, "cpu": 128, "ram_gb": 1024},
}

_RACKS = ["rack-a", "rack-b", "rack-c"]

# Per-rack layout: 2 standard, 2 highmem, 1 fat
_RACK_LAYOUT = [
    ("standard", 2),
    ("highmem", 2),
    ("fat", 1),
]


def _build_default_nodes() -> dict[str, dict]:
    """Create 15 nodes across 3 racks (deterministic ordering)."""
    nodes: dict[str, dict] = {}
    for rack in _RACKS:
        idx = 0
        for node_type, count in _RACK_LAYOUT:
            spec = _NODE_SPECS[node_type]
            for _ in range(count):
                node_id = f"{rack}-{node_type[0]}{idx}"
                nodes[node_id] = {
                    "rack": rack,
                    "type": node_type,
                    "status": "Ready",
                    "gpu_total": spec["gpu"],
                    "gpu_mem_gb": spec["gpu_mem_gb"],
                    "cpu_total": spec["cpu"],
                    "ram_total_gb": spec["ram_gb"],
                    # Usage counters — recomputed by _recompute_node_usage
                    "gpu_used": 0,
                    "cpu_used": 0,
                    "ram_used_gb": 0,
                }
                idx += 1
    return nodes


def _build_default_jobs(node_ids: list[str], rng: random.Random) -> dict[str, dict]:
    """Generate 60-80 jobs with randomised resource requests.

    Uses *rng* for full reproducibility (seed=42 at call site).
    """
    priorities = ["urgent", "normal", "preemptible"]
    priority_weights = [0.1, 0.6, 0.3]  # ~10% urgent, 60% normal, 30% preemptible

    # Job groups — each job belongs to a named experiment/project
    job_groups = [
        "llm-pretrain", "llm-finetune", "eval-suite", "data-prep",
        "hparam-sweep", "diffusion-train", "rl-train", "embedding-gen",
        "benchmark", "distillation",
    ]

    # 72 total GPUs across 15 nodes. Target ~55-65% utilization (40-47 GPUs used)
    # so scenarios have headroom for rescheduling after failures.
    # avg GPU/job ≈ 1.33 with this distribution, 33 jobs × 1.33 ≈ 44 GPUs ≈ 61%
    num_jobs = rng.randint(30, 35)
    jobs: dict[str, dict] = {}
    for i in range(num_jobs):
        job_id = f"job-{i:04d}"
        priority = rng.choices(priorities, weights=priority_weights, k=1)[0]
        group = rng.choice(job_groups)

        # Resource requests — mostly small, few large
        gpu = rng.choice([1, 1, 1, 1, 2, 2])
        cpu = rng.choice([4, 8, 8, 16, 16, 32])
        ram_gb = rng.choice([16, 32, 32, 64, 64, 128])

        # ~20% of jobs have rack affinity
        rack_affinity: Optional[str] = None
        if rng.random() < 0.2:
            rack_affinity = rng.choice(_RACKS)

        jobs[job_id] = {
            "group": group,
            "gpu": gpu,
            "cpu": cpu,
            "ram_gb": ram_gb,
            "priority": priority,
            "rack_affinity": rack_affinity,
            "status": "Pending",  # will become Running or Queued after packing
        }
    return jobs


def _greedy_pack(
    nodes: dict[str, dict],
    jobs: dict[str, dict],
) -> dict[str, str]:
    """Greedy first-fit bin-packing: assign jobs to nodes.

    Returns assignments mapping job_id -> node_id.
    Jobs that cannot be placed remain with status "Queued".
    Successfully placed jobs get status "Running".
    """
    assignments: dict[str, str] = {}

    # Remaining capacity tracker (avoid mutating node dicts during packing)
    remaining = {
        nid: {
            "gpu": n["gpu_total"],
            "cpu": n["cpu_total"],
            "ram_gb": n["ram_total_gb"],
        }
        for nid, n in nodes.items()
        if n["status"] == "Ready"
    }

    # Sort jobs by priority (urgent first) then by gpu descending for tighter packing
    priority_order = {"urgent": 0, "normal": 1, "preemptible": 2}
    sorted_job_ids = sorted(
        jobs.keys(),
        key=lambda jid: (priority_order.get(jobs[jid]["priority"], 9), -jobs[jid]["gpu"]),
    )

    node_id_list = sorted(remaining.keys())

    for jid in sorted_job_ids:
        job = jobs[jid]
        placed = False

        # Filter nodes by rack affinity if specified
        candidates = node_id_list
        if job["rack_affinity"]:
            candidates = [
                nid for nid in node_id_list
                if nodes[nid]["rack"] == job["rack_affinity"]
            ]

        for nid in candidates:
            if nid not in remaining:
                continue
            r = remaining[nid]
            if (r["gpu"] >= job["gpu"]
                    and r["cpu"] >= job["cpu"]
                    and r["ram_gb"] >= job["ram_gb"]):
                assignments[jid] = nid
                r["gpu"] -= job["gpu"]
                r["cpu"] -= job["cpu"]
                r["ram_gb"] -= job["ram_gb"]
                job["status"] = "Running"
                placed = True
                break

        if not placed:
            job["status"] = "Queued"

    return assignments


class ClusterManager(BaseSystemManager):
    """GPU cluster scheduler simulator. Pure Python, no external dependencies.

    Implements BaseSystemManager for SiLR verification compatibility.
    """

    def __init__(self) -> None:
        self._time: float = 0.0
        self._nodes: dict[str, dict] = _build_default_nodes()

        rng = random.Random(42)
        self._jobs: dict[str, dict] = _build_default_jobs(
            sorted(self._nodes.keys()), rng
        )
        self._assignments: dict[str, str] = _greedy_pack(self._nodes, self._jobs)
        self._recompute_node_usage()

    # ------------------------------------------------------------------
    # BaseSystemManager interface
    # ------------------------------------------------------------------

    @property
    def sim_time(self) -> float:
        return self._time

    @property
    def base_mva(self) -> float:
        return 1.0  # no per-unit system in cluster scheduling

    @property
    def system_state(self) -> dict:
        """Domain state for constraint checkers."""
        return {
            "nodes": self._nodes,
            "jobs": self._jobs,
            "assignments": self._assignments,
        }

    def create_shadow_copy(self) -> ClusterManager:
        """Create independent deep copy for SiLR verification."""
        shadow = object.__new__(ClusterManager)
        shadow._time = self._time
        shadow._nodes = copy.deepcopy(self._nodes)
        shadow._jobs = copy.deepcopy(self._jobs)
        shadow._assignments = copy.deepcopy(self._assignments)
        return shadow

    def solve(self) -> bool:
        """Recompute node utilisation from current assignments.

        Always returns True (cluster state is always 'converged').
        """
        self._recompute_node_usage()
        self._time += 1.0
        return True

    # ------------------------------------------------------------------
    # Domain-specific operations
    # ------------------------------------------------------------------

    def fail_node(self, node_id: str) -> bool:
        """Mark a node as NotReady and re-queue all its jobs.

        Returns True if the node existed and was Ready.
        """
        if node_id not in self._nodes:
            return False
        node = self._nodes[node_id]
        if node["status"] != "Ready":
            return False

        node["status"] = "NotReady"

        # Re-queue every job assigned to this node
        evicted = [jid for jid, nid in self._assignments.items() if nid == node_id]
        for jid in evicted:
            self._jobs[jid]["status"] = "Queued"
            del self._assignments[jid]

        self._recompute_node_usage()
        return True

    def restore_node(self, node_id: str) -> bool:
        """Restore a failed node. Returns True if it was NotReady."""
        if node_id not in self._nodes:
            return False
        node = self._nodes[node_id]
        if node["status"] != "NotReady":
            return False

        node["status"] = "Ready"
        self._recompute_node_usage()
        return True

    def fail_rack(self, rack: str) -> list[str]:
        """Fail all Ready nodes in a rack. Returns list of failed node IDs."""
        failed: list[str] = []
        for nid, node in self._nodes.items():
            if node["rack"] == rack and node["status"] == "Ready":
                self.fail_node(nid)
                failed.append(nid)
        return failed

    def add_jobs(self, job_defs: list[dict]) -> list[str]:
        """Add new jobs (all start as Queued). Returns new job IDs."""
        new_ids: list[str] = []
        next_idx = len(self._jobs)
        for jdef in job_defs:
            jid = f"job-{next_idx:04d}"
            next_idx += 1
            self._jobs[jid] = {
                "group": jdef.get("group", f"dynamic-{jid}"),
                "gpu": jdef.get("gpu", 1),
                "cpu": jdef.get("cpu", 8),
                "ram_gb": jdef.get("ram_gb", 32),
                "priority": jdef.get("priority", "normal"),
                "rack_affinity": jdef.get("rack_affinity", None),
                "status": "Queued",
            }
            new_ids.append(jid)
        return new_ids

    def get_queued_jobs(self) -> list[str]:
        """Return IDs of all Queued jobs."""
        return [jid for jid, j in self._jobs.items() if j["status"] == "Queued"]

    def get_schedulable_nodes(self) -> list[str]:
        """Return IDs of all Ready nodes."""
        return sorted(
            nid for nid, n in self._nodes.items() if n["status"] == "Ready"
        )

    def get_node_ids(self) -> list[str]:
        """Return all node IDs (sorted)."""
        return sorted(self._nodes.keys())

    def get_job_ids(self) -> list[str]:
        """Return all job IDs (sorted)."""
        return sorted(self._jobs.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recompute_node_usage(self) -> None:
        """Recalculate gpu_used / cpu_used / ram_used_gb from assignments."""
        # Zero out all counters first
        for node in self._nodes.values():
            node["gpu_used"] = 0
            node["cpu_used"] = 0
            node["ram_used_gb"] = 0

        # Sum resource usage from assigned jobs
        for jid, nid in self._assignments.items():
            job = self._jobs[jid]
            node = self._nodes[nid]
            node["gpu_used"] += job["gpu"]
            node["cpu_used"] += job["cpu"]
            node["ram_used_gb"] += job["ram_gb"]
