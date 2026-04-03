# GPU Cluster Scheduling + GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GPU cluster job scheduling domain and step-level GRPO trainer to SILR-Agent, producing a measurable SFT→GRPO improvement.

**Architecture:** New `domains/cluster/` implements `BaseSystemManager`, 5 constraint checkers, 6 tools, and a scenario loader following the exact patterns in `domains/network/`. A new `silr/training/grpo_trainer.py` implements the iterative offline GRPO loop using step-level (observation, action, reward) tuples grouped by (scenario, step_number).

**Tech Stack:** Python 3.10+, pure stdlib for domain (zero dependencies), PyTorch + TRL + PEFT for training.

---

## File Structure

```
domains/cluster/
├── __init__.py              — Public API exports
├── manager.py               — ClusterManager(BaseSystemManager): 15-node GPU cluster state
├── checkers.py              — 5 constraint checkers (BaseConstraintChecker)
├── tools.py                 — 6 agent tools (BaseTool) + create_cluster_toolset()
├── config.py                — build_cluster_domain_config() → DomainConfig
├── observation.py           — ClusterObserver(BaseObserver)
├── failsafe.py              — ClusterFailsafe(BaseFailsafe)
├── prompts/
│   ├── __init__.py
│   ├── system_prompt.py     — build_cluster_system_prompt()
│   └── tool_schemas.py      — build_cluster_tool_schemas()
└── scenarios/
    ├── __init__.py
    └── loader.py            — ClusterScenario + ClusterScenarioLoader

silr/training/
└── grpo_trainer.py          — StepLevelGRPOTrainer (NEW)

examples/
└── cluster_scheduling.py    — Runnable demo

tests/
├── test_cluster_manager.py
├── test_cluster_checkers.py
├── test_cluster_tools.py
├── test_cluster_scenarios.py
├── test_cluster_integration.py
└── test_grpo_trainer.py
```

---

### Task 1: ClusterManager — Core Data Model

**Files:**
- Create: `domains/cluster/__init__.py`
- Create: `domains/cluster/manager.py`
- Test: `tests/test_cluster_manager.py`

This is the foundation. The manager holds the cluster state (nodes, jobs, assignments) and implements `BaseSystemManager`.

- [ ] **Step 1: Write failing tests for ClusterManager**

```python
# tests/test_cluster_manager.py
"""Tests for GPU cluster manager."""

import pytest
from domains.cluster.manager import ClusterManager


class TestClusterManagerInit:
    def test_default_topology(self):
        mgr = ClusterManager()
        state = mgr.system_state
        assert len(state["nodes"]) == 15
        assert len(state["jobs"]) >= 60

    def test_node_types(self):
        mgr = ClusterManager()
        nodes = mgr.system_state["nodes"]
        standard = [n for n in nodes.values() if n["type"] == "standard"]
        highmem = [n for n in nodes.values() if n["type"] == "highmem"]
        fat = [n for n in nodes.values() if n["type"] == "fat"]
        assert len(standard) == 6
        assert len(highmem) == 6
        assert len(fat) == 3

    def test_racks(self):
        mgr = ClusterManager()
        nodes = mgr.system_state["nodes"]
        racks = {n["rack"] for n in nodes.values()}
        assert racks == {"rack-a", "rack-b", "rack-c"}

    def test_gpu_counts(self):
        mgr = ClusterManager()
        nodes = mgr.system_state["nodes"]
        for n in nodes.values():
            if n["type"] == "fat":
                assert n["gpu_total"] == 8
            else:
                assert n["gpu_total"] == 4


class TestClusterManagerInterface:
    def test_sim_time(self):
        mgr = ClusterManager()
        assert mgr.sim_time == 0.0

    def test_base_mva(self):
        mgr = ClusterManager()
        assert mgr.base_mva == 1.0

    def test_system_state_keys(self):
        mgr = ClusterManager()
        state = mgr.system_state
        assert "nodes" in state
        assert "jobs" in state
        assert "assignments" in state

    def test_shadow_copy_isolation(self):
        mgr = ClusterManager()
        shadow = mgr.create_shadow_copy()
        # Modify shadow
        shadow.fail_node("node-01")
        # Original unaffected
        assert mgr.system_state["nodes"]["node-01"]["status"] == "Ready"
        assert shadow.system_state["nodes"]["node-01"]["status"] == "NotReady"

    def test_run_pflow_returns_bool(self):
        mgr = ClusterManager()
        result = mgr.run_pflow()
        assert isinstance(result, bool)


class TestClusterManagerOperations:
    def test_fail_node(self):
        mgr = ClusterManager()
        mgr.fail_node("node-01")
        node = mgr.system_state["nodes"]["node-01"]
        assert node["status"] == "NotReady"

    def test_fail_node_requeues_jobs(self):
        mgr = ClusterManager()
        # Find a job on node-01
        jobs_on_01 = [
            jid for jid, assignment in mgr.system_state["assignments"].items()
            if assignment == "node-01"
        ]
        mgr.fail_node("node-01")
        for jid in jobs_on_01:
            assert mgr.system_state["jobs"][jid]["status"] == "Queued"
            assert jid not in mgr.system_state["assignments"]

    def test_restore_node(self):
        mgr = ClusterManager()
        mgr.fail_node("node-01")
        mgr.restore_node("node-01")
        assert mgr.system_state["nodes"]["node-01"]["status"] == "Ready"

    def test_run_pflow_computes_utilization(self):
        mgr = ClusterManager()
        mgr.run_pflow()
        for node in mgr.system_state["nodes"].values():
            assert "gpu_used" in node
            assert "cpu_used" in node
            assert "ram_used" in node
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

Run: `cd /mnt/d/SciTokyo/SILR-Agent && python -m pytest tests/test_cluster_manager.py -v`
Expected: `ModuleNotFoundError: No module named 'domains.cluster'`

- [ ] **Step 3: Implement ClusterManager**

```python
# domains/cluster/__init__.py
"""GPU cluster scheduling domain for SiLR-Agent."""

from .manager import ClusterManager

__all__ = ["ClusterManager"]
```

```python
# domains/cluster/manager.py
"""ClusterManager: GPU cluster simulator for SiLR framework.

Topology: 15 GPU nodes across 3 racks (rack-a, rack-b, rack-c).
  - 6 standard: 4 GPU (80GB), 64 CPU, 256 GB RAM
  - 6 highmem:  4 GPU (80GB), 64 CPU, 512 GB RAM
  - 3 fat:      8 GPU (80GB), 128 CPU, 1 TB RAM

Jobs have resource requests (gpu, cpu, ram), priority classes
(urgent/normal/preemptible), and optional rack-affinity for
multi-node training.
"""

from __future__ import annotations

import copy
from typing import Any

from silr.core.interfaces import BaseSystemManager


# --- Default cluster topology ---

def _build_default_nodes() -> dict[str, dict]:
    """Create the 15-node heterogeneous cluster."""
    nodes = {}
    rack_names = ["rack-a", "rack-b", "rack-c"]

    idx = 1
    for rack in rack_names:
        # 2 standard per rack
        for _ in range(2):
            nid = f"node-{idx:02d}"
            nodes[nid] = {
                "type": "standard",
                "rack": rack,
                "gpu_total": 4,
                "gpu_mem_gb": 80,
                "cpu_total": 64,
                "ram_total_gb": 256,
                "gpu_used": 0,
                "cpu_used": 0,
                "ram_used_gb": 0,
                "status": "Ready",  # Ready | NotReady | Cordoned
            }
            idx += 1

        # 2 highmem per rack
        for _ in range(2):
            nid = f"node-{idx:02d}"
            nodes[nid] = {
                "type": "highmem",
                "rack": rack,
                "gpu_total": 4,
                "gpu_mem_gb": 80,
                "cpu_total": 64,
                "ram_total_gb": 512,
                "gpu_used": 0,
                "cpu_used": 0,
                "ram_used_gb": 0,
                "status": "Ready",
            }
            idx += 1

    # 3 fat nodes, one per rack
    for i, rack in enumerate(rack_names):
        nid = f"node-{idx:02d}"
        nodes[nid] = {
            "type": "fat",
            "rack": rack,
            "gpu_total": 8,
            "gpu_mem_gb": 80,
            "cpu_total": 128,
            "ram_total_gb": 1024,
            "gpu_used": 0,
            "cpu_used": 0,
            "ram_used_gb": 0,
            "status": "Ready",
        }
        idx += 1

    return nodes


def _build_default_jobs(nodes: dict[str, dict]) -> tuple[dict, dict]:
    """Create 70 jobs with varied resource requests and priorities.

    Returns (jobs_dict, assignments_dict).
    """
    import random
    rng = random.Random(42)

    job_groups = [
        ("llm-pretrain", "urgent", 4, 32, 128),     # large GPU jobs
        ("llm-finetune", "normal", 2, 16, 64),      # medium GPU jobs
        ("eval-suite", "normal", 1, 8, 32),          # small GPU jobs
        ("data-prep", "preemptible", 0, 16, 64),     # CPU-only
        ("hparam-sweep", "preemptible", 1, 4, 16),   # small preemptible
        ("diffusion-train", "normal", 4, 32, 128),   # large GPU jobs
        ("rl-train", "normal", 2, 16, 64),           # medium GPU jobs
        ("embedding-gen", "preemptible", 1, 8, 32),  # small preemptible
        ("benchmark", "urgent", 2, 16, 64),          # urgent medium
        ("distillation", "normal", 2, 16, 64),       # medium
    ]

    jobs = {}
    assignments = {}
    job_id = 1

    # Build list of schedulable nodes
    ready_nodes = [nid for nid, n in nodes.items() if n["status"] == "Ready"]

    for group_name, priority, gpu_req, cpu_req, ram_req in job_groups:
        count = rng.randint(5, 9)
        for i in range(count):
            jid = f"job-{job_id:03d}"
            # Add some variation to resource requests
            gpu_var = max(0, gpu_req + rng.choice([-1, 0, 0, 1]))
            cpu_var = max(1, cpu_req + rng.randint(-4, 4))
            ram_var = max(8, ram_req + rng.randint(-16, 16))
            rack_affinity = rng.choice(["rack-a", "rack-b", "rack-c", None, None])
            jobs[jid] = {
                "group": group_name,
                "priority": priority,
                "gpu_req": gpu_var,
                "cpu_req": cpu_var,
                "ram_req_gb": ram_var,
                "rack_affinity": rack_affinity,
                "status": "Running",  # Running | Queued
            }
            job_id += 1

    # Greedy initial assignment: pack jobs onto nodes
    node_usage = {nid: {"gpu": 0, "cpu": 0, "ram": 0} for nid in ready_nodes}
    job_list = sorted(jobs.keys(), key=lambda j: -jobs[j]["gpu_req"])

    for jid in job_list:
        job = jobs[jid]
        placed = False
        # Try nodes (prefer affinity rack)
        candidates = ready_nodes[:]
        if job["rack_affinity"]:
            candidates.sort(
                key=lambda n: (0 if nodes[n]["rack"] == job["rack_affinity"] else 1)
            )
        for nid in candidates:
            node = nodes[nid]
            usage = node_usage[nid]
            if (usage["gpu"] + job["gpu_req"] <= node["gpu_total"]
                    and usage["cpu"] + job["cpu_req"] <= node["cpu_total"]
                    and usage["ram"] + job["ram_req_gb"] <= node["ram_total_gb"]):
                assignments[jid] = nid
                usage["gpu"] += job["gpu_req"]
                usage["cpu"] += job["cpu_req"]
                usage["ram"] += job["ram_req_gb"]
                placed = True
                break
        if not placed:
            job["status"] = "Queued"

    return jobs, assignments


class ClusterManager(BaseSystemManager):
    """GPU cluster simulator. Pure Python, no external dependencies.

    Implements BaseSystemManager for SiLR verification compatibility.
    """

    def __init__(self):
        self._time: float = 0.0
        self._nodes: dict[str, dict] = _build_default_nodes()
        self._jobs: dict[str, dict]
        self._assignments: dict[str, str]  # job_id -> node_id
        self._jobs, self._assignments = _build_default_jobs(self._nodes)
        self._recompute_node_usage()

    # --- BaseSystemManager interface ---

    @property
    def sim_time(self) -> float:
        return self._time

    @property
    def base_mva(self) -> float:
        return 1.0

    @property
    def system_state(self) -> dict:
        return {
            "nodes": self._nodes,
            "jobs": self._jobs,
            "assignments": self._assignments,
        }

    def create_shadow_copy(self) -> ClusterManager:
        shadow = ClusterManager.__new__(ClusterManager)
        shadow._time = self._time
        shadow._nodes = copy.deepcopy(self._nodes)
        shadow._jobs = copy.deepcopy(self._jobs)
        shadow._assignments = copy.deepcopy(self._assignments)
        return shadow

    def run_pflow(self) -> bool:
        """Recompute node utilization from current assignments.

        Returns True if all jobs are either Running or cluster has no
        capacity to schedule remaining Queued jobs (i.e., state is consistent).
        Always returns True since this is a bookkeeping step, not a solver.
        """
        self._recompute_node_usage()
        self._time += 1.0
        return True

    # --- Domain-specific operations ---

    def fail_node(self, node_id: str) -> None:
        """Simulate node failure. All jobs on this node become Queued."""
        node = self._nodes[node_id]
        node["status"] = "NotReady"
        # Re-queue all jobs on this node
        evicted = [jid for jid, nid in self._assignments.items() if nid == node_id]
        for jid in evicted:
            self._jobs[jid]["status"] = "Queued"
            del self._assignments[jid]
        self._recompute_node_usage()

    def restore_node(self, node_id: str) -> None:
        """Bring a failed node back online."""
        self._nodes[node_id]["status"] = "Ready"

    def fail_rack(self, rack: str) -> None:
        """Simulate rack failure. All nodes in rack go down."""
        for nid, node in self._nodes.items():
            if node["rack"] == rack:
                self.fail_node(nid)

    def add_jobs(self, new_jobs: list[dict]) -> list[str]:
        """Add new jobs to the cluster (simulates job surge). Returns job IDs."""
        added = []
        base = len(self._jobs) + 1
        for i, spec in enumerate(new_jobs):
            jid = f"job-{base + i:03d}"
            self._jobs[jid] = {
                "group": spec.get("group", "dynamic"),
                "priority": spec.get("priority", "normal"),
                "gpu_req": spec.get("gpu_req", 1),
                "cpu_req": spec.get("cpu_req", 8),
                "ram_req_gb": spec.get("ram_req_gb", 32),
                "rack_affinity": spec.get("rack_affinity"),
                "status": "Queued",
            }
            added.append(jid)
        return added

    def get_node_ids(self) -> list[str]:
        return sorted(self._nodes.keys())

    def get_job_ids(self) -> list[str]:
        return sorted(self._jobs.keys())

    def get_queued_jobs(self) -> list[str]:
        return [jid for jid, j in self._jobs.items() if j["status"] == "Queued"]

    def get_schedulable_nodes(self) -> list[str]:
        return [nid for nid, n in self._nodes.items()
                if n["status"] == "Ready"]

    # --- Internal ---

    def _recompute_node_usage(self) -> None:
        """Recalculate gpu_used/cpu_used/ram_used from assignments."""
        for node in self._nodes.values():
            node["gpu_used"] = 0
            node["cpu_used"] = 0
            node["ram_used_gb"] = 0

        for jid, nid in self._assignments.items():
            job = self._jobs[jid]
            node = self._nodes[nid]
            node["gpu_used"] += job["gpu_req"]
            node["cpu_used"] += job["cpu_req"]
            node["ram_used_gb"] += job["ram_req_gb"]
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd /mnt/d/SciTokyo/SILR-Agent && python -m pytest tests/test_cluster_manager.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add domains/cluster/__init__.py domains/cluster/manager.py tests/test_cluster_manager.py
git commit -m "feat: add ClusterManager for GPU cluster scheduling domain"
```

---

### Task 2: Constraint Checkers (5 checkers)

**Files:**
- Create: `domains/cluster/checkers.py`
- Test: `tests/test_cluster_checkers.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cluster_checkers.py
"""Tests for GPU cluster constraint checkers."""

import pytest
from domains.cluster.manager import ClusterManager
from domains.cluster.checkers import (
    ResourceCapacityChecker,
    AffinityChecker,
    RackSpreadChecker,
    PriorityChecker,
    QueueChecker,
)


@pytest.fixture
def mgr():
    return ClusterManager()


class TestResourceCapacityChecker:
    def test_clean_state_passes(self, mgr):
        checker = ResourceCapacityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.passed
        assert result.checker_name == "resource_capacity"

    def test_overloaded_node_fails(self, mgr):
        # Force a node over GPU capacity
        mgr._nodes["node-01"]["gpu_used"] = 99
        checker = ResourceCapacityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed
        assert len(result.violations) >= 1
        assert result.violations[0].constraint_type == "resource_capacity"

    def test_summary_has_max_utilization(self, mgr):
        checker = ResourceCapacityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert "max_gpu_util" in result.summary


class TestAffinityChecker:
    def test_no_violations_in_default(self, mgr):
        checker = AffinityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        # May or may not pass depending on default placement
        assert result.checker_name == "affinity"

    def test_violation_when_job_in_wrong_rack(self, mgr):
        # Find a job with rack_affinity and move it to wrong rack
        for jid, job in mgr._jobs.items():
            if job["rack_affinity"] and jid in mgr._assignments:
                target_rack = job["rack_affinity"]
                wrong_node = next(
                    nid for nid, n in mgr._nodes.items()
                    if n["rack"] != target_rack and n["status"] == "Ready"
                )
                mgr._assignments[jid] = wrong_node
                break
        checker = AffinityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed


class TestPriorityChecker:
    def test_no_urgent_queued_passes(self, mgr):
        checker = PriorityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.checker_name == "priority"

    def test_urgent_queued_while_preemptible_running_fails(self, mgr):
        # Create an urgent queued job
        mgr._jobs["job-urgent"] = {
            "group": "deadline", "priority": "urgent",
            "gpu_req": 1, "cpu_req": 4, "ram_req_gb": 16,
            "rack_affinity": None, "status": "Queued",
        }
        # Ensure a preemptible job is running
        has_preemptible = any(
            j["priority"] == "preemptible" and j["status"] == "Running"
            for j in mgr._jobs.values()
        )
        if not has_preemptible:
            pytest.skip("No preemptible running job in default state")
        checker = PriorityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed


class TestQueueChecker:
    def test_all_running_passes(self, mgr):
        # Force all jobs to Running
        for job in mgr._jobs.values():
            job["status"] = "Running"
        checker = QueueChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.passed

    def test_queued_jobs_fail(self, mgr):
        mgr._jobs["job-001"]["status"] = "Queued"
        if "job-001" in mgr._assignments:
            del mgr._assignments["job-001"]
        checker = QueueChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed
        assert result.summary["queued_count"] >= 1


class TestRackSpreadChecker:
    def test_checker_name(self, mgr):
        checker = RackSpreadChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.checker_name == "rack_spread"
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m pytest tests/test_cluster_checkers.py -v`
Expected: `ImportError`

- [ ] **Step 3: Implement 5 checkers**

```python
# domains/cluster/checkers.py
"""GPU cluster constraint checkers for SiLR verification.

Five constraints:
1. Resource capacity — no node exceeds GPU/CPU/RAM limits
2. Rack affinity — jobs with rack_affinity are on correct rack
3. Rack spread — fault-tolerant groups span 2+ racks
4. Priority — no urgent job queued while preemptible running
5. Queue — all jobs scheduled (recovery target)
"""

from __future__ import annotations

from typing import Any

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation


class ResourceCapacityChecker(BaseConstraintChecker):
    """No node exceeds GPU, CPU, or RAM capacity."""

    name = "resource_capacity"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        nodes = system_state["nodes"]
        violations = []
        max_gpu_util = 0.0

        for nid, node in nodes.items():
            if node["status"] == "NotReady":
                continue
            gpu_util = node["gpu_used"] / node["gpu_total"] if node["gpu_total"] else 0
            max_gpu_util = max(max_gpu_util, gpu_util)

            if node["gpu_used"] > node["gpu_total"]:
                violations.append(Violation(
                    constraint_type="resource_capacity",
                    device_type="node",
                    device_id=nid,
                    metric="gpu_used",
                    value=float(node["gpu_used"]),
                    limit=float(node["gpu_total"]),
                    unit="GPUs",
                    severity="critical",
                    detail=f"{nid}: {node['gpu_used']}/{node['gpu_total']} GPUs",
                ))
            if node["cpu_used"] > node["cpu_total"]:
                violations.append(Violation(
                    constraint_type="resource_capacity",
                    device_type="node",
                    device_id=nid,
                    metric="cpu_used",
                    value=float(node["cpu_used"]),
                    limit=float(node["cpu_total"]),
                    unit="cores",
                    severity="violation",
                    detail=f"{nid}: {node['cpu_used']}/{node['cpu_total']} CPU cores",
                ))
            if node["ram_used_gb"] > node["ram_total_gb"]:
                violations.append(Violation(
                    constraint_type="resource_capacity",
                    device_type="node",
                    device_id=nid,
                    metric="ram_used_gb",
                    value=float(node["ram_used_gb"]),
                    limit=float(node["ram_total_gb"]),
                    unit="GB",
                    severity="violation",
                    detail=f"{nid}: {node['ram_used_gb']}/{node['ram_total_gb']} GB RAM",
                ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "max_gpu_util": round(max_gpu_util, 3),
                "overloaded_nodes": len({v.device_id for v in violations}),
                "n_violations": len(violations),
            },
            violations=violations,
        )


class AffinityChecker(BaseConstraintChecker):
    """Jobs with rack_affinity must be placed in the correct rack."""

    name = "affinity"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        jobs = system_state["jobs"]
        nodes = system_state["nodes"]
        assignments = system_state["assignments"]
        violations = []

        for jid, job in jobs.items():
            if job["status"] != "Running" or not job.get("rack_affinity"):
                continue
            if jid not in assignments:
                continue
            nid = assignments[jid]
            actual_rack = nodes[nid]["rack"]
            if actual_rack != job["rack_affinity"]:
                violations.append(Violation(
                    constraint_type="affinity",
                    device_type="job",
                    device_id=jid,
                    metric="rack_match",
                    value=0.0,
                    limit=1.0,
                    unit="bool",
                    severity="warning",
                    detail=f"{jid} wants {job['rack_affinity']} but on {nid} ({actual_rack})",
                ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "affinity_violations": len(violations),
                "n_violations": len(violations),
            },
            violations=violations,
        )


class RackSpreadChecker(BaseConstraintChecker):
    """Urgent job groups should have replicas spread across 2+ racks."""

    name = "rack_spread"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        jobs = system_state["jobs"]
        nodes = system_state["nodes"]
        assignments = system_state["assignments"]
        violations = []

        # Group urgent running jobs by group name
        group_racks: dict[str, set[str]] = {}
        group_counts: dict[str, int] = {}
        for jid, job in jobs.items():
            if job["priority"] != "urgent" or job["status"] != "Running":
                continue
            g = job["group"]
            group_counts[g] = group_counts.get(g, 0) + 1
            if jid in assignments:
                rack = nodes[assignments[jid]]["rack"]
                group_racks.setdefault(g, set()).add(rack)

        for g, count in group_counts.items():
            if count >= 2:  # Only check spread for groups with 2+ jobs
                racks_used = len(group_racks.get(g, set()))
                if racks_used < 2:
                    violations.append(Violation(
                        constraint_type="rack_spread",
                        device_type="group",
                        device_id=g,
                        metric="racks_covered",
                        value=float(racks_used),
                        limit=2.0,
                        unit="racks",
                        severity="warning",
                        detail=f"Group '{g}' has {count} urgent jobs in only {racks_used} rack(s)",
                    ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "groups_checked": len(group_counts),
                "spread_violations": len(violations),
                "n_violations": len(violations),
            },
            violations=violations,
        )


class PriorityChecker(BaseConstraintChecker):
    """No urgent job should be Queued while preemptible jobs are Running."""

    name = "priority"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        jobs = system_state["jobs"]
        violations = []

        urgent_queued = [
            jid for jid, j in jobs.items()
            if j["priority"] == "urgent" and j["status"] == "Queued"
        ]
        preemptible_running = [
            jid for jid, j in jobs.items()
            if j["priority"] == "preemptible" and j["status"] == "Running"
        ]

        if urgent_queued and preemptible_running:
            for jid in urgent_queued:
                violations.append(Violation(
                    constraint_type="priority",
                    device_type="job",
                    device_id=jid,
                    metric="urgent_queued",
                    value=1.0,
                    limit=0.0,
                    unit="bool",
                    severity="critical",
                    detail=(
                        f"Urgent job {jid} queued while {len(preemptible_running)} "
                        f"preemptible job(s) running"
                    ),
                ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "urgent_queued": len(urgent_queued),
                "preemptible_running": len(preemptible_running),
                "n_violations": len(violations),
            },
            violations=violations,
        )


class QueueChecker(BaseConstraintChecker):
    """All jobs must be scheduled (no Queued jobs remaining)."""

    name = "queue"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        jobs = system_state["jobs"]
        violations = []

        queued = [jid for jid, j in jobs.items() if j["status"] == "Queued"]
        total = len(jobs)

        for jid in queued:
            job = jobs[jid]
            violations.append(Violation(
                constraint_type="queue",
                device_type="job",
                device_id=jid,
                metric="queued",
                value=1.0,
                limit=0.0,
                unit="bool",
                severity="violation",
                detail=f"{jid} ({job['group']}, {job['priority']}) needs {job['gpu_req']} GPUs",
            ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "queued_count": len(queued),
                "total_jobs": total,
                "queue_ratio": round(len(queued) / total, 3) if total else 0,
                "n_violations": len(violations),
            },
            violations=violations,
        )
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `python -m pytest tests/test_cluster_checkers.py -v`

- [ ] **Step 5: Commit**

```bash
git add domains/cluster/checkers.py tests/test_cluster_checkers.py
git commit -m "feat: add 5 constraint checkers for cluster domain"
```

---

### Task 3: Tools (6 agent tools)

**Files:**
- Create: `domains/cluster/tools.py`
- Test: `tests/test_cluster_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cluster_tools.py
"""Tests for GPU cluster agent tools."""

import pytest
from domains.cluster.manager import ClusterManager
from domains.cluster.tools import (
    AssignJobTool,
    MigrateJobTool,
    PreemptJobTool,
    ScaleJobTool,
    DrainNodeTool,
    RestoreNodeTool,
    create_cluster_toolset,
)
from silr.exceptions import ValidationError, DeviceNotFoundError


@pytest.fixture
def mgr():
    m = ClusterManager()
    # Ensure at least one job is Queued for testing
    m._jobs["job-001"]["status"] = "Queued"
    if "job-001" in m._assignments:
        del m._assignments["job-001"]
    m._recompute_node_usage()
    return m


class TestAssignJobTool:
    def test_assign_queued_job(self, mgr):
        tool = AssignJobTool(mgr)
        # Find a node with capacity
        node_id = mgr.get_schedulable_nodes()[0]
        result = tool.execute(job_id="job-001", node_id=node_id)
        assert result["status"] == "success"
        assert mgr._jobs["job-001"]["status"] == "Running"
        assert mgr._assignments["job-001"] == node_id

    def test_assign_nonexistent_job_fails(self, mgr):
        tool = AssignJobTool(mgr)
        result = tool.execute(job_id="job-999", node_id="node-01")
        assert result["status"] == "error"

    def test_assign_running_job_fails(self, mgr):
        tool = AssignJobTool(mgr)
        running = next(j for j, d in mgr._jobs.items() if d["status"] == "Running")
        result = tool.execute(job_id=running, node_id="node-01")
        assert result["status"] == "error"


class TestPreemptJobTool:
    def test_preempt_running_job(self, mgr):
        running = next(j for j, d in mgr._jobs.items() if d["status"] == "Running")
        tool = PreemptJobTool(mgr)
        result = tool.execute(job_id=running)
        assert result["status"] == "success"
        assert mgr._jobs[running]["status"] == "Queued"
        assert running not in mgr._assignments


class TestDrainNodeTool:
    def test_drain_node(self, mgr):
        tool = DrainNodeTool(mgr)
        result = tool.execute(node_id="node-01")
        assert result["status"] == "success"
        assert mgr._nodes["node-01"]["status"] == "Cordoned"


class TestRestoreNodeTool:
    def test_restore_failed_node(self, mgr):
        mgr.fail_node("node-01")
        tool = RestoreNodeTool(mgr)
        result = tool.execute(node_id="node-01")
        assert result["status"] == "success"
        assert mgr._nodes["node-01"]["status"] == "Ready"


class TestCreateToolset:
    def test_returns_all_tools(self, mgr):
        toolset = create_cluster_toolset(mgr)
        expected = {"assign_job", "migrate_job", "preempt_job",
                    "scale_job", "drain_node", "restore_node"}
        assert set(toolset.keys()) == expected
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m pytest tests/test_cluster_tools.py -v`

- [ ] **Step 3: Implement 6 tools**

```python
# domains/cluster/tools.py
"""GPU cluster tools for SiLR agent actions.

6 tools: assign_job, migrate_job, preempt_job, scale_job, drain_node, restore_node.
"""

from __future__ import annotations

from silr.tools.base import BaseTool
from silr.exceptions import DeviceNotFoundError, ValidationError, SystemStateError


class AssignJobTool(BaseTool):
    """Assign a Queued job to a node."""

    name = "assign_job"
    description = "Place a queued job on a specific GPU node"

    def _validate_params(self, job_id: str = "", node_id: str = "", **kw) -> None:
        if not job_id:
            raise ValidationError("job_id is required")
        if not node_id:
            raise ValidationError("node_id is required")
        if job_id not in self.manager._jobs:
            raise DeviceNotFoundError(f"Job {job_id} not found")
        if node_id not in self.manager._nodes:
            raise DeviceNotFoundError(f"Node {node_id} not found")
        if self.manager._jobs[job_id]["status"] != "Queued":
            raise SystemStateError(f"Job {job_id} is not Queued (status: {self.manager._jobs[job_id]['status']})")
        if self.manager._nodes[node_id]["status"] != "Ready":
            raise SystemStateError(f"Node {node_id} is not Ready (status: {self.manager._nodes[node_id]['status']})")

    def _run(self, job_id: str = "", node_id: str = "", **kw) -> dict:
        mgr = self.manager
        job = mgr._jobs[job_id]
        job["status"] = "Running"
        mgr._assignments[job_id] = node_id
        mgr._recompute_node_usage()
        return {"job_id": job_id, "node_id": node_id, "assigned": True}


class MigrateJobTool(BaseTool):
    """Move a running job to a different node (checkpoint + restart)."""

    name = "migrate_job"
    description = "Migrate a running job to a different node"

    def _validate_params(self, job_id: str = "", target_node: str = "", **kw) -> None:
        if not job_id:
            raise ValidationError("job_id is required")
        if not target_node:
            raise ValidationError("target_node is required")
        if job_id not in self.manager._jobs:
            raise DeviceNotFoundError(f"Job {job_id} not found")
        if target_node not in self.manager._nodes:
            raise DeviceNotFoundError(f"Node {target_node} not found")
        if self.manager._jobs[job_id]["status"] != "Running":
            raise SystemStateError(f"Job {job_id} is not Running")
        if self.manager._nodes[target_node]["status"] != "Ready":
            raise SystemStateError(f"Node {target_node} is not Ready")

    def _run(self, job_id: str = "", target_node: str = "", **kw) -> dict:
        mgr = self.manager
        old_node = mgr._assignments.get(job_id, "none")
        mgr._assignments[job_id] = target_node
        mgr._recompute_node_usage()
        return {"job_id": job_id, "from_node": old_node, "to_node": target_node, "migrated": True}


class PreemptJobTool(BaseTool):
    """Suspend a running job, freeing its resources. Job becomes Queued."""

    name = "preempt_job"
    description = "Preempt a running job to free resources (job re-queued)"

    def _validate_params(self, job_id: str = "", **kw) -> None:
        if not job_id:
            raise ValidationError("job_id is required")
        if job_id not in self.manager._jobs:
            raise DeviceNotFoundError(f"Job {job_id} not found")
        if self.manager._jobs[job_id]["status"] != "Running":
            raise SystemStateError(f"Job {job_id} is not Running")

    def _run(self, job_id: str = "", **kw) -> dict:
        mgr = self.manager
        old_node = mgr._assignments.pop(job_id, "none")
        mgr._jobs[job_id]["status"] = "Queued"
        mgr._recompute_node_usage()
        return {"job_id": job_id, "freed_node": old_node, "preempted": True}


class ScaleJobTool(BaseTool):
    """Adjust GPU allocation for a running job (elastic training)."""

    name = "scale_job"
    description = "Change GPU count for a running job (elastic training)"

    def _validate_params(self, job_id: str = "", gpu_count: int = 0, **kw) -> None:
        if not job_id:
            raise ValidationError("job_id is required")
        if gpu_count <= 0:
            raise ValidationError("gpu_count must be positive")
        if job_id not in self.manager._jobs:
            raise DeviceNotFoundError(f"Job {job_id} not found")
        if self.manager._jobs[job_id]["status"] != "Running":
            raise SystemStateError(f"Job {job_id} is not Running")

    def _run(self, job_id: str = "", gpu_count: int = 0, **kw) -> dict:
        mgr = self.manager
        old_count = mgr._jobs[job_id]["gpu_req"]
        mgr._jobs[job_id]["gpu_req"] = gpu_count
        mgr._recompute_node_usage()
        return {"job_id": job_id, "old_gpu": old_count, "new_gpu": gpu_count, "scaled": True}


class DrainNodeTool(BaseTool):
    """Mark a node as unschedulable (no new jobs placed here)."""

    name = "drain_node"
    description = "Cordon a node — no new jobs will be assigned to it"

    def _validate_params(self, node_id: str = "", **kw) -> None:
        if not node_id:
            raise ValidationError("node_id is required")
        if node_id not in self.manager._nodes:
            raise DeviceNotFoundError(f"Node {node_id} not found")

    def _run(self, node_id: str = "", **kw) -> dict:
        mgr = self.manager
        old_status = mgr._nodes[node_id]["status"]
        mgr._nodes[node_id]["status"] = "Cordoned"
        return {"node_id": node_id, "old_status": old_status, "drained": True}


class RestoreNodeTool(BaseTool):
    """Bring a node back online (from NotReady or Cordoned)."""

    name = "restore_node"
    description = "Restore a failed or cordoned node to Ready status"

    def _validate_params(self, node_id: str = "", **kw) -> None:
        if not node_id:
            raise ValidationError("node_id is required")
        if node_id not in self.manager._nodes:
            raise DeviceNotFoundError(f"Node {node_id} not found")
        if self.manager._nodes[node_id]["status"] == "Ready":
            raise SystemStateError(f"Node {node_id} is already Ready")

    def _run(self, node_id: str = "", **kw) -> dict:
        mgr = self.manager
        mgr._nodes[node_id]["status"] = "Ready"
        return {"node_id": node_id, "restored": True}


def create_cluster_toolset(manager) -> dict:
    """Create toolset for the GPU cluster domain."""
    tools = [
        AssignJobTool(manager),
        MigrateJobTool(manager),
        PreemptJobTool(manager),
        ScaleJobTool(manager),
        DrainNodeTool(manager),
        RestoreNodeTool(manager),
    ]
    return {t.name: t for t in tools}
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `python -m pytest tests/test_cluster_tools.py -v`

- [ ] **Step 5: Commit**

```bash
git add domains/cluster/tools.py tests/test_cluster_tools.py
git commit -m "feat: add 6 agent tools for cluster domain"
```

---

### Task 4: Scenarios + Loader

**Files:**
- Create: `domains/cluster/scenarios/__init__.py`
- Create: `domains/cluster/scenarios/loader.py`
- Test: `tests/test_cluster_scenarios.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cluster_scenarios.py
"""Tests for cluster scenario loader."""

import pytest
from domains.cluster.manager import ClusterManager
from domains.cluster.scenarios.loader import ClusterScenarioLoader, ClusterScenario


@pytest.fixture
def loader():
    return ClusterScenarioLoader()


class TestScenarioLoader:
    def test_load_known_scenario(self, loader):
        s = loader.load("node_failure_single")
        assert isinstance(s, ClusterScenario)
        assert s.id == "node_failure_single"

    def test_load_unknown_raises(self, loader):
        with pytest.raises(KeyError):
            loader.load("nonexistent")

    def test_load_all_returns_list(self, loader):
        scenarios = loader.load_all()
        assert len(scenarios) >= 6

    def test_setup_episode_creates_queued_jobs(self, loader):
        mgr = ClusterManager()
        scenario = loader.load("node_failure_single")
        loader.setup_episode(mgr, scenario)
        queued = mgr.get_queued_jobs()
        assert len(queued) > 0

    def test_all_scenarios_have_difficulty(self, loader):
        for s in loader.load_all():
            assert s.difficulty in ("easy", "medium", "hard")

    def test_parameterized_generation(self, loader):
        all_s = loader.load_all()
        assert len(all_s) >= 15  # base + parameterized
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m pytest tests/test_cluster_scenarios.py -v`

- [ ] **Step 3: Implement scenarios and loader**

```python
# domains/cluster/scenarios/__init__.py
"""Cluster scenario definitions."""

from .loader import ClusterScenario, ClusterScenarioLoader

__all__ = ["ClusterScenario", "ClusterScenarioLoader"]
```

```python
# domains/cluster/scenarios/loader.py
"""Scenario definitions and loader for GPU cluster scheduling domain.

6 base scenario types, parameterized to 50+ variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..manager import ClusterManager


@dataclass
class ClusterScenario:
    """GPU cluster incident scenario."""

    id: str
    description: str
    difficulty: str = "easy"  # easy | medium | hard
    node_failures: list[str] = field(default_factory=list)
    rack_failure: str | None = None
    new_jobs: list[dict] = field(default_factory=list)
    force_queued: list[str] = field(default_factory=list)  # job IDs to force Queued


def _base_scenarios() -> list[ClusterScenario]:
    """6 base scenario types."""
    return [
        # 1. Single node failure
        ClusterScenario(
            id="node_failure_single",
            description="Single GPU node failure — reschedule displaced jobs",
            difficulty="easy",
            node_failures=["node-03"],
        ),
        # 2. Rack failure
        ClusterScenario(
            id="rack_failure",
            description="Full rack power failure — major rescheduling needed",
            difficulty="hard",
            rack_failure="rack-b",
        ),
        # 3. Job surge
        ClusterScenario(
            id="job_surge",
            description="Burst of urgent jobs requiring immediate placement",
            difficulty="medium",
            new_jobs=[
                {"group": "deadline-exp", "priority": "urgent", "gpu_req": 2, "cpu_req": 16, "ram_req_gb": 64},
                {"group": "deadline-exp", "priority": "urgent", "gpu_req": 2, "cpu_req": 16, "ram_req_gb": 64},
                {"group": "deadline-exp", "priority": "urgent", "gpu_req": 4, "cpu_req": 32, "ram_req_gb": 128},
                {"group": "deadline-exp", "priority": "urgent", "gpu_req": 1, "cpu_req": 8, "ram_req_gb": 32},
            ],
        ),
        # 4. Resource fragmentation
        ClusterScenario(
            id="fragmentation",
            description="Jobs queued due to GPU fragmentation — consolidation needed",
            difficulty="medium",
            new_jobs=[
                {"group": "large-train", "priority": "normal", "gpu_req": 4, "cpu_req": 32, "ram_req_gb": 128},
                {"group": "large-train", "priority": "normal", "gpu_req": 4, "cpu_req": 32, "ram_req_gb": 128},
            ],
        ),
        # 5. Priority conflict
        ClusterScenario(
            id="priority_conflict",
            description="Urgent jobs queued — preempt lower-priority jobs",
            difficulty="easy",
            new_jobs=[
                {"group": "critical-eval", "priority": "urgent", "gpu_req": 2, "cpu_req": 16, "ram_req_gb": 64},
                {"group": "critical-eval", "priority": "urgent", "gpu_req": 2, "cpu_req": 16, "ram_req_gb": 64},
            ],
        ),
        # 6. Compound: node failure + job surge
        ClusterScenario(
            id="compound_failure_surge",
            description="Node failure during job surge — cascading scheduling crisis",
            difficulty="hard",
            node_failures=["node-05"],
            new_jobs=[
                {"group": "emergency", "priority": "urgent", "gpu_req": 2, "cpu_req": 16, "ram_req_gb": 64},
                {"group": "emergency", "priority": "urgent", "gpu_req": 4, "cpu_req": 32, "ram_req_gb": 128},
            ],
        ),
    ]


def _parameterize() -> list[ClusterScenario]:
    """Generate parameterized variants from base scenarios."""
    variants = []

    # Node failure variants: different nodes, different types
    for node_id in ["node-01", "node-07", "node-13"]:
        variants.append(ClusterScenario(
            id=f"node_failure_{node_id}",
            description=f"Single node failure: {node_id}",
            difficulty="easy",
            node_failures=[node_id],
        ))

    # Multi-node failure (same rack)
    variants.append(ClusterScenario(
        id="multi_node_rack_a",
        description="Two nodes fail in rack-a",
        difficulty="medium",
        node_failures=["node-01", "node-03"],
    ))
    variants.append(ClusterScenario(
        id="multi_node_rack_c",
        description="Two nodes fail in rack-c",
        difficulty="medium",
        node_failures=["node-09", "node-11"],
    ))

    # Rack failure variants
    for rack in ["rack-a", "rack-c"]:
        variants.append(ClusterScenario(
            id=f"rack_failure_{rack}",
            description=f"Full rack failure: {rack}",
            difficulty="hard",
            rack_failure=rack,
        ))

    # Job surge variants: different sizes
    variants.append(ClusterScenario(
        id="job_surge_small",
        description="Small burst of normal jobs",
        difficulty="easy",
        new_jobs=[
            {"group": "batch", "priority": "normal", "gpu_req": 1, "cpu_req": 8, "ram_req_gb": 32},
            {"group": "batch", "priority": "normal", "gpu_req": 1, "cpu_req": 8, "ram_req_gb": 32},
        ],
    ))
    variants.append(ClusterScenario(
        id="job_surge_large_gpu",
        description="Large GPU job surge — fat nodes needed",
        difficulty="hard",
        new_jobs=[
            {"group": "mega-train", "priority": "urgent", "gpu_req": 8, "cpu_req": 64, "ram_req_gb": 256},
            {"group": "mega-train", "priority": "normal", "gpu_req": 4, "cpu_req": 32, "ram_req_gb": 128},
        ],
    ))

    # Compound variants
    for node in ["node-02", "node-08"]:
        variants.append(ClusterScenario(
            id=f"compound_{node}_surge",
            description=f"Node {node} failure + urgent job surge",
            difficulty="hard",
            node_failures=[node],
            new_jobs=[
                {"group": "emergency", "priority": "urgent", "gpu_req": 2, "cpu_req": 16, "ram_req_gb": 64},
            ],
        ))

    # Priority conflict with fat node requirement
    variants.append(ClusterScenario(
        id="priority_fat_node",
        description="Urgent 8-GPU job needs fat node — preempt occupants",
        difficulty="hard",
        new_jobs=[
            {"group": "deadline-large", "priority": "urgent", "gpu_req": 8, "cpu_req": 64, "ram_req_gb": 256},
        ],
    ))

    return variants


_ALL_SCENARIOS = _base_scenarios() + _parameterize()
_SCENARIO_MAP = {s.id: s for s in _ALL_SCENARIOS}


class ClusterScenarioLoader:
    """Load and apply GPU cluster scheduling scenarios."""

    def load(self, scenario_id: str) -> ClusterScenario:
        if scenario_id not in _SCENARIO_MAP:
            raise KeyError(
                f"Unknown scenario: {scenario_id}. "
                f"Available: {sorted(_SCENARIO_MAP.keys())}"
            )
        return _SCENARIO_MAP[scenario_id]

    def load_all(self) -> list[ClusterScenario]:
        return list(_ALL_SCENARIOS)

    def setup_episode(self, manager: ClusterManager, scenario: ClusterScenario) -> None:
        """Apply scenario faults and additions to the cluster."""
        # Apply node failures
        for node_id in scenario.node_failures:
            manager.fail_node(node_id)

        # Apply rack failure
        if scenario.rack_failure:
            manager.fail_rack(scenario.rack_failure)

        # Add new jobs
        if scenario.new_jobs:
            manager.add_jobs(scenario.new_jobs)

        # Force specific jobs to Queued
        for jid in scenario.force_queued:
            if jid in manager._jobs:
                manager._jobs[jid]["status"] = "Queued"
                manager._assignments.pop(jid, None)

        manager.run_pflow()
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `python -m pytest tests/test_cluster_scenarios.py -v`

- [ ] **Step 5: Commit**

```bash
git add domains/cluster/scenarios/ tests/test_cluster_scenarios.py
git commit -m "feat: add scenario loader with 6 base + parameterized variants"
```

---

### Task 5: Observer + Prompts + Failsafe + Config

**Files:**
- Create: `domains/cluster/observation.py`
- Create: `domains/cluster/failsafe.py`
- Create: `domains/cluster/prompts/__init__.py`
- Create: `domains/cluster/prompts/system_prompt.py`
- Create: `domains/cluster/prompts/tool_schemas.py`
- Create: `domains/cluster/config.py`
- Update: `domains/cluster/__init__.py`
- Test: `tests/test_cluster_integration.py`

This task wires everything together into a `DomainConfig` and tests the full SiLR verification pipeline.

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_cluster_integration.py
"""Integration tests: full SiLR verification on cluster domain."""

import pytest
from domains.cluster import ClusterManager
from domains.cluster.config import build_cluster_domain_config
from domains.cluster.scenarios.loader import ClusterScenarioLoader
from silr.verifier import SiLRVerifier
from silr.verifier.types import Verdict


@pytest.fixture
def setup():
    mgr = ClusterManager()
    config = build_cluster_domain_config()
    loader = ClusterScenarioLoader()
    scenario = loader.load("node_failure_single")
    loader.setup_episode(mgr, scenario)
    verifier = SiLRVerifier(mgr, domain_config=config)
    return mgr, verifier


class TestSiLRVerification:
    def test_valid_assign_passes(self, setup):
        mgr, verifier = setup
        queued = mgr.get_queued_jobs()
        assert len(queued) > 0
        schedulable = mgr.get_schedulable_nodes()
        result = verifier.verify({
            "tool_name": "assign_job",
            "params": {"job_id": queued[0], "node_id": schedulable[0]},
        })
        assert result.verdict in (Verdict.PASS, Verdict.FAIL)
        assert result.pflow_converged

    def test_invalid_tool_returns_error(self, setup):
        _, verifier = setup
        result = verifier.verify({
            "tool_name": "nonexistent_tool",
            "params": {},
        })
        assert result.verdict == Verdict.ERROR

    def test_observer_produces_observation(self):
        mgr = ClusterManager()
        config = build_cluster_domain_config()
        observer = config.create_observer(mgr)
        obs = observer.observe()
        assert isinstance(obs.compressed_json, str)
        assert isinstance(obs.is_stable, bool)

    def test_failsafe_suggests_action(self):
        mgr = ClusterManager()
        config = build_cluster_domain_config()
        loader = ClusterScenarioLoader()
        loader.setup_episode(mgr, loader.load("priority_conflict"))
        mgr.run_pflow()
        failsafe = config.create_failsafe(mgr)
        observer = config.create_observer(mgr)
        obs = observer.observe()
        if not obs.is_stable:
            action = failsafe.suggest(obs)
            assert action is None or "tool_name" in action

    def test_domain_config_has_all_fields(self):
        config = build_cluster_domain_config()
        assert config.domain_name == "gpu_cluster"
        assert len(config.checkers) == 5
        assert len(config.allowed_actions) == 6
        assert config.create_toolset is not None
        assert config.create_observer is not None
        assert config.create_failsafe is not None
        assert config.build_system_prompt is not None
        assert config.build_tool_schemas is not None
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m pytest tests/test_cluster_integration.py -v`

- [ ] **Step 3: Implement observer**

```python
# domains/cluster/observation.py
"""Cluster domain observer for LLM agent."""

from __future__ import annotations

import json

from silr.agent.observation import BaseObserver
from silr.agent.types import Observation
from .checkers import (
    ResourceCapacityChecker, AffinityChecker, RackSpreadChecker,
    PriorityChecker, QueueChecker,
)


class ClusterObserver(BaseObserver):
    """Observe GPU cluster state and produce compressed JSON for LLM."""

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

        # Compressed view for LLM
        down_nodes = [
            {"node": nid, "rack": n["rack"], "type": n["type"]}
            for nid, n in nodes.items() if n["status"] == "NotReady"
        ]
        cordoned = [nid for nid, n in nodes.items() if n["status"] == "Cordoned"]
        queued_jobs = [
            {"job": jid, "group": j["group"], "priority": j["priority"],
             "gpu_req": j["gpu_req"]}
            for jid, j in jobs.items() if j["status"] == "Queued"
        ]
        # Node utilization for ready nodes
        busy_nodes = []
        for nid, n in nodes.items():
            if n["status"] != "Ready":
                continue
            gpu_util = n["gpu_used"] / n["gpu_total"] if n["gpu_total"] else 0
            if gpu_util > 0.7:
                busy_nodes.append({
                    "node": nid, "gpu_util": round(gpu_util * 100, 1),
                    "gpu_free": n["gpu_total"] - n["gpu_used"],
                })

        compressed = {
            "down_nodes": down_nodes,
            "cordoned_nodes": cordoned,
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
```

- [ ] **Step 4: Implement failsafe**

```python
# domains/cluster/failsafe.py
"""Priority-first scheduling failsafe for GPU cluster domain."""

from __future__ import annotations

from typing import Optional

from silr.agent.failsafe import BaseFailsafe
from silr.agent.types import Observation


class ClusterFailsafe(BaseFailsafe):
    """Rule-based failsafe: preempt lowest-priority job, assign highest-priority queued."""

    def __init__(self, manager):
        self._manager = manager

    def suggest(self, obs: Observation) -> Optional[dict]:
        mgr = self._manager
        queued = mgr.get_queued_jobs()
        if not queued:
            return None

        # Find highest-priority queued job
        priority_order = {"urgent": 0, "normal": 1, "preemptible": 2}
        queued.sort(key=lambda j: priority_order.get(mgr._jobs[j]["priority"], 1))
        target_job = queued[0]
        job = mgr._jobs[target_job]

        # Find a node that can fit it
        for nid in mgr.get_schedulable_nodes():
            node = mgr._nodes[nid]
            if (node["gpu_total"] - node["gpu_used"] >= job["gpu_req"]
                    and node["cpu_total"] - node["cpu_used"] >= job["cpu_req"]
                    and node["ram_total_gb"] - node["ram_used_gb"] >= job["ram_req_gb"]):
                return {"tool_name": "assign_job", "params": {"job_id": target_job, "node_id": nid}}

        # No space — preempt a preemptible job to make room
        preemptible = [
            jid for jid, j in mgr._jobs.items()
            if j["priority"] == "preemptible" and j["status"] == "Running"
        ]
        if preemptible:
            return {"tool_name": "preempt_job", "params": {"job_id": preemptible[0]}}

        return None

    def suggest_escalated(self, obs: Observation, last_rejected: Optional[dict] = None) -> Optional[dict]:
        return self.suggest(obs)
```

- [ ] **Step 5: Implement prompts**

```python
# domains/cluster/prompts/__init__.py
"""Cluster domain prompt builders."""

from .system_prompt import build_cluster_system_prompt
from .tool_schemas import build_cluster_tool_schemas

__all__ = ["build_cluster_system_prompt", "build_cluster_tool_schemas"]
```

```python
# domains/cluster/prompts/system_prompt.py
"""System prompt for GPU cluster scheduling agent."""


def build_cluster_system_prompt(manager, tool_schemas: list) -> str:
    nodes = manager.system_state["nodes"]
    node_summary = []
    for nid, n in sorted(nodes.items()):
        node_summary.append(
            f"  {nid}: {n['type']}, {n['rack']}, "
            f"{n['gpu_total']} GPU, {n['cpu_total']} CPU, "
            f"{n['ram_total_gb']}GB RAM, status={n['status']}"
        )

    tool_desc = "\n".join(
        f"  - {s['function']['name']}: {s['function']['description']}"
        for s in tool_schemas
    )

    return f"""You are a GPU cluster scheduling agent managing a 15-node cluster.
Your task: resolve scheduling incidents by assigning queued jobs to available nodes,
migrating jobs for load balancing, or preempting low-priority jobs for urgent ones.

Cluster topology (3 racks, InfiniBand interconnect):
{chr(10).join(node_summary)}

Available actions:
{tool_desc}

Constraints you must satisfy:
1. Resource capacity: GPU/CPU/RAM must not exceed node limits
2. Rack affinity: jobs with rack_affinity must be placed in the correct rack
3. Rack spread: urgent job groups with 2+ jobs should span 2+ racks
4. Priority: no urgent job should be queued while preemptible jobs are running
5. Queue: all jobs must be scheduled (no queued jobs remaining)

Response format: output a JSON action.
{{"tool_name": "<action>", "params": {{...}}}}

If the system is stable and all jobs are scheduled, respond with:
{{"tool_name": "none", "params": {{}}}}

Strategy guidelines:
- Assign queued jobs to nodes with available capacity
- For priority conflicts, preempt preemptible jobs first
- For node failures, redistribute displaced jobs across remaining nodes
- Prefer nodes in the correct rack for affinity-sensitive jobs
- If rejected for capacity overflow, try a different node or preempt first
"""
```

```python
# domains/cluster/prompts/tool_schemas.py
"""Tool schemas for LLM function calling."""


def build_cluster_tool_schemas(manager) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "assign_job",
                "description": "Place a queued job on a specific GPU node",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Job ID (e.g. job-001)"},
                        "node_id": {"type": "string", "description": "Target node ID (e.g. node-01)"},
                    },
                    "required": ["job_id", "node_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "migrate_job",
                "description": "Migrate a running job to a different node",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Job ID to migrate"},
                        "target_node": {"type": "string", "description": "Destination node ID"},
                    },
                    "required": ["job_id", "target_node"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "preempt_job",
                "description": "Preempt a running job to free resources (job re-queued)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Job ID to preempt"},
                    },
                    "required": ["job_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scale_job",
                "description": "Change GPU count for a running job (elastic training)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Job ID to scale"},
                        "gpu_count": {"type": "integer", "description": "New GPU count"},
                    },
                    "required": ["job_id", "gpu_count"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "drain_node",
                "description": "Cordon a node — no new jobs will be assigned to it",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "Node ID to drain"},
                    },
                    "required": ["node_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "restore_node",
                "description": "Restore a failed or cordoned node to Ready status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "Node ID to restore"},
                    },
                    "required": ["node_id"],
                },
            },
        },
    ]
```

- [ ] **Step 6: Implement domain config**

```python
# domains/cluster/config.py
"""Cluster DomainConfig factory."""

from silr.core.config import DomainConfig
from .checkers import (
    ResourceCapacityChecker, AffinityChecker, RackSpreadChecker,
    PriorityChecker, QueueChecker,
)
from .tools import create_cluster_toolset
from .observation import ClusterObserver
from .failsafe import ClusterFailsafe
from .prompts import build_cluster_system_prompt, build_cluster_tool_schemas


def build_cluster_domain_config() -> DomainConfig:
    """Build DomainConfig for the GPU cluster scheduling domain."""
    return DomainConfig(
        domain_name="gpu_cluster",
        checkers=[
            ResourceCapacityChecker(),
            AffinityChecker(),
            RackSpreadChecker(),
            PriorityChecker(),
            QueueChecker(),
        ],
        allowed_actions=frozenset([
            "assign_job", "migrate_job", "preempt_job",
            "scale_job", "drain_node", "restore_node",
        ]),
        create_toolset=create_cluster_toolset,
        create_observer=lambda mgr: ClusterObserver(mgr),
        create_failsafe=lambda mgr: ClusterFailsafe(mgr),
        build_system_prompt=build_cluster_system_prompt,
        build_tool_schemas=build_cluster_tool_schemas,
    )
```

- [ ] **Step 7: Update `__init__.py` exports**

```python
# domains/cluster/__init__.py
"""GPU cluster scheduling domain for SiLR-Agent."""

from .manager import ClusterManager
from .config import build_cluster_domain_config
from .scenarios.loader import ClusterScenarioLoader, ClusterScenario

__all__ = [
    "ClusterManager",
    "build_cluster_domain_config",
    "ClusterScenarioLoader",
    "ClusterScenario",
]
```

- [ ] **Step 8: Run tests — expect PASS**

Run: `python -m pytest tests/test_cluster_integration.py -v`

- [ ] **Step 9: Run all cluster tests together**

Run: `python -m pytest tests/test_cluster_*.py -v`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add domains/cluster/ tests/test_cluster_integration.py
git commit -m "feat: complete cluster domain with observer, failsafe, prompts, and config"
```

---

### Task 6: GRPO Trainer — Step-Level

**Files:**
- Create: `silr/training/grpo_trainer.py`
- Test: `tests/test_grpo_trainer.py`

This is the core training innovation. Implements the iterative offline GRPO loop with step-level grouping.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_grpo_trainer.py
"""Tests for step-level GRPO trainer (unit tests with mock model)."""

import pytest
from dataclasses import dataclass
from silr.training.grpo_trainer import (
    GRPOConfig,
    StepSample,
    compute_advantages,
)


class TestGRPOConfig:
    def test_defaults(self):
        cfg = GRPOConfig()
        assert cfg.num_iterations == 5
        assert cfg.rollouts_per_scenario == 8
        assert cfg.clip_eps == 0.2
        assert cfg.kl_coeff == 0.1
        assert cfg.lr == 1e-5

    def test_custom(self):
        cfg = GRPOConfig(num_iterations=3, rollouts_per_scenario=4)
        assert cfg.num_iterations == 3
        assert cfg.rollouts_per_scenario == 4


class TestStepSample:
    def test_creation(self):
        s = StepSample(
            obs_text='{"test": true}',
            action_text='Thought: test\n{"tool_name": "assign_job", "params": {}}',
            reward=1.0,
            group_key=("scenario_1", 1),
        )
        assert s.advantage == 0.0  # default


class TestComputeAdvantages:
    def test_single_group(self):
        samples = [
            StepSample("obs", "act1", reward=1.5, group_key=("s1", 1)),
            StepSample("obs", "act2", reward=0.5, group_key=("s1", 1)),
            StepSample("obs", "act3", reward=1.0, group_key=("s1", 1)),
        ]
        compute_advantages(samples)
        # reward 1.5 should have positive advantage
        assert samples[0].advantage > 0
        # reward 0.5 should have negative advantage
        assert samples[1].advantage < 0
        # Should be zero-mean within group
        mean_adv = sum(s.advantage for s in samples) / len(samples)
        assert abs(mean_adv) < 1e-6

    def test_multiple_groups(self):
        samples = [
            StepSample("obs", "a", reward=1.0, group_key=("s1", 1)),
            StepSample("obs", "b", reward=0.0, group_key=("s1", 1)),
            StepSample("obs", "c", reward=2.0, group_key=("s2", 1)),
            StepSample("obs", "d", reward=1.0, group_key=("s2", 1)),
        ]
        compute_advantages(samples)
        # Group 1: rewards [1.0, 0.0] -> advantages [+, -]
        assert samples[0].advantage > 0
        assert samples[1].advantage < 0
        # Group 2: rewards [2.0, 1.0] -> advantages [+, -]
        assert samples[2].advantage > 0
        assert samples[3].advantage < 0

    def test_single_sample_group_zero_advantage(self):
        samples = [
            StepSample("obs", "a", reward=1.0, group_key=("s1", 1)),
        ]
        compute_advantages(samples)
        assert samples[0].advantage == 0.0

    def test_all_same_reward_zero_advantage(self):
        samples = [
            StepSample("obs", "a", reward=1.0, group_key=("s1", 1)),
            StepSample("obs", "b", reward=1.0, group_key=("s1", 1)),
        ]
        compute_advantages(samples)
        assert samples[0].advantage == 0.0
        assert samples[1].advantage == 0.0
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m pytest tests/test_grpo_trainer.py -v`

- [ ] **Step 3: Implement GRPO trainer**

```python
# silr/training/grpo_trainer.py
"""Step-level GRPO (Group Relative Policy Optimization) trainer.

Iterative offline GRPO loop:
1. Run current policy on scenarios → collect (obs, action, reward) per step
2. Group by (scenario_id, step_number), compute group-relative advantages
3. GRPO policy update with clipped objective + KL penalty against SFT ref
4. Repeat

This module provides the data structures and advantage computation.
The full training loop (model loading, tokenization, gradient updates)
is in the training script, since it depends on PyTorch/TRL/PEFT.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    """Configuration for step-level GRPO training."""

    # Rollout
    num_iterations: int = 5
    rollouts_per_scenario: int = 8

    # GRPO objective
    clip_eps: float = 0.2
    kl_coeff: float = 0.1

    # Training
    lr: float = 1e-5
    batch_size: int = 4
    grpo_epochs: int = 1
    max_seq_len: int = 4096

    # Model
    base_model: str = "Qwen/Qwen3-14B"
    sft_adapter_path: str = ""
    output_dir: str = "outputs/grpo"

    # Reward
    step_cost: float = 0.05


@dataclass
class StepSample:
    """A single step-level training sample for GRPO."""

    obs_text: str
    action_text: str
    reward: float
    group_key: tuple  # (scenario_id, step_number)
    advantage: float = 0.0
    log_prob: float = 0.0  # filled during training


def compute_advantages(samples: list[StepSample]) -> None:
    """Compute group-relative advantages in-place.

    Groups samples by group_key, then within each group:
        advantage_i = (reward_i - mean) / (std + eps)

    Single-sample groups or groups with zero variance get advantage=0.
    """
    groups: dict[tuple, list[StepSample]] = defaultdict(list)
    for s in samples:
        groups[s.group_key].append(s)

    for group_samples in groups.values():
        if len(group_samples) <= 1:
            for s in group_samples:
                s.advantage = 0.0
            continue

        rewards = [s.reward for s in group_samples]
        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = var_r ** 0.5

        if std_r < 1e-8:
            for s in group_samples:
                s.advantage = 0.0
        else:
            for s in group_samples:
                s.advantage = (s.reward - mean_r) / std_r
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `python -m pytest tests/test_grpo_trainer.py -v`

- [ ] **Step 5: Commit**

```bash
git add silr/training/grpo_trainer.py tests/test_grpo_trainer.py
git commit -m "feat: add step-level GRPO trainer with advantage computation"
```

---

### Task 7: Reward Extension for Cluster Domain

**Files:**
- Modify: `silr/training/reward.py`
- Test: `tests/test_reward.py` (add new test cases)

- [ ] **Step 1: Read existing reward tests**

Run: `cat tests/test_reward.py` to see existing test structure.

- [ ] **Step 2: Add cluster-domain reward tests**

Append to `tests/test_reward.py`:

```python
class TestClusterReward:
    """Test reward computation with cluster domain checkers."""

    def test_pass_with_margin(self):
        result = VerificationResult(
            verdict=Verdict.PASS,
            action={"tool_name": "assign_job", "params": {}},
            check_results=[
                CheckResult(
                    checker_name="resource_capacity",
                    passed=True,
                    summary={"max_gpu_util": 0.6, "n_violations": 0},
                ),
                CheckResult(
                    checker_name="queue",
                    passed=True,
                    summary={"queued_count": 0, "total_jobs": 70, "n_violations": 0},
                ),
            ],
        )
        config = RewardConfig(thresholds={
            "resource_capacity": {"max_util": 1.0},
            "queue": {},
        })
        reward = compute_grpo_reward(result, config)
        assert reward >= 1.0  # PASS base

    def test_fail_with_severity(self):
        result = VerificationResult(
            verdict=Verdict.FAIL,
            action={"tool_name": "assign_job", "params": {}},
            check_results=[
                CheckResult(
                    checker_name="resource_capacity",
                    passed=False,
                    summary={"n_violations": 1},
                    violations=[Violation(
                        constraint_type="resource_capacity",
                        device_type="node", device_id="node-01",
                        metric="gpu_used", value=5.0, limit=4.0,
                        unit="GPUs", severity="critical",
                        detail="node-01: 5/4 GPUs",
                    )],
                ),
            ],
        )
        reward = compute_grpo_reward(result)
        assert reward <= -0.3
```

- [ ] **Step 3: Add cluster margin logic to `_margin_for_check`**

Add to `silr/training/reward.py`, in the `_margin_for_check` function, before the generic fallback:

```python
    if name == "resource_capacity":
        max_util = _get_first(summary, "max_gpu_util")
        if max_util is not None:
            return max(0.0, min(1.0, 1.0 - max_util))

    if name == "queue":
        ratio = _get_first(summary, "queue_ratio")
        if ratio is not None:
            return max(0.0, min(1.0, 1.0 - ratio))
```

- [ ] **Step 4: Run all reward tests**

Run: `python -m pytest tests/test_reward.py -v`

- [ ] **Step 5: Commit**

```bash
git add silr/training/reward.py tests/test_reward.py
git commit -m "feat: add cluster domain margin computation to reward function"
```

---

### Task 8: Example Script + Full Regression

**Files:**
- Create: `examples/cluster_scheduling.py`

- [ ] **Step 1: Write example script**

```python
# examples/cluster_scheduling.py
"""Demo: GPU cluster scheduling with SiLR verification.

Shows the domain working end-to-end:
1. Create cluster + apply node failure scenario
2. Verify a scheduling action via SiLR
3. Observer produces LLM-readable state
"""

from domains.cluster import (
    ClusterManager,
    build_cluster_domain_config,
    ClusterScenarioLoader,
)
from silr.verifier import SiLRVerifier, Verdict


def main():
    # Setup
    mgr = ClusterManager()
    config = build_cluster_domain_config()
    loader = ClusterScenarioLoader()

    # Apply scenario: single node failure
    scenario = loader.load("node_failure_single")
    loader.setup_episode(mgr, scenario)
    print(f"Scenario: {scenario.description}")
    print(f"Queued jobs: {len(mgr.get_queued_jobs())}")

    # Create verifier
    verifier = SiLRVerifier(mgr, domain_config=config)

    # Try assigning a queued job
    queued = mgr.get_queued_jobs()
    schedulable = mgr.get_schedulable_nodes()
    if queued and schedulable:
        action = {
            "tool_name": "assign_job",
            "params": {"job_id": queued[0], "node_id": schedulable[0]},
        }
        result = verifier.verify(action)
        print(f"\nAction: assign {queued[0]} -> {schedulable[0]}")
        print(f"Verdict: {result.verdict.value}")
        if result.check_results:
            for cr in result.check_results:
                status = "PASS" if cr.passed else "FAIL"
                print(f"  {cr.checker_name}: {status} — {cr.summary}")

    # Observer output
    observer = config.create_observer(mgr)
    obs = observer.observe()
    print(f"\nSystem stable: {obs.is_stable}")
    print(f"Violations: {len(obs.violations)}")
    print(f"Compressed (first 200 chars): {obs.compressed_json[:200]}...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run example**

Run: `cd /mnt/d/SciTokyo/SILR-Agent && python examples/cluster_scheduling.py`
Expected: Prints scenario info, verification result, observer output

- [ ] **Step 3: Run full test suite (regression check)**

Run: `python -m pytest tests/ -v`
Expected: All existing tests still PASS, all new cluster tests PASS

- [ ] **Step 4: Commit**

```bash
git add examples/cluster_scheduling.py
git commit -m "feat: add cluster scheduling example script"
```

---

### Task 9: CI + Packaging Update

**Files:**
- Modify: `pyproject.toml` (add cluster to optional deps if needed)
- Modify: `.github/workflows/ci.yml` (ensure cluster tests run)

- [ ] **Step 1: Check if CI needs cluster tests added**

Read `.github/workflows/ci.yml` — if it runs `pytest tests/` globally, no change needed. If it lists specific test files, add the new ones.

- [ ] **Step 2: Update pyproject.toml if needed**

The cluster domain is zero-dependency (like network), so no new dependencies. But the `[training]` extra should note GRPO if TRL version matters.

- [ ] **Step 3: Run CI locally**

Run: `python -m pytest tests/ -v --tb=short`

- [ ] **Step 4: Commit if changes were needed**

```bash
git add pyproject.toml .github/workflows/ci.yml
git commit -m "infra: ensure cluster domain tests run in CI"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] ClusterManager with 15 nodes / 3 racks / 3 types → Task 1
- [x] 5 constraint checkers → Task 2
- [x] 6 agent tools → Task 3
- [x] 6 scenario types + parameterization → Task 4
- [x] Observer, failsafe, prompts, DomainConfig → Task 5
- [x] GRPO trainer (advantage computation + config) → Task 6
- [x] Cluster reward margins → Task 7
- [x] Example script → Task 8
- [x] CI update → Task 9

**Placeholder scan:** No TBD/TODO found. All code blocks are complete.

**Type consistency:**
- `ClusterManager` used consistently across all tasks
- `ClusterScenario` / `ClusterScenarioLoader` consistent
- `StepSample` / `compute_advantages` / `GRPOConfig` consistent in Task 6
- `build_cluster_domain_config` returns `DomainConfig` consistently
- Checker names match between `checkers.py` and `observation.py`
- Tool names match between `tools.py`, `config.py`, and `tool_schemas.py`

**Note:** The GRPO training *script* (model loading, tokenization, actual gradient computation) is not in this plan — it requires PyTorch/TRL which cannot run on local WSL. That script will be written for the Intel server in Week 3 once the domain framework is tested and SFT data is collected.
