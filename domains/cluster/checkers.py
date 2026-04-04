"""GPU cluster constraint checkers for SiLR verification.

Five constraints:
1. Resource capacity -- no node exceeds GPU/CPU/RAM limits
2. Rack affinity -- jobs with rack_affinity are on correct rack
3. Rack spread -- fault-tolerant groups span 2+ racks
4. Priority -- no urgent job queued while preemptible running
5. Queue -- all jobs scheduled (recovery target)
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
        violations: list[Violation] = []
        max_gpu_util = 0.0

        for nid, node in nodes.items():
            if node["status"] != "Ready":
                continue

            gpu_total = node["gpu_total"]
            gpu_util = node["gpu_used"] / gpu_total if gpu_total > 0 else 0.0
            max_gpu_util = max(max_gpu_util, gpu_util)

            if node["gpu_used"] > gpu_total:
                violations.append(Violation(
                    constraint_type="resource_capacity",
                    device_type="node",
                    device_id=nid,
                    metric="gpu_used",
                    value=float(node["gpu_used"]),
                    limit=float(gpu_total),
                    unit="GPUs",
                    severity="critical",
                    detail=f"{nid}: {node['gpu_used']}/{gpu_total} GPUs",
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
                    detail=(
                        f"{nid}: {node['ram_used_gb']}/{node['ram_total_gb']} GB RAM"
                    ),
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
        violations: list[Violation] = []

        for jid, job in jobs.items():
            if job["status"] != "Running":
                continue
            if not job.get("rack_affinity"):
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
                    detail=(
                        f"{jid} wants {job['rack_affinity']} "
                        f"but on {nid} ({actual_rack})"
                    ),
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
    """Urgent job groups with 2+ jobs should span 2+ racks."""

    name = "rack_spread"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        jobs = system_state["jobs"]
        nodes = system_state["nodes"]
        assignments = system_state["assignments"]
        violations: list[Violation] = []

        # Group urgent running jobs by group name
        group_racks: dict[str, set[str]] = {}
        group_counts: dict[str, int] = {}

        for jid, job in jobs.items():
            if job["priority"] != "urgent" or job["status"] != "Running":
                continue
            group = job.get("group")
            if group is None:
                continue
            group_counts[group] = group_counts.get(group, 0) + 1
            if jid in assignments:
                rack = nodes[assignments[jid]]["rack"]
                group_racks.setdefault(group, set()).add(rack)

        for group, count in group_counts.items():
            if count < 2:
                continue
            racks_used = len(group_racks.get(group, set()))
            if racks_used < 2:
                violations.append(Violation(
                    constraint_type="rack_spread",
                    device_type="group",
                    device_id=group,
                    metric="racks_covered",
                    value=float(racks_used),
                    limit=2.0,
                    unit="racks",
                    severity="warning",
                    detail=(
                        f"Group '{group}' has {count} urgent jobs "
                        f"in only {racks_used} rack(s)"
                    ),
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
        violations: list[Violation] = []

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
                        f"Urgent job {jid} queued while "
                        f"{len(preemptible_running)} preemptible job(s) running"
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
    """Urgent and normal jobs must be scheduled. Preemptible may stay queued.

    Recovery target: no urgent or normal jobs in the queue.
    Preemptible jobs are by definition non-critical — leaving them
    queued when capacity is exhausted is the correct triage decision.
    """

    name = "queue"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        jobs = system_state["jobs"]
        violations: list[Violation] = []

        queued = [jid for jid, j in jobs.items() if j["status"] == "Queued"]
        critical_queued = [
            jid for jid in queued
            if jobs[jid]["priority"] in ("urgent", "normal")
        ]
        preemptible_queued = [
            jid for jid in queued
            if jobs[jid]["priority"] == "preemptible"
        ]
        total = len(jobs)

        for jid in critical_queued:
            job = jobs[jid]
            gpu = job.get("gpu", job.get("gpu_req", "?"))
            group = job.get("group", "unknown")
            severity = "critical" if job["priority"] == "urgent" else "violation"
            violations.append(Violation(
                constraint_type="queue",
                device_type="job",
                device_id=jid,
                metric="queued",
                value=1.0,
                limit=0.0,
                unit="bool",
                severity=severity,
                detail=f"{jid} ({group}, {job['priority']}) needs {gpu} GPUs",
            ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "queued_count": len(queued),
                "critical_queued": len(critical_queued),
                "preemptible_queued": len(preemptible_queued),
                "total_jobs": total,
                "queue_ratio": round(len(critical_queued) / total, 3) if total else 0,
                "n_violations": len(violations),
            },
            violations=violations,
        )
