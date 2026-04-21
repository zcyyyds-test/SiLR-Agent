"""5 constraint checkers for cluster_v2023 domain.

Per-action gate (SiLRVerifier.checkers): ResourceCapacity + Affinity only.
Observer-only (episode-level signals): Priority / Queue / Fragmentation.

Rationale (per spec §5.1): Queue/Priority evaluated per-action would
reject every intermediate step (LS still queued while scheduling is in
progress → 100% fail). This mirrors cluster v1 and finance designs.
"""

from __future__ import annotations

from typing import Any

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation


class ResourceCapacityChecker(BaseConstraintChecker):
    """No Ready node may exceed cpu_milli / ram_mib / gpu_total."""

    name = "resource_capacity"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        nodes = system_state["nodes"]
        violations: list[Violation] = []
        max_gpu_util = 0.0

        for nid, node in nodes.items():
            if node["status"] != "Ready":
                continue
            g_used, g_total = node["gpu_used"], node["gpu_total"]
            if g_total > 0:
                max_gpu_util = max(max_gpu_util, g_used / g_total)
            if g_used > g_total:
                violations.append(Violation(
                    constraint_type="resource_capacity", device_type="node",
                    device_id=nid, metric="gpu_used",
                    value=float(g_used), limit=float(g_total), unit="GPUs",
                    severity="critical",
                    detail=f"{nid}: {g_used}/{g_total} GPUs",
                ))
            if node["cpu_used"] > node["cpu_total"]:
                violations.append(Violation(
                    constraint_type="resource_capacity", device_type="node",
                    device_id=nid, metric="cpu_milli",
                    value=float(node["cpu_used"]), limit=float(node["cpu_total"]),
                    unit="cpu_milli", severity="violation",
                    detail=f"{nid}: cpu {node['cpu_used']}/{node['cpu_total']}",
                ))
            if node["ram_used_mib"] > node["ram_total_mib"]:
                violations.append(Violation(
                    constraint_type="resource_capacity", device_type="node",
                    device_id=nid, metric="ram_mib",
                    value=float(node["ram_used_mib"]),
                    limit=float(node["ram_total_mib"]),
                    unit="MiB", severity="violation",
                    detail=f"{nid}: ram {node['ram_used_mib']}/{node['ram_total_mib']}",
                ))

        return CheckResult(
            checker_name=self.name,
            passed=not violations,
            summary={
                "max_gpu_util": round(max_gpu_util, 3),
                "n_violations": len(violations),
            },
            violations=violations,
        )


class AffinityChecker(BaseConstraintChecker):
    """Running jobs with gpu_spec_required must be placed on matching node.model."""

    name = "affinity"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        jobs = system_state["jobs"]
        nodes = system_state["nodes"]
        assignments = system_state["assignments"]
        violations: list[Violation] = []

        for jid, job in jobs.items():
            if job["status"] != "Running":
                continue
            req = job.get("gpu_spec_required")
            if not req:
                continue
            nid = assignments.get(jid)
            if nid is None:
                continue
            node_model = nodes[nid]["model"]
            if node_model != req:
                violations.append(Violation(
                    constraint_type="affinity", device_type="job",
                    device_id=jid, metric="gpu_spec_match",
                    value=0.0, limit=1.0, unit="bool", severity="warning",
                    detail=f"{jid} needs {req} but on {nid} ({node_model})",
                ))

        return CheckResult(
            checker_name=self.name, passed=not violations,
            summary={"n_violations": len(violations)},
            violations=violations,
        )


class PriorityChecker(BaseConstraintChecker):
    """No LS job queued while BE job running (higher-priority preemption).

    Per spec §5.1: OBSERVER-ONLY (not in SiLRVerifier.checkers). Used
    by observer for reward shaping and recovery-target signalling.
    """

    name = "priority"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        jobs = system_state["jobs"]
        ls_queued = [j for j, v in jobs.items()
                     if v["qos"] == "LS" and v["status"] == "Queued"]
        be_running = [j for j, v in jobs.items()
                      if v["qos"] == "BE" and v["status"] == "Running"]
        violations: list[Violation] = []
        if ls_queued and be_running:
            for jid in ls_queued:
                violations.append(Violation(
                    constraint_type="priority", device_type="job",
                    device_id=jid, metric="ls_queued_while_be_running",
                    value=1.0, limit=0.0, unit="bool", severity="critical",
                    detail=(f"LS {jid} queued while {len(be_running)} "
                            f"BE running"),
                ))
        return CheckResult(
            checker_name=self.name, passed=not violations,
            summary={"ls_queued": len(ls_queued),
                     "be_running": len(be_running),
                     "n_violations": len(violations)},
            violations=violations,
        )
