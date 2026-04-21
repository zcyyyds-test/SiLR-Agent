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
