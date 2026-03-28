"""Network domain constraint checkers for SiLR verification.

Two constraints:
1. Link utilization must stay below 90% (overload threshold)
2. All demands must be routable (connectivity check)
"""

from __future__ import annotations

from typing import Any

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation


class LinkUtilizationChecker(BaseConstraintChecker):
    """Check that no link exceeds utilization threshold."""

    name = "link_utilization"

    def __init__(self, max_utilization_pct: float = 90.0):
        self._max_util = max_utilization_pct

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        links = system_state["links"]
        violations = []

        for (src, dst), data in links.items():
            if not data["up"]:
                continue
            if data["capacity"] <= 0:
                continue
            util = data["traffic"] / data["capacity"] * 100
            if util > self._max_util:
                severity = "critical" if util > 100 else "violation"
                violations.append(Violation(
                    constraint_type="link_utilization",
                    device_type="link",
                    device_id=f"{src}-{dst}",
                    metric="utilization_pct",
                    value=round(util, 1),
                    limit=self._max_util,
                    unit="%",
                    severity=severity,
                    detail=f"Link {src}-{dst}: {util:.1f}% utilization (limit: {self._max_util}%)",
                ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "total_up_links": sum(1 for d in links.values() if d["up"]),
                "overloaded": len(violations),
                "max_utilization_pct": max(
                    (d["traffic"] / d["capacity"] * 100
                     for d in links.values()
                     if d["up"] and d["capacity"] > 0),
                    default=0,
                ),
                "n_violations": len(violations),
            },
            violations=violations,
        )


class ConnectivityChecker(BaseConstraintChecker):
    """Check that all demand pairs remain reachable."""

    name = "connectivity"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        links = system_state["links"]
        demands = system_state["demands"]
        nodes = system_state["nodes"]
        violations = []

        # Build adjacency from up links
        adj: dict[int, set[int]] = {n: set() for n in nodes}
        for (src, dst), data in links.items():
            if data["up"]:
                adj[src].add(dst)
                adj[dst].add(src)

        # BFS reachability for each demand
        unreachable = 0
        for (src, dst), demand in demands.items():
            if not self._is_reachable(src, dst, adj):
                unreachable += 1
                violations.append(Violation(
                    constraint_type="connectivity",
                    device_type="demand",
                    device_id=f"{src}->{dst}",
                    metric="reachable",
                    value=0.0,
                    limit=1.0,
                    unit="bool",
                    severity="critical",
                    detail=f"Demand {src}->{dst} ({demand} Mbps) is unreachable",
                ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "total_demands": len(demands),
                "unreachable": unreachable,
                "n_violations": len(violations),
            },
            violations=violations,
        )

    @staticmethod
    def _is_reachable(src: int, dst: int, adj: dict[int, set[int]]) -> bool:
        """BFS reachability check."""
        visited = {src}
        queue = [src]
        while queue:
            node = queue.pop(0)
            if node == dst:
                return True
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False
