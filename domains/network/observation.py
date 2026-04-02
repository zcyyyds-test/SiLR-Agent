"""Network domain observer for coordinator multi-agent support."""

from __future__ import annotations

import json

from silr.agent.observation import BaseObserver
from silr.agent.types import Observation
from .checkers import LinkUtilizationChecker, ConnectivityChecker


class NetworkObserver(BaseObserver):
    """Observer for the toy network domain.

    Queries system state and all checkers to produce a compressed
    observation suitable for LLM consumption.
    """

    def __init__(self, manager):
        self._manager = manager
        self._checkers = [LinkUtilizationChecker(), ConnectivityChecker()]

    def observe(self) -> Observation:
        state = self._manager.system_state

        # Run checkers to detect violations
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

        # Build compressed JSON for LLM
        links_summary = []
        for (src, dst), data in state["links"].items():
            if not data["up"]:
                links_summary.append({"link": f"{src}-{dst}", "status": "DOWN"})
            else:
                util = data["traffic"] / data["capacity"] * 100 if data["capacity"] > 0 else 0
                if util > 70:
                    links_summary.append({
                        "link": f"{src}-{dst}",
                        "utilization_pct": round(util, 1),
                    })

        compressed = {
            "down_links": [l for l in links_summary if l.get("status") == "DOWN"],
            "stressed_links": [l for l in links_summary if "utilization_pct" in l],
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
