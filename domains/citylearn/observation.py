"""CityLearn district-storage domain observer.

Runs the SoC / district-import / district-export checkers and emits a compact
JSON summary highlighting out-of-bound batteries and over-limit feeder flows,
plus the discrete set-point catalog so the LLM can infer the feasible action
range without trial-and-error. ``is_stable`` is True iff all three checkers
pass.
"""

from __future__ import annotations

import json

from silr.agent.observation import BaseObserver
from silr.agent.types import Observation

from . import simulator as sim
from .checkers import (
    SoCChecker,
    DistrictImportChecker,
    DistrictExportChecker,
)


class CityLearnObserver(BaseObserver):
    """Observer for the district-storage domain.

    All three constraint families are actionable (fixable by adjusting
    per-building battery set-points), so every checker contributes to
    ``is_stable``.
    """

    def __init__(self, manager):
        self._manager = manager
        self._checkers = [
            SoCChecker(),
            DistrictImportChecker(),
            DistrictExportChecker(),
        ]

    def observe(self) -> Observation:
        state = self._manager.system_state

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

        # Per-building summary with current set-point + SoC headroom.
        buildings_summary = []
        for b in state["buildings"]:
            buildings_summary.append({
                "id": b["id"],
                "index": b["index"],
                "soc_kwh": round(b["soc_kwh"], 3),
                "soc_min_kwh": b["soc_min_kwh"],
                "soc_max_kwh": b["soc_max_kwh"],
                "action_kw": b["action_kw"],
            })

        compressed = {
            "t": state["t"],
            "price": state["price"],
            "buildings": buildings_summary,
            "district_import_kw": round(state["district_import_kw"], 3),
            "district_import_limit_kw": state["district_import_limit_kw"],
            "district_export_kw": round(state["district_export_kw"], 3),
            "district_export_limit_kw": state["district_export_limit_kw"],
            "peak_import_kw": round(state["peak_import_kw"], 3),
            "cost": round(state["cost"], 3),
            "action_choices_kw": list(sim.ACTIONS_PER_BUILDING),
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
