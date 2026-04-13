"""Portfolio domain observer for coordinator multi-agent support."""

from __future__ import annotations

import json

from silr.agent.observation import BaseObserver
from silr.agent.types import Observation
from .checkers import (
    PositionConcentrationChecker,
    SectorExposureChecker,
    DrawdownChecker,
    CashReserveChecker,
)


class FinanceObserver(BaseObserver):
    """Observer for the portfolio compliance domain.

    Queries system state and all 4 checkers to produce a compressed
    observation suitable for LLM consumption.
    """

    def __init__(self, manager):
        self._manager = manager
        self._checkers = [
            PositionConcentrationChecker(),
            SectorExposureChecker(),
            DrawdownChecker(),
            CashReserveChecker(),
        ]

    def observe(self) -> Observation:
        state = self._manager.system_state

        # Run all checkers
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

        # Violated positions (weight > 25%)
        violated_positions = [
            {"symbol": sym, "weight_pct": round(pos["weight"] * 100, 1)}
            for sym, pos in state["positions"].items()
            if pos["weight"] * 100 > 25.0
        ]

        # Violated sectors (weight > 45%)
        violated_sectors = [
            {"sector": sec, "weight_pct": round(w * 100, 1)}
            for sec, w in state["sector_exposure"].items()
            if w * 100 > 45.0
        ]

        # Portfolio summary for LLM
        positions_summary = []
        for sym in sorted(state["positions"].keys()):
            pos = state["positions"][sym]
            if pos["qty"] > 0:
                positions_summary.append({
                    "symbol": sym,
                    "qty": pos["qty"],
                    "price": pos["price"],
                    "weight_pct": round(pos["weight"] * 100, 1),
                    "sector": pos["sector"],
                })

        compressed = {
            "positions": positions_summary,
            "cash": round(state["cash"], 2),
            "cash_pct": round(state["cash"] / state["portfolio_value"] * 100, 1)
                        if state["portfolio_value"] > 0 else 0,
            "portfolio_value": round(state["portfolio_value"], 2),
            "drawdown_pct": round(state["drawdown_pct"], 1),
            "sector_exposure": {
                s: round(w * 100, 1) for s, w in state["sector_exposure"].items()
            },
            "violated_positions": violated_positions,
            "violated_sectors": violated_sectors,
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
