"""Portfolio compliance constraint checkers for SiLR verification.

Four constraints:
1. Position concentration: no single stock > 25% of portfolio
2. Sector exposure: no single sector > 45% of portfolio
3. Drawdown: portfolio drawdown from peak must stay within 10%
4. Cash reserve: cash must be at least 5% of portfolio
"""

from __future__ import annotations

from typing import Any

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation


class PositionConcentrationChecker(BaseConstraintChecker):
    """Check that no single stock exceeds a weight threshold."""

    name = "position_concentration"

    def __init__(self, max_weight_pct: float = 25.0):
        self._max_weight = max_weight_pct

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        positions = system_state["positions"]
        violations = []
        max_observed = 0.0

        for symbol, pos in positions.items():
            weight_pct = pos["weight"] * 100
            max_observed = max(max_observed, weight_pct)
            if weight_pct > self._max_weight:
                violations.append(Violation(
                    constraint_type="position_concentration",
                    device_type="position",
                    device_id=symbol,
                    metric="weight_pct",
                    value=round(weight_pct, 2),
                    limit=self._max_weight,
                    unit="%",
                    severity="critical" if weight_pct > self._max_weight * 1.2 else "violation",
                    detail=f"{symbol}: {weight_pct:.1f}% weight (limit: {self._max_weight}%)",
                ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "max_position_weight_pct": round(max_observed, 2),
                "n_violations": len(violations),
            },
            violations=violations,
        )


class SectorExposureChecker(BaseConstraintChecker):
    """Check that no single sector exceeds an exposure threshold."""

    name = "sector_exposure"

    def __init__(self, max_sector_weight_pct: float = 45.0):
        self._max_sector = max_sector_weight_pct

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        sector_exposure = system_state["sector_exposure"]
        violations = []
        max_observed = 0.0

        for sector, weight in sector_exposure.items():
            weight_pct = weight * 100
            max_observed = max(max_observed, weight_pct)
            if weight_pct > self._max_sector:
                violations.append(Violation(
                    constraint_type="sector_exposure",
                    device_type="sector",
                    device_id=sector,
                    metric="sector_weight_pct",
                    value=round(weight_pct, 2),
                    limit=self._max_sector,
                    unit="%",
                    severity="critical" if weight_pct > self._max_sector * 1.2 else "violation",
                    detail=f"Sector {sector}: {weight_pct:.1f}% exposure (limit: {self._max_sector}%)",
                ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "max_sector_weight_pct": round(max_observed, 2),
                "sector_weights": {s: round(w * 100, 2) for s, w in sector_exposure.items()},
                "n_violations": len(violations),
            },
            violations=violations,
        )


class DrawdownChecker(BaseConstraintChecker):
    """Check that portfolio drawdown from peak is within tolerance."""

    name = "drawdown"

    def __init__(self, max_drawdown_pct: float = 10.0):
        self._max_dd = max_drawdown_pct

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        dd_pct = system_state["drawdown_pct"]
        violations = []

        if dd_pct > self._max_dd:
            violations.append(Violation(
                constraint_type="drawdown",
                device_type="portfolio",
                device_id="portfolio",
                metric="drawdown_pct",
                value=round(dd_pct, 2),
                limit=self._max_dd,
                unit="%",
                severity="critical" if dd_pct > self._max_dd * 1.5 else "violation",
                detail=f"Portfolio drawdown {dd_pct:.1f}% exceeds {self._max_dd}% limit",
            ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "drawdown_pct": round(dd_pct, 2),
                "peak_value": round(system_state["peak_value"], 2),
                "current_value": round(system_state["portfolio_value"], 2),
                "n_violations": len(violations),
            },
            violations=violations,
        )


class CashReserveChecker(BaseConstraintChecker):
    """Check that cash is at least a minimum fraction of portfolio value."""

    name = "cash_reserve"

    def __init__(self, min_cash_ratio_pct: float = 5.0):
        self._min_ratio = min_cash_ratio_pct

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        cash = system_state["cash"]
        pv = system_state["portfolio_value"]
        cash_ratio_pct = (cash / pv * 100) if pv > 0 else 0.0
        violations = []

        if cash_ratio_pct < self._min_ratio:
            violations.append(Violation(
                constraint_type="cash_reserve",
                device_type="account",
                device_id="cash",
                metric="cash_ratio_pct",
                value=round(cash_ratio_pct, 2),
                limit=self._min_ratio,
                unit="%",
                severity="critical" if cash_ratio_pct < self._min_ratio * 0.5 else "violation",
                detail=f"Cash {cash_ratio_pct:.1f}% of portfolio (minimum: {self._min_ratio}%)",
            ))

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "cash": round(cash, 2),
                "cash_ratio_pct": round(cash_ratio_pct, 2),
                "n_violations": len(violations),
            },
            violations=violations,
        )
