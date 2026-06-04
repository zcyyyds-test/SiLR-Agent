"""District-storage constraint checkers for SiLR verification.

Three constraint families, each emitting Violations with numeric value/limit so
the verifier's L3 magnitude guard (progress_mag) and severity score work:
1. Battery SoC bounds (per building): soc_min <= soc <= soc_max
2. District import limit: aggregate feeder import <= import_limit
3. District export limit: aggregate feeder back-feed <= export_limit

Ported from the ANM storage-SoC checker pattern (per-component bound with
distance-to-limit), so progress_mag transfers directly across domains.
"""

from __future__ import annotations

from typing import Any

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation


class SoCChecker(BaseConstraintChecker):
    """Each battery's state-of-charge must stay within [soc_min, soc_max]."""

    name = "battery_soc"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        violations = []
        worst_margin = 0.0
        for b in system_state["buildings"]:
            soc = b["soc_kwh"]
            smin = b["soc_min_kwh"]
            smax = b["soc_max_kwh"]
            if soc < smin:
                margin = smin - soc
                worst_margin = max(worst_margin, margin)
                violations.append(Violation(
                    constraint_type="soc_min",
                    device_type="battery",
                    device_id=b["id"],
                    metric="soc_kwh",
                    value=round(soc, 4),
                    limit=round(smin, 4),
                    unit="kWh",
                    severity="critical" if margin > 0.2 * (smax - smin) else "violation",
                    detail=f"{b['id']}: SoC {soc:.2f} kWh below min {smin:.2f} kWh",
                ))
            elif soc > smax:
                margin = soc - smax
                worst_margin = max(worst_margin, margin)
                violations.append(Violation(
                    constraint_type="soc_max",
                    device_type="battery",
                    device_id=b["id"],
                    metric="soc_kwh",
                    value=round(soc, 4),
                    limit=round(smax, 4),
                    unit="kWh",
                    severity="critical" if margin > 0.2 * (smax - smin) else "violation",
                    detail=f"{b['id']}: SoC {soc:.2f} kWh above max {smax:.2f} kWh",
                ))
        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={"worst_soc_margin_kwh": round(worst_margin, 4), "n_violations": len(violations)},
            violations=violations,
        )


class DistrictImportChecker(BaseConstraintChecker):
    """Aggregate district feeder import must not exceed the import limit."""

    name = "district_import"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        imp = system_state["district_import_kw"]
        limit = system_state["district_import_limit_kw"]
        violations = []
        if imp > limit:
            violations.append(Violation(
                constraint_type="import_limit",
                device_type="feeder",
                device_id="district",
                metric="import_kw",
                value=round(imp, 4),
                limit=round(limit, 4),
                unit="kW",
                severity="critical" if imp > 1.2 * limit else "violation",
                detail=f"District import {imp:.2f} kW exceeds limit {limit:.2f} kW",
            ))
        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={"district_import_kw": round(imp, 4), "limit_kw": limit, "n_violations": len(violations)},
            violations=violations,
        )


class DistrictExportChecker(BaseConstraintChecker):
    """Aggregate district back-feed (export) must not exceed the export limit."""

    name = "district_export"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        exp = system_state["district_export_kw"]
        limit = system_state["district_export_limit_kw"]
        violations = []
        if exp > limit:
            violations.append(Violation(
                constraint_type="export_limit",
                device_type="feeder",
                device_id="district",
                metric="export_kw",
                value=round(exp, 4),
                limit=round(limit, 4),
                unit="kW",
                severity="critical" if exp > 1.2 * limit else "violation",
                detail=f"District export {exp:.2f} kW exceeds limit {limit:.2f} kW",
            ))
        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={"district_export_kw": round(exp, 4), "limit_kw": limit, "n_violations": len(violations)},
            violations=violations,
        )
