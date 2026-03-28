"""Bus voltage constraint checker."""

import numpy as np

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation
from ..config import VOLTAGE_MIN_PU, VOLTAGE_MAX_PU


class VoltageChecker(BaseConstraintChecker):
    """Check bus voltage magnitudes against [VOLTAGE_MIN_PU, VOLTAGE_MAX_PU]."""

    name = "voltage"

    def check(self, ss, base_mva: float) -> CheckResult:
        v = np.asarray(ss.Bus.v.v, dtype=float)
        bus_idx = list(ss.Bus.idx.v)

        violations = []

        # Detect NaN/Inf voltages before normal range checks
        bad_mask = ~np.isfinite(v)
        if np.any(bad_mask):
            n_bad = int(np.sum(bad_mask))
            violations.append(Violation(
                constraint_type="voltage",
                device_type="bus",
                device_id="system",
                metric="v_pu",
                value=float('nan'),
                limit=VOLTAGE_MIN_PU,
                unit="p.u.",
                severity="critical",
                detail=f"{n_bad} buses have NaN/Inf voltage",
            ))

        for i, bid in enumerate(bus_idx):
            vi = float(v[i])
            if not np.isfinite(vi):
                continue  # already reported above
            if vi < VOLTAGE_MIN_PU:
                severity = "critical" if vi < VOLTAGE_MIN_PU - 0.05 else "violation"
                violations.append(Violation(
                    constraint_type="voltage",
                    device_type="bus",
                    device_id=bid,
                    metric="v_pu",
                    value=vi,
                    limit=VOLTAGE_MIN_PU,
                    unit="p.u.",
                    severity=severity,
                    detail=f"Bus {bid}: V = {vi:.4f} p.u. < {VOLTAGE_MIN_PU} p.u.",
                ))
            elif vi > VOLTAGE_MAX_PU:
                severity = "critical" if vi > VOLTAGE_MAX_PU + 0.05 else "violation"
                violations.append(Violation(
                    constraint_type="voltage",
                    device_type="bus",
                    device_id=bid,
                    metric="v_pu",
                    value=vi,
                    limit=VOLTAGE_MAX_PU,
                    unit="p.u.",
                    severity=severity,
                    detail=f"Bus {bid}: V = {vi:.4f} p.u. > {VOLTAGE_MAX_PU} p.u.",
                ))

        summary = {
            "min_pu": float(np.nanmin(v)) if np.any(np.isfinite(v)) else float('nan'),
            "max_pu": float(np.nanmax(v)) if np.any(np.isfinite(v)) else float('nan'),
            "n_violations": len(violations),
        }

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary=summary,
            violations=violations,
        )
