"""Transmission line loading constraint checker."""

import numpy as np

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation
from ..tools.observation import compute_line_flows
from ..utils.unit_converter import pu_to_mw
from ..config import LINE_LOADING_NORMAL_PCT, LINE_LOADING_EMERGENCY_PCT


class LineLoadingChecker(BaseConstraintChecker):
    """Check line loading percentages. rate_a=0 lines are skipped (NaN)."""

    name = "line_loading"

    def check(self, ss, base_mva: float) -> CheckResult:
        flows = compute_line_flows(ss.Line, base_mva)
        line_idx_list = list(ss.Line.idx.v)

        violations = []
        max_loading = 0.0
        n_rated = 0  # lines with valid rate_a

        for i in range(ss.Line.n):
            loading = flows["loading_pct"][i]
            if np.isnan(loading):
                continue

            n_rated += 1
            loading_val = float(loading)
            if loading_val > max_loading:
                max_loading = loading_val

            if loading_val > LINE_LOADING_NORMAL_PCT:
                if loading_val > LINE_LOADING_EMERGENCY_PCT:
                    severity = "critical"
                else:
                    severity = "warning"
                violations.append(Violation(
                    constraint_type="line_loading",
                    device_type="line",
                    device_id=line_idx_list[i],
                    metric="loading_pct",
                    value=loading_val,
                    limit=LINE_LOADING_NORMAL_PCT,
                    unit="%",
                    severity=severity,
                    detail=(
                        f"Line {line_idx_list[i]}: loading = {loading_val:.1f}% "
                        f"> {LINE_LOADING_NORMAL_PCT}% "
                        f"(S = {float(pu_to_mw(flows['S_from_mag'][i], base_mva)):.1f} MVA)"
                    ),
                ))

        summary = {
            "max_loading_pct": max_loading if n_rated > 0 else None,
            "n_rated_lines": n_rated,
            "n_total_lines": ss.Line.n,
            "n_violations": len(violations),
        }

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary=summary,
            violations=violations,
        )
