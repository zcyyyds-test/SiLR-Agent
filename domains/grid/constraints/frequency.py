"""Generator frequency constraint checker."""

import numpy as np

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation
from ..utils.unit_converter import omega_to_delta_f
from ..config import FREQ_DEV_MAX_HZ


class FrequencyChecker(BaseConstraintChecker):
    """Check generator frequency deviations against FREQ_DEV_MAX_HZ."""

    name = "frequency"

    def check(self, ss, base_mva: float) -> CheckResult:
        violations = []
        all_delta_f = []

        for model_name in ["GENROU", "GENCLS"]:
            mdl = getattr(ss, model_name, None)
            if mdl is None or mdl.n == 0:
                continue

            omega_raw = np.asarray(mdl.omega.v, dtype=float)
            has_omega = omega_raw.size == mdl.n
            idx_list = list(mdl.idx.v)

            for i in range(mdl.n):
                w = float(omega_raw[i]) if has_omega else 1.0
                df = omega_to_delta_f(w)
                all_delta_f.append(df)

                if abs(df) > FREQ_DEV_MAX_HZ:
                    severity = "critical" if abs(df) > FREQ_DEV_MAX_HZ * 1.5 else "violation"
                    direction = "high" if df > 0 else "low"
                    violations.append(Violation(
                        constraint_type="frequency",
                        device_type="generator",
                        device_id=idx_list[i],
                        metric="delta_f_hz",
                        value=df,  # signed: positive=high, negative=low
                        limit=FREQ_DEV_MAX_HZ,
                        unit="Hz",
                        severity=severity,
                        detail=(
                            f"{model_name}_{idx_list[i]}: "
                            f"delta_f = {df:+.2f} Hz ({direction}), "
                            f"|delta_f| = {abs(df):.2f} Hz > limit {FREQ_DEV_MAX_HZ} Hz"
                        ),
                    ))

        max_abs = float(max(abs(d) for d in all_delta_f)) if all_delta_f else 0.0

        summary = {
            "max_abs_delta_f_hz": max_abs,
            "n_generators": len(all_delta_f),
            "n_violations": len(violations),
        }

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary=summary,
            violations=violations,
        )
