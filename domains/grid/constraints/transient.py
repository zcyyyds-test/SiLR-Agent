"""Transient stability (rotor angle separation) constraint checker."""

import numpy as np

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation
from ..utils.unit_converter import rad_to_deg
from ..config import ROTOR_ANGLE_MAX_DEG


class TransientStabilityChecker(BaseConstraintChecker):
    """Check rotor angle separation (COI-referenced).

    If TDS time-series data is available, analyzes peak separation over
    the entire simulation window. Otherwise, falls back to final-state check.
    """

    name = "transient"

    def check(self, ss, base_mva: float) -> CheckResult:
        ts_info = self._analyze_time_series(ss)

        if ts_info["has_time_series"]:
            return self._check_from_time_series(ss, ts_info)
        else:
            return self._check_final_state(ss)

    def _analyze_time_series(self, ss) -> dict:
        """Extract peak rotor angle separation from TDS time-series."""
        if not hasattr(ss.dae, 'ts') or ss.dae.ts is None:
            return {"has_time_series": False}

        ts_t = getattr(ss.dae.ts, 't', None)
        ts_x = getattr(ss.dae.ts, 'x', None)

        if ts_t is None or ts_x is None:
            return {"has_time_series": False}

        ts_t_arr = np.array(ts_t)
        ts_x_arr = np.array(ts_x)

        if ts_t_arr.size < 2 or ts_x_arr.ndim < 2:
            return {"has_time_series": False}

        # Collect delta addresses and inertia from all generator models
        delta_addrs = []
        M_values = []
        gen_ids = []

        for model_name in ["GENROU", "GENCLS"]:
            mdl = getattr(ss, model_name, None)
            if mdl is None or mdl.n == 0:
                continue

            if not hasattr(mdl, 'delta') or not hasattr(mdl.delta, 'a'):
                continue

            addrs = mdl.delta.a
            if addrs is None or (hasattr(addrs, 'size') and addrs.size == 0):
                continue

            delta_addrs.extend(np.asarray(addrs).tolist())

            # Inertia: M = 2H
            if hasattr(mdl, 'M') and mdl.M.v is not None:
                M_values.extend(np.asarray(mdl.M.v, dtype=float).tolist())
            elif hasattr(mdl, 'H') and mdl.H.v is not None:
                M_values.extend((np.asarray(mdl.H.v, dtype=float) * 2).tolist())
            else:
                M_values.extend([2.0] * mdl.n)

            gen_ids.extend(list(mdl.idx.v))

        if len(delta_addrs) < 2:
            return {"has_time_series": False}

        # Validate addresses are within bounds
        max_addr = ts_x_arr.shape[1] - 1 if ts_x_arr.ndim == 2 else 0
        if any(a > max_addr for a in delta_addrs):
            return {"has_time_series": False}

        # Extract delta time-series: (n_steps, n_gens)
        delta_ts = ts_x_arr[:, delta_addrs]

        # COI-referenced separation at each timestep
        M_arr = np.array(M_values)
        M_total = M_arr.sum()
        if M_total <= 0:
            return {"has_time_series": False}

        coi_ts = (delta_ts @ M_arr) / M_total  # (n_steps,)
        delta_coi_ts = delta_ts - coi_ts[:, np.newaxis]
        separation_ts = np.max(delta_coi_ts, axis=1) - np.min(delta_coi_ts, axis=1)
        separation_deg_ts = np.degrees(separation_ts)

        peak_idx = int(np.argmax(separation_deg_ts))

        return {
            "peak_separation_deg": float(separation_deg_ts[peak_idx]),
            "peak_time": float(ts_t_arr[peak_idx]),
            "final_separation_deg": float(separation_deg_ts[-1]),
            "has_time_series": True,
            "gen_ids": gen_ids,
            "delta_coi_ts": delta_coi_ts,
            "separation_deg_ts": separation_deg_ts,
            "ts_t": ts_t_arr,
        }

    def _check_from_time_series(self, ss, ts_info: dict) -> CheckResult:
        """Check constraints using time-series peak values."""
        peak_sep = ts_info["peak_separation_deg"]
        final_sep = ts_info["final_separation_deg"]
        peak_time = ts_info["peak_time"]

        violations = []
        if peak_sep > ROTOR_ANGLE_MAX_DEG:
            violations.append(Violation(
                constraint_type="transient",
                device_type="generator",
                device_id="system",
                metric="separation_deg",
                value=peak_sep,
                limit=ROTOR_ANGLE_MAX_DEG,
                unit="deg",
                severity="critical",
                detail=(
                    f"Peak rotor angle separation {peak_sep:.1f} deg "
                    f"> {ROTOR_ANGLE_MAX_DEG} deg at t = {peak_time:.2f}s"
                ),
            ))

        summary = {
            "peak_separation_deg": peak_sep,
            "peak_time": peak_time,
            "final_separation_deg": final_sep,
            "limit_deg": ROTOR_ANGLE_MAX_DEG,
            "method": "time_series",
        }

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary=summary,
            violations=violations,
        )

    def _check_final_state(self, ss) -> CheckResult:
        """Fallback: check final-state rotor angle separation."""
        all_delta = []
        all_M = []
        gen_info = []

        for model_name in ["GENROU", "GENCLS"]:
            mdl = getattr(ss, model_name, None)
            if mdl is None or mdl.n == 0:
                continue

            delta_raw = np.asarray(mdl.delta.v, dtype=float)
            has_delta = delta_raw.size == mdl.n

            if hasattr(mdl, 'M') and mdl.M.v is not None:
                M = np.asarray(mdl.M.v, dtype=float)
            elif hasattr(mdl, 'H') and mdl.H.v is not None:
                M = np.asarray(mdl.H.v, dtype=float) * 2.0
            else:
                M = np.ones(mdl.n) * 2.0

            idx_list = list(mdl.idx.v)
            for i in range(mdl.n):
                all_delta.append(float(delta_raw[i]) if has_delta else 0.0)
                all_M.append(float(M[i]))
                gen_info.append(idx_list[i])

        violations = []
        max_sep_deg = 0.0

        if len(all_delta) > 1:
            all_delta_arr = np.array(all_delta)
            all_M_arr = np.array(all_M)
            total_M = np.sum(all_M_arr)
            coi = np.sum(all_M_arr * all_delta_arr) / total_M if total_M > 0 else 0.0

            angles_coi = all_delta_arr - coi
            max_sep_deg = float(rad_to_deg(np.max(angles_coi) - np.min(angles_coi)))

            if max_sep_deg > ROTOR_ANGLE_MAX_DEG:
                violations.append(Violation(
                    constraint_type="transient",
                    device_type="generator",
                    device_id="system",
                    metric="separation_deg",
                    value=max_sep_deg,
                    limit=ROTOR_ANGLE_MAX_DEG,
                    unit="deg",
                    severity="critical",
                    detail=(
                        f"Final rotor angle separation {max_sep_deg:.1f} deg "
                        f"> {ROTOR_ANGLE_MAX_DEG} deg"
                    ),
                ))

        summary = {
            "max_separation_deg": max_sep_deg,
            "limit_deg": ROTOR_ANGLE_MAX_DEG,
            "n_generators": len(all_delta),
            "method": "final_state",
        }

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary=summary,
            violations=violations,
        )
