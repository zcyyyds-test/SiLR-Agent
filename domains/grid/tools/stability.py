"""Stability check tool: 4-category constraint assessment."""

import numpy as np

from silr.tools.base import BaseTool
from .observation import compute_line_flows
from ..simulator import SystemState
from ..utils.unit_converter import pu_to_mw, rad_to_deg
from ..config import (
    VOLTAGE_MIN_PU, VOLTAGE_MAX_PU,
    FREQ_DEV_MAX_HZ,
    LINE_LOADING_NORMAL_PCT, LINE_LOADING_EMERGENCY_PCT,
    ROTOR_ANGLE_MAX_DEG,
    SYSTEM_FREQ_HZ,
)


class CheckStabilityTool(BaseTool):
    name = "check_stability"
    description = "Check system stability: voltage, frequency, line loading, transient."

    def _validate_params(self, **kw):
        pass

    def _run(self, **kw):
        self.manager._require_min_state(SystemState.PFLOW_DONE)
        ss = self.manager.ss
        base = self.manager.base_mva
        checks = {}

        # 1. Voltage check
        v = np.asarray(ss.Bus.v.v, dtype=float)
        bus_idx = list(ss.Bus.idx.v)
        v_violations = []
        for i, bid in enumerate(bus_idx):
            if v[i] < VOLTAGE_MIN_PU or v[i] > VOLTAGE_MAX_PU:
                v_violations.append({
                    "bus_id": bid,
                    "v_pu": float(v[i]),
                    "limit_low": VOLTAGE_MIN_PU,
                    "limit_high": VOLTAGE_MAX_PU,
                })
        checks["voltage"] = {
            "ok": len(v_violations) == 0,
            "min_pu": float(np.min(v)),
            "max_pu": float(np.max(v)),
            "violations": v_violations,
        }

        # 2. Frequency check
        f_violations = []
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
                df = float((w - 1.0) * SYSTEM_FREQ_HZ)
                all_delta_f.append(df)
                if abs(df) > FREQ_DEV_MAX_HZ:
                    f_violations.append({
                        "gen_id": idx_list[i],
                        "delta_f_hz": df,
                        "limit": FREQ_DEV_MAX_HZ,
                    })
        checks["frequency"] = {
            "ok": len(f_violations) == 0,
            "max_abs_delta_f_hz": (
                float(max(abs(d) for d in all_delta_f)) if all_delta_f else 0.0
            ),
            "violations": f_violations,
        }

        # 3. Line loading check
        flows = compute_line_flows(ss.Line, base)
        line_idx_list = list(ss.Line.idx.v)
        overloaded = []
        for i in range(ss.Line.n):
            loading = flows["loading_pct"][i]
            if not np.isnan(loading) and loading > LINE_LOADING_NORMAL_PCT:
                overloaded.append({
                    "line_id": line_idx_list[i],
                    "loading_pct": float(loading),
                    "rate_a_mva": float(pu_to_mw(flows["rate_a"][i], base)),
                    "s_mva": float(pu_to_mw(flows["S_from_mag"][i], base)),
                    "emergency": loading > LINE_LOADING_EMERGENCY_PCT,
                })
        checks["line_loading"] = {
            "ok": len(overloaded) == 0,
            "overloaded": overloaded,
        }

        # 4. Transient stability (rotor angle separation)
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

        all_delta = np.array(all_delta)
        all_M = np.array(all_M)
        total_M = np.sum(all_M)
        coi = np.sum(all_M * all_delta) / total_M if total_M > 0 else 0.0

        angle_violations = []
        max_sep_deg = 0.0
        if len(all_delta) > 1:
            angles_coi = all_delta - coi
            max_sep_deg = float(rad_to_deg(
                np.max(angles_coi) - np.min(angles_coi)
            ))
            if max_sep_deg > ROTOR_ANGLE_MAX_DEG:
                angles_coi_deg = rad_to_deg(angles_coi)
                i_max = int(np.argmax(angles_coi_deg))
                i_min = int(np.argmin(angles_coi_deg))
                angle_violations.append({
                    "gen_leading": gen_info[i_max],
                    "gen_lagging": gen_info[i_min],
                    "separation_deg": max_sep_deg,
                    "limit_deg": ROTOR_ANGLE_MAX_DEG,
                })

        checks["transient"] = {
            "ok": len(angle_violations) == 0,
            "max_separation_deg": max_sep_deg,
            "coi_angle_deg": float(rad_to_deg(coi)),
            "violations": angle_violations,
        }

        # Overall
        stable = all(c["ok"] for c in checks.values())

        return {
            "stable": stable,
            "checks": checks,
            "total_violations": (
                len(v_violations) + len(f_violations)
                + len(overloaded) + len(angle_violations)
            ),
        }
