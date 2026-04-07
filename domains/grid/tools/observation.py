"""Observation tools: get_bus_voltage, get_frequency, get_rotor_angle, get_line_flow."""

import logging

import numpy as np

from silr.tools.base import BaseTool
from ..simulator import SystemState
from ..utils.unit_converter import pu_to_mw, rad_to_deg, omega_to_delta_f
from ..utils.validators import validate_device_idx
from ..config import (
    SYSTEM_FREQ_HZ,
    VOLTAGE_MIN_PU, VOLTAGE_MAX_PU,
    FREQ_DEV_MAX_HZ,
    ROTOR_ANGLE_MAX_DEG,
)
from silr.exceptions import ValidationError


class GetBusVoltageTool(BaseTool):
    name = "get_bus_voltage"
    description = "Get bus voltage magnitude (p.u.) and angle (deg)."

    def _validate_params(self, bus_id=None, **kw):
        if bus_id is not None and bus_id != "all":
            validate_device_idx(self.manager.ss.Bus, bus_id, "Bus")

    def _run(self, bus_id=None, **kw):
        self.manager._require_min_state(SystemState.PFLOW_DONE)
        ss = self.manager.ss

        v_mag = np.asarray(ss.Bus.v.v, dtype=float)
        v_ang = np.asarray(ss.Bus.a.v, dtype=float)
        idx_list = list(ss.Bus.idx.v)

        if bus_id is not None and bus_id != "all":
            indices = [idx_list.index(bus_id)]
        else:
            indices = range(len(idx_list))

        entries = []
        for i in indices:
            v = float(v_mag[i])
            violation = None
            if v < VOLTAGE_MIN_PU or v > VOLTAGE_MAX_PU:
                violation = {
                    "violated": True,
                    "limit_low": VOLTAGE_MIN_PU,
                    "limit_high": VOLTAGE_MAX_PU,
                    "value": v,
                    "unit": "p.u.",
                }
            entries.append({
                "bus_id": idx_list[i],
                "v_pu": v,
                "v_angle_deg": float(rad_to_deg(v_ang[i])),
                "violation": violation,
            })

        return {"buses": entries, "count": len(entries)}


class GetFrequencyTool(BaseTool):
    name = "get_frequency"
    description = "Get generator frequency from GENROU/GENCLS omega."

    def _validate_params(self, gen_id=None, **kw):
        if gen_id is not None and gen_id != "all":
            all_idx = self.manager.get_all_syn_gen_idx()
            if gen_id not in all_idx:
                raise ValidationError(
                    f"Generator '{gen_id}' not found. Available: {all_idx}"
                )

    def _run(self, gen_id=None, **kw):
        self.manager._require_min_state(SystemState.PFLOW_DONE)
        ss = self.manager.ss
        entries = []

        for model_name in ["GENROU", "GENCLS"]:
            mdl = getattr(ss, model_name, None)
            if mdl is None or mdl.n == 0:
                continue

            idx_list = list(mdl.idx.v)
            omega_raw = np.asarray(mdl.omega.v, dtype=float)
            has_omega = omega_raw.size == mdl.n

            for i, gid in enumerate(idx_list):
                if gen_id is not None and gen_id != "all" and gid != gen_id:
                    continue

                omega = float(omega_raw[i]) if has_omega else 1.0
                delta_f = omega_to_delta_f(omega)

                violation = None
                if abs(delta_f) > FREQ_DEV_MAX_HZ:
                    violation = {
                        "violated": True,
                        "limit": FREQ_DEV_MAX_HZ,
                        "value": float(abs(delta_f)),
                        "unit": "Hz",
                    }

                entries.append({
                    "gen_id": gid,
                    "model": model_name,
                    "freq_hz": float(SYSTEM_FREQ_HZ + delta_f),
                    "delta_f_hz": float(delta_f),
                    "omega_pu": omega,
                    "violation": violation,
                })

        return {"generators": entries, "count": len(entries)}


class GetRotorAngleTool(BaseTool):
    name = "get_rotor_angle"
    description = "Get rotor angle relative to COI (Center of Inertia)."

    def _validate_params(self, gen_id=None, **kw):
        if gen_id is not None and gen_id != "all":
            all_idx = self.manager.get_all_syn_gen_idx()
            if gen_id not in all_idx:
                raise ValidationError(
                    f"Generator '{gen_id}' not found. Available: {all_idx}"
                )

    def _run(self, gen_id=None, **kw):
        self.manager._require_min_state(SystemState.PFLOW_DONE)
        ss = self.manager.ss

        # Collect all generators' delta and inertia
        all_delta = []
        all_M = []
        all_idx = []
        all_models = []

        for model_name in ["GENROU", "GENCLS"]:
            mdl = getattr(ss, model_name, None)
            if mdl is None or mdl.n == 0:
                continue

            idx_list = list(mdl.idx.v)
            delta_raw = np.asarray(mdl.delta.v, dtype=float)
            has_delta = delta_raw.size == mdl.n

            if hasattr(mdl, 'M') and mdl.M.v is not None:
                M_vals = np.asarray(mdl.M.v, dtype=float)
            elif hasattr(mdl, 'H') and mdl.H.v is not None:
                M_vals = np.asarray(mdl.H.v, dtype=float) * 2.0
            else:
                M_vals = np.ones(mdl.n) * 2.0

            for i in range(mdl.n):
                delta_val = float(delta_raw[i]) if has_delta else 0.0
                all_delta.append(delta_val)
                all_M.append(float(M_vals[i]))
                all_idx.append(idx_list[i])
                all_models.append(model_name)

        all_delta_arr = np.array(all_delta)
        all_M_arr = np.array(all_M)
        total_M = np.sum(all_M_arr)
        coi_angle = (np.sum(all_M_arr * all_delta_arr) / total_M
                     if total_M > 0 else 0.0)

        entries = []
        for i in range(len(all_idx)):
            if gen_id is not None and gen_id != "all" and all_idx[i] != gen_id:
                continue

            angle_deg = float(rad_to_deg(all_delta_arr[i]))
            angle_coi_deg = float(rad_to_deg(all_delta_arr[i] - coi_angle))

            violation = None
            if abs(angle_coi_deg) > ROTOR_ANGLE_MAX_DEG:
                violation = {
                    "violated": True,
                    "limit": ROTOR_ANGLE_MAX_DEG,
                    "value": float(abs(angle_coi_deg)),
                    "unit": "deg",
                }

            entries.append({
                "gen_id": all_idx[i],
                "model": all_models[i],
                "angle_deg": angle_deg,
                "angle_coi_deg": angle_coi_deg,
                "violation": violation,
            })

        # Max separation among all generators
        if len(all_delta_arr) > 1:
            angles_coi = all_delta_arr - coi_angle
            max_sep = float(rad_to_deg(np.max(angles_coi) - np.min(angles_coi)))
        else:
            max_sep = 0.0

        return {
            "generators": entries,
            "count": len(entries),
            "coi_angle_deg": float(rad_to_deg(coi_angle)),
            "max_separation_deg": max_sep,
        }


# --- Line flow helper ---

def compute_line_flows(line_model, base_mva: float):
    """Compute line P/Q flows from bus voltages and line parameters.

    Returns dict of numpy arrays (all in p.u. except loading_pct).
    """
    line = line_model
    n = line.n

    v1 = np.asarray(line.v1.v, dtype=float)
    v2 = np.asarray(line.v2.v, dtype=float)
    a1 = np.asarray(line.a1.v, dtype=float)
    a2 = np.asarray(line.a2.v, dtype=float)

    r = np.asarray(line.r.v, dtype=float)
    x = np.asarray(line.x.v, dtype=float)
    b_total = np.asarray(line.b.v, dtype=float)

    def _get_or_zeros(attr_name):
        if hasattr(line, attr_name):
            val = getattr(line, attr_name).v
            if val is not None:
                return np.asarray(val, dtype=float)
        return np.zeros(n)

    g_total = _get_or_zeros('g')
    b1 = _get_or_zeros('b1')
    b2 = _get_or_zeros('b2')
    g1 = _get_or_zeros('g1')
    g2 = _get_or_zeros('g2')

    tap = np.asarray(line.tap.v, dtype=float)
    tap = np.where(tap == 0, 1.0, tap)

    # Phase shift — ANDES may use 'shift' or 'phi'
    phi = np.zeros(n)
    for attr in ('shift', 'phi'):
        if hasattr(line, attr):
            val = getattr(line, attr).v
            if val is not None:
                phi = np.asarray(val, dtype=float)
                break

    u = np.asarray(line.u.v, dtype=float)
    rate_a = _get_or_zeros('rate_a')

    # Series admittance
    z = r + 1j * x
    zero_z_mask = np.abs(z) <= 1e-12
    if np.any(zero_z_mask):
        _log = logging.getLogger(__name__)
        n_zero = int(np.sum(zero_z_mask))
        _log.warning(f"{n_zero} lines have zero impedance — flows will be zero")
    y_s = np.where(~zero_z_mask, 1.0 / z, 0.0 + 0j)

    # Shunt admittance at each end
    y_sh_from = (g_total / 2 + g1) + 1j * (b_total / 2 + b1)
    y_sh_to = (g_total / 2 + g2) + 1j * (b_total / 2 + b2)

    # Admittance matrix elements (pi-model with tap and phase shift)
    y_ff = (y_s + y_sh_from) / (tap ** 2)
    y_ft = -y_s / (tap * np.exp(-1j * phi))
    y_tf = -y_s / (tap * np.exp(1j * phi))
    y_tt = y_s + y_sh_to

    # Complex bus voltages at line terminals
    V1 = v1 * np.exp(1j * a1)
    V2 = v2 * np.exp(1j * a2)

    # Complex power (p.u.)
    I_from = y_ff * V1 + y_ft * V2
    I_to = y_tf * V1 + y_tt * V2
    S_from = V1 * np.conj(I_from) * u
    S_to = V2 * np.conj(I_to) * u

    # Loading percentage
    S_from_mag = np.abs(S_from)
    loading_pct = np.where(rate_a > 0, S_from_mag / rate_a * 100, np.nan)

    return {
        "P_from": np.real(S_from),
        "Q_from": np.imag(S_from),
        "P_to": np.real(S_to),
        "Q_to": np.imag(S_to),
        "S_from_mag": S_from_mag,
        "S_to_mag": np.abs(S_to),
        "loading_pct": loading_pct,
        "rate_a": rate_a,
        "u": u,
    }


class GetLineFlowTool(BaseTool):
    name = "get_line_flow"
    description = "Get line active/reactive power flow and loading percentage."

    def _validate_params(self, line_id=None, **kw):
        if line_id is not None and line_id != "all":
            validate_device_idx(self.manager.ss.Line, line_id, "Line")

    def _run(self, line_id=None, **kw):
        self.manager._require_min_state(SystemState.PFLOW_DONE)
        ss = self.manager.ss
        base = self.manager.base_mva

        flows = compute_line_flows(ss.Line, base)
        idx_list = list(ss.Line.idx.v)
        bus1_list = list(ss.Line.bus1.v)
        bus2_list = list(ss.Line.bus2.v)

        entries = []
        for i in range(ss.Line.n):
            lid = idx_list[i]
            if line_id is not None and line_id != "all" and lid != line_id:
                continue

            loading = flows["loading_pct"][i]
            violation = None
            if not np.isnan(loading) and loading > 100.0:
                violation = {
                    "violated": True,
                    "limit": 100.0,
                    "value": float(loading),
                    "unit": "%",
                }

            entries.append({
                "line_id": lid,
                "from_bus": bus1_list[i],
                "to_bus": bus2_list[i],
                "status": int(flows["u"][i]),
                "p_from_mw": float(pu_to_mw(flows["P_from"][i], base)),
                "q_from_mvar": float(pu_to_mw(flows["Q_from"][i], base)),
                "p_to_mw": float(pu_to_mw(flows["P_to"][i], base)),
                "q_to_mvar": float(pu_to_mw(flows["Q_to"][i], base)),
                "s_from_mva": float(pu_to_mw(flows["S_from_mag"][i], base)),
                "loading_pct": None if np.isnan(loading) else float(loading),
                "violation": violation,
            })

        return {"lines": entries, "count": len(entries)}
