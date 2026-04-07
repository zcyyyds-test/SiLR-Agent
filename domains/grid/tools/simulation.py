"""Simulation tools: inject_fault, run_tds, run_powerflow."""

import numpy as np

from silr.tools.base import BaseTool
from ..simulator import SystemState
from ..utils.unit_converter import pu_to_mw
from ..utils.validators import validate_device_idx, validate_positive
from ..config import SYSTEM_FREQ_HZ
from silr.exceptions import ValidationError


class InjectFaultTool(BaseTool):
    name = "inject_fault"
    description = "Inject a fault (bus_fault or line_trip) into the system."

    def _validate_params(self, fault_type=None, target_id=None,
                         tf=None, tc=None, **kw):
        if fault_type not in ("bus_fault", "line_trip", "line_toggle"):
            raise ValidationError(
                f"fault_type must be 'bus_fault', 'line_trip', or "
                f"'line_toggle', got '{fault_type}'"
            )
        if tf is None:
            raise ValidationError("tf (fault time) is required")
        validate_positive(tf, "tf")
        if fault_type == "bus_fault":
            if tc is None:
                raise ValidationError("tc (clear time) required for bus_fault")
            if tc <= tf:
                raise ValidationError(f"tc ({tc}) must be > tf ({tf})")

    def _run(self, fault_type=None, target_id=None,
             tf=None, tc=None, xf=0.0001, **kw):
        ss = self.manager.ss

        if fault_type == "bus_fault":
            validate_device_idx(ss.Bus, target_id, "Bus")
            params = {"bus": target_id, "tf": tf, "tc": tc, "xf": xf}
            self.manager.register_event("Fault", params)
            self.manager.rebuild_with_events()
            return {
                "fault_type": "bus_fault",
                "target": target_id,
                "tf": tf, "tc": tc, "xf": xf,
                "method": "event_rebuild",
            }

        elif fault_type == "line_trip":
            validate_device_idx(ss.Line, target_id, "Line")
            status = self.manager.get_line_status(target_id)
            if status == 0:
                raise ValidationError(
                    f"Line '{target_id}' is already disconnected."
                )
            ss.Line.alter('u', target_id, 0)
            return {
                "fault_type": "line_trip",
                "target": target_id,
                "method": "alter_u",
                "previous_status": 1, "new_status": 0,
            }

        elif fault_type == "line_toggle":
            validate_device_idx(ss.Line, target_id, "Line")
            params = {"model": "Line", "dev": target_id, "t": tf}
            self.manager.register_event("Toggle", params)
            self.manager.rebuild_with_events()
            return {
                "fault_type": "line_toggle",
                "target": target_id,
                "tf": tf,
                "method": "toggle_rebuild",
            }


class RunTdsTool(BaseTool):
    name = "run_tds"
    description = "Run time-domain simulation to specified end time."

    def _validate_params(self, t_end=None, step_size=None, **kw):
        if t_end is None:
            raise ValidationError("t_end is required")
        validate_positive(t_end, "t_end")
        if step_size is not None:
            validate_positive(step_size, "step_size")

    def _run(self, t_end=None, step_size=None, **kw):
        self.manager.run_tds(t_end, step_size)
        ss = self.manager.ss

        summary = {"t_end": float(ss.dae.t)}

        # Voltage summary
        v = np.asarray(ss.Bus.v.v, dtype=float)
        summary["voltage"] = {
            "min_pu": float(np.min(v)),
            "max_pu": float(np.max(v)),
            "mean_pu": float(np.mean(v)),
        }

        # Frequency summary
        for model_name in ["GENROU", "GENCLS"]:
            mdl = getattr(ss, model_name, None)
            if mdl is not None and mdl.n > 0:
                omega = np.asarray(mdl.omega.v, dtype=float)
                delta_f = (omega - 1.0) * SYSTEM_FREQ_HZ
                summary["frequency"] = {
                    "min_delta_f_hz": float(np.min(delta_f)),
                    "max_delta_f_hz": float(np.max(delta_f)),
                }
                break

        return summary


class RunPowerFlowTool(BaseTool):
    name = "run_powerflow"
    description = "Run AC power flow calculation."

    def _validate_params(self, **kw):
        pass

    def _run(self, **kw):
        converged = self.manager.solve()
        ss = self.manager.ss
        base = self.manager.base_mva
        data = {"converged": converged}

        if converged:
            v = np.asarray(ss.Bus.v.v, dtype=float)
            data["voltage_summary"] = {
                "min_pu": float(np.min(v)),
                "max_pu": float(np.max(v)),
                "mean_pu": float(np.mean(v)),
            }

            total_gen_p = 0.0
            for mdl_name in ["PV", "Slack"]:
                mdl = getattr(ss, mdl_name, None)
                if mdl is not None and mdl.n > 0:
                    total_gen_p += float(np.sum(mdl.p.v))

            total_load_p = 0.0
            if ss.PQ.n > 0:
                total_load_p = float(np.sum(ss.PQ.p0.v))

            data["power_summary"] = {
                "total_gen_pu": total_gen_p,
                "total_load_pu": total_load_p,
                "total_gen_mw": float(pu_to_mw(total_gen_p, base)),
                "total_load_mw": float(pu_to_mw(total_load_p, base)),
            }

        return data
