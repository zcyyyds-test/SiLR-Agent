"""Action tools: adjust_gen, shed_load, trip_line, close_line."""

import numpy as np

from silr.tools.base import BaseTool
from ..simulator import SystemState
from ..utils.unit_converter import pu_to_mw, mw_to_pu
from ..utils.validators import validate_device_idx, clamp
from silr.exceptions import ValidationError, DeviceNotFoundError


class AdjustGenTool(BaseTool):
    name = "adjust_gen"
    description = "Adjust generator active power output by delta_p_mw."

    def _validate_params(self, gen_id=None, delta_p_mw=None, **kw):
        if gen_id is None:
            raise ValidationError("gen_id is required")
        if delta_p_mw is None:
            raise ValidationError("delta_p_mw is required")
        try:
            delta_p_mw = float(delta_p_mw)
        except (TypeError, ValueError):
            raise ValidationError(f"delta_p_mw must be a number, got {type(delta_p_mw).__name__}")

    def _run(self, gen_id=None, delta_p_mw=None, **kw):
        ss = self.manager.ss
        base = self.manager.base_mva

        # Find which syn gen model this belongs to
        syn_model = self.manager.get_gen_model_name(gen_id)
        if syn_model is None:
            raise DeviceNotFoundError(
                f"Generator '{gen_id}' not found in GENROU or GENCLS."
            )

        # Get static gen (PV/Slack) mapping
        static_gen_idx = self.manager.get_static_gen_for_syn(gen_id, syn_model)
        limits = self.manager.get_gen_limits(static_gen_idx)

        current_p_pu = limits["p0"]
        delta_p_pu = mw_to_pu(delta_p_mw, base)
        new_p_pu = current_p_pu + delta_p_pu

        clamped_p_pu, was_clamped = clamp(new_p_pu, limits["pmin"], limits["pmax"])

        # Apply to static gen model
        for mdl_name in ["PV", "Slack"]:
            mdl = getattr(ss, mdl_name, None)
            if mdl is not None and static_gen_idx in list(mdl.idx.v):
                mdl.alter('p0', static_gen_idx, clamped_p_pu)
                break

        actual_delta_mw = pu_to_mw(clamped_p_pu - current_p_pu, base)

        return {
            "gen_id": gen_id,
            "static_gen_id": static_gen_idx,
            "previous_p_mw": float(pu_to_mw(current_p_pu, base)),
            "new_p_mw": float(pu_to_mw(clamped_p_pu, base)),
            "requested_delta_mw": float(delta_p_mw),
            "actual_delta_mw": float(actual_delta_mw),
            "was_clamped": was_clamped,
            "limits_mw": {
                "pmin": float(pu_to_mw(limits["pmin"], base)),
                "pmax": float(pu_to_mw(limits["pmax"], base)),
            },
        }


class ShedLoadTool(BaseTool):
    name = "shed_load"
    description = "Shed load at a specified bus by amount_mw."

    def _validate_params(self, bus_id=None, amount_mw=None, **kw):
        if bus_id is None:
            raise ValidationError("bus_id is required")
        try:
            amount_mw = float(amount_mw) if amount_mw is not None else None
        except (TypeError, ValueError):
            raise ValidationError(f"amount_mw must be a number, got {type(amount_mw).__name__}")
        if amount_mw is None or amount_mw <= 0:
            raise ValidationError(
                f"amount_mw must be positive, got {amount_mw}"
            )

    def _run(self, bus_id=None, amount_mw=None, **kw):
        ss = self.manager.ss
        base = self.manager.base_mva
        validate_device_idx(ss.Bus, bus_id, "Bus")

        pq_loads = self.manager.get_pq_on_bus(bus_id)
        if not pq_loads:
            raise ValidationError(f"No PQ loads found on bus '{bus_id}'.")

        total_p_pu = sum(ld["p0"] for ld in pq_loads)
        if total_p_pu <= 0:
            raise ValidationError(
                f"No positive load on bus '{bus_id}' to shed."
            )

        amount_pu = mw_to_pu(amount_mw, base)
        actual_shed_pu, was_clamped = clamp(amount_pu, 0, total_p_pu)
        shed_ratio = actual_shed_pu / total_p_pu

        pq_idx_list = list(ss.PQ.idx.v)
        in_tds = self.manager.state in (
            SystemState.TDS_INITIALIZED, SystemState.TDS_RUNNING
        )

        shed_details = []
        for load in pq_loads:
            idx = load["idx"]
            old_p = load["p0"]
            old_q = load["q0"]
            new_p = old_p * (1 - shed_ratio)
            new_q = old_q * (1 - shed_ratio)

            internal_i = pq_idx_list.index(idx)

            if in_tds:
                # During TDS, modify Ppf/Qpf directly for immediate effect
                if hasattr(ss.PQ, 'Ppf'):
                    ss.PQ.Ppf.v[internal_i] = new_p
                if hasattr(ss.PQ, 'Qpf'):
                    ss.PQ.Qpf.v[internal_i] = new_q
            else:
                ss.PQ.alter('p0', idx, new_p)
                ss.PQ.alter('q0', idx, new_q)

            shed_details.append({
                "pq_id": idx,
                "old_p_mw": float(pu_to_mw(old_p, base)),
                "new_p_mw": float(pu_to_mw(new_p, base)),
                "old_q_mvar": float(pu_to_mw(old_q, base)),
                "new_q_mvar": float(pu_to_mw(new_q, base)),
            })

        return {
            "bus_id": bus_id,
            "requested_mw": float(amount_mw),
            "actual_shed_mw": float(pu_to_mw(actual_shed_pu, base)),
            "was_clamped": was_clamped,
            "total_bus_load_mw": float(pu_to_mw(total_p_pu, base)),
            "shed_ratio": float(shed_ratio),
            "details": shed_details,
        }


class TripLineTool(BaseTool):
    name = "trip_line"
    description = "Trip (disconnect) a transmission line."

    def _validate_params(self, line_id=None, **kw):
        if line_id is None:
            raise ValidationError("line_id is required")

    def _run(self, line_id=None, **kw):
        ss = self.manager.ss
        validate_device_idx(ss.Line, line_id, "Line")

        status = self.manager.get_line_status(line_id)
        if status == 0:
            raise ValidationError(
                f"Line '{line_id}' is already disconnected."
            )

        ss.Line.alter('u', line_id, 0)

        return {
            "line_id": line_id,
            "previous_status": 1,
            "new_status": 0,
            "action": "tripped",
        }


class CloseLineTool(BaseTool):
    name = "close_line"
    description = "Close (reconnect) a transmission line."

    def _validate_params(self, line_id=None, **kw):
        if line_id is None:
            raise ValidationError("line_id is required")

    def _run(self, line_id=None, **kw):
        ss = self.manager.ss
        validate_device_idx(ss.Line, line_id, "Line")

        status = self.manager.get_line_status(line_id)
        if status == 1:
            raise ValidationError(
                f"Line '{line_id}' is already connected."
            )

        ss.Line.alter('u', line_id, 1)

        return {
            "line_id": line_id,
            "previous_status": 0,
            "new_status": 1,
            "action": "closed",
        }
