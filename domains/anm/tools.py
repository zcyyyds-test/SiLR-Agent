"""gym-anm domain tools for SiLR verification.

Three tools following the SiLR pattern (observe + act):
  - ``get_grid_status``       — observe bus voltages, branch loadings, storage SoC
  - ``set_generator_setpoint``— set a renewable generator's active/reactive set-point
  - ``set_storage_setpoint``  — set a storage unit's active/reactive set-point

The continuous gym-anm action vector is exposed to the LLM as discrete, physically
interpretable set-point tools (values in the env's native unit; MW/MVAr for
ANM6-Easy). Action tools mutate the manager's pending ``_P_set`` / ``_Q_set``;
``manager.solve()`` then re-solves the power flow for the verifier to check.

Bound checking: ``_validate_params`` rejects non-finite / out-of-device-range
set-points by raising ``ValidationError``. The verifier translates these to
``Verdict.ERROR`` (not ``FAIL``), keeping parameter typos out of safety-FAIL
training signals.
"""

from __future__ import annotations

import math

from silr.tools.base import BaseTool
from silr.exceptions import DeviceNotFoundError, ValidationError


def _check_finite(name: str, value) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return v


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    if value < lo or value > hi:
        raise ValidationError(
            f"{name}={value:.4f} outside device limits [{lo:.4f}, {hi:.4f}]"
        )


class GetGridStatusTool(BaseTool):
    """Observe bus voltages, branch loadings, storage state of charge, and
    actionable device bounds (potential, p_min/p_max, q_min/q_max, soc limits).

    Exposing bounds is critical for LLM decision quality: without them the
    agent cannot infer the feasible action range and emits out-of-bound
    set-points that the tool layer rejects, polluting the training signal.
    """

    name = "get_grid_status"
    description = (
        "Read bus voltage magnitudes, branch loadings, storage SoC, and "
        "actionable device bounds (P_pot for generators, p/q limits, SoC range)"
    )

    def _validate_params(self, **kwargs) -> None:
        pass

    def _run(self, **kwargs) -> dict:
        mgr = self.manager
        sim = mgr._sim
        buses = [
            {
                "bus": bid,
                "v_pu": round(float(abs(b.v)), 4),
                "v_min": float(b.v_min),
                "v_max": float(b.v_max),
            }
            for bid, b in sim.buses.items()
        ]
        branches = []
        for br in sim.branches.values():
            rate = float(br.rate)
            s = float(br.s_apparent_max)
            loading = (
                round(abs(s) / rate * 100.0, 1)
                if rate > 0 and math.isfinite(s)
                else None
            )
            branches.append(
                {
                    "branch": f"{br.f_bus}-{br.t_bus}",
                    "loading_pct": loading,
                    "rate_pu": round(rate, 4),
                }
            )
        base = mgr.base_mva  # bounds shown to the LLM in MW/MVAr (action unit)
        generators = []
        for g in mgr._gen_ids:
            dev = sim.devices[g]
            generators.append(
                {
                    "gen_id": g,
                    "P_pot_mw": round(float(mgr._P_pot.get(g, 0.0)), 4),
                    "p_min_mw": float(dev.p_min) * base,
                    "p_max_mw": float(dev.p_max) * base,
                    "q_min_mvar": float(dev.q_min) * base,
                    "q_max_mvar": float(dev.q_max) * base,
                }
            )
        storage = []
        for s in mgr._des_ids:
            dev = sim.devices[s]
            storage.append(
                {
                    "storage_id": s,
                    "soc": round(float(dev.soc), 4),
                    "soc_min": float(dev.soc_min),
                    "soc_max": float(dev.soc_max),
                    "p_min_mw": float(dev.p_min) * base,
                    "p_max_mw": float(dev.p_max) * base,
                    "q_min_mvar": float(dev.q_min) * base,
                    "q_max_mvar": float(dev.q_max) * base,
                }
            )
        return {
            "buses": buses,
            "branches": branches,
            "generators": generators,
            "storage": storage,
        }


class SetGeneratorSetpointTool(BaseTool):
    """Set a non-slack (renewable) generator's active/reactive power set-point.

    Rejects (ValidationError → Verdict.ERROR): non-finite values, p outside
    ``[device.p_min, min(device.p_max, P_pot[gen_id])]``, q outside
    ``[device.q_min, device.q_max]``. ``P_pot[gen_id]`` is the current realized
    available potential (renewables cannot generate more than that).
    """

    name = "set_generator_setpoint"
    description = (
        "Set a renewable generator's active power set-point (curtailment ceiling) "
        "and optional reactive power set-point"
    )

    def _validate_params(self, gen_id=None, p_mw=None, q_mvar=None, **kwargs) -> None:
        if gen_id is None or p_mw is None:
            raise ValidationError("gen_id and p_mw are required")
        mgr = self.manager
        if gen_id not in mgr._gen_ids:
            raise DeviceNotFoundError(
                f"Generator {gen_id} not found. Available: {mgr._gen_ids}"
            )
        p = _check_finite("p_mw", p_mw)
        dev = mgr._sim.devices[gen_id]
        base = mgr.base_mva  # gym-anm: device limits are p.u.; action / P_pot are MW
        p_lo = float(dev.p_min) * base
        p_hi = min(
            float(dev.p_max) * base,
            float(mgr._P_pot.get(gen_id, dev.p_max * base)),
        )
        _check_range(f"p_mw (gen {gen_id})", p, p_lo, p_hi)
        if q_mvar is not None:
            q = _check_finite("q_mvar", q_mvar)
            _check_range(
                f"q_mvar (gen {gen_id})",
                q,
                float(dev.q_min) * base,
                float(dev.q_max) * base,
            )

    def _run(self, gen_id=None, p_mw=None, q_mvar=None, **kwargs) -> dict:
        mgr = self.manager
        mgr._P_set[gen_id] = float(p_mw)
        if q_mvar is not None:
            mgr._Q_set[gen_id] = float(q_mvar)
        return {
            "gen_id": gen_id,
            "p_set": float(p_mw),
            "q_set": mgr._Q_set.get(gen_id),
        }


class SetStorageSetpointTool(BaseTool):
    """Set a storage unit's active power set-point (negative=charge, positive=discharge).

    Rejects (ValidationError → Verdict.ERROR): non-finite values, p/q outside
    ``[device.p_min, device.p_max]`` / ``[device.q_min, device.q_max]``. SoC
    bounds are enforced by the simulator (and observed by ``ANMStorageSoCChecker``).
    """

    name = "set_storage_setpoint"
    description = (
        "Set a storage unit's active power set-point "
        "(negative = charge, positive = discharge) and optional reactive set-point"
    )

    def _validate_params(self, storage_id=None, p_mw=None, q_mvar=None, **kwargs) -> None:
        if storage_id is None or p_mw is None:
            raise ValidationError("storage_id and p_mw are required")
        mgr = self.manager
        if storage_id not in mgr._des_ids:
            raise DeviceNotFoundError(
                f"Storage {storage_id} not found. Available: {mgr._des_ids}"
            )
        p = _check_finite("p_mw", p_mw)
        dev = mgr._sim.devices[storage_id]
        base = mgr.base_mva  # gym-anm: device limits are p.u.; action is MW
        _check_range(
            f"p_mw (storage {storage_id})",
            p,
            float(dev.p_min) * base,
            float(dev.p_max) * base,
        )
        if q_mvar is not None:
            q = _check_finite("q_mvar", q_mvar)
            _check_range(
                f"q_mvar (storage {storage_id})",
                q,
                float(dev.q_min) * base,
                float(dev.q_max) * base,
            )

    def _run(self, storage_id=None, p_mw=None, q_mvar=None, **kwargs) -> dict:
        mgr = self.manager
        mgr._P_set[storage_id] = float(p_mw)
        if q_mvar is not None:
            mgr._Q_set[storage_id] = float(q_mvar)
        return {
            "storage_id": storage_id,
            "p_set": float(p_mw),
            "q_set": mgr._Q_set.get(storage_id),
        }


def create_anm_toolset(manager) -> dict:
    """Create the gym-anm toolset bound to a manager instance."""
    tools = [
        GetGridStatusTool(manager),
        SetGeneratorSetpointTool(manager),
        SetStorageSetpointTool(manager),
    ]
    return {t.name: t for t in tools}
