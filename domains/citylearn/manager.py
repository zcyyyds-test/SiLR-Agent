"""CityLearnManager: SiLR adapter over the district-storage simulator.

Implements BaseSystemManager. The recovery task is ANM-isomorphic: a single
hour `t` is fixed, the manager holds per-building battery set-points that the
agent mutates via tools, and solve() re-evaluates the district steady state
WITHOUT advancing time. The verifier shadow-copies the manager, applies a
proposed set-point change, re-solves, and gates on whether SoC / district
import / export constraints improved (terminal vs progress_mag).

solve() deliberately does NOT advance `t` (unlike FinanceManager.solve, which
self-increments time harmlessly because finance prices are static). CityLearn
load/PV/price are hour-dependent, so advancing time inside solve would make the
verifier's baseline-vs-post comparison read two different hours and invalidate
the gate.
"""

from __future__ import annotations

from typing import Any

from silr.core.interfaces import BaseSystemManager
from domains.citylearn import simulator as sim
from domains.citylearn.simulator import CityLearnState


class CityLearnManager(BaseSystemManager):
    """District battery-storage simulator. Pure Python, no external deps."""

    def __init__(
        self,
        fixed_t: int = 18,
        initial_soc: tuple[float, ...] | None = None,
        initial_actions: tuple[float, ...] | None = None,
        peak_import_kw: float = 0.0,
    ):
        self._state = CityLearnState(
            t=int(fixed_t),
            soc_kwh=tuple(initial_soc) if initial_soc is not None else sim.INITIAL_SOC_KWH,
            peak_import_kw=float(peak_import_kw),
        )
        # Pending per-building set-points; the agent mutates these via tools,
        # solve() applies them. Initial values come from the scenario and may
        # be constraint-violating (that is the post-violation start state).
        self._actions: list[float] = (
            list(initial_actions) if initial_actions is not None
            else [0.0] * sim.N_BUILDINGS
        )
        self._derived: dict = {}
        self._penalty: float = 0.0
        self.solve()

    # ── BaseSystemManager interface ──────────────────────────────
    @property
    def sim_time(self) -> float:
        return float(self._state.t)

    @property
    def base_mva(self) -> float:
        return 1.0  # not applicable

    @property
    def last_penalty(self) -> float:
        """Scalar penalty = total constraint-violation magnitude (0 = feasible).

        Consumed by the scalar_progress gating policy. Zero iff every SoC bound
        and the district import/export limits are satisfied.
        """
        return self._penalty

    @property
    def system_state(self) -> dict:
        d = self._derived
        buildings = []
        for b in range(sim.N_BUILDINGS):
            buildings.append({
                "id": f"battery_{b}",
                "index": b,
                "soc_kwh": d["soc_next_kwh"][b],
                "soc_min_kwh": sim.SOC_MIN_KWH[b],
                "soc_max_kwh": sim.SOC_MAX_KWH[b],
                "action_kw": self._actions[b],
            })
        return {
            "t": d["t"],
            "price": d["price"],
            "buildings": buildings,
            "district_import_kw": d["district_import_kw"],
            "district_import_limit_kw": sim.DISTRICT_IMPORT_LIMIT_KW,
            "district_export_kw": d["district_export_kw"],
            "district_export_limit_kw": sim.DISTRICT_EXPORT_LIMIT_KW,
            "peak_import_kw": d["peak_import_kw"],
            "cost": d["cost"],
        }

    def create_shadow_copy(self) -> "CityLearnManager":
        """Independent copy for SiLR verification.

        _state is a frozen dataclass (immutable) so sharing the reference is
        safe; the mutable per-building action list is deep-copied.
        """
        shadow = CityLearnManager.__new__(CityLearnManager)
        shadow._state = self._state
        shadow._actions = list(self._actions)
        shadow._derived = dict(self._derived)
        shadow._penalty = self._penalty
        return shadow

    def solve(self) -> bool:
        """Re-evaluate district steady state for the pending set-points at the
        FIXED hour t. Pure arithmetic, never fails to converge."""
        self._derived = sim.evaluate(self._state, tuple(self._actions))
        self._penalty = self._compute_penalty()
        return True

    # ── helpers ──────────────────────────────────────────────────
    def _compute_penalty(self) -> float:
        d = self._derived
        pen = 0.0
        for b in range(sim.N_BUILDINGS):
            s = d["soc_next_kwh"][b]
            pen += max(0.0, sim.SOC_MIN_KWH[b] - s)
            pen += max(0.0, s - sim.SOC_MAX_KWH[b])
        pen += max(0.0, d["district_import_kw"] - sim.DISTRICT_IMPORT_LIMIT_KW)
        pen += max(0.0, d["district_export_kw"] - sim.DISTRICT_EXPORT_LIMIT_KW)
        return round(pen, 6)

    # ── domain operations (used by tools / scenario injection) ───
    def set_building_action(self, index: int, value: float) -> bool:
        """Set a building's battery set-point (discrete charge/discharge kW)."""
        if index < 0 or index >= sim.N_BUILDINGS:
            return False
        if value not in sim.ACTIONS_PER_BUILDING:
            return False
        self._actions[index] = float(value)
        return True

    def get_building_indices(self) -> list[int]:
        return list(range(sim.N_BUILDINGS))

    def get_action_choices(self) -> list[float]:
        return list(sim.ACTIONS_PER_BUILDING)
