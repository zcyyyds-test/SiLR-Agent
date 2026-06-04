"""Stress scenarios for the CityLearn district-storage domain.

Each scenario freezes a single hour ``t`` plus an initial per-building SoC and
an initial set of (deliberately constraint-violating) battery set-points. The
SiLR-gated agent must adjust the per-building set-points until every battery's
SoC is within bounds and the district feeder import/export limits are
satisfied (= recovery). Time does not advance — one (snapshot, set-point) pair
is one decision the verifier gates (ANM-isomorphic).

Design notes:
  - Recoverability is the empirical question this domain exists to study, so
    each shipped snapshot is initially VIOLATING but has at least one feasible
    joint set-point (verified at design time with ``simulator.evaluate`` — see
    the per-scenario ``notes``).
  - ``source_seed`` records provenance; the CityLearn profiles are
    deterministic (a single summer-day trace), so the seed is informational.
  - ``setup_episode`` installs the frozen state + initial set-points and solves
    once; it does not silently repair an unrecoverable snapshot.

Hours were chosen to exercise distinct constraint families:
  - t=18 (evening load peak, near-zero PV) → discharging from a low SoC drives
    batteries below ``soc_min`` (SoC-floor violation).
  - t=12 (midday PV peak, negative net load) → charging from a high SoC drives
    batteries above ``soc_max`` (SoC-ceiling violation); discharging hard
    pushes aggregate back-feed past the export limit (export violation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import simulator as sim
from .manager import CityLearnManager
from .simulator import CityLearnState


@dataclass
class CityLearnScenario:
    """A frozen (hour, initial SoC, initial set-points) operating point.

    ``initial_actions`` are the deliberately-violating starting set-points
    (kW per building, from ``simulator.ACTIONS_PER_BUILDING``). ``initial_soc``
    is the per-building SoC at the start of the fixed hour.
    """

    id: str
    fixed_t: int
    initial_soc: tuple[float, ...]
    initial_actions: tuple[float, ...]
    peak_import_kw: float = 0.0
    source_seed: Optional[int] = None
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    notes: str = ""


SCENARIOS: list[CityLearnScenario] = [
    # SoC-floor violation: evening load peak, low SoC, all batteries discharging
    # hard. soc_next = soc - 3.0/0.95 ≈ soc - 3.16 → all three fall below the
    # 0.5 kWh floor. Recovery: charge instead (feasible joint action e.g.
    # (-3.0, -3.0, -3.0) keeps district import well under the 16 kW limit).
    CityLearnScenario(
        id="cl_multi_1",
        fixed_t=18,
        initial_soc=(0.9, 0.8, 0.7),
        initial_actions=(3.0, 3.0, 3.0),
        difficulty="medium",
        source_seed=0,
        notes=(
            "t=18 (load peak, PV≈0). All three batteries discharge 3.0 kW from "
            "near-empty SoC → soc_next≈(-2.26,-2.36,-2.46), all below the 0.5 "
            "kWh floor (penalty≈8.57). Feasible recovery: charge "
            "(-3.0,-3.0,-3.0) → SoC rises into range, district import ≈ "
            "5.11+~7.5 charge ≈ 12.6 kW < 16 limit."
        ),
    ),
    # SoC-ceiling violation: midday, high SoC, all batteries charging hard.
    # soc_next = soc + 0.95*3.0 = soc + 2.85 → all three exceed soc_max.
    # Recovery: stop charging (feasible joint action (0,0,0)).
    CityLearnScenario(
        id="cl_multi_2",
        fixed_t=12,
        initial_soc=(5.0, 4.5, 3.2),
        initial_actions=(-3.0, -3.0, -3.0),
        difficulty="medium",
        source_seed=0,
        notes=(
            "t=12 (PV peak). All three batteries charge 3.0 kW from high SoC → "
            "soc_next=(7.85,7.35,6.05) vs soc_max=(6.4,5.6,4.0), all above the "
            "ceiling (penalty=5.25). Feasible recovery: idle (0,0,0) leaves SoC "
            "unchanged (within bounds) and district net at PV-driven export "
            "2.83 kW < 8 limit."
        ),
    ),
    # District export violation: midday PV peak, all batteries discharging hard
    # from a comfortable SoC. Aggregate back-feed = |net(-2.83) - discharge 9.0|
    # = 11.83 kW > 8 kW export limit, while SoC stays in range. Recovery:
    # reduce discharge (feasible joint action e.g. (-1.5,-1.5,0.0)).
    CityLearnScenario(
        id="cl_multi_3",
        fixed_t=12,
        initial_soc=(3.8, 3.8, 3.8),
        initial_actions=(3.0, 3.0, 3.0),
        difficulty="hard",
        source_seed=0,
        notes=(
            "t=12 (PV peak, net load -2.83 kW). All three discharge 3.0 kW → "
            "soc_next≈0.64 kWh (still ≥0.5 floor) but aggregate export = 11.83 "
            "kW > 8 kW limit (penalty≈3.83). Feasible recovery: cut discharge, "
            "e.g. (-1.5,-1.5,0.0), to bring export under the limit."
        ),
    ),
]


def _load_mined_scenarios() -> list["CityLearnScenario"]:
    """Optionally extend the curated library with mining-pipeline output.

    If ``domains/citylearn/scenarios_mined.json`` exists (written by
    ``scripts/citylearn_select_multi_action.py``), each record is promoted to a
    ``CityLearnScenario`` and merged with the hand-curated list -- mirrors ANM's
    loader, letting the multi-type band grow without bloating this module. The
    file is optional: when absent (before mining has run) the library stays at
    the original three scenarios.
    """
    import json
    from pathlib import Path

    path = Path(__file__).with_name("scenarios_mined.json")
    if not path.exists():
        return []
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return []
    mined: list[CityLearnScenario] = []
    for r in data.get("scenarios", []):
        try:
            mined.append(
                CityLearnScenario(
                    id=r["id"],
                    fixed_t=int(r["fixed_t"]),
                    initial_soc=tuple(float(x) for x in r["initial_soc"]),
                    initial_actions=tuple(float(x) for x in r["initial_actions"]),
                    peak_import_kw=float(r.get("peak_import_kw", 0.0)),
                    source_seed=r.get("source_seed"),
                    difficulty=r.get("difficulty", "medium"),
                    notes=(
                        f"class={r.get('class')} families={r.get('families')} "
                        f"default_penalty={r.get('default_penalty')} "
                        f"n_feasible={r.get('n_feasible')} "
                        f"single_action_solvable={r.get('single_action_solvable')}"
                    ),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return mined


SCENARIOS.extend(_load_mined_scenarios())
_SCENARIO_MAP = {s.id: s for s in SCENARIOS}


class CityLearnScenarioLoader:
    """Load CityLearn district-storage stress scenarios and apply them."""

    def load(self, scenario_id: str) -> CityLearnScenario:
        if scenario_id not in _SCENARIO_MAP:
            raise KeyError(
                f"Unknown CityLearn scenario: {scenario_id!r}. "
                f"Available: {sorted(_SCENARIO_MAP)}"
            )
        return _SCENARIO_MAP[scenario_id]

    def load_all(self) -> list[CityLearnScenario]:
        return list(SCENARIOS)

    def setup_episode(
        self,
        manager: CityLearnManager,
        scenario: CityLearnScenario,
    ) -> bool:
        """Install the frozen hour + initial SoC + violating set-points on
        ``manager`` and solve once (no time advance).

        Returns the convergence flag from ``solve()`` (always True for this
        pure-arithmetic simulator).
        """
        manager._state = CityLearnState(
            t=int(scenario.fixed_t),
            soc_kwh=tuple(scenario.initial_soc),
            peak_import_kw=float(scenario.peak_import_kw),
        )
        manager._actions = list(scenario.initial_actions)
        return manager.solve()
