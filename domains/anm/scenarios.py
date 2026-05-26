"""Stress scenarios for the gym-anm ANM6-Easy distribution-network domain.

Each scenario freezes a specific ``(P_load, P_pot)`` snapshot that the SiLR-gated
agent must keep within voltage / branch-loading / storage-SoC limits, by emitting
set-points for the non-slack generators and the storage unit.

Design notes (see decisions.md 2026-05-22 panel#2):
  - Frozen conditions (no stochastic advance) match the SiLR verifier semantics:
    one (snapshot, set-point) pair is one decision the verifier must gate.
  - Recoverability is the empirical question this domain exists to study —
    we ship snapshots of varying stress, not only "easy" ones. ``setup_episode``
    refuses to silently fix unrecoverable snapshots; the eval pipeline reports
    when the agent cannot find a feasible action.
  - ``seed`` records provenance ("snapshot was sampled from ANM6-Easy at this
    seed / step") so independent replication can reproduce the (P_load, P_pot)
    pair from gym-anm rather than trusting the hand-typed numbers.

Add new scenarios from the mining script (``scripts/mine_anm_scenarios.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .manager import GymANMManager


@dataclass
class ANMScenario:
    """A frozen (P_load, P_pot) operating point + provenance metadata.

    Keys of ``P_load`` must equal the manager's load device ids; keys of
    ``P_pot`` must equal the non-slack generator ids. The manager's
    ``set_conditions`` enforces this and raises on mismatch.
    """

    id: str
    description: str
    difficulty: str  # "easy" | "medium" | "hard"
    P_load: dict[int, float]
    P_pot: dict[int, float]
    # Optional initial set-points to apply on top of defaults (mostly None;
    # default initialisation = renewables at full P_pot, storage idle).
    initial_P_set: Optional[dict[int, float]] = None
    initial_Q_set: Optional[dict[int, float]] = None
    # Optional storage state-of-charge override for mined scenarios that were
    # selected near a storage boundary. Without this, "near_min/near_max" mined
    # records silently replay with the environment's native SoC.
    initial_soc: Optional[dict[int, float]] = None
    # Provenance: which env seed / step this snapshot was sampled from.
    source_seed: Optional[int] = None
    source_step: Optional[int] = None
    notes: str = ""


# ANM6-Easy device ids (loads / non-slack gens / storage / slack=0).
_LOAD_IDS = (1, 3, 5)
_GEN_IDS = (2, 4)
_DES_IDS = (6,)


SCENARIOS: list[ANMScenario] = [
    ANMScenario(
        id="easy_lightload",
        description="Light load, modest renewables — feasible at default set-points.",
        difficulty="easy",
        P_load={1: -0.5, 3: -0.5, 5: -0.5},
        P_pot={2: 1.0, 4: 1.0},
        notes="Smoke-test control case: default = PASS.",
    ),
    ANMScenario(
        id="medium_seed42_default",
        description=(
            "ANM6-Easy at seed=42 step=0: low wind potential, moderate PV "
            "potential, branch 2-4 (PV → load) overloaded if PV runs at full."
        ),
        difficulty="medium",
        P_load={1: -1.0, 3: -4.0, 5: 0.0},
        P_pot={2: 0.0, 4: 40.0},
        source_seed=42,
        source_step=0,
        notes=(
            "Default (PV at P_pot=40MW) overloads branch 2-4 (rate 18 MVA → "
            "≈200%). Curtailing PV alone may not relieve it (load-driven "
            "overload) — storage discharge typically required."
        ),
    ),
    ANMScenario(
        id="hard_renewable_surge",
        description=(
            "Heavy renewable surge + light load: branch reverse flow stress, "
            "requires curtailment + storage charging to absorb surplus."
        ),
        difficulty="hard",
        P_load={1: -0.2, 3: -0.5, 5: -0.2},
        P_pot={2: 28.0, 4: 48.0},
        notes=(
            "Renewables near p_max (gen 2: 30 MW, gen 4: 50 MW) with very "
            "light load — surplus pushes branches near rating in reverse. "
            "Recovery: curtail and/or charge storage."
        ),
    ),
]


def _load_mined_scenarios() -> list[ANMScenario]:
    """Optionally extend the curated library with mining-pipeline output.

    If ``domains/anm/scenarios_mined.json`` exists (written by
    ``scripts/anm_select_mined.py``), each record is promoted to an
    ``ANMScenario`` and merged with the hand-curated list. This lets the
    paper's evaluation table grow without bloating the static module
    body. The file is optional — when absent (e.g., before the mining
    pipeline has been run), the library stays at the original 3
    scenarios.
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
    mined: list[ANMScenario] = []
    for r in data.get("scenarios", []):
        # Skip records that are missing the frozen conditions outright —
        # an earlier selector version emitted ``P_load: null`` for some
        # classes and would crash the loader at module import time.
        if not isinstance(r.get("P_load"), dict) or not isinstance(r.get("P_pot"), dict):
            continue
        try:
            mined.append(
                ANMScenario(
                    id=r["id"],
                    description=r.get("description", ""),
                    difficulty=r.get("difficulty", "medium"),
                    P_load={int(k): float(v) for k, v in r["P_load"].items()},
                    P_pot={int(k): float(v) for k, v in r["P_pot"].items()},
                    initial_soc=(
                        {int(k): float(v) for k, v in r["initial_soc"].items()}
                        if isinstance(r.get("initial_soc"), dict)
                        else None
                    ),
                    source_seed=r.get("source_seed"),
                    notes=(
                        f"class={r.get('class')} "
                        f"mpc_recovered={r.get('mpc_recovered')} "
                        f"single_action_solvable={r.get('single_action_solvable')}"
                    ),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return mined


SCENARIOS.extend(_load_mined_scenarios())
_SCENARIO_MAP = {s.id: s for s in SCENARIOS}


class ANMScenarioLoader:
    """Load gym-anm ANM6-Easy stress scenarios and apply them to a manager."""

    def load(self, scenario_id: str) -> ANMScenario:
        if scenario_id not in _SCENARIO_MAP:
            raise KeyError(
                f"Unknown ANM scenario: {scenario_id!r}. "
                f"Available: {sorted(_SCENARIO_MAP)}"
            )
        return _SCENARIO_MAP[scenario_id]

    def load_all(self) -> list[ANMScenario]:
        return list(SCENARIOS)

    def setup_episode(
        self,
        manager: GymANMManager,
        scenario: ANMScenario,
        solve: bool = True,
    ) -> bool:
        """Apply ``scenario`` to ``manager``: freeze conditions, reset set-points,
        optionally overlay ``initial_P_set`` / ``initial_Q_set``, and solve.

        Returns the convergence flag from the final ``solve()`` (True if
        gym-anm's power flow converged).
        """
        # set_conditions validates keys and resets pending set-points to defaults
        # (renewables at full P_pot, storage idle) + re-solves once.
        manager.set_conditions(
            P_load=scenario.P_load,
            P_pot=scenario.P_pot,
            reset_setpoints=True,
            solve=False,  # we may need to overlay set-points before solving
        )
        if scenario.initial_soc is not None:
            for did, val in scenario.initial_soc.items():
                if did not in manager._des_ids:
                    raise ValueError(
                        f"initial_soc device id {did} not in storage ids "
                        f"{manager._des_ids}"
                    )
                manager._sim.devices[did].soc = float(val)
        if scenario.initial_P_set is not None:
            for did, val in scenario.initial_P_set.items():
                manager._P_set[did] = float(val)
        if scenario.initial_Q_set is not None:
            for did, val in scenario.initial_Q_set.items():
                manager._Q_set[did] = float(val)
        if solve:
            return manager.solve()
        return True
