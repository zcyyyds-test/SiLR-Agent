"""Mine CityLearn district-storage stress scenarios by exhaustive perturbation.

Unlike ANM (whose recoverability needs an approximate MPC oracle), the CityLearn
recovery task is a single-hour snapshot over a *discrete* joint action set
(5 set-points x 3 buildings = 125 joint actions), so recoverability is decided
*exactly* by brute force: a snapshot is recoverable iff some joint action drives
the penalty to zero, and ``single_action`` iff some feasible action differs from
the (violating) start in exactly one building. No LLM, no MPC, no GPU.

Perturbation space:
  - fixed hour ``t`` (spans load peaks t=7,8,17-19 and the PV peak t=11-13,
    exercising soc_min / soc_max / import / export families);
  - per-building initial SoC sampled at fractions of [soc_min, soc_max];
  - per-building initial set-point = the (deliberately violating) start state,
    swept over the full discrete action set.

Each candidate is classified trivial / single_action / multi_action /
unrecoverable, with the active constraint families recorded so the selector can
prioritise *multi-type* (physically incomparable) scenarios -- the regime where
the product-order reward should separate most sharply from a scalar surrogate.

Run from repo root (TSUBAME silr-vllm env, but CPU-only -- no GPU needed):
    PYTHONPATH=. python scripts/citylearn_scenario_mine.py \\
        --output mined_scenarios_citylearn.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from typing import Any

from domains.citylearn import simulator as sim
from domains.citylearn.checkers import (
    DistrictExportChecker,
    DistrictImportChecker,
    SoCChecker,
)
from domains.citylearn.manager import CityLearnManager

# Hours chosen for constraint-family diversity (see module docstring).
DEFAULT_HOURS = (7, 8, 11, 12, 13, 17, 18, 19)
# Per-building SoC sampling fractions of the [soc_min, soc_max] band.
DEFAULT_SOC_FRACS = (0.1, 0.5, 0.9)

_CHECKERS = (SoCChecker(), DistrictImportChecker(), DistrictExportChecker())
_ACTIONS = sim.ACTIONS_PER_BUILDING
_JOINT_ACTIONS = tuple(itertools.product(_ACTIONS, repeat=sim.N_BUILDINGS))


def _soc_from_fracs(fracs: tuple[float, ...]) -> tuple[float, ...]:
    """Map a per-building fraction tuple to absolute SoC (kWh)."""
    return tuple(
        sim.SOC_MIN_KWH[b] + f * (sim.SOC_MAX_KWH[b] - sim.SOC_MIN_KWH[b])
        for b, f in enumerate(fracs)
    )


def _penalty(t: int, soc: tuple[float, ...], action: tuple[float, ...]) -> float:
    mgr = CityLearnManager(fixed_t=t, initial_soc=soc, initial_actions=action)
    return mgr.last_penalty


def _families(t: int, soc: tuple[float, ...], action: tuple[float, ...]) -> list[str]:
    """Active constraint families (constraint_type) of the start state."""
    mgr = CityLearnManager(fixed_t=t, initial_soc=soc, initial_actions=action)
    state = mgr.system_state
    fams: list[str] = []
    for checker in _CHECKERS:
        cr = checker.check(state, mgr.base_mva)
        for v in cr.violations:
            if v.constraint_type not in fams:
                fams.append(v.constraint_type)
    return fams


def _hamming(a: tuple[float, ...], b: tuple[float, ...]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def classify(t: int, soc: tuple[float, ...], init_action: tuple[float, ...],
             penalty_map: dict[tuple[float, ...], float]) -> dict[str, Any] | None:
    """Classify one (t, soc, init_action) start state. Returns None if the
    start is already feasible (not a recovery scenario)."""
    default_pen = penalty_map[init_action]
    if default_pen <= 1e-6:
        return None  # not a post-violation start

    feasible = [a for a, p in penalty_map.items() if p <= 1e-6]
    if not feasible:
        return {"class": "unrecoverable", "default_penalty": default_pen,
                "n_feasible": 0, "single_action_solvable": False}

    single = any(_hamming(a, init_action) == 1 for a in feasible)
    return {
        "class": "single_action" if single else "multi_action",
        "default_penalty": round(default_pen, 4),
        "n_feasible": len(feasible),
        "single_action_solvable": single,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, nargs="+", default=list(DEFAULT_HOURS))
    parser.add_argument("--soc-fracs", type=float, nargs="+",
                        default=list(DEFAULT_SOC_FRACS))
    parser.add_argument("--output", default="mined_scenarios_citylearn.json")
    args = parser.parse_args()

    soc_configs = list(itertools.product(args.soc_fracs, repeat=sim.N_BUILDINGS))
    n_cells = len(args.hours) * len(soc_configs)
    print(f"Mining CityLearn: {len(args.hours)} hours x {len(soc_configs)} SoC "
          f"configs x {len(_JOINT_ACTIONS)} start actions "
          f"= {n_cells * len(_JOINT_ACTIONS)} candidates "
          f"({n_cells} (t,soc) cells)")

    catalogue: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    t0 = time.time()

    for ci, (t, soc_fracs) in enumerate(itertools.product(args.hours, soc_configs)):
        soc = _soc_from_fracs(soc_fracs)
        # One exact penalty sweep over the joint action set for this (t, soc).
        penalty_map = {a: _penalty(t, soc, a) for a in _JOINT_ACTIONS}

        for init_action in _JOINT_ACTIONS:
            cls = classify(t, soc, init_action, penalty_map)
            if cls is None:
                continue
            counts[cls["class"]] = counts.get(cls["class"], 0) + 1

            fams = _families(t, soc, init_action)
            family_key = "multi_type" if len(fams) > 1 else (fams[0] if fams else "none")
            family_counts[family_key] = family_counts.get(family_key, 0) + 1

            catalogue.append({
                "fixed_t": t,
                "soc_fracs": list(soc_fracs),
                "initial_soc": [round(x, 4) for x in soc],
                "initial_actions": list(init_action),
                "families": fams,
                "n_families": len(fams),
                **cls,
            })

        if (ci + 1) % 8 == 0 or ci + 1 == n_cells:
            print(f"  [progress] {ci+1}/{n_cells} cells, counts={counts}, "
                  f"families={family_counts}")

    dt = time.time() - t0
    print(f"\n=== mined {len(catalogue)} violating candidates in {dt:.1f}s ===")
    print(f"  class counts: {counts}")
    print(f"  family diversity: {family_counts}")

    with open(args.output, "w") as f:
        json.dump({"counts": counts, "family_diversity": family_counts,
                   "catalogue": catalogue}, f, indent=2)
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
