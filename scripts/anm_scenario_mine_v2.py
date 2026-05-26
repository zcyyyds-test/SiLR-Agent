"""Mining v2: stress diversity via synthetic perturbation + larger seed sweep.

What v1 missed (panel-flagged):
  - All 110 non-trivial snapshots stressed branch_loading only — gym-anm's
    natural ANM6-Easy stochastic process never drives voltage or SoC
    violations, so the library inherited that bias.
  - `manager.step()` does not advance the env's time-of-day aux variable,
    so within a single seed steps 1..N return the same (P_load, P_pot) as
    step 0. v2 uses only step=0 and varies the seed for independent samples.

What v2 adds:
  - load multipliers ``{0.25, 1.0, 2.0, 3.0}``: 0.25× pushes the network
    toward overvoltage / reverse flow; 2-3× pushes toward branch overload
    + undervoltage on the heaviest-loaded buses.
  - gen multipliers ``{0.0, 1.0}``: 0.0× disables one or both renewables,
    forcing the slack bus + storage to cover all demand, which can
    surface SoC stress when initial SoC is near its lower bound.
  - SoC perturbations ``{native, near_min, near_max}``: directly mutate
    the storage device's SoC before solving so we can sample the SoC-
    boundary region that the natural process never reaches.
  - All (seed × load_mul × gen_mul × soc_pert) combinations classified
    as in v1: trivial / single_action / multi_action / mpc_unsolved,
    with per-checker stress type recorded.

Run from repo root (AMD ``silr-anm`` env):
    PYTHONPATH=. python scripts/anm_scenario_mine_v2.py \\
        --n-seeds 25 --output mined_scenarios_v2.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from typing import Any

import numpy as np

from domains.anm import GymANMManager, build_anm_domain_config
from gym_anm import MPCAgentConstant
from gym_anm.simulator.components import StorageUnit


LOAD_MULTIPLIERS = (0.25, 1.0, 2.0, 3.0)
GEN_MULTIPLIERS = (0.0, 1.0)
SOC_PERTURBATIONS = ("native", "near_min", "near_max")


def violation_summary(mgr, cfg) -> dict[str, Any]:
    by_checker = {}
    total = 0
    for checker in cfg.checkers:
        cr = checker.check(mgr.system_state, mgr.base_mva)
        n = 0 if cr.passed else len(cr.violations)
        by_checker[checker.name] = n
        total += n
    return {"total": total, "by_checker": by_checker, "penalty": float(mgr.last_penalty)}


def try_mpc(mgr) -> tuple[bool, float, list[float]]:
    env = mgr._env
    agent = MPCAgentConstant(
        simulator=env.simulator,
        action_space=env.action_space,
        gamma=env.gamma,
        safety_margin=0.9,
        planning_steps=8,
    )
    action = np.asarray(agent.act(env), dtype=float)
    n_gen = len(mgr._gen_ids)
    n_des = len(mgr._des_ids)
    P_set, Q_set = {}, {}
    for a, did in zip(action[:n_gen], mgr._gen_ids):
        P_set[int(did)] = float(a)
    for a, did in zip(action[n_gen:2 * n_gen], mgr._gen_ids):
        Q_set[int(did)] = float(a)
    for a, did in zip(action[2 * n_gen:2 * n_gen + n_des], mgr._des_ids):
        P_set[int(did)] = float(a)
    for a, did in zip(action[2 * n_gen + n_des:], mgr._des_ids):
        Q_set[int(did)] = float(a)
    mgr._P_set = P_set
    mgr._Q_set = Q_set
    mgr.solve()
    return mgr.last_penalty < 1e-6, float(mgr.last_penalty), [float(x) for x in action]


def try_single_actions(mgr, cfg, P_load, P_pot, soc_setter) -> bool:
    """Coarse grid over each single set-point. Replays the SoC perturbation
    on each retry since set_conditions resets the manager state."""
    for gen_id in mgr._gen_ids:
        device = mgr._sim.devices[gen_id]
        p_max_mw = min(float(device.p_max) * mgr.base_mva,
                       float(P_pot.get(gen_id, 0.0)) or 1e-3)
        grid = np.linspace(0.0, p_max_mw, 9)
        for p in grid:
            mgr.set_conditions(P_load, P_pot, reset_setpoints=True, solve=False)
            soc_setter(mgr)
            mgr._P_set[gen_id] = float(p)
            if mgr.solve() and violation_summary(mgr, cfg)["total"] == 0:
                return True
    for storage_id in mgr._des_ids:
        device = mgr._sim.devices[storage_id]
        p_min_mw = float(device.p_min) * mgr.base_mva
        p_max_mw = float(device.p_max) * mgr.base_mva
        grid = np.linspace(p_min_mw, p_max_mw, 9)
        for p in grid:
            mgr.set_conditions(P_load, P_pot, reset_setpoints=True, solve=False)
            soc_setter(mgr)
            mgr._P_set[storage_id] = float(p)
            if mgr.solve() and violation_summary(mgr, cfg)["total"] == 0:
                return True
    return False


def _soc_setter(perturb: str):
    """Return a closure that mutates each storage device's SoC to the
    requested boundary region just before solve()."""
    def apply(mgr):
        for sid in mgr._des_ids:
            dev = mgr._sim.devices[sid]
            if perturb == "native":
                pass
            elif perturb == "near_min":
                dev.soc = float(dev.soc_min) + 0.01 * (dev.soc_max - dev.soc_min)
            elif perturb == "near_max":
                dev.soc = float(dev.soc_max) - 0.01 * (dev.soc_max - dev.soc_min)
            else:
                raise ValueError(f"unknown soc perturbation: {perturb}")
    return apply


def classify(seed: int, load_mul: float, gen_mul: float, soc_pert: str,
             cfg) -> dict[str, Any]:
    mgr = GymANMManager(seed=seed)
    base_P_load = {k: v for k, v in mgr._P_load.items()}
    base_P_pot = {k: v for k, v in mgr._P_pot.items()}
    P_load = {k: v * load_mul for k, v in base_P_load.items()}
    P_pot = {k: v * gen_mul for k, v in base_P_pot.items()}
    soc_setter = _soc_setter(soc_pert)

    # 1. Default state
    try:
        mgr.set_conditions(P_load, P_pot, reset_setpoints=True, solve=False)
        soc_setter(mgr)
        if not mgr.solve():
            return {"class": "infeasible_solver", "reason": "default solve diverged"}
        default = violation_summary(mgr, cfg)
    except Exception as e:
        return {"class": "infeasible_conditions", "reason": f"{type(e).__name__}: {e}"}

    if default["total"] == 0:
        return {"class": "trivial", "default": default, "mpc": None}

    # 2. MPC
    mgr.set_conditions(P_load, P_pot, reset_setpoints=True, solve=False)
    soc_setter(mgr)
    mgr.solve()
    try:
        mpc_ok, mpc_pen, mpc_action = try_mpc(mgr)
    except Exception as e:
        return {"class": "mpc_error", "reason": f"{type(e).__name__}: {e}",
                "default": default}
    mpc_post = violation_summary(mgr, cfg)
    mpc_info = {"recovered": mpc_ok, "penalty": mpc_pen,
                "post_violations": mpc_post["total"], "action": mpc_action}

    if not mpc_ok:
        return {"class": "mpc_unsolved", "default": default, "mpc": mpc_info}

    # 3. Single-action probe
    mgr.set_conditions(P_load, P_pot, reset_setpoints=True, solve=False)
    soc_setter(mgr)
    mgr.solve()
    try:
        single = try_single_actions(mgr, cfg, P_load, P_pot, soc_setter)
    except Exception:
        single = False
    return {
        "class": "single_action" if single else "multi_action",
        "default": default,
        "mpc": mpc_info,
        "single_action_solvable": single,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=25)
    parser.add_argument("--output", default="mined_scenarios_v2.json")
    args = parser.parse_args()

    cfg = build_anm_domain_config()

    total_candidates = args.n_seeds * len(LOAD_MULTIPLIERS) * len(GEN_MULTIPLIERS) * len(SOC_PERTURBATIONS)
    print(f"Mining v2: {args.n_seeds} seeds × {len(LOAD_MULTIPLIERS)} load × "
          f"{len(GEN_MULTIPLIERS)} gen × {len(SOC_PERTURBATIONS)} soc = "
          f"{total_candidates} candidates")
    print()

    catalogue: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    type_counts = {"voltage": 0, "branch_loading": 0, "storage_soc": 0,
                   "multi_type": 0}
    t0 = time.time()
    next_print = 0

    for seed in range(args.n_seeds):
        for load_mul, gen_mul, soc_pert in itertools.product(
            LOAD_MULTIPLIERS, GEN_MULTIPLIERS, SOC_PERTURBATIONS
        ):
            cls = classify(seed, load_mul, gen_mul, soc_pert, cfg)
            counts[cls["class"]] = counts.get(cls["class"], 0) + 1

            # Track stress-type diversity on non-trivial entries
            d = cls.get("default")
            if d and d["total"] > 0:
                active = [k for k, n in d["by_checker"].items() if n > 0]
                if len(active) == 1:
                    type_counts[active[0]] = type_counts.get(active[0], 0) + 1
                elif len(active) > 1:
                    type_counts["multi_type"] += 1

            entry = {
                "source_seed": seed,
                "load_mul": load_mul,
                "gen_mul": gen_mul,
                "soc_pert": soc_pert,
                **cls,
            }
            catalogue.append(entry)

            if cls["class"] in ("multi_action", "mpc_unsolved"):
                d = cls.get("default", {})
                if d:
                    print(f"  seed={seed} load×{load_mul} gen×{gen_mul} "
                          f"soc={soc_pert:<9} -> {cls['class']:<12} "
                          f"viol={d['total']} {d['by_checker']} "
                          f"pen={d['penalty']:.2f}")

        done = (seed + 1) * len(LOAD_MULTIPLIERS) * len(GEN_MULTIPLIERS) * len(SOC_PERTURBATIONS)
        if done >= next_print:
            print(f"  [progress] seed {seed+1}/{args.n_seeds}, "
                  f"done {done}/{total_candidates}, "
                  f"counts={counts}, type_diversity={type_counts}")
            next_print = done + 50

    dt = time.time() - t0
    print(f"\n=== mined {len(catalogue)} candidates in {dt:.1f}s ===")
    print(f"  class counts: {counts}")
    print(f"  per-checker stress diversity (non-trivial entries): {type_counts}")

    with open(args.output, "w") as f:
        json.dump({"counts": counts, "stress_diversity": type_counts,
                   "catalogue": catalogue}, f, indent=2)
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
