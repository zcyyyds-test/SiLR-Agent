"""Mine diverse stressed ANM6-Easy snapshots and classify their difficulty.

Why this exists (paper context):
  - the initial scenarios library has 3 hand-picked snapshots, all stressing
    branch-loading constraints only — panel reviewers flagged the lack of
    voltage- and SoC-driven stress, and the lack of "MPC-hard" / boundary
    cases that would let the paper claim a meaningful limitation.
  - this script sweeps the gym-anm stochastic-process snapshot space
    (env seed × time-step) plus a few synthetic perturbations, evaluates
    each candidate under default / single-action / MPC control, and bins
    them into a small taxonomy:

        trivial        default state is already feasible.
        single_action  some single set-point change clears all violations.
        multi_action   MPC clears it but no single action does
                       (the deadlock-inducing class).
        mpc_unsolved   MPC fails to reach a zero-violation state (boundary
                       case — paper limitation / future-work motivation).

  - the script produces a JSON catalogue that downstream eval can promote
    into the static ``scenarios.py`` library after manual inspection.

CPU-only (numpy/cvxpy/scipy), so it can run in parallel with vLLM serving.

Run from repo root (AMD ``silr-anm`` env):

    PYTHONPATH=. python scripts/anm_scenario_mine.py \\
        --n-seeds 40 --n-steps 6 --output mined_scenarios.json
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from domains.anm import GymANMManager, build_anm_domain_config
from gym_anm import MPCAgentConstant


def violation_summary(mgr, cfg) -> dict[str, Any]:
    """Run all checkers; return per-checker counts + total + penalty."""
    by_checker = {}
    total = 0
    for checker in cfg.checkers:
        cr = checker.check(mgr.system_state, mgr.base_mva)
        n = 0 if cr.passed else len(cr.violations)
        by_checker[checker.name] = n
        total += n
    return {"total": total, "by_checker": by_checker, "penalty": float(mgr.last_penalty)}


def try_mpc(mgr) -> tuple[bool, float, list[float]]:
    """Apply MPC's recommended action and return (recovered, penalty, action)."""
    env = mgr._env
    agent = MPCAgentConstant(
        simulator=env.simulator,
        action_space=env.action_space,
        gamma=env.gamma,
        safety_margin=0.9,
        planning_steps=8,
    )
    action = np.asarray(agent.act(env), dtype=float)
    # Layout: [P_gen, Q_gen, P_des, Q_des] matching manager._gen_ids / _des_ids order.
    n_gen = len(mgr._gen_ids)
    n_des = len(mgr._des_ids)
    P_set = {}
    Q_set = {}
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


def try_single_actions(mgr, cfg, P_load, P_pot) -> bool:
    """Sweep a coarse grid over each single set-point and check whether any
    single change clears all violations. Used to classify a snapshot as
    single-action-solvable vs multi-action-required."""
    # Coarse grid: 9 points per device covering full range.
    for gen_id in mgr._gen_ids:
        device = mgr._sim.devices[gen_id]
        p_max_mw = float(device.p_max) * mgr.base_mva
        p_max_mw = min(p_max_mw, float(P_pot.get(gen_id, p_max_mw)))
        grid = np.linspace(0.0, p_max_mw, 9)
        for p in grid:
            mgr.set_conditions(P_load, P_pot, reset_setpoints=True, solve=False)
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
            mgr._P_set[storage_id] = float(p)
            if mgr.solve() and violation_summary(mgr, cfg)["total"] == 0:
                return True
    return False


def classify_snapshot(mgr, cfg, P_load, P_pot) -> dict[str, Any]:
    """Run default / MPC / single-action probes and bin the snapshot."""
    # 1. Default state
    mgr.set_conditions(P_load, P_pot, reset_setpoints=True, solve=True)
    default = violation_summary(mgr, cfg)

    if default["total"] == 0:
        return {"class": "trivial", "default": default, "mpc": None,
                "single_action_solvable": None}

    # 2. MPC
    mgr.set_conditions(P_load, P_pot, reset_setpoints=True, solve=True)
    mpc_ok, mpc_pen, mpc_action = try_mpc(mgr)
    mpc_post = violation_summary(mgr, cfg)
    mpc_info = {"recovered": mpc_ok, "penalty": mpc_pen,
                "post_violations": mpc_post["total"], "action": mpc_action}

    if not mpc_ok:
        return {"class": "mpc_unsolved", "default": default, "mpc": mpc_info,
                "single_action_solvable": None}

    # 3. Single-action probe (only if MPC succeeded — otherwise we know it's hard)
    single = try_single_actions(mgr, cfg, P_load, P_pot)

    return {
        "class": "single_action" if single else "multi_action",
        "default": default,
        "mpc": mpc_info,
        "single_action_solvable": single,
    }


def sample_natural(mgr_factory, cfg, seed: int, step: int) -> tuple[dict, dict] | None:
    """Pull a snapshot from gym-anm's natural stochastic process at (seed, step)."""
    try:
        mgr = mgr_factory(seed=seed)
    except Exception:
        return None
    # Advance the env step times to reach the snapshot.
    for _ in range(step):
        mgr.step(reset_setpoints=True)
    P_load = dict(mgr._P_load)
    P_pot = dict(mgr._P_pot)
    return P_load, P_pot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=6,
                        help="For each seed, sample snapshots at step 0..N-1.")
    parser.add_argument("--output", default="mined_scenarios.json")
    parser.add_argument("--limit-multi", type=int, default=30,
                        help="Stop sampling once we have at least this many "
                             "multi_action and at least one mpc_unsolved.")
    args = parser.parse_args()

    cfg = build_anm_domain_config()

    catalogue: list[dict[str, Any]] = []
    counts: dict[str, int] = {"trivial": 0, "single_action": 0,
                              "multi_action": 0, "mpc_unsolved": 0}
    t0 = time.time()

    print(f"Mining ANM6-Easy stress snapshots (n_seeds={args.n_seeds}, "
          f"n_steps={args.n_steps}, ~{args.n_seeds * args.n_steps} candidates) ...")

    for seed in range(args.n_seeds):
        for step in range(args.n_steps):
            sample = sample_natural(GymANMManager, cfg, seed, step)
            if sample is None:
                continue
            P_load, P_pot = sample
            mgr = GymANMManager(seed=seed)
            for _ in range(step):
                mgr.step(reset_setpoints=True)
            cls = classify_snapshot(mgr, cfg, P_load, P_pot)
            counts[cls["class"]] += 1
            entry = {
                "source_seed": seed,
                "source_step": step,
                "P_load": {str(k): v for k, v in P_load.items()},
                "P_pot": {str(k): v for k, v in P_pot.items()},
                **cls,
            }
            catalogue.append(entry)
            if cls["class"] != "trivial":
                print(f"  seed={seed} step={step} -> {cls['class']:<14} "
                      f"default_viol={cls['default']['total']} "
                      f"by_checker={cls['default']['by_checker']} "
                      f"penalty={cls['default']['penalty']:.2f}")
            # Early-exit: enough multi_action and at least one mpc_unsolved
            if counts["multi_action"] >= args.limit_multi and counts["mpc_unsolved"] >= 1:
                break
        else:
            continue
        break

    dt = time.time() - t0
    print(f"\n=== mined {len(catalogue)} candidates in {dt:.1f}s ===")
    print(f"  class counts: {counts}")
    print(f"  stress diversity (non-trivial):")
    type_breakdown = {"voltage": 0, "branch_loading": 0, "storage_soc": 0}
    for e in catalogue:
        if e["class"] == "trivial":
            continue
        for ck, n in e["default"]["by_checker"].items():
            if n > 0:
                type_breakdown[ck] = type_breakdown.get(ck, 0) + 1
    print(f"  per-checker presence (entries with ≥1 viol of that type): {type_breakdown}")

    with open(args.output, "w") as f:
        json.dump({"counts": counts, "diversity": type_breakdown,
                   "catalogue": catalogue}, f, indent=2)
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
