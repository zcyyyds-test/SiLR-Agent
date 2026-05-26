"""MPC baseline on the ANM scenario library — external policy reference.

Runs gym-anm's built-in :class:`MPCAgentConstant` (Henry & Ernst 2021, Eq. 4.8)
on each ANM scenario and compares against the "no-control" baseline (default
set-points = renewables at full P_pot, storage idle).

Why this script exists (paper context):
  - one of the WISE prior-panel hard-issues was "no external baseline". MPC is
    free (ships with gym-anm), DC-OPF-based (peer-reviewed approximation), and
    is the natural non-LLM control reference for ANM.
  - we report per scenario: default penalty / violations vs MPC penalty /
    violations, plus the MPC action vector for sanity-checking what it did.
  - the SiLR-gated agent's eval will be compared against the SAME default and
    MPC baselines on the SAME scenario library — so the numbers in this script
    are the "external" column in the eval table.

Run from repo root (AMD ``silr-anm`` env):

    PYTHONPATH=. python scripts/anm_mpc_baseline.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from domains.anm import (
    ANMScenarioLoader,
    GymANMManager,
    build_anm_domain_config,
)
from gym_anm import MPCAgentConstant
from scripts.anm_artifact_provenance import code_fingerprint, scenario_manifest


@dataclass
class ScenarioResult:
    scenario_id: str
    difficulty: str
    default_converged: bool
    default_penalty: float
    default_reward: float
    default_violations: int
    default_breakdown: dict[str, int]
    mpc_converged: bool
    mpc_penalty: float
    mpc_reward: float
    mpc_violations: int
    mpc_breakdown: dict[str, int]
    mpc_action: list[float]


def _checker_status(mgr: GymANMManager, cfg: Any) -> tuple[int, dict[str, int]]:
    """Run all configured checkers; return (total violations, per-checker count)."""
    breakdown: dict[str, int] = {}
    total = 0
    for checker in cfg.checkers:
        cr = checker.check(mgr.system_state, mgr.base_mva)
        n = 0 if cr.passed else len(cr.violations)
        breakdown[checker.name] = n
        total += n
    return total, breakdown


def _unpack_mpc_action(mgr: GymANMManager, action: np.ndarray) -> tuple[dict[int, float], dict[int, float]]:
    """Unpack the env action vector into ``P_set`` / ``Q_set`` dicts.

    Layout (mirrors ``ANMEnv.step``):
        [P_set_gen..., Q_set_gen..., P_set_des..., Q_set_des...]
    Device order follows ``simulator.devices.items()`` iteration, which is the
    same as ``manager._gen_ids`` / ``manager._des_ids`` (both filter that dict).
    """
    n_gen = len(mgr._gen_ids)
    n_des = len(mgr._des_ids)
    expected = 2 * n_gen + 2 * n_des
    if action.shape != (expected,):
        raise AssertionError(
            f"MPC action shape {action.shape} != expected ({expected},); "
            f"n_gen={n_gen}, n_des={n_des}"
        )
    P_set: dict[int, float] = {}
    Q_set: dict[int, float] = {}
    for a, did in zip(action[:n_gen], mgr._gen_ids):
        P_set[int(did)] = float(a)
    for a, did in zip(action[n_gen : 2 * n_gen], mgr._gen_ids):
        Q_set[int(did)] = float(a)
    for a, did in zip(action[2 * n_gen : 2 * n_gen + n_des], mgr._des_ids):
        P_set[int(did)] = float(a)
    for a, did in zip(action[2 * n_gen + n_des :], mgr._des_ids):
        Q_set[int(did)] = float(a)
    return P_set, Q_set


def run_scenario(
    scenario,
    cfg,
    planning_steps: int = 8,
    safety_margin: float = 0.9,
    seed_fallback: int = 42,
) -> ScenarioResult:
    seed = scenario.source_seed if scenario.source_seed is not None else seed_fallback
    mgr = GymANMManager(seed=seed)
    loader = ANMScenarioLoader()
    default_converged = loader.setup_episode(mgr, scenario, solve=True)
    default_penalty = mgr.last_penalty
    default_reward = mgr.last_reward
    default_total, default_breakdown = _checker_status(mgr, cfg)

    env = mgr._env
    agent = MPCAgentConstant(
        simulator=env.simulator,
        action_space=env.action_space,
        gamma=env.gamma,
        safety_margin=safety_margin,
        planning_steps=planning_steps,
    )
    action = np.asarray(agent.act(env), dtype=float)

    P_set, Q_set = _unpack_mpc_action(mgr, action)
    mgr._P_set = P_set
    mgr._Q_set = Q_set
    mpc_converged = mgr.solve()
    mpc_penalty = mgr.last_penalty
    mpc_reward = mgr.last_reward
    mpc_total, mpc_breakdown = _checker_status(mgr, cfg)

    return ScenarioResult(
        scenario_id=scenario.id,
        difficulty=scenario.difficulty,
        default_converged=default_converged,
        default_penalty=default_penalty,
        default_reward=default_reward,
        default_violations=default_total,
        default_breakdown=default_breakdown,
        mpc_converged=mpc_converged,
        mpc_penalty=mpc_penalty,
        mpc_reward=mpc_reward,
        mpc_violations=mpc_total,
        mpc_breakdown=mpc_breakdown,
        mpc_action=[round(float(x), 4) for x in action.tolist()],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None, help="Optional JSON output path.")
    ap.add_argument("--scenarios", nargs="+", default=None,
                    help="Optional subset of scenario ids to run.")
    ap.add_argument("--planning-steps", type=int, default=8)
    ap.add_argument("--safety-margin", type=float, default=0.9)
    args = ap.parse_args()

    cfg = build_anm_domain_config()
    loader = ANMScenarioLoader()
    scenarios = (
        [loader.load(sid) for sid in args.scenarios]
        if args.scenarios is not None
        else loader.load_all()
    )

    print(f"MPC baseline on {len(scenarios)} ANM scenarios")
    print(
        "  agent: MPCAgentConstant("
        f"planning_steps={args.planning_steps}, safety_margin={args.safety_margin})"
    )
    print()

    results: list[ScenarioResult] = []
    for s in scenarios:
        print(f"=== {s.id} [{s.difficulty}] ===")
        r = run_scenario(
            s,
            cfg,
            planning_steps=args.planning_steps,
            safety_margin=args.safety_margin,
        )
        results.append(r)
        print(
            f"  default : converged={r.default_converged}  "
            f"penalty={r.default_penalty:8.3f}  reward={r.default_reward:8.3f}  "
            f"viol={r.default_violations} {r.default_breakdown}"
        )
        print(
            f"  MPC     : converged={r.mpc_converged}  "
            f"penalty={r.mpc_penalty:8.3f}  reward={r.mpc_reward:8.3f}  "
            f"viol={r.mpc_violations} {r.mpc_breakdown}"
        )
        print(f"  action  : {r.mpc_action}")
        print()

    # --- summary table ---
    print(f"{'scenario':<30} {'diff':<7} {'def viol':>9} {'mpc viol':>9} "
          f"{'def pen':>10} {'mpc pen':>10} {'Δ pen':>10}")
    print("-" * 100)
    for r in results:
        dpen, mpen = r.default_penalty, r.mpc_penalty
        delta = dpen - mpen
        print(
            f"{r.scenario_id:<30} {r.difficulty:<7} "
            f"{r.default_violations:>9} {r.mpc_violations:>9} "
            f"{dpen:>10.3f} {mpen:>10.3f} {delta:>10.3f}"
        )

    # --- assertions: invariants we expect on the current library ---
    by_id = {r.scenario_id: r for r in results}

    # MPC must not break an already-feasible snapshot.
    easy = by_id["easy_lightload"]
    assert easy.mpc_converged, "MPC produced infeasible action on easy snapshot"
    assert easy.mpc_violations == 0, (
        f"MPC broke easy_lightload: viol {easy.default_violations} -> {easy.mpc_violations}"
    )

    # MPC should at minimum not regress penalty on stressed snapshots (DC-OPF
    # is an approximation: it may not fully eliminate violations, but it must
    # not make things strictly worse than no-control).
    for sid in ("medium_seed42_default", "hard_renewable_surge"):
        r = by_id[sid]
        assert r.mpc_converged, f"MPC infeasible on {sid}"
        assert r.mpc_penalty <= r.default_penalty + 1e-3, (
            f"MPC regressed on {sid}: penalty "
            f"{r.default_penalty:.3f} -> {r.mpc_penalty:.3f}"
        )

    if args.output:
        out = {
            "config": {
                "agent": "MPCAgentConstant",
                "planning_steps": args.planning_steps,
                "safety_margin": args.safety_margin,
            },
            "scenario_manifest": scenario_manifest([s.id for s in scenarios]),
            "code_fingerprint": code_fingerprint(extra_paths=("scripts/anm_mpc_baseline.py",)),
            "results": [asdict(r) for r in results],
        }
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")

    print("\nMPC BASELINE OK")


if __name__ == "__main__":
    main()
