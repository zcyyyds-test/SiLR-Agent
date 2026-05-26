"""Measure ANM solver penalty jitter on repeated solves of the same state.

This tests Mimo's claim that ε calibration for a magnitude-aware admission
predicate is "domain-breaking" because Newton-Raphson jitter is in the
0.01-0.1 range. We solve the same state N times via deepcopy + solve and
report (last_penalty, total_violation_severity_score) statistics.

Sanity output:
  per-scenario:
    last_penalty:    mean / std / min / max / range
    severity_score:  mean / std / min / max / range
  proposed admission ε candidates (5×σ heuristic) and whether they admit
  an adversary's 0.5 worsening drift step (toy threat from the adversarial
  sweep finding).

Usage:
  python scripts/anm_solver_jitter.py --reps 100
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path

from domains.anm.config import build_anm_domain_config
from domains.anm.manager import GymANMManager
from domains.anm.scenarios import ANMScenarioLoader


def _severity_score(check_results) -> float:
    """Domain-agnostic magnitude proxy: sum of |value - limit| over violations."""
    score = 0.0
    for cr in check_results:
        for v in cr.violations:
            try:
                val = float(v.value)
                lim = float(v.limit)
            except (TypeError, ValueError):
                score += 1.0
                continue
            if not (math.isfinite(val) and math.isfinite(lim)):
                score += 1000.0
                continue
            score += abs(val - lim)
    return score


def jitter_one_scenario(scenario_id: str, reps: int) -> dict:
    loader = ANMScenarioLoader()
    domain_cfg = build_anm_domain_config()
    checkers = list(domain_cfg.checkers)

    manager_seed = 42
    manager = GymANMManager(seed=manager_seed)
    scenario = loader.load(scenario_id)
    loader.setup_episode(manager, scenario)

    pens = []
    scores = []
    for _ in range(reps):
        shadow = manager.create_shadow_copy()
        ok = shadow.solve()
        pen = shadow.last_penalty if ok else float("nan")
        checks = [c.check(shadow.system_state, shadow.base_mva) for c in checkers]
        score = _severity_score(checks)
        if math.isfinite(pen):
            pens.append(pen)
        scores.append(score)
        del shadow

    def _stats(name, xs):
        if not xs:
            return {"n": 0}
        m = statistics.mean(xs)
        sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return {
            "n": len(xs),
            "mean": round(m, 6),
            "std": round(sd, 6),
            "min": round(min(xs), 6),
            "max": round(max(xs), 6),
            "range": round(max(xs) - min(xs), 6),
        }

    return {
        "scenario": scenario_id,
        "manager_seed": manager_seed,
        "reps": reps,
        "last_penalty": _stats("pen", pens),
        "severity_score": _stats("score", scores),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "easy_lightload",
            "medium_seed42_default",
            "hard_renewable_surge",
        ],
    )
    ap.add_argument("--output", default="solver_jitter_v1.json")
    args = ap.parse_args()

    print(f"[jitter] reps={args.reps} scenarios={args.scenarios}")
    out = {"reps": args.reps, "scenarios": []}
    for sid in args.scenarios:
        print(f"[jitter] scenario={sid}")
        rec = jitter_one_scenario(sid, args.reps)
        out["scenarios"].append(rec)
        print(json.dumps(rec, indent=2))

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"[jitter] wrote {args.output}")

    # ε feasibility analysis
    print("\n[jitter] ε-feasibility analysis:")
    print("  threshold heuristic: ε = max(5·σ, 0.001) on last_penalty")
    print("  attack budget: an adversary can drift Δ/step; recovery sweeps showed Δ ≈ 0.5 between steps")
    print("  if ε >= 0.5: every adversarial step is admitted (vulnerability survives)")
    print("  if ε < 0.5 but >= 5·σ: guard catches drift while tolerating jitter")
    for rec in out["scenarios"]:
        sigma = rec["last_penalty"].get("std", 0.0) or 0.0
        eps = max(5 * sigma, 0.001)
        decisive = "✗ FAILS (jitter > attack drift)" if eps >= 0.5 else "✓ OK"
        print(f"  {rec['scenario']:30s} σ={sigma:.4e} → 5σ={5*sigma:.4f}, ε={eps:.4f}, {decisive}")


if __name__ == "__main__":
    main()
