"""Trajectory-level offline analysis for binary vs graded GRPO reward.

This script does NOT require a GPU or LLM call. It re-analyzes existing
episode-level JSON results (`eval_sweep_v1.json`, `eval_mined_v1.json`,
`adversarial_v1.json`) to demonstrate that a graded verifier-derived
reward signal — `PASS=1, SAFE_PROGRESS=0.5, FAIL=0` — gives non-degenerate
GRPO-style advantages where the binary alternative (`PASS=1, else 0`)
collapses to all-zero or all-one within a group, killing the training
signal.

Paper Table-4 raw material: "advantage distribution under binary vs
graded reward for each scenario × policy cell".
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _reward_binary(tally: dict, recovered: bool) -> float:
    """Episode-level binary reward: 1 iff recovered to PASS, else 0."""
    return 1.0 if recovered else 0.0


def _reward_graded_episode(tally: dict, recovered: bool) -> float:
    """Episode-level graded reward.

    Each verdict contributes a weight, averaged over total verifier calls:
        PASS = 1.0
        SAFE_PROGRESS = 0.5
        FAIL = 0.0
        ERROR = excluded (does not count toward denominator)

    Then a terminal bonus of +0.5 is added if recovered (so a fully
    recovered episode > a partially recovering episode > a stalled one).
    """
    p = tally.get("PASS", 0)
    sp = tally.get("SAFE_PROGRESS", 0)
    f = tally.get("FAIL", 0)
    denom = p + sp + f
    if denom == 0:
        # No verifier calls: only happens on the trivial OFF path on
        # easy_lightload where the system is already stable.
        return 1.0 if recovered else 0.0
    step_avg = (p * 1.0 + sp * 0.5 + f * 0.0) / denom
    return step_avg + (0.5 if recovered else 0.0)


def _group_advantage(rewards: list[float]) -> list[float]:
    """GRPO-style group advantage: (r_i - mean) / std (or 0 if std=0)."""
    if not rewards:
        return []
    mu = statistics.mean(rewards)
    sd = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    if sd == 0.0:
        return [0.0] * len(rewards)
    return [(r - mu) / sd for r in rewards]


def _is_degenerate(advs: list[float]) -> bool:
    """A group's advantage is degenerate iff every element ≈ 0
    (group-mean reward was identical across reps → 0 training signal)."""
    return all(abs(a) < 1e-9 for a in advs)


def analyze(json_paths: list[Path]) -> dict:
    # group_key = (scenario, policy) → list of (rep_seed, tally, recovered)
    groups: dict[tuple, list] = defaultdict(list)
    for p in json_paths:
        if not p.exists():
            print(f"  [skip] {p} not found")
            continue
        data = json.loads(p.read_text())
        eps = data.get("episodes", [])
        # adversarial json uses "attack" — fold into policy as "progress/attack"
        for ep in eps:
            scn = ep.get("scenario", "?")
            pol = ep.get("policy", ep.get("attack", "?"))
            key = (scn, str(pol))
            groups[key].append({
                "rep_seed": ep.get("rep_seed"),
                "tally": ep.get("verdict_tally", {}) or {},
                "recovered": bool(ep.get("recovered", False)),
                "final_penalty": float(ep.get("final_penalty", 0.0)),
                "source": p.name,
            })

    summary = []
    for key, eps in sorted(groups.items()):
        if len(eps) < 2:
            continue  # need ≥2 reps for group advantage
        r_bin = [_reward_binary(e["tally"], e["recovered"]) for e in eps]
        r_gra = [_reward_graded_episode(e["tally"], e["recovered"]) for e in eps]
        a_bin = _group_advantage(r_bin)
        a_gra = _group_advantage(r_gra)
        summary.append({
            "scenario": key[0],
            "policy": key[1],
            "n_reps": len(eps),
            "r_binary": {
                "mean": round(statistics.mean(r_bin), 3),
                "std": round(statistics.pstdev(r_bin), 3) if len(r_bin) > 1 else 0.0,
                "all": r_bin,
            },
            "r_graded": {
                "mean": round(statistics.mean(r_gra), 3),
                "std": round(statistics.pstdev(r_gra), 3) if len(r_gra) > 1 else 0.0,
                "all": [round(r, 3) for r in r_gra],
            },
            "advantage_binary_degenerate": _is_degenerate(a_bin),
            "advantage_graded_degenerate": _is_degenerate(a_gra),
            "adv_bin_range": round(max(a_bin) - min(a_bin), 3) if a_bin else 0,
            "adv_gra_range": round(max(a_gra) - min(a_gra), 3) if a_gra else 0,
            "source": eps[0]["source"],
        })
    return {
        "n_groups_total": len(groups),
        "n_groups_analyzed": len(summary),
        "summary": summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=[
        "eval_sweep_v1.json",
        "eval_mined_v1.json",
        "adversarial_v1.json",
        "adversarial_stallmit_v1.json",
    ])
    ap.add_argument("--output", default="trajectory_offline_v1.json")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    result = analyze(paths)

    # Headline stat: in how many groups did graded reward give non-degenerate
    # advantage while binary collapsed to zero?
    rescued = sum(
        1 for s in result["summary"]
        if s["advantage_binary_degenerate"] and not s["advantage_graded_degenerate"]
    )
    both_deg = sum(
        1 for s in result["summary"]
        if s["advantage_binary_degenerate"] and s["advantage_graded_degenerate"]
    )
    neither = sum(
        1 for s in result["summary"]
        if not s["advantage_binary_degenerate"]
    )

    print(f"=== Trajectory offline GRPO-advantage analysis ===")
    print(f"  groups total / with ≥2 reps : {result['n_groups_total']} / {result['n_groups_analyzed']}")
    print(f"  binary degenerate, graded saved signal : {rescued}/{result['n_groups_analyzed']} ← key paper number")
    print(f"  both degenerate (same reward across reps): {both_deg}")
    print(f"  neither degenerate (binary signal already non-zero): {neither}")
    print()
    print("=== per-group (sorted by rescue impact) ===")
    for s in sorted(
        result["summary"],
        key=lambda x: (
            not (x["advantage_binary_degenerate"] and not x["advantage_graded_degenerate"]),
            -x["adv_gra_range"],
        ),
    ):
        flag = "RESCUE" if (s["advantage_binary_degenerate"] and not s["advantage_graded_degenerate"]) else ""
        print(f"  {s['scenario']:<35} pol={s['policy']:<14} n={s['n_reps']} "
              f"r_bin μ={s['r_binary']['mean']:.2f} σ={s['r_binary']['std']:.2f} | "
              f"r_gra μ={s['r_graded']['mean']:.3f} σ={s['r_graded']['std']:.3f} "
              f"| Δadv_bin={s['adv_bin_range']:.2f} Δadv_gra={s['adv_gra_range']:.2f} {flag}")

    out = {
        "rescued": rescued,
        "both_degenerate": both_deg,
        "neither_degenerate": neither,
        **result,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
