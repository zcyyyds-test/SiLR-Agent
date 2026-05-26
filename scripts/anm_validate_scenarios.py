"""Validate ANM scenarios: load each, apply, run the verifier's checkers on the
default state, report the stress profile.

What this answers, end to end:
  - does ``ANMScenarioLoader.setup_episode`` actually configure the manager
    (no silent key-mismatch / divergence)?
  - what is the *default-state* status of each scenario (PASS = trivially safe,
    FAIL = stress to recover from)?
  - which checker(s) violate? — useful to ensure scenarios exercise voltage,
    branch loading, and SoC each at least once across the library.

Run from repo root:

    PYTHONPATH=. python scripts/anm_validate_scenarios.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from domains.anm import (
    ANMScenarioLoader,
    GymANMManager,
    build_anm_domain_config,
)
from scripts.anm_artifact_provenance import code_fingerprint, scenario_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    cfg = build_anm_domain_config()
    loader = ANMScenarioLoader()
    scenarios = loader.load_all()
    print(f"Loaded {len(scenarios)} scenarios:")
    for s in scenarios:
        print(f"  - [{s.difficulty:6s}] {s.id} — {s.description}")
    print()

    # Reuse one manager for all scenarios — set_conditions just overwrites.
    mgr = GymANMManager(seed=0)

    rows = []
    for s in scenarios:
        converged = loader.setup_episode(mgr, s, solve=True)
        # Run all configured checkers on the default state.
        violations_by_checker = {}
        any_violation = False
        for checker in cfg.checkers:
            cr = checker.check(mgr.system_state, mgr.base_mva)
            violations_by_checker[checker.name] = {
                "passed": cr.passed,
                "summary": cr.summary,
                "n_violations": len(cr.violations),
                "details": [v.detail for v in cr.violations[:3]],
            }
            if not cr.passed:
                any_violation = True

        rows.append(
            {
                "scenario_id": s.id,
                "difficulty": s.difficulty,
                "converged": converged,
                "default_state": "FAIL" if any_violation else "PASS",
                "checkers": violations_by_checker,
                "gym_anm_penalty": round(mgr.last_penalty, 4),
                "gym_anm_reward": round(mgr.last_reward, 4),
            }
        )

    # --- print compact summary ---
    print(f"{'id':<28} {'diff':<7} {'conv':<6} {'default':<8} {'penalty':<8}  violations")
    print("-" * 100)
    for row in rows:
        viols = [
            f"{name}({info['n_violations']})"
            for name, info in row["checkers"].items()
            if not info["passed"]
        ]
        viol_str = " ".join(viols) if viols else "(none)"
        print(
            f"{row['scenario_id']:<28} "
            f"{row['difficulty']:<7} "
            f"{str(row['converged']):<6} "
            f"{row['default_state']:<8} "
            f"{row['gym_anm_penalty']:<8} "
            f"{viol_str}"
        )

    print()
    # --- assertions: invariants the library should satisfy ---
    ids = [r["scenario_id"] for r in rows]
    assert len(set(ids)) == len(ids), f"duplicate scenario ids: {ids}"
    by_diff = {d: [r for r in rows if r["difficulty"] == d] for d in {"easy", "medium", "hard"}}
    assert by_diff["easy"], "library must include at least one easy (PASS) scenario"
    # at least one stress scenario:
    assert any(r["default_state"] == "FAIL" for r in rows), (
        "library must include at least one FAIL stress scenario for SiLR to gate"
    )
    # all scenarios should solver-converge (else the scenario is broken, not stressed)
    for r in rows:
        assert r["converged"], (
            f"scenario {r['scenario_id']} did not converge — broken, not stressed"
        )
    if args.output:
        out = {
            "n_scenarios": len(rows),
            "scenario_manifest": scenario_manifest([s.id for s in scenarios]),
            "code_fingerprint": code_fingerprint(extra_paths=("scripts/anm_validate_scenarios.py",)),
            "rows": rows,
        }
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    print("SCENARIO VALIDATION OK")


if __name__ == "__main__":
    main()
