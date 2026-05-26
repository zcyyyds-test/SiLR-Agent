"""Cross-check ANM scenario replay artifacts.

This is a CPU-only guardrail after mining/selection changes. It verifies that
the selected scenario JSON, default-state validation, and MPC baseline all
describe the same frozen operating points.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.anm_artifact_provenance import code_fingerprint, scenario_manifest


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> Any:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def close(a: float, b: float, tol: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="domains/anm/scenarios_mined.json")
    ap.add_argument("--validation", default="scenario_validation_v1.json")
    ap.add_argument("--mpc", default="mpc_baseline_v1.json")
    ap.add_argument("--output", default="scenario_consistency_v1.json")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    selected = load_json(args.scenarios).get("scenarios", [])
    validation = load_json(args.validation)
    mpc = load_json(args.mpc)

    selected_by_id = {r["id"]: r for r in selected}
    validation_by_id = {r["scenario_id"]: r for r in validation["rows"]}
    mpc_by_id = {r["scenario_id"]: r for r in mpc["results"]}

    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for sid, rec in selected_by_id.items():
        v = validation_by_id.get(sid)
        m = mpc_by_id.get(sid)
        if v is None or m is None:
            failures.append(f"{sid}: missing validation or MPC row")
            continue

        validation_violations = sum(
            c["n_violations"] for c in v["checkers"].values() if not c["passed"]
        )
        checks = {
            "validation_matches_mpc_default_penalty": close(
                v["gym_anm_penalty"], m["default_penalty"], args.tol
            ),
            "selected_matches_mpc_default_penalty": close(
                rec["default_penalty"], m["default_penalty"], args.tol
            ),
            "selected_matches_validation_violation_count": (
                int(rec["default_violation_count"]) == validation_violations
            ),
        }

        if rec.get("mpc_penalty") is not None:
            checks["selected_matches_mpc_penalty"] = close(
                rec["mpc_penalty"], m["mpc_penalty"], args.tol
            )
        if rec.get("mpc_post_violations") is not None:
            checks["selected_matches_mpc_post_violations"] = (
                int(rec["mpc_post_violations"]) == int(m["mpc_violations"])
            )
        if rec.get("mpc_recovered") is not None:
            checks["selected_matches_mpc_recovered"] = (
                bool(rec["mpc_recovered"]) == (int(m["mpc_violations"]) == 0)
            )

        ok = all(checks.values())
        if not ok:
            bad = [name for name, passed in checks.items() if not passed]
            failures.append(f"{sid}: {', '.join(bad)}")

        rows.append(
            {
                "scenario_id": sid,
                "class": rec.get("class"),
                "soc_pert": rec.get("soc_pert"),
                "initial_soc": rec.get("initial_soc"),
                "selected_default_penalty": rec["default_penalty"],
                "validation_default_penalty": v["gym_anm_penalty"],
                "mpc_default_penalty": m["default_penalty"],
                "selected_mpc_penalty": rec.get("mpc_penalty"),
                "mpc_penalty": m["mpc_penalty"],
                "checks": checks,
                "ok": ok,
            }
        )

    out = {
        "config": vars(args),
        "scenario_manifest": scenario_manifest(selected_by_id.keys()),
        "code_fingerprint": code_fingerprint(
            extra_paths=(
                "scripts/anm_scenario_consistency.py",
                args.scenarios,
                args.validation,
                args.mpc,
            )
        ),
        "n_selected": len(selected_by_id),
        "n_failures": len(failures),
        "failures": failures,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Checked {len(rows)} mined scenarios")
    for row in rows:
        status = "OK" if row["ok"] else "FAIL"
        print(
            f"{status} {row['scenario_id']}: "
            f"default={row['mpc_default_penalty']:.4f}, "
            f"mpc={row['mpc_penalty']:.4f}, soc={row['initial_soc']}"
        )
    print(f"Wrote {args.output}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("SCENARIO CONSISTENCY OK")


if __name__ == "__main__":
    main()
