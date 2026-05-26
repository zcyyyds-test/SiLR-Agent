"""Generate narrative-facing ANM tables from evaluation artifacts.

The full report is deliberately exhaustive. This script creates the smaller
tables that the paper story needs: the headline multi-action comparison,
hard-case containment ratios, and selected trajectory traces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anm_eval_report import (  # noqa: E402
    build_cell_rows,
    group_rows,
    mpc_index,
    scenario_index,
)


POLICY_ORDER = ["OFF", "terminal", "scalar_progress", "progress", "progress_mag"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a JSON object")
    return data


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def policy_sort(policy: str) -> int:
    try:
        return POLICY_ORDER.index(policy)
    except ValueError:
        return len(POLICY_ORDER)


def trajectory_points(ep: dict[str, Any]) -> list[float]:
    trace = ep.get("step_trace") or []
    if not trace:
        return []
    points = [trace[0].get("pre_penalty")]
    points.extend(item.get("post_penalty") for item in trace)
    return [float(p) for p in points if p is not None]


def arrow_trace(points: list[float]) -> str:
    return " -> ".join(fmt(p) for p in points)


def write_story_tables(
    output: Path,
    eval_json: Path,
    mpc_json: Path | None,
    trajectory_json: Path | None,
) -> None:
    data = load_json(eval_json)
    cells = build_cell_rows(data, scenario_index(), mpc_index(mpc_json))
    family_rows = group_rows(cells, "family")

    by_family_policy = {
        (row["family"], row["policy"]): row
        for row in family_rows
    }

    lines: list[str] = []
    lines.append("# ANM Story Tables")
    lines.append("")
    lines.append(f"Source: `{eval_json.name}`")
    lines.append("")

    lines.append("## Headline: Mined Multi-Action")
    lines.append("")
    rows = []
    for policy in POLICY_ORDER:
        row = by_family_policy.get(("mined_multi_action", policy))
        if not row:
            continue
        rows.append([
            policy,
            f"{row['successes']}/{row['n']}",
            fmt(row["recovery_rate"]),
            fmt(row["final_penalty_mean"]),
            fmt(row["worsening_rate"]),
        ])
    lines.extend(md_table(
        ["policy", "recovery", "rate", "penalty mean", "worse"],
        rows,
    ))
    lines.append("")
    lines.append(
        "Narrative use: binary terminal gating deadlocks; scalar_progress tests "
        "whether a single penalty threshold is enough; structured graded verdicts "
        "restore liveness, with progress_mag giving the cleanest safety profile."
    )
    lines.append("")

    lines.append("## Hard Residual: Damage Containment")
    lines.append("")
    off = by_family_policy.get(("mined_mpc_residual", "OFF"))
    off_penalty = float(off["final_penalty_mean"]) if off else float("nan")
    rows = []
    for policy in POLICY_ORDER:
        row = by_family_policy.get(("mined_mpc_residual", policy))
        if not row:
            continue
        penalty = float(row["final_penalty_mean"])
        containment = (
            (off_penalty - penalty) / off_penalty
            if off_penalty and not math.isnan(off_penalty)
            else float("nan")
        )
        rows.append([
            policy,
            f"{row['successes']}/{row['n']}",
            fmt(penalty),
            fmt(row["worsening_rate"]),
            fmt(containment),
        ])
    lines.extend(md_table(
        ["policy", "recovery", "penalty mean", "worse", "containment vs OFF"],
        rows,
    ))
    lines.append("")
    lines.append(
        "Narrative use: do not claim MPC dominance. Use this as an honest "
        "limitation plus containment result."
    )
    lines.append("")

    if trajectory_json and trajectory_json.exists():
        traj = load_json(trajectory_json)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for ep in traj.get("episodes", []):
            grouped[(ep.get("scenario", ""), ep.get("policy", ""))].append(ep)

        lines.append("## Trajectory Snippets")
        lines.append("")
        preferred = [
            ("mined_multi_action_3_l0p25g1p0_s12", "progress"),
            ("mined_multi_action_3_l0p25g1p0_s12", "progress_mag"),
            ("mined_mpc_unsolved_2_l2p0g1p0_s20", "progress"),
            ("mined_mpc_unsolved_2_l2p0g1p0_s20", "progress_mag"),
        ]
        rows = []
        for key in preferred:
            for ep in sorted(grouped.get(key, []), key=lambda e: int(e.get("rep_seed", 0))):
                points = trajectory_points(ep)
                if not points:
                    continue
                rows.append([
                    key[0],
                    key[1],
                    str(ep.get("rep_seed")),
                    str(bool(ep.get("recovered"))),
                    arrow_trace(points),
                ])
        if rows:
            lines.extend(md_table(
                ["scenario", "policy", "seed", "recovered", "penalty trace"],
                rows,
            ))
        else:
            lines.append("- No selected trajectory snippets available yet.")
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", type=Path, required=True)
    ap.add_argument("--mpc-json", type=Path, default=None)
    ap.add_argument("--trajectory-json", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    write_story_tables(
        output=args.output,
        eval_json=args.eval_json,
        mpc_json=args.mpc_json,
        trajectory_json=args.trajectory_json,
    )


if __name__ == "__main__":
    main()
