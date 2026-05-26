"""Report whether structured progress admission beats a scalar penalty gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
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


POLICIES = ["terminal", "scalar_progress", "progress", "progress_mag"]
FAMILIES = ["mined_multi_action", "mined_mpc_residual"]


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


def policy_row(row: dict[str, Any]) -> list[str]:
    return [
        row["policy"],
        f"{row['successes']}/{row['n']}",
        fmt(row["recovery_rate"]),
        fmt(row["final_penalty_mean"]),
        fmt(row["worsening_rate"]),
    ]


def contrast_verdict(family_rows: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    lines = []
    for family in FAMILIES:
        scalar = family_rows.get((family, "scalar_progress"))
        structured = family_rows.get((family, "progress_mag"))
        if not scalar or not structured:
            lines.append(f"- `{family}`: pending; scalar or progress_mag row missing.")
            continue
        scalar_rec = float(scalar["recovery_rate"])
        structured_rec = float(structured["recovery_rate"])
        scalar_pen = float(scalar["final_penalty_mean"])
        structured_pen = float(structured["final_penalty_mean"])
        rec_gap = structured_rec - scalar_rec
        pen_gap = scalar_pen - structured_pen
        if rec_gap >= 0.25 or pen_gap >= max(1.0, 0.25 * max(abs(scalar_pen), 1.0)):
            label = "supports structured admission"
        elif abs(rec_gap) <= 0.10 and abs(pen_gap) <= max(0.5, 0.10 * max(abs(scalar_pen), 1.0)):
            label = "weakens structured-admission claim"
        else:
            label = "mixed"
        lines.append(
            f"- `{family}`: {label}; "
            f"progress_mag - scalar recovery gap `{fmt(rec_gap)}`, "
            f"scalar - progress_mag penalty gap `{fmt(pen_gap)}`."
        )
    return lines


def warning_lines(data: dict[str, Any], cells: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    status = data.get("status")
    if isinstance(status, dict) and not status.get("complete", True):
        completed = status.get("completed_episodes")
        expected = status.get("expected_episodes")
        if completed != expected:
            warnings.append(f"Incomplete artifact: {completed}/{expected} episodes.")
        else:
            warnings.append("Artifact is marked incomplete; inspect merge source statuses.")
    for row in cells:
        if row["policy"] in POLICIES and row["family"] in FAMILIES and int(row["n"]) < 3:
            warnings.append(f"Cell {row['scenario']}__{row['policy']} has n={row['n']} (<3).")
    return warnings


def write_report(eval_json: Path, mpc_json: Path | None, output: Path) -> None:
    data = load_json(eval_json)
    cells = build_cell_rows(data, scenario_index(), mpc_index(mpc_json))
    family_rows_list = group_rows(cells, "family")
    family_rows = {
        (row["family"], row["policy"]): row
        for row in family_rows_list
    }
    cell_rows = {
        (row["scenario"], row["policy"]): row
        for row in cells
    }

    lines: list[str] = []
    lines.append("# Scalar vs Structured Admission Contrast")
    lines.append("")
    lines.append(f"Source: `{eval_json.name}`")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    warnings = warning_lines(data, cells)
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Decision Gate")
    lines.append("")
    lines.extend(contrast_verdict(family_rows))
    lines.append("")
    lines.append(
        "Use this conservatively: the claim is not that scalar gates are useless, "
        "but that post-violation recovery needs more than a one-dimensional "
        "non-worsening threshold on the challenging cases."
    )
    lines.append("")

    lines.append("## Family-Level Contrast")
    lines.append("")
    for family in FAMILIES:
        rows = []
        for policy in POLICIES:
            row = family_rows.get((family, policy))
            if row:
                rows.append(policy_row(row))
        lines.append(f"### `{family}`")
        lines.append("")
        lines.extend(md_table(
            ["policy", "recovery", "rate", "penalty mean", "worse"],
            rows,
        ))
        lines.append("")

    lines.append("## Scenario-Level Contrast")
    lines.append("")
    scenarios = sorted({
        row["scenario"]
        for row in cells
        if row["family"] in FAMILIES and row["policy"] in POLICIES
    })
    for scenario in scenarios:
        rows = []
        for policy in POLICIES:
            row = cell_rows.get((scenario, policy))
            if row:
                rows.append(policy_row(row))
        lines.append(f"### `{scenario}`")
        lines.append("")
        lines.extend(md_table(
            ["policy", "recovery", "rate", "penalty mean", "worse"],
            rows,
        ))
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True, type=Path)
    parser.add_argument("--mpc-json", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    write_report(args.eval_json, args.mpc_json, args.output)


if __name__ == "__main__":
    main()
