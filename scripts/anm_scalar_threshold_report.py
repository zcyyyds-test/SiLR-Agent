"""Summarize scalar-progress threshold-sensitivity runs.

Inputs are JSON artifacts from ``scripts/anm_eval_sweep.py`` with
``scalar_progress`` only, typically produced by
``scripts/run_amd_scalar_threshold_multi_action_gpu0.bat``.

Example:
    python3 scripts/anm_scalar_threshold_report.py \\
      --inputs \\
        0p00=eval_scalar_threshold_0p00_gpu0.json \\
        0p05=eval_scalar_threshold_0p05_gpu0.json \\
      --reference-inputs progress_mag=eval_mined_refresh_plus_n5_gpu0_v1.json \\
      --output experiments/scalar_threshold_report_gpu0.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def parse_inputs(items: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected label=path input, got {item!r}")
        label, path = item.split("=", 1)
        if not label:
            raise ValueError(f"Empty label in {item!r}")
        parsed.append((label, Path(path)))
    return parsed


def summarize(label: str, data: dict[str, Any], policy_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in data.get("episodes", []):
        if ep.get("policy") != policy_name:
            continue
        grouped[str(ep.get("scenario"))].append(ep)

    rows: list[dict[str, Any]] = []
    for scenario, eps in sorted(grouped.items()):
        n = len(eps)
        recovered = sum(1 for ep in eps if ep.get("recovered"))
        penalty = [float(ep.get("final_penalty", 0.0)) for ep in eps]
        worsening = sum(1 for ep in eps if ep.get("worsened"))
        proposals = [float(ep.get("total_proposals", 0.0)) for ep in eps]
        rejections = [float(ep.get("total_rejections", 0.0)) for ep in eps]
        rows.append({
            "label": label,
            "policy": policy_name,
            "scenario": scenario,
            "n": n,
            "recovered": recovered,
            "recovery_rate": recovered / n if n else float("nan"),
            "penalty_mean": sum(penalty) / n if n else float("nan"),
            "worse_rate": worsening / n if n else float("nan"),
            "proposals_mean": sum(proposals) / n if n else float("nan"),
            "reject_rate": (
                sum(rejections) / sum(proposals)
                if sum(proposals) > 0
                else float("nan")
            ),
        })
    return rows


def write_report(
    inputs: list[tuple[str, Path]],
    reference_inputs: list[tuple[str, Path]],
    reference_policy: str,
    output: Path,
) -> None:
    all_rows: list[dict[str, Any]] = []
    statuses: list[list[str]] = []
    series = [
        (label, path, "scalar_progress")
        for label, path in inputs
    ] + [
        (label, path, reference_policy)
        for label, path in reference_inputs
    ]
    series_order = [label for label, _, _ in series]
    for label, path, policy_name in series:
        data = load_json(path)
        status = data.get("status", {})
        statuses.append([
            label,
            policy_name,
            path.name,
            str(status.get("complete")),
            str(status.get("completed_episodes")),
            str(status.get("expected_episodes")),
        ])
        all_rows.extend(summarize(label, data, policy_name))

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_label[row["label"]].append(row)

    lines: list[str] = []
    lines.append("# Scalar Threshold Sensitivity Report")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.extend(md_table(
        ["label", "policy", "file", "complete", "completed", "expected"],
        statuses,
    ))
    lines.append("")

    family_rows = []
    for label in series_order:
        rows = [
            row for row in by_label.get(label, [])
            if row["scenario"].startswith("mined_multi_action")
        ]
        n = sum(row["n"] for row in rows)
        recovered = sum(row["recovered"] for row in rows)
        penalty_weighted = sum(row["penalty_mean"] * row["n"] for row in rows)
        family_rows.append([
            label,
            f"{recovered}/{n}",
            fmt(recovered / n if n else float("nan")),
            fmt(penalty_weighted / n if n else float("nan")),
        ])
    lines.append("## Multi-Action Aggregate")
    lines.append("")
    lines.extend(md_table(
        ["series", "recovery", "rate", "penalty mean"],
        family_rows,
    ))
    lines.append("")

    lines.append("## Scenario Detail")
    lines.append("")
    detail_rows = []
    for row in all_rows:
        detail_rows.append([
            row["label"],
            row["policy"],
            row["scenario"],
            f"{row['recovered']}/{row['n']}",
            fmt(row["recovery_rate"]),
            fmt(row["penalty_mean"]),
            fmt(row["worse_rate"]),
            fmt(row["proposals_mean"]),
            fmt(row["reject_rate"]),
        ])
    lines.extend(md_table(
        [
            "series",
            "policy",
            "scenario",
            "recovery",
            "rate",
            "penalty mean",
            "worse",
            "props mean",
            "reject/prop",
        ],
        detail_rows,
    ))
    lines.append("")
    lines.append(
        "Interpret cautiously: this report tests whether scalar threshold "
        "relaxation closes the recovery gap. If a relaxed scalar gate matches "
        "`progress_mag`, the product-order claim should be demoted."
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--reference-inputs", nargs="*", default=[])
    parser.add_argument("--reference-policy", default="progress_mag")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_report(
        parse_inputs(args.inputs),
        parse_inputs(args.reference_inputs),
        args.reference_policy,
        args.output,
    )


if __name__ == "__main__":
    main()
