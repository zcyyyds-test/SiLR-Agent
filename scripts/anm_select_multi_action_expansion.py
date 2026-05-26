"""Build the P0 multi-action expansion scenario set.

The v2 mining catalogue contains 24 multi-action rows, but they are 8 unique
operating points repeated under three SoC perturbations. For the P0 rerun we
deduplicate by (source_seed, load_mul, gen_mul), prefer the native SoC row, and
write an expanded scenarios_mined.json that preserves the existing single-action
and MPC-residual records while replacing the multi-action band with all 8 unique
operating points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anm_select_mined import _make_scenario


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a JSON object")
    return data


def _select_multi_action(catalogue: list[dict[str, Any]], max_points: int = 8) -> list[dict[str, Any]]:
    multi = [row for row in catalogue if row.get("class") == "multi_action"]
    grouped: dict[tuple[int, float, float], list[dict[str, Any]]] = {}
    for row in multi:
        key = (
            int(row["source_seed"]),
            float(row["load_mul"]),
            float(row["gen_mul"]),
        )
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        native = [row for row in rows if row.get("soc_pert") == "native"]
        chosen = native[0] if native else sorted(rows, key=lambda r: str(r.get("soc_pert")))[0]
        selected.append(chosen)
    return selected[:max_points]


def _selection_report(records: list[dict[str, Any]], output_json: Path) -> str:
    lines = [
        "# Multi-Action Expansion Selection",
        "",
        "Date: 2026-05-25",
        "",
        "## Rule",
        "",
        "- Source pool: `mined_scenarios_v2.json`.",
        "- Filter: `class == multi_action`.",
        "- The pool has 24 rows, but they are 8 unique operating points repeated across three SoC perturbations.",
        "- Deduplicate by `(source_seed, load_mul, gen_mul)`.",
        "- Prefer `soc_pert == native` within each group.",
        "- Keep all 8 unique operating points, including the previous `multi_1/2/3` as the first three IDs.",
        "",
        f"Scenario library written to `{output_json}`.",
        "",
        "## Selected Scenarios",
        "",
        "| id | seed | load_mul | gen_mul | soc | default viol | default penalty | MPC recovered |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for rec in records:
        lines.append(
            "| {id} | {seed} | {load:.2f} | {gen:.2f} | {soc} | {viol} | {pen:.3f} | {mpc} |".format(
                id=rec["id"],
                seed=rec["source_seed"],
                load=float(rec["load_mul"]),
                gen=float(rec["gen_mul"]),
                soc=rec["soc_pert"],
                viol=rec["default_violation_count"],
                pen=float(rec["default_penalty"]),
                mpc=rec["mpc_recovered"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, default=Path("mined_scenarios_v2.json"))
    parser.add_argument(
        "--existing",
        type=Path,
        default=Path("domains/anm/scenarios_mined.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("domains/anm/scenarios_mined.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/multi_action_expansion_selection_2026-05-25.md"),
    )
    parser.add_argument("--max-points", type=int, default=8)
    args = parser.parse_args()

    catalogue = _load_json(args.catalogue).get("catalogue", [])
    existing = _load_json(args.existing).get("scenarios", [])
    selected_entries = _select_multi_action(catalogue, max_points=args.max_points)
    selected_records = [
        _make_scenario(entry, idx, "multi_action")
        for idx, entry in enumerate(selected_entries)
    ]

    preserved = [
        rec for rec in existing
        if rec.get("class") != "multi_action"
    ]
    merged = []
    inserted = False
    for rec in preserved:
        if not inserted and rec.get("class") == "mpc_unsolved":
            merged.extend(selected_records)
            inserted = True
        merged.append(rec)
    if not inserted:
        merged.extend(selected_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"scenarios": merged}, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        _selection_report(selected_records, args.output),
        encoding="utf-8",
    )

    print(f"selected {len(selected_records)} unique multi-action operating points")
    for rec in selected_records:
        print(
            f"  {rec['id']} seed={rec['source_seed']} "
            f"load={rec['load_mul']} gen={rec['gen_mul']} "
            f"pen={rec['default_penalty']:.3f}"
        )
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
