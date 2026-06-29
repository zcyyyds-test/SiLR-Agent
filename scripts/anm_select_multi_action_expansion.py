"""Build or extend the ANM multi-action expansion scenario set.

The v2 mining catalogue contains 24 multi-action rows, but they are 8 unique
operating points repeated under three SoC perturbations. For the P0 rerun we
deduplicate by (source_seed, load_mul, gen_mul), prefer the native SoC row, and
write an expanded scenarios_mined.json that preserves the existing single-action
and MPC-residual records while replacing the multi-action band with all 8 unique
operating points.

The 2026-06-14b extension mode starts from a larger mining catalogue, excludes
the operating points already registered in ``scenarios_mined.json``, and appends
new MPC-recoverable multi-action operating points with all available SoC
variants.  This is the path used to increase the independent operating-point
count beyond the original 8.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
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


def _op_key(row: dict[str, Any]) -> tuple[int, float, float]:
    return (
        int(row["source_seed"]),
        float(row["load_mul"]),
        float(row["gen_mul"]),
    )


def _registered_multi_action_keys(records: list[dict[str, Any]]) -> set[tuple[int, float, float]]:
    keys: set[tuple[int, float, float]] = set()
    for row in records:
        if row.get("class") != "multi_action":
            continue
        keys.add(_op_key(row))
    return keys


def _group_multi_action(catalogue: list[dict[str, Any]]) -> dict[tuple[int, float, float], list[dict[str, Any]]]:
    grouped: dict[tuple[int, float, float], list[dict[str, Any]]] = {}
    for row in catalogue:
        if row.get("class") != "multi_action":
            continue
        if not row.get("mpc", {}).get("recovered", False):
            continue
        grouped.setdefault(_op_key(row), []).append(row)
    return grouped


def _has_complete_soc_variants(rows: list[dict[str, Any]]) -> bool:
    return {"native", "near_min", "near_max"}.issubset(
        {str(row.get("soc_pert", "native")) for row in rows}
    )


def _select_new_operating_points(
    catalogue: list[dict[str, Any]],
    existing_keys: set[tuple[int, float, float]],
    max_new_points: int,
) -> list[tuple[tuple[int, float, float], list[dict[str, Any]]]]:
    grouped = _group_multi_action(catalogue)
    candidates = [
        (key, rows)
        for key, rows in grouped.items()
        if key not in existing_keys
        and _has_complete_soc_variants(rows)
    ]
    # Stable, transparent rule: balance load/gen coverage relative to the
    # existing band, and within each operating-condition bucket prefer source
    # seeds not yet selected.  The mining catalogue already guarantees
    # MPC-recoverable multi-action status; selection is not conditioned on any
    # LLM outcome.
    by_condition: dict[
        tuple[float, float],
        list[tuple[tuple[int, float, float], list[dict[str, Any]]]],
    ] = {}
    for item in sorted(candidates, key=lambda item: (item[0][1], item[0][2], item[0][0])):
        condition = (item[0][1], item[0][2])
        by_condition.setdefault(condition, []).append(item)

    condition_counts = Counter((key[1], key[2]) for key in existing_keys)
    selected_seeds: set[int] = set()

    selected: list[tuple[tuple[int, float, float], list[dict[str, Any]]]] = []
    while len(selected) < max_new_points:
        progressed = False
        for condition in sorted(by_condition, key=lambda c: (condition_counts[c], c[0], c[1])):
            bucket = by_condition[condition]
            if not bucket:
                continue
            pick_idx = next(
                (idx for idx, item in enumerate(bucket) if item[0][0] not in selected_seeds),
                0,
            )
            item = bucket.pop(pick_idx)
            selected.append(item)
            selected_seeds.add(item[0][0])
            condition_counts[condition] += 1
            progressed = True
            if len(selected) >= max_new_points:
                break
        if not progressed:
            break
    return selected


def _soc_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    order = {"native": 0, "near_min": 1, "near_max": 2}
    soc = str(row.get("soc_pert", "native"))
    return (order.get(soc, 99), soc)


def _records_for_operating_points(
    selected: list[tuple[tuple[int, float, float], list[dict[str, Any]]]],
    start_idx: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for offset, (_key, rows) in enumerate(selected):
        idx = start_idx + offset
        for row in sorted(rows, key=_soc_sort_key):
            rec = _make_scenario(row, idx, "multi_action")
            if row.get("soc_pert") != "native":
                rec["id"] = f"{rec['id']}_soc{row['soc_pert']}"
            records.append(rec)
    return records


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


def _extension_report(
    selected: list[tuple[tuple[int, float, float], list[dict[str, Any]]]],
    records: list[dict[str, Any]],
    output_json: Path,
    existing_key_count: int,
) -> str:
    lines = [
        "# Multi-Action Operating-Point Extension",
        "",
        "Date: 2026-06-14",
        "",
        "## Rule",
        "",
        "- Source pool: mining catalogue passed via `--catalogue`.",
        "- Filter: `class == multi_action` and `mpc.recovered == true`.",
        "- Exclude operating points already registered in `--existing`.",
        "- Require all three SoC variants (`native`, `near_min`, `near_max`) so each selected operating point contributes the same number of scenarios.",
        "- Operating point key: `(source_seed, load_mul, gen_mul)`.",
        "- Selection balances load/gen coverage relative to the existing band and prefers new source seeds within each load/gen bucket.",
        "- Append all SoC variants for each selected operating point.",
        "- Selection is independent of LLM outcomes and verifier policy verdicts.",
        "",
        f"Existing operating points excluded: {existing_key_count}.",
        f"New operating points selected: {len(selected)}.",
        f"New scenario records appended: {len(records)}.",
        f"Scenario library written to `{output_json}`.",
        "",
        "## Selected Operating Points",
        "",
        "| new_index | seed | load_mul | gen_mul | soc variants | default penalty(native if present) |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for i, (key, rows) in enumerate(selected, start=1):
        native = [r for r in rows if r.get("soc_pert") == "native"]
        ref = native[0] if native else sorted(rows, key=_soc_sort_key)[0]
        lines.append(
            "| {i} | {seed} | {load:.2f} | {gen:.2f} | {soc} | {pen:.3f} |".format(
                i=i,
                seed=key[0],
                load=key[1],
                gen=key[2],
                soc=", ".join(str(r.get("soc_pert")) for r in sorted(rows, key=_soc_sort_key)),
                pen=float(ref["default"]["penalty"]),
            )
        )
    lines.extend([
        "",
        "## Appended Scenario IDs",
        "",
    ])
    lines.extend(f"- `{rec['id']}`" for rec in records)
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
    parser.add_argument(
        "--extend-existing",
        action="store_true",
        help="Append new operating points instead of replacing the old multi-action band.",
    )
    parser.add_argument("--max-new-points", type=int, default=8)
    args = parser.parse_args()

    catalogue = _load_json(args.catalogue).get("catalogue", [])
    existing = _load_json(args.existing).get("scenarios", [])

    if args.extend_existing:
        existing_keys = _registered_multi_action_keys(existing)
        selected = _select_new_operating_points(
            catalogue,
            existing_keys=existing_keys,
            max_new_points=args.max_new_points,
        )
        if len(selected) < args.max_new_points:
            raise SystemExit(
                f"Only found {len(selected)}/{args.max_new_points} new "
                "MPC-recoverable multi-action operating points. "
                "Run mining on a larger seed/multiplier pool first."
            )
        selected_records = _records_for_operating_points(
            selected,
            start_idx=len(existing_keys),
        )
        existing_ids = {rec["id"] for rec in existing}
        duplicate_ids = sorted(rec["id"] for rec in selected_records if rec["id"] in existing_ids)
        if duplicate_ids:
            raise SystemExit(f"Generated duplicate scenario ids: {duplicate_ids[:5]}")

        merged = [*existing, *selected_records]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"scenarios": merged}, indent=2) + "\n",
            encoding="utf-8",
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            _extension_report(
                selected,
                selected_records,
                args.output,
                existing_key_count=len(existing_keys),
            ),
            encoding="utf-8",
        )
        print(
            f"selected {len(selected)} new operating points; "
            f"appended {len(selected_records)} scenario records"
        )
        for rec in selected_records:
            print(
                f"  + {rec['id']} seed={rec['source_seed']} "
                f"load={rec['load_mul']} gen={rec['gen_mul']} soc={rec['soc_pert']}"
            )
        print(f"wrote {args.output}")
        print(f"wrote {args.report}")
        return

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
