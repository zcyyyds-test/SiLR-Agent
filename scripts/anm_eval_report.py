"""Generate paper-facing reports for ANM evaluation artifacts.

The report is intentionally CPU-only. It turns raw sweep JSON files into a
small evidence packet with confidence intervals, safety deltas, MPC comparison,
and explicit warnings about incomplete or non-paper-facing artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.anm import ANMScenarioLoader

POLICY_ORDER = ["OFF", "terminal", "scalar_progress", "progress", "progress_mag"]
DEFAULT_WORSE_ABS_TOL = 1e-3
DEFAULT_WORSE_REL_TOL = 1e-3


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a JSON object")
    return data


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], 0.0)
    return (statistics.mean(values), statistics.stdev(values))


def worse_threshold(default_penalty: float, abs_tol: float, rel_tol: float) -> float:
    """Tolerance for counting a final penalty as materially worse.

    Solver and floating-point jitter can move tiny penalties by about 1e-4 in
    the ANM residual cases. Paper-facing worsening should ignore that noise.
    """
    return max(abs_tol, abs(default_penalty) * rel_tol)


def scenario_index() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for s in ANMScenarioLoader().load_all():
        if s.id.startswith("mined_single_action"):
            family = "mined_single_action"
        elif s.id.startswith("mined_multi_action"):
            family = "mined_multi_action"
        elif s.id.startswith("mined_mpc_unsolved"):
            family = "mined_mpc_residual"
        elif s.id.startswith("mined_"):
            family = "mined_other"
        else:
            family = "hand"
        out[s.id] = {
            "difficulty": s.difficulty,
            "family": family,
        }
    return out


def mpc_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = load_json(path)
    return {
        r["scenario_id"]: r
        for r in data.get("results", [])
        if isinstance(r, dict) and "scenario_id" in r
    }


def policy_sort_key(policy: str) -> int:
    try:
        return POLICY_ORDER.index(policy)
    except ValueError:
        return len(POLICY_ORDER)


def aggregate_episodes(
    episodes: list[dict[str, Any]],
    worse_abs_tol: float = DEFAULT_WORSE_ABS_TOL,
    worse_rel_tol: float = DEFAULT_WORSE_REL_TOL,
) -> dict[str, Any]:
    n = len(episodes)
    successes = sum(1 for e in episodes if e.get("recovered"))
    ci_low, ci_high = wilson_ci(successes, n)
    final_penalties = [float(e["final_penalty"]) for e in episodes if "final_penalty" in e]
    proposals = [float(e["total_proposals"]) for e in episodes if "total_proposals" in e]
    rejections = [float(e["total_rejections"]) for e in episodes if "total_rejections" in e]
    wallclock = [float(e["wallclock_s"]) for e in episodes if "wallclock_s" in e]
    default_penalties = [
        float(e["default_penalty"]) for e in episodes if "default_penalty" in e
    ]
    worsening = 0
    comparable = 0
    for e in episodes:
        if "default_penalty" in e and "final_penalty" in e:
            comparable += 1
            default_penalty = float(e["default_penalty"])
            final_penalty = float(e["final_penalty"])
            threshold = worse_threshold(default_penalty, worse_abs_tol, worse_rel_tol)
            if final_penalty > default_penalty + threshold:
                worsening += 1
    pen_mean, pen_std = mean_std(final_penalties)
    prop_mean, prop_std = mean_std(proposals)
    rej_mean, rej_std = mean_std(rejections)
    wall_mean, wall_std = mean_std(wallclock)
    default_mean, default_std = mean_std(default_penalties)
    return {
        "n": n,
        "successes": successes,
        "recovery_rate": successes / n if n else float("nan"),
        "recovery_ci_low": ci_low,
        "recovery_ci_high": ci_high,
        "final_penalty_mean": pen_mean,
        "final_penalty_std": pen_std,
        "default_penalty_mean": default_mean,
        "default_penalty_std": default_std,
        "proposals_mean": prop_mean,
        "proposals_std": prop_std,
        "rejections_mean": rej_mean,
        "rejections_std": rej_std,
        "wallclock_mean": wall_mean,
        "wallclock_std": wall_std,
        "worsening_rate": worsening / comparable if comparable else float("nan"),
    }


def build_cell_rows(
    data: dict[str, Any],
    scenario_meta: dict[str, dict[str, Any]],
    mpc_meta: dict[str, dict[str, Any]],
    worse_abs_tol: float = DEFAULT_WORSE_ABS_TOL,
    worse_rel_tol: float = DEFAULT_WORSE_REL_TOL,
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ep in data.get("episodes", []):
        scenario = ep.get("scenario")
        policy = ep.get("policy")
        if scenario and policy:
            by_cell[(scenario, policy)].append(ep)

    rows: list[dict[str, Any]] = []
    for (scenario, policy), episodes in sorted(
        by_cell.items(), key=lambda kv: (kv[0][0], policy_sort_key(kv[0][1]))
    ):
        agg = aggregate_episodes(
            episodes,
            worse_abs_tol=worse_abs_tol,
            worse_rel_tol=worse_rel_tol,
        )
        meta = scenario_meta.get(scenario, {})
        mpc = mpc_meta.get(scenario, {})
        mpc_penalty = mpc.get("mpc_penalty")
        penalty_gap = (
            agg["final_penalty_mean"] - float(mpc_penalty)
            if mpc_penalty is not None and not math.isnan(agg["final_penalty_mean"])
            else float("nan")
        )
        rows.append({
            "scenario": scenario,
            "family": meta.get("family", "unknown"),
            "difficulty": meta.get("difficulty", "unknown"),
            "policy": policy,
            **agg,
            "mpc_penalty": float(mpc_penalty) if mpc_penalty is not None else float("nan"),
            "mpc_violations": mpc.get("mpc_violations", ""),
            "penalty_minus_mpc": penalty_gap,
        })
    return rows


def group_rows(cell_rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        grouped[(str(row[group_key]), row["policy"])].append(row)

    rows = []
    for (group, policy), cells in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], policy_sort_key(kv[0][1]))
    ):
        n = sum(int(c["n"]) for c in cells)
        successes = sum(int(c["successes"]) for c in cells)
        ci_low, ci_high = wilson_ci(successes, n)
        penalties = []
        worsening_weighted = 0.0
        for c in cells:
            if not math.isnan(float(c["final_penalty_mean"])):
                penalties.extend([float(c["final_penalty_mean"])] * int(c["n"]))
            if not math.isnan(float(c["worsening_rate"])):
                worsening_weighted += float(c["worsening_rate"]) * int(c["n"])
        pen_mean, pen_std = mean_std(penalties)
        rows.append({
            group_key: group,
            "policy": policy,
            "n": n,
            "successes": successes,
            "recovery_rate": successes / n if n else float("nan"),
            "recovery_ci_low": ci_low,
            "recovery_ci_high": ci_high,
            "final_penalty_mean": pen_mean,
            "final_penalty_std": pen_std,
            "worsening_rate": worsening_weighted / n if n else float("nan"),
        })
    return rows


def fmt_float(value: Any, digits: int = 3) -> str:
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


def warning_lines(data: dict[str, Any], cell_rows: list[dict[str, Any]]) -> list[str]:
    warnings = []
    status = data.get("status")
    if isinstance(status, dict) and not status.get("complete", True):
        warnings.append(
            f"Incomplete artifact: {status.get('completed_episodes')}/"
            f"{status.get('expected_episodes')} episodes."
        )
    if "scenario_manifest" not in data:
        warnings.append("Missing scenario_manifest; do not use as paper-facing mined evidence.")
    fp = data.get("code_fingerprint", {})
    if isinstance(fp, dict) and fp.get("git_commit") is None:
        warnings.append("code_fingerprint.git_commit is null; preserve file hashes for provenance.")
    if any("step_trace" not in ep for ep in data.get("episodes", [])):
        warnings.append("Episodes do not all include step_trace; trajectory figures need a rerun.")
    for row in cell_rows:
        if row["n"] < 3:
            warnings.append(
                f"Cell {row['scenario']}__{row['policy']} has n={row['n']} (<3)."
            )
    return warnings


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    eval_path: Path,
    data: dict[str, Any],
    cell_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    difficulty_rows: list[dict[str, Any]],
    worse_abs_tol: float,
    worse_rel_tol: float,
) -> None:
    lines: list[str] = []
    lines.append(f"# ANM Evaluation Report: `{eval_path.name}`")
    lines.append("")
    status = data.get("status", {})
    lines.append("## Status")
    lines.append("")
    lines.extend(md_table(
        ["field", "value"],
        [
            ["complete", str(status.get("complete", "unknown"))],
            ["completed_episodes", str(status.get("completed_episodes", len(data.get("episodes", []))))],
            ["expected_episodes", str(status.get("expected_episodes", ""))],
            ["model", str(data.get("config", {}).get("model", ""))],
            ["policies", ", ".join(data.get("policies", []))],
            ["worse_abs_tol", fmt_float(worse_abs_tol, digits=6)],
            ["worse_rel_tol", fmt_float(worse_rel_tol, digits=6)],
        ],
    ))
    lines.append("")

    warnings = warning_lines(data, cell_rows)
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Cell Results")
    lines.append("")
    lines.extend(md_table(
        [
            "scenario",
            "family",
            "diff",
            "policy",
            "n",
            "rec",
            "95% CI",
            "penalty",
            "worse",
            "mpc pen",
            "gap",
        ],
        [
            [
                r["scenario"],
                r["family"],
                r["difficulty"],
                r["policy"],
                str(r["n"]),
                fmt_float(r["recovery_rate"]),
                f"[{fmt_float(r['recovery_ci_low'])}, {fmt_float(r['recovery_ci_high'])}]",
                f"{fmt_float(r['final_penalty_mean'])} ({fmt_float(r['final_penalty_std'])})",
                fmt_float(r["worsening_rate"]),
                fmt_float(r["mpc_penalty"]),
                fmt_float(r["penalty_minus_mpc"]),
            ]
            for r in cell_rows
        ],
    ))
    lines.append("")

    for title, key, rows in [
        ("By Scenario Family", "family", family_rows),
        ("By Difficulty", "difficulty", difficulty_rows),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(md_table(
            [key, "policy", "n", "rec", "95% CI", "penalty", "worse"],
            [
                [
                    r[key],
                    r["policy"],
                    str(r["n"]),
                    fmt_float(r["recovery_rate"]),
                    f"[{fmt_float(r['recovery_ci_low'])}, {fmt_float(r['recovery_ci_high'])}]",
                    f"{fmt_float(r['final_penalty_mean'])} ({fmt_float(r['final_penalty_std'])})",
                    fmt_float(r["worsening_rate"]),
                ]
                for r in rows
            ],
        ))
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True, type=Path)
    ap.add_argument("--mpc-json", type=Path, default=None)
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument(
        "--worse-abs-tol",
        type=float,
        default=DEFAULT_WORSE_ABS_TOL,
        help="Absolute penalty increase required before counting worsening.",
    )
    ap.add_argument(
        "--worse-rel-tol",
        type=float,
        default=DEFAULT_WORSE_REL_TOL,
        help="Relative penalty increase required before counting worsening.",
    )
    args = ap.parse_args()

    data = load_json(args.eval_json)
    cell_rows = build_cell_rows(
        data,
        scenario_index(),
        mpc_index(args.mpc_json),
        worse_abs_tol=args.worse_abs_tol,
        worse_rel_tol=args.worse_rel_tol,
    )
    family_rows = group_rows(cell_rows, "family")
    difficulty_rows = group_rows(cell_rows, "difficulty")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, cell_rows)
    write_markdown(
        args.output_md,
        args.eval_json,
        data,
        cell_rows,
        family_rows,
        difficulty_rows,
        worse_abs_tol=args.worse_abs_tol,
        worse_rel_tol=args.worse_rel_tol,
    )
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
