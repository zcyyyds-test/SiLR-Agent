"""Summarize scalar-trap spectrum diagnostics from existing ANM traces.

This is a CPU-only, artifact-only analysis. It does not run the LLM or the
power-flow simulator. The goal is to check whether the scalar projection trap is
only a two-scenario anecdote or whether the existing multi-action traces form a
smooth-to-hard spectrum useful for paper writing.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


MULTI_ACTION = [
    "mined_multi_action_1_l0p25g1p0_s5",
    "mined_multi_action_2_l1p0g1p0_s5",
    "mined_multi_action_3_l0p25g1p0_s12",
]


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


def scenario_meta(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("scenarios", []):
        if isinstance(row, dict) and row.get("id"):
            out[row["id"]] = row
    return out


def scalar_rel20_episodes(paths: list[Path]) -> dict[str, dict[int, dict[str, Any]]]:
    by_scenario_seed: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        data = load_json(path)
        for ep in data.get("episodes", []):
            if ep.get("policy") != "scalar_progress":
                continue
            scenario = str(ep.get("scenario", ""))
            if scenario not in MULTI_ACTION:
                continue
            seed = int(ep.get("rep_seed", -1))
            # Extra file has seeds 1003/1004; primary file has 1000/1001/1002.
            by_scenario_seed[scenario][seed] = ep
    return by_scenario_seed


def progress_mag_episodes(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    data = load_json(path)
    out: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for ep in data.get("episodes", []):
        if ep.get("policy") != "progress_mag":
            continue
        scenario = str(ep.get("scenario", ""))
        if scenario not in MULTI_ACTION:
            continue
        out[scenario][int(ep.get("rep_seed", -1))] = ep
    return out


def first_safe_progress(ep: dict[str, Any]) -> dict[str, Any] | None:
    for step in ep.get("step_trace") or []:
        if not step.get("applied"):
            continue
        verdicts = step.get("verdicts") or []
        if "SAFE_PROGRESS" in verdicts:
            return step
    return None


def summarize_scalar_episode(ep: dict[str, Any]) -> dict[str, Any]:
    first = first_safe_progress(ep)
    final_penalty = float(ep.get("final_penalty", float("nan")))
    first_post = (
        float(first.get("post_penalty"))
        if first and first.get("post_penalty") is not None
        else float("nan")
    )
    plateau = (
        first is not None
        and not ep.get("recovered")
        and math.isfinite(first_post)
        and abs(final_penalty - first_post) <= 1e-6
    )
    return {
        "recovered": bool(ep.get("recovered")),
        "final_penalty": final_penalty,
        "total_proposals": int(ep.get("total_proposals", 0)),
        "total_rejections": int(ep.get("total_rejections", 0)),
        "first_step": int(first.get("step")) if first else None,
        "first_pre_penalty": float(first.get("pre_penalty")) if first else float("nan"),
        "first_post_penalty": first_post,
        "first_violation_count": (
            int(first.get("last_violation_count"))
            if first and first.get("last_violation_count") is not None
            else None
        ),
        "first_severity_score": (
            float(first.get("last_severity_score"))
            if first and first.get("last_severity_score") is not None
            else float("nan")
        ),
        "plateau_after_first": plateau,
    }


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def write_report(
    output: Path,
    scenario_json: Path,
    scalar_jsons: list[Path],
    progress_mag_json: Path,
) -> None:
    meta = scenario_meta(scenario_json)
    scalar = scalar_rel20_episodes(scalar_jsons)
    structured = progress_mag_episodes(progress_mag_json)

    lines: list[str] = []
    lines.append("# Scalar Trap Spectrum Diagnostic")
    lines.append("")
    lines.append("Date: 2026-05-25")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "Check whether the scalar projection trap reads as a two-scenario anecdote "
        "or as a smooth-to-hard spectrum over the mined multi-action family. This "
        "is a trace-level diagnostic from existing artifacts; it does not run new "
        "LLM or GPU experiments."
    )
    lines.append("")

    rows = []
    for scenario in MULTI_ACTION:
        m = meta.get(scenario, {})
        scalar_eps = [scalar.get(scenario, {}).get(seed) for seed in sorted(scalar.get(scenario, {}))]
        scalar_eps = [ep for ep in scalar_eps if ep is not None]
        scalar_summaries = [summarize_scalar_episode(ep) for ep in scalar_eps]
        structured_eps = list(structured.get(scenario, {}).values())
        n_scalar = len(scalar_summaries)
        scalar_rec = sum(1 for s in scalar_summaries if s["recovered"])
        scalar_plateau = sum(1 for s in scalar_summaries if s["plateau_after_first"])
        structured_rec = sum(1 for ep in structured_eps if ep.get("recovered"))
        first_counts = [
            float(s["first_violation_count"])
            for s in scalar_summaries
            if s["first_violation_count"] is not None
        ]
        first_sev = [float(s["first_severity_score"]) for s in scalar_summaries]
        first_post = [float(s["first_post_penalty"]) for s in scalar_summaries]
        props = [float(s["total_proposals"]) for s in scalar_summaries]
        rejs = [float(s["total_rejections"]) for s in scalar_summaries]
        rows.append([
            scenario.replace("mined_multi_action_", "multi_"),
            str(m.get("default_violation_count", "")),
            fmt(m.get("default_penalty")),
            f"{scalar_rec}/{n_scalar}",
            f"{structured_rec}/{len(structured_eps)}",
            fmt(mean(first_counts)),
            fmt(mean(first_sev), 1),
            fmt(mean(first_post)),
            f"{scalar_plateau}/{n_scalar}",
            fmt(mean(props), 1),
            fmt(mean(rejs), 1),
        ])

    lines.append("## Spectrum Table")
    lines.append("")
    lines.extend(md_table(
        [
            "scenario",
            "initial branch viol",
            "initial penalty",
            "rel20 scalar rec",
            "progress_mag rec",
            "first scalar residual viol",
            "first scalar residual severity",
            "first scalar residual penalty",
            "plateau after first",
            "scalar props",
            "scalar rejects",
        ],
        rows,
    ))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Raw initial stress is not the mechanism: `multi_1` starts with higher "
        "initial count/penalty than `multi_2`, yet scalar recovers `multi_1`."
    )
    lines.append(
        "- The useful diagnostic is what happens after the first scalar-admitted "
        "local descent. `multi_1` reaches a low residual and then PASSes; "
        "`multi_2` often accepts the same residual step but then stalls unless "
        "the next proposal escapes; `multi_3` accepts one local descent and "
        "plateaus in all tested seeds."
    )
    lines.append(
        "- This supports a cautious spectrum claim: scalar slack is sufficient on "
        "the smooth case, unstable on the transitional case, and consistently "
        "trapped on the hard residual-geometry case. It does not yet prove a "
        "population-level correlation between physical coupling and scalar "
        "failure."
    )
    lines.append("")

    lines.append("## Paper Wording")
    lines.append("")
    lines.append("Use:")
    lines.append("")
    lines.append(
        "> Across the mined multi-action family, scalar admission exhibits a "
        "smooth-to-hard spectrum. It solves the smooth case, becomes seed-sensitive "
        "on the transitional case, and in the hardest branch-overload geometry "
        "accepts one locally improving step before plateauing in every tested seed. "
        "Thus the scalar projection trap is not a single cherry-picked trace; it is "
        "the endpoint of a spectrum where scalar descent increasingly decouples "
        "from branch-level recovery geometry."
    )
    lines.append("")
    lines.append("Avoid:")
    lines.append("")
    lines.append(
        "- Claiming statistical correlation from three scenarios."
    )
    lines.append(
        "- Calling this a full physical coupling metric unless a simulator-based "
        "branch sensitivity analysis is added."
    )
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-json", type=Path, default=Path("domains/anm/scenarios_mined.json"))
    ap.add_argument(
        "--scalar-json",
        type=Path,
        action="append",
        required=True,
        help="Scalar rel20 JSON artifact. Pass multiple times to merge seeds.",
    )
    ap.add_argument("--progress-mag-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    write_report(
        output=args.output,
        scenario_json=args.scenario_json,
        scalar_jsons=args.scalar_json,
        progress_mag_json=args.progress_mag_json,
    )


if __name__ == "__main__":
    main()
