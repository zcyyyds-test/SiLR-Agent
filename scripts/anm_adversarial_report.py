"""Generate a compact paper-facing report for ANM adversarial sweeps."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ATTACK_ORDER = ["none", "prompt_injection", "observation_poison", "stall", "stall_rag"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a JSON object")
    return data


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], 0.0)
    return (statistics.mean(values), statistics.stdev(values))


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


def attack_sort_key(attack: str) -> int:
    try:
        return ATTACK_ORDER.index(attack)
    except ValueError:
        return len(ATTACK_ORDER)


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    if n == 0:
        return {}

    penalties = [float(e["final_penalty"]) for e in episodes if "final_penalty" in e]
    defaults = [float(e["default_penalty"]) for e in episodes if "default_penalty" in e]
    proposals = [float(e["total_proposals"]) for e in episodes if "total_proposals" in e]
    rejections = [float(e["total_rejections"]) for e in episodes if "total_rejections" in e]
    wallclock = [float(e["wallclock_s"]) for e in episodes if "wallclock_s" in e]

    verdict_total: Counter[str] = Counter()
    for e in episodes:
        for verdict, count in (e.get("verdict_tally") or {}).items():
            verdict_total[str(verdict)] += int(count)

    penalty_mean, penalty_std = mean_std(penalties)
    default_mean, default_std = mean_std(defaults)
    prop_mean, prop_std = mean_std(proposals)
    reject_mean, reject_std = mean_std(rejections)
    wall_mean, wall_std = mean_std(wallclock)
    total_prop = sum(proposals)
    total_rej = sum(rejections)

    return {
        "n": n,
        "attack_successes": sum(1 for e in episodes if e.get("attack_succeeded")),
        "false_recoveries": sum(1 for e in episodes if e.get("false_recovery")),
        "worsened": sum(1 for e in episodes if e.get("worsened")),
        "recovered": sum(1 for e in episodes if e.get("recovered")),
        "stall_terminated": sum(1 for e in episodes if e.get("stall_terminated")),
        "final_penalty_mean": penalty_mean,
        "final_penalty_std": penalty_std,
        "default_penalty_mean": default_mean,
        "default_penalty_std": default_std,
        "proposals_mean": prop_mean,
        "proposals_std": prop_std,
        "rejections_mean": reject_mean,
        "rejections_std": reject_std,
        "rejection_rate": (total_rej / total_prop) if total_prop else 0.0,
        "wallclock_mean": wall_mean,
        "wallclock_std": wall_std,
        "verdict_total": dict(verdict_total),
    }


def group_episodes(data: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ep in data.get("episodes", []):
        scenario = ep.get("scenario", "")
        attack = ep.get("attack", "")
        if scenario and attack:
            grouped[(scenario, attack)].append(ep)
    return grouped


def write_report(input_path: Path, output_path: Path) -> None:
    data = load_json(input_path)
    status = data.get("status", {})
    config = data.get("config", {})
    grouped = group_episodes(data)

    lines: list[str] = []
    lines.append("# ANM Adversarial Sweep Report")
    lines.append("")
    lines.append(f"Source: `{input_path.name}`")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(
        f"- Completion: `{status.get('completed_episodes')}/"
        f"{status.get('expected_episodes')}` "
        f"(complete={status.get('complete')})"
    )
    lines.append(f"- Gating policy: `{config.get('gating_policy')}`")
    lines.append(f"- Reps: `{config.get('reps')}`, max_steps: `{config.get('max_steps')}`")
    lines.append("")

    rows = []
    for (scenario, attack), episodes in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], attack_sort_key(kv[0][1]))
    ):
        agg = aggregate(episodes)
        n = agg["n"]
        rows.append([
            scenario,
            attack,
            f"{agg['attack_successes']}/{n}",
            f"{agg['false_recoveries']}/{n}",
            f"{agg['worsened']}/{n}",
            f"{agg['recovered']}/{n}",
            f"{agg['stall_terminated']}/{n}",
            fmt(agg["final_penalty_mean"]),
            fmt(agg["rejection_rate"]),
        ])
    lines.append("## Attack Mitigation")
    lines.append("")
    lines.extend(md_table(
        [
            "scenario",
            "attack",
            "attack succ",
            "false rec",
            "worse",
            "recovered",
            "stall term",
            "penalty mean",
            "reject/prop",
        ],
        rows,
    ))
    lines.append("")

    lines.append("## Verdict Totals")
    lines.append("")
    rows = []
    for (scenario, attack), episodes in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], attack_sort_key(kv[0][1]))
    ):
        agg = aggregate(episodes)
        rows.append([
            scenario,
            attack,
            json.dumps(agg["verdict_total"], sort_keys=True),
        ])
    lines.extend(md_table(["scenario", "attack", "verdict totals"], rows))
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.input, args.output)


if __name__ == "__main__":
    main()
