"""Plot per-step ANM penalty trajectories from eval_sweep artifacts.

This script consumes JSON produced by ``scripts/anm_eval_sweep.py`` after
``step_trace`` support was enabled. It does not replay the simulator; it only
plots recorded pre/post penalties, so it can run locally once JSON is pulled
back from a remote server.

Example:
    python3 scripts/anm_plot_trajectory.py \\
      --eval-json eval_trajectory_v1_gpu0.json \\
      --scenario mined_multi_action_3_l0p25g1p0_s12 \\
      --policies progress progress_mag \\
      --output-prefix figures/mined_multi_action_3_trajectory
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


POLICY_ORDER = ["OFF", "terminal", "scalar_progress", "progress", "progress_mag"]
COLORS = {
    "OFF": "#999999",
    "terminal": "#D55E00",
    "scalar_progress": "#CC79A7",
    "progress": "#0072B2",
    "progress_mag": "#009E73",
}
LINESTYLES = {
    "OFF": ":",
    "terminal": "--",
    "scalar_progress": (0, (1, 1)),
    "progress": "-.",
    "progress_mag": "-",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a JSON object")
    return data


def trajectory_points(ep: dict[str, Any]) -> list[tuple[int, float]]:
    trace = ep.get("step_trace") or []
    if not trace:
        return []
    first_pre = trace[0].get("pre_penalty")
    if first_pre is None:
        return []
    points = [(0, float(first_pre))]
    for item in trace:
        step = int(item.get("step", len(points)))
        post = item.get("post_penalty")
        if post is None:
            continue
        points.append((step, float(post)))
    # Deduplicate repeated terminal recovered rows at the same penalty only if
    # they share a step number; keep flat tails because they communicate stall.
    out: list[tuple[int, float]] = []
    seen: set[int] = set()
    for step, penalty in points:
        if step in seen:
            continue
        seen.add(step)
        out.append((step, penalty))
    return out


def policy_sort(policy: str) -> int:
    try:
        return POLICY_ORDER.index(policy)
    except ValueError:
        return len(POLICY_ORDER)


def select_episodes(
    data: dict[str, Any],
    scenario: str,
    policies: list[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    allowed = set(policies) if policies else None
    for ep in data.get("episodes", []):
        if ep.get("scenario") != scenario:
            continue
        policy = str(ep.get("policy"))
        if allowed is not None and policy not in allowed:
            continue
        if trajectory_points(ep):
            grouped[policy].append(ep)
    return dict(sorted(grouped.items(), key=lambda kv: policy_sort(kv[0])))


def write_points_csv(path: Path, grouped: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "policy",
            "rep_seed",
            "recovered",
            "step",
            "penalty",
            "verdicts",
        ])
        for policy, episodes in grouped.items():
            for ep in episodes:
                verdict_by_step = {
                    int(item.get("step", -1)): ",".join(item.get("verdicts", []))
                    for item in ep.get("step_trace", [])
                }
                for step, penalty in trajectory_points(ep):
                    writer.writerow([
                        ep.get("scenario"),
                        policy,
                        ep.get("rep_seed"),
                        ep.get("recovered"),
                        step,
                        f"{penalty:.10g}",
                        verdict_by_step.get(step, ""),
                    ])


def plot(
    grouped: dict[str, list[dict[str, Any]]],
    scenario: str,
    out_prefix: Path,
    title: str | None,
) -> None:
    if not grouped:
        raise ValueError(f"No step_trace episodes found for scenario={scenario}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    max_step = 0
    max_penalty = 0.0

    for policy, episodes in grouped.items():
        color = COLORS.get(policy, "#333333")
        linestyle = LINESTYLES.get(policy, "-")
        for ep in sorted(episodes, key=lambda e: int(e.get("rep_seed", 0))):
            pts = trajectory_points(ep)
            if not pts:
                continue
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            max_step = max(max_step, max(xs))
            max_penalty = max(max_penalty, max(ys))
            label = f"{policy}" if ep is episodes[0] else None
            alpha = 0.95 if ep.get("recovered") else 0.55
            marker = "o" if ep.get("recovered") else "x"
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0 if ep.get("recovered") else 1.4,
                markersize=4.5,
                alpha=alpha,
                label=label,
            )
            if ep.get("recovered"):
                ax.annotate(
                    "recovered",
                    (xs[-1], ys[-1]),
                    textcoords="offset points",
                    xytext=(4, 5),
                    fontsize=7,
                    color=color,
                )

    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("ReAct step")
    ax.set_ylabel("Native ANM penalty")
    ax.set_xticks(range(0, max_step + 1))
    ax.set_ylim(bottom=-0.03 * max(1.0, max_penalty), top=max(1.0, max_penalty) * 1.12)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=max(1, len(grouped)))
    ax.set_title(title or scenario)

    fig.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
        print(f"wrote {out_prefix.with_suffix(suffix)}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", type=Path, required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--policies", nargs="*", default=None)
    ap.add_argument("--output-prefix", type=Path, required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    data = load_json(args.eval_json)
    grouped = select_episodes(data, args.scenario, args.policies)
    write_points_csv(args.output_prefix.with_suffix(".csv"), grouped)
    print(f"wrote {args.output_prefix.with_suffix('.csv')}")
    plot(grouped, args.scenario, args.output_prefix, args.title)


if __name__ == "__main__":
    main()
