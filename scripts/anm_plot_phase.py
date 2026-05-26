"""Plot violation-count/severity phase trajectories from enriched step traces.

This consumes JSON artifacts produced by ``scripts/anm_eval_sweep.py`` after
``last_violation_count`` and ``last_severity_score`` were added to step_trace.
Older artifacts without those fields are skipped with a clear error.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


POLICY_ORDER = ["terminal", "scalar_progress", "progress", "progress_mag"]
COLORS = {
    "terminal": "#D55E00",
    "scalar_progress": "#CC79A7",
    "progress": "#0072B2",
    "progress_mag": "#009E73",
}
LINESTYLES = {
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


def phase_points(ep: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in ep.get("step_trace") or []:
        if "last_violation_count" not in item or "last_severity_score" not in item:
            continue
        points.append((
            float(item["last_violation_count"]),
            float(item["last_severity_score"]),
        ))
    if ep.get("recovered") and points and points[-1] != (0.0, 0.0):
        points.append((0.0, 0.0))
    return points


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
        if phase_points(ep):
            grouped[policy].append(ep)
    return dict(sorted(grouped.items(), key=lambda kv: policy_sort(kv[0])))


def plot(
    grouped: dict[str, list[dict[str, Any]]],
    scenario: str,
    out_prefix: Path,
    title: str | None,
) -> None:
    if not grouped:
        raise ValueError(
            "No enriched phase points found. Rerun anm_eval_sweep.py after "
            "step_trace enrichment or choose another artifact."
        )

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

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    max_x = 1.0
    max_y = 1.0

    for policy, episodes in grouped.items():
        color = COLORS.get(policy, "#333333")
        linestyle = LINESTYLES.get(policy, "-")
        for idx, ep in enumerate(sorted(episodes, key=lambda e: int(e.get("rep_seed", 0)))):
            pts = phase_points(ep)
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            max_x = max(max_x, max(xs))
            max_y = max(max_y, max(ys))
            label = policy if idx == 0 else None
            marker = "o" if ep.get("recovered") else "x"
            alpha = 0.9 if ep.get("recovered") else 0.55
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8 if ep.get("recovered") else 1.3,
                markersize=4.2,
                alpha=alpha,
                label=label,
            )

    ax.scatter([0], [0], color="#222222", marker="*", s=60, label="terminal")
    ax.set_xlabel("Violation count")
    ax.set_ylabel("Aggregate severity")
    ax.set_xlim(left=-0.1, right=max_x + 0.4)
    ax.set_ylim(bottom=-0.03 * max_y, top=max_y * 1.12)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(title or scenario)
    fig.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
        print(f"wrote {out_prefix.with_suffix(suffix)}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    data = load_json(args.eval_json)
    grouped = select_episodes(data, args.scenario, args.policies)
    plot(grouped, args.scenario, args.output_prefix, args.title)


if __name__ == "__main__":
    main()
