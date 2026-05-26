"""Create an SVG mechanism figure for scalar projection failure.

The script uses only the Python standard library so it can run on the local
Codex desktop without plotting dependencies.
"""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path


OUT = Path("figures/scalar_projection_trap_mechanism.svg")
TRACE_CSV = Path("figures/mined_multi_action_3_scalar_vs_structured_partial.csv")


def load_trace(policy: str, seed: int) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    penalties: list[float] = []
    with TRACE_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["policy"] == policy and int(row["rep_seed"]) == seed:
                steps.append(int(row["step"]))
                penalties.append(float(row["penalty"]))
    if not steps:
        raise ValueError(f"No trace for {policy} seed={seed}")
    return steps, penalties


def line(points: list[tuple[float, float]], color: str, width: float = 2.0,
         dash: str | None = None) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" '
        f'stroke-linejoin="round"{dash_attr}/>'
    )


def circle(x: float, y: float, r: float, color: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}"/>'


def text(x: float, y: float, s: str, size: int = 12, color: str = "#222",
         anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, DejaVu Sans, sans-serif" fill="{color}" '
        f'text-anchor="{anchor}">{escape(s)}</text>'
    )


def arrow(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="2.2" marker-end="url(#arrow)"/>'
    )


def panel_a() -> str:
    left, top, width, height = 45, 55, 265, 210

    def xy(a: float, b: float) -> tuple[float, float]:
        return left + a / 7.2 * width, top + height - b / 7.2 * height

    start = xy(4.0, 4.0)
    false_progress = xy(0.8, 6.5)
    good = [xy(*p) for p in [(4.0, 4.0), (3.0, 2.5), (1.6, 1.1), (0.0, 0.0)]]

    parts = [
        text(left, 30, "A. Scalar projection can hide branch geometry", 13),
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#fff" stroke="#ddd"/>',
    ]
    # Scalar level sets P=a+b.
    for level in (8.0, 7.3, 5.5):
        parts.append(line([xy(0, level), xy(level, 0)], "#bbbbbb", 1.0, "4 4"))
    parts.append(arrow(*start, *false_progress, "#D55E00"))
    parts.append(line(good, "#009E73", 2.4))
    for p in good:
        parts.append(circle(*p, 4.0, "#009E73"))
    parts.append(circle(*start, 5.2, "#222222"))
    parts.append(circle(*false_progress, 5.2, "#D55E00"))
    parts.append(text(false_progress[0] + 8, false_progress[1] - 6,
                      "lower scalar, wrong branch mix", 11, "#D55E00"))
    parts.append(text(good[-1][0] + 8, good[-1][1] - 6, "terminal", 11))
    parts.append(text(left + width / 2, top + height + 34, "Overload on branch A", 11, anchor="middle"))
    parts.append(text(left - 34, top + height / 2, "Branch B", 11))
    parts.append(text(left + 8, top + 16, "structured path", 11, "#009E73"))
    return "\n".join(parts)


def panel_b() -> str:
    left, top, width, height = 370, 55, 300, 210
    scalar_steps, scalar_penalties = load_trace("scalar_progress", 1001)
    pm_steps, pm_penalties = load_trace("progress_mag", 1001)
    max_step = 6
    max_pen = 18.0

    def xy(step: float, penalty: float) -> tuple[float, float]:
        return left + step / max_step * width, top + height - penalty / max_pen * height

    scalar_pts = [xy(s, p) for s, p in zip(scalar_steps, scalar_penalties)]
    pm_pts = [xy(s, p) for s, p in zip(pm_steps, pm_penalties)]
    parts = [
        text(left, 30, "B. Empirical multi_3: scalar plateau vs structured recovery", 13),
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#fff" stroke="#ddd"/>',
    ]
    for i in range(0, 7):
        x = left + i / max_step * width
        parts.append(f'<line x1="{x:.1f}" y1="{top + height:.1f}" x2="{x:.1f}" y2="{top + height + 4:.1f}" stroke="#777"/>')
        parts.append(text(x, top + height + 18, str(i), 9, "#555", "middle"))
    for pen in (0, 5, 10, 15):
        y = top + height - pen / max_pen * height
        parts.append(f'<line x1="{left - 4:.1f}" y1="{y:.1f}" x2="{left:.1f}" y2="{y:.1f}" stroke="#777"/>')
        parts.append(text(left - 8, y + 3, str(pen), 9, "#555", "end"))
        parts.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="#eeeeee"/>')
    parts.append(line(scalar_pts, "#CC79A7", 2.2))
    parts.append(line(pm_pts, "#009E73", 2.4))
    for p in scalar_pts:
        parts.append(circle(*p, 3.8, "#CC79A7"))
    for p in pm_pts:
        parts.append(circle(*p, 4.2, "#009E73"))
    parts.append(text(left + 132, top + 50, "scalar accepts local descent,", 11, "#CC79A7"))
    parts.append(text(left + 132, top + 64, "then plateaus", 11, "#CC79A7"))
    parts.append(text(left + 100, top + 142, "reject then admit", 11, "#444444"))
    parts.append(arrow(left + 134, top + 137, pm_pts[1][0], pm_pts[1][1], "#444444"))
    parts.append(text(left + 115, top + height + 36, "Verifier-mediated step", 11))
    parts.append(text(left - 34, top + 12, "Penalty", 11))
    parts.append(text(left + width - 118, top + 18, "progress_mag", 11, "#009E73"))
    parts.append(text(left + width - 118, top + 34, "scalar", 11, "#CC79A7"))
    return "\n".join(parts)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="330" viewBox="0 0 720 330">
<defs>
  <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L6,3 z" fill="#444"/>
  </marker>
</defs>
<rect width="720" height="330" fill="#ffffff"/>
{panel_a()}
{panel_b()}
</svg>
'''
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
