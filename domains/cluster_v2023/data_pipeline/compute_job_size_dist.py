"""Compute p(g) = P(num_gpu == g) from v2023 trace.

Used by FragmentationChecker (FGD ATC'23 formula) so the fragmentation
metric is comparable to the original paper. Without this, F values are
computed against a made-up distribution and cannot be claimed as
"matching FGD" in the README / resume bullets.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def compute_dist(jobs_csv: Path) -> dict[int, float]:
    """Return p(g) = P(num_gpu == g) for all jobs with num_gpu > 0.

    Raises ValueError if the CSV has no valid rows — silently returning
    {} would let downstream FragmentationChecker report F=0 always
    (Codex review Q4).
    """
    counts: dict[int, int] = {}
    total = 0
    with open(Path(jobs_csv), newline="") as f:
        for row in csv.DictReader(f):
            g = int(row.get("num_gpu") or 0)
            if g <= 0:
                continue
            counts[g] = counts.get(g, 0) + 1
            total += 1
    if total == 0:
        raise ValueError(
            f"No rows with num_gpu > 0 in {jobs_csv}; cannot compute p(g). "
            f"Downstream fragmentation metric would collapse to zero silently."
        )
    return {g: c / total for g, c in sorted(counts.items())}


def save_dist(dist: dict[int, float], path: Path) -> None:
    if not dist:
        raise ValueError(
            "Refusing to save empty p(g); would silently break "
            "FragmentationChecker downstream.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON requires str keys
    path.write_text(json.dumps({str(k): v for k, v in dist.items()}, indent=2))


def load_dist(path: Path) -> dict[int, float]:
    raw = json.loads(Path(path).read_text())
    return {int(k): float(v) for k, v in raw.items()}
