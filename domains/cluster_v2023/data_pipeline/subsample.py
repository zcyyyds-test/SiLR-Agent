"""Stratified node subsampling — preserve gpu_spec / model distribution."""

from __future__ import annotations

import csv
import random
from pathlib import Path


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def stratified_nodes(
    nodes_csv: Path,
    *,
    target: int,
    seed: int = 42,
) -> list[dict]:
    """Sample `target` nodes, preserving model-column proportions.

    Returns list of dicts with native CSV keys (sn, cpu_milli,
    memory_mib, gpu, model, ...).
    """
    rows = _load_csv(Path(nodes_csv))
    if target >= len(rows):
        return rows[:]

    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    total = len(rows)
    rng = random.Random(seed)
    models_sorted = sorted(by_model)  # deterministic order
    n_models = len(models_sorted)

    # Kimi review Q1: when target < n_models, the proportional-with-min-1
    # allocator starves tail buckets (the last few get take=0 once
    # `remaining` is exhausted). Diversity beats proportional fidelity at
    # this scale: randomly pick `target` models, one row from each.
    if target < n_models:
        chosen_models = rng.sample(models_sorted, target)
        return [rng.choice(by_model[m]) for m in chosen_models]

    out: list[dict] = []
    remaining = target
    for i, model in enumerate(models_sorted):
        group = by_model[model]
        if i == n_models - 1:
            # Clamp to len(group): if earlier rounding left too much remaining
            # (e.g. target=8 over sizes [3,3,3,1]), do NOT try to sample more
            # than the group holds — rng.sample raises ValueError otherwise.
            take = min(remaining, len(group))
        else:
            take = max(1, round(target * len(group) / total))
            take = min(take, len(group), remaining)
        remaining -= take
        picked = rng.sample(group, take)
        out.extend(picked)

    return out
