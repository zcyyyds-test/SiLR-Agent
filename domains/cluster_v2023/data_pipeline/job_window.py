"""Slice jobs into a time window, apply node-capacity filter."""

from __future__ import annotations

import csv
from pathlib import Path


def select_window(
    jobs_csv: Path,
    *,
    start: float,
    end: float,
    max_jobs: int,
    max_cpu_milli_per_job: int | None = None,
    max_mem_mib_per_job: int | None = None,
    max_gpus_per_job: int | None = None,
) -> list[dict]:
    """Return jobs whose creation_time ∈ [start, end) that fit node caps."""
    out: list[dict] = []
    with open(Path(jobs_csv), newline="") as f:
        for row in csv.DictReader(f):
            ct = float(row.get("creation_time") or 0)
            if not (start <= ct < end):
                continue
            if max_cpu_milli_per_job and int(row.get("cpu_milli") or 0) > max_cpu_milli_per_job:
                continue
            if max_mem_mib_per_job and int(row.get("memory_mib") or 0) > max_mem_mib_per_job:
                continue
            if max_gpus_per_job and int(row.get("num_gpu") or 0) > max_gpus_per_job:
                continue
            out.append(row)

    out.sort(key=lambda r: float(r.get("creation_time") or 0))
    return out[:max_jobs]
