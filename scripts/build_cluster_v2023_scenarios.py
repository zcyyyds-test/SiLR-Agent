"""Generate N recovery scenarios from Alibaba v2023 trace.

Usage (production):
    python scripts/build_cluster_v2023_scenarios.py \\
        --raw-dir domains/cluster_v2023/data_pipeline/raw \\
        --out-dir domains/cluster_v2023/scenarios/data \\
        --n 25
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.cluster_v2023.checkers import FragmentationChecker
from domains.cluster_v2023.data_pipeline.inject_faults import (
    inject_fragmentation_surge,
    inject_gpu_spec_mismatch,
    inject_node_failure,
    inject_qos_pressure,
)
from domains.cluster_v2023.data_pipeline.job_window import select_window
from domains.cluster_v2023.data_pipeline.subsample import stratified_nodes
from domains.cluster_v2023.expert import BestFitExpert
from domains.cluster_v2023.manager import ClusterV2023Manager
from domains.cluster_v2023.scenarios.loader import ScenarioLoader

logger = logging.getLogger(__name__)

# Scenario-type distribution per spec §6 (Philly ATC'19 fault ratios)
FAULT_DIST = [
    ("node_failure", 0.50),
    ("gpu_spec_mismatch", 0.20),
    ("qos_pressure", 0.20),
    ("fragmentation_surge", 0.10),
]


def _normalize_nodes(rows: list[dict]) -> dict:
    return {
        r["sn"]: {
            "model": r["model"],
            "cpu_total": int(r.get("cpu_milli") or 0),
            "ram_total_mib": int(r.get("memory_mib") or 0),
            "gpu_total": int(r.get("gpu") or 0),
            "status": "Ready",
            "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0,
        } for r in rows
    }


def _normalize_jobs(rows: list[dict]) -> dict:
    qos_map = {"LS": "LS", "Burstable": "Burstable", "BE": "BE",
               "LatencySensitive": "LS"}
    return {
        r["name"]: {
            "cpu": int(r.get("cpu_milli") or 0),
            "ram_mib": int(r.get("memory_mib") or 0),
            "gpu": int(r.get("num_gpu") or 0),
            "gpu_spec_required": None,
            "qos": qos_map.get((r.get("qos") or "BE").strip(), "BE"),
            "status": "Queued",
        } for r in rows if int(r.get("num_gpu") or 0) > 0
    }


def _apply_fault(fault_type: str, mgr, seed: int) -> dict:
    if fault_type == "node_failure":
        return inject_node_failure(mgr, n_nodes=2, seed=seed)
    if fault_type == "gpu_spec_mismatch":
        return inject_gpu_spec_mismatch(mgr, n_jobs=3, seed=seed)
    if fault_type == "qos_pressure":
        return inject_qos_pressure(mgr, n_ls_queued=5, seed=seed)
    if fault_type == "fragmentation_surge":
        return inject_fragmentation_surge(mgr, seed=seed)
    raise ValueError(fault_type)


def _prescheduled_mgr(nodes: dict, jobs: dict) -> ClusterV2023Manager:
    mgr = ClusterV2023Manager(nodes=nodes, jobs=jobs)
    actions = BestFitExpert().plan(mgr, max_steps=4 * max(len(jobs), 1))
    for a in actions:
        BestFitExpert.apply(mgr, a)
    mgr.solve()
    return mgr


def _cycle_distribution(n: int, *, seed: int):
    """Expand FAULT_DIST to a shuffled bucket of length n.

    Uses the caller's seed (fixed to 0 historically — now parameterised).
    """
    counts = {t: round(p * n) for t, p in FAULT_DIST}
    diff = n - sum(counts.values())
    counts["node_failure"] += diff  # round-off absorbed by dominant class
    bucket: list[str] = []
    for t, c in counts.items():
        bucket.extend([t] * c)
    random.Random(seed).shuffle(bucket)
    for t in bucket:
        yield t


def build_scenarios(*, out_dir: Path, nodes_csv: Path, jobs_csv: Path,
                    target_nodes: int, window_start: float, window_end: float,
                    max_jobs: int, n_scenarios: int, seed: int = 42) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    node_rows = stratified_nodes(nodes_csv, target=target_nodes, seed=seed)
    base_nodes = _normalize_nodes(node_rows)
    max_gpu_per_node = max((n["gpu_total"] for n in base_nodes.values()),
                           default=1)
    jobs_rows = select_window(
        jobs_csv, start=window_start, end=window_end, max_jobs=max_jobs,
        max_gpus_per_job=max_gpu_per_node,
    )
    base_jobs = _normalize_jobs(jobs_rows)

    built = 0
    attempts = 0
    types_iter = _cycle_distribution(n_scenarios, seed=seed)
    while built < n_scenarios and attempts < n_scenarios * 5:
        attempts += 1
        try:
            fault_type = next(types_iter)
        except StopIteration:
            # Rebuild the bucket if we ran through it
            types_iter = _cycle_distribution(n_scenarios, seed=seed + attempts)
            fault_type = next(types_iter)
        ft_seed = rng.randint(0, 10_000_000)
        mgr = _prescheduled_mgr(base_nodes, base_jobs)

        # Baseline F on the clean prescheduled state
        frag = FragmentationChecker(f_threshold=1e9).check(
            mgr.system_state, 1.0)
        f_bestfit_baseline = max(frag.summary["F"], 1.0)

        fault_meta = _apply_fault(fault_type, mgr, seed=ft_seed)
        mgr.solve()

        expert_actions = BestFitExpert(seed=ft_seed).plan(mgr, max_steps=15)
        if not expert_actions:
            logger.warning("unsolvable attempt (%s seed=%d); retrying",
                           fault_type, ft_seed)
            continue

        sid = f"v2023_{fault_type}_{built:02d}"
        scenario = {
            "scenario_id": sid,
            "fault_type": fault_type,
            "fault_meta": fault_meta,
            "nodes": mgr.system_state["nodes"],
            "jobs": mgr.system_state["jobs"],
            "assignments": mgr.system_state["assignments"],
            "expert_solution": expert_actions,
            "f_bestfit_baseline": round(f_bestfit_baseline, 3),
            "f_threshold": round(1.2 * f_bestfit_baseline, 3),
        }
        ScenarioLoader.save(scenario, out_dir / f"{sid}.json")
        built += 1
    return built


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-nodes", type=int, default=40)
    p.add_argument("--window-start", type=float, default=0)
    p.add_argument("--window-end", type=float, default=6000)
    p.add_argument("--max-jobs", type=int, default=400)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    raw = Path(args.raw_dir)
    built = build_scenarios(
        out_dir=Path(args.out_dir),
        nodes_csv=raw / "openb_node_list_gpu_node.csv",
        jobs_csv=raw / "openb_pod_list_default.csv",
        target_nodes=args.target_nodes,
        window_start=args.window_start, window_end=args.window_end,
        max_jobs=args.max_jobs, n_scenarios=args.n, seed=args.seed,
    )
    logger.info("built %d scenarios in %s", built, args.out_dir)
    return 0 if built == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
