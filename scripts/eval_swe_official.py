"""Wrap the official SWE-bench harness — runs on macOS with Docker.

Invokes `python -m swebench.harness.run_evaluation` and parses the
resulting report.json into a single summary line.

Usage:
    python scripts/eval_swe_official.py \\
        --predictions outputs/swe_eval/predictions-14B-zs.jsonl \\
        --run-id 14B-zs
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", args.dataset,
        "--split", args.split,
        "--predictions_path", args.predictions,
        "--run_id", args.run_id,
        "--max_workers", str(args.workers),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # Parse report. SWE-bench 2.x may place it under multiple path variants.
    candidates = [
        Path(f"logs/run_evaluation/{args.run_id}/{args.run_id}/report.json"),
    ]
    candidates.extend(Path("logs/run_evaluation").rglob(f"{args.run_id}/report.json"))
    report = next((c for c in candidates if c.exists()), None)
    if report is None:
        print("report.json not found", file=sys.stderr)
        sys.exit(2)

    data = json.loads(report.read_text())
    resolved = data.get("resolved_ids") or data.get("resolved", [])
    total = data.get("total_instances") or len(data.get("instance_ids", []))
    pct = 100.0 * len(resolved) / max(total, 1)
    print(f"\n=== {args.run_id}: {len(resolved)}/{total} resolved  ({pct:.1f}%) ===")


if __name__ == "__main__":
    main()
