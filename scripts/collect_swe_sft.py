"""Filter open-source SWE-Gym trajectories down to SWE-bench Lite's 12 repos.

Input : SWE-Gym JSONL (one record per instance, with `patch` field).
Output: SFT-ready JSONL of {"messages": [...]} records, matching the shape
        train_swe_sft.py expects. Drops records >4K tokens (tokenizer-agnostic
        proxy: total content byte length / 3 > 4096).

Usage:
    python scripts/collect_swe_sft.py \\
        --input D:/zcy/silr-swe-cache/swegym.jsonl \\
        --output D:/zcy/silr-swe-cache/swe_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# NOTE: SWE-Gym is intentionally disjoint from SWE-bench Lite's repos
# (pandas, moto, mypy, dvc, dask, ... vs django, sympy, ...). The original
# plan's "filter to LITE_REPOS" would drop every record. Default is now
# no filter; --repos can optionally restrict.
LITE_REPOS = {
    "django/django", "sympy/sympy", "astropy/astropy",
    "scikit-learn/scikit-learn", "matplotlib/matplotlib",
    "sphinx-doc/sphinx", "pytest-dev/pytest", "psf/requests",
    "pydata/xarray", "pylint-dev/pylint", "pallets/flask",
    "mwaskom/seaborn",
}


def swegym_record_to_messages(rec: dict) -> list[dict]:
    """Convert one SWE-Gym record into a two-step Agentless trajectory."""
    system = (
        "You are a code-repair agent. Step 1 localize, Step 2 patch (unified diff)."
    )
    user = json.dumps({
        "instance_id": rec["instance_id"],
        "problem_statement": rec["problem_statement"],
    })
    # The final assistant message wraps the gold patch as a `patch` tool call.
    assistant = json.dumps({"tool_name": "patch", "params": {"patch": rec["patch"]}})
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def filter_swegym(
    src: Path,
    out: Path,
    lite_repos: set | None = None,
    max_bytes: int = 12_000,
) -> int:
    kept = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if lite_repos is not None and rec.get("repo") not in lite_repos:
                continue
            msgs = swegym_record_to_messages(rec)
            # Rough byte-size filter: < 12KB ≈ < 4K tokens (3 bytes/token avg).
            total = sum(len(m["content"]) for m in msgs)
            if total > max_bytes:
                continue
            fh.write(json.dumps({"messages": msgs}) + "\n")
            kept += 1
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--repos",
        choices=["all", "lite"],
        default="all",
        help="'all' (default) keeps every record; 'lite' restricts to SWE-bench"
        " Lite's 12 repos (intended for test-time contamination studies — note"
        " that SWE-Gym itself does NOT overlap with Lite, so 'lite' yields 0).",
    )
    ap.add_argument("--max-bytes", type=int, default=12_000)
    args = ap.parse_args()
    lite = LITE_REPOS if args.repos == "lite" else None
    n = filter_swegym(Path(args.input), Path(args.output), lite, args.max_bytes)
    print(f"kept {n} records → {args.output}")


if __name__ == "__main__":
    main()
