"""Merge ANM evaluation JSON artifacts for post-hoc reporting.

The focused N expansion intentionally writes a small separate artifact instead
of mutating the v3 sweep. This helper combines raw episode rows while preserving
source provenance so `anm_eval_report.py` and `anm_story_tables.py` can generate
paper-facing tables over the merged evidence block.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a JSON object")
    return data


def episode_key(ep: dict[str, Any]) -> tuple[Any, ...]:
    return (
        ep.get("scenario"),
        ep.get("policy"),
        ep.get("rep_seed"),
        ep.get("stall_budget"),
        ep.get("total_steps"),
        ep.get("final_penalty"),
    )


def write_output_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def merge_artifacts(paths: list[Path], dedupe: bool) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    scenario_manifest: dict[str, Any] = {}
    manifest_conflicts: list[str] = []
    source_records: list[dict[str, Any]] = []
    policies: list[str] = []
    all_complete = True

    for path in paths:
        data = load_json(path)
        status = data.get("status", {})
        source_records.append({
            "path": str(path),
            "status": status,
            "config": data.get("config", {}),
            "policies": data.get("policies", []),
            "code_fingerprint": data.get("code_fingerprint", {}),
        })
        if isinstance(status, dict) and not status.get("complete", True):
            all_complete = False
        for policy in data.get("policies", []):
            if policy not in policies:
                policies.append(policy)
        for sid, record in data.get("scenario_manifest", {}).items():
            if sid in scenario_manifest and scenario_manifest[sid] != record:
                manifest_conflicts.append(sid)
            else:
                scenario_manifest[sid] = record
        for ep in data.get("episodes", []):
            key = episode_key(ep)
            if dedupe and key in seen:
                continue
            seen.add(key)
            ep_out = dict(ep)
            ep_out.setdefault("source_artifact", path.name)
            episodes.append(ep_out)

    return {
        "status": {
            "complete": all_complete,
            "completed_episodes": len(episodes),
            "expected_episodes": len(episodes),
        },
        "config": {
            "merge_inputs": [str(p) for p in paths],
            "dedupe": dedupe,
        },
        "scenario_manifest": scenario_manifest,
        "code_fingerprint": {
            "merge_sources": source_records,
            "manifest_conflicts": sorted(set(manifest_conflicts)),
        },
        "policies": policies,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    payload = merge_artifacts(paths, dedupe=not args.no_dedupe)
    write_output_atomic(Path(args.output), payload)
    print(
        f"wrote {args.output}: "
        f"{payload['status']['completed_episodes']} episodes from {len(paths)} inputs"
    )


if __name__ == "__main__":
    main()
