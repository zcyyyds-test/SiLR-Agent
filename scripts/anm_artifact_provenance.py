"""Small provenance helpers shared by ANM experiment scripts.

The ANM evidence is assembled from multiple runs. These helpers keep each JSON
artifact tied to the scenario definitions and code surface that produced it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

from domains.anm import ANMScenarioLoader


ROOT = Path(__file__).resolve().parents[1]

CODE_FINGERPRINT_PATHS = (
    "domains/anm/checkers.py",
    "domains/anm/config.py",
    "domains/anm/manager.py",
    "domains/anm/observation.py",
    "domains/anm/scenarios.py",
    "domains/anm/tools.py",
    "silr/agent/action_parser.py",
    "silr/agent/config.py",
    "silr/agent/react_loop.py",
    "silr/agent/types.py",
    "silr/core/config.py",
    "silr/core/interfaces.py",
    "silr/verifier/types.py",
    "silr/verifier/verifier.py",
)


def _float_dict(data: dict[int, float] | None) -> dict[str, float] | None:
    if data is None:
        return None
    return {str(k): float(v) for k, v in sorted(data.items())}


def scenario_record(s: Any) -> dict[str, Any]:
    return {
        "id": s.id,
        "difficulty": s.difficulty,
        "P_load": _float_dict(s.P_load),
        "P_pot": _float_dict(s.P_pot),
        "initial_P_set": _float_dict(s.initial_P_set),
        "initial_Q_set": _float_dict(s.initial_Q_set),
        "initial_soc": _float_dict(s.initial_soc),
        "source_seed": s.source_seed,
        "source_step": s.source_step,
    }


def record_hash(record: dict[str, Any]) -> str:
    blob = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def scenario_manifest(
    scenario_ids: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    loader = ANMScenarioLoader()
    scenarios = (
        loader.load_all()
        if scenario_ids is None
        else [loader.load(sid) for sid in scenario_ids]
    )
    out: dict[str, dict[str, Any]] = {}
    for s in scenarios:
        record = scenario_record(s)
        out[s.id] = {
            "sha256": record_hash(record),
            "record": record,
        }
    return out


def file_sha256(path: str) -> str | None:
    p = ROOT / path
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def code_fingerprint(extra_paths: Iterable[str] = ()) -> dict[str, Any]:
    status = _git_output(["status", "--short"]) or ""
    tracked_diff = _git_output(["diff", "--name-only"]) or ""
    git_commit = _git_output(["rev-parse", "HEAD"])
    env_commit = os.environ.get("SILR_CODE_COMMIT")
    env_dirty = os.environ.get("SILR_CODE_DIRTY")
    env_sync_id = os.environ.get("SILR_SYNC_ID")
    paths = list(dict.fromkeys([*CODE_FINGERPRINT_PATHS, *extra_paths]))
    return {
        "git_commit": git_commit or env_commit,
        "git_commit_source": "git" if git_commit else ("env" if env_commit else None),
        "git_dirty": bool(status) or env_dirty in {"1", "true", "True", "yes"},
        "source_dirty": env_dirty,
        "sync_id": env_sync_id,
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_files": [p for p in tracked_diff.splitlines() if p],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "gym-anm": _package_version("gym-anm"),
            "numpy": _package_version("numpy"),
            "cvxpy": _package_version("cvxpy"),
        },
        "file_sha256": {path: file_sha256(path) for path in paths},
    }


def sanitized_config(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    for key in ("api_key", "openai_api_key"):
        if out.get(key) not in (None, "", "EMPTY"):
            out[key] = "***REDACTED***"
    for key in (
        "SILR_MAX_TOKENS",
        "SILR_SCALAR_PROGRESS_RELATIVE_SLACK",
        "SILR_SYNC_ID",
        "CUDA_VISIBLE_DEVICES",
    ):
        value = os.environ.get(key)
        if value not in (None, ""):
            out[f"env_{key}"] = value
    return out
