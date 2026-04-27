"""SWE-bench Lite scenario loader.

Expects a manifest JSONL (one record per instance) and a local repo cache
directory pre-populated with bare clones of each repo. The loader checks
out the repo at `base_commit` into a per-instance temp worktree, then
returns a ready-to-use `Instance`.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from .manager import Instance


def _parse_test_list(val: Any) -> list[str]:
    """FAIL_TO_PASS / PASS_TO_PASS may arrive as a JSON-encoded string.

    Verified on princeton-nlp/SWE-bench_Lite (April 2026): the `datasets`
    export delivers these fields as `str` like `'["t1", "t2"]'`, not native
    `list`. A naive `list(val)` on a string yields a list of characters
    (silent reward corruption in GRPO). Parse JSON if string; pass through
    if already list; empty list otherwise.
    """
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    return []


class SWEBenchLoader:
    """Load SWE-bench Lite instances from a manifest JSONL.

    manifest  : path to swe-bench-lite.jsonl (one JSON per line)
    repo_cache: directory containing one sub-dir per repo (mirror clones
                named e.g. `django__django`, `sympy__sympy`, ...)
    """

    def __init__(self, manifest: Path, repo_cache: Path):
        self._manifest = Path(manifest)
        self._repo_cache = Path(repo_cache)
        self._records: dict[str, dict] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        for line in self._manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            self._records[rec["instance_id"]] = rec

    def list_instance_ids(self) -> list[str]:
        return sorted(self._records.keys())

    def load(self, instance_id: str, work_root: Optional[Path] = None) -> Instance:
        rec = self._records[instance_id]
        repo_name = rec["repo"].replace("/", "__")
        src = self._repo_cache / repo_name
        if not (src / ".git").exists():
            raise FileNotFoundError(
                f"Repo cache missing: {src}. "
                f"Populate with `git clone <upstream> {src}` first."
            )
        # Materialize a fresh checkout at base_commit
        if work_root is None:
            work_root = Path(tempfile.mkdtemp(prefix=f"silr-swe-chk-{instance_id}-"))
        shutil.copytree(src, work_root, dirs_exist_ok=True)
        # `-f` (force) is required on Windows: the repo cache may have
        # autocrlf-adjusted line endings that show as "modified", blocking
        # a plain checkout with "Please commit your changes before switch".
        # Force-checkout discards any such spurious modifications and makes
        # the working tree match the requested commit byte-for-byte.
        # 60s timeout guards against disk-I/O stalls on large repos
        # (sympy/sklearn checkouts can touch tens of thousands of files).
        subprocess.run(
            ["git", "checkout", "-f", "-q", rec["base_commit"]],
            cwd=work_root, check=True, timeout=60,
        )
        return Instance(
            instance_id=rec["instance_id"],
            repo=str(work_root),
            base_commit=rec["base_commit"],
            problem_statement=rec["problem_statement"],
            fail_to_pass=_parse_test_list(rec.get("FAIL_TO_PASS")),
            pass_to_pass=_parse_test_list(rec.get("PASS_TO_PASS")),
            conda_env=rec.get("conda_env"),
        )
