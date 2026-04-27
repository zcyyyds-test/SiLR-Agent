"""RepoManager: SWE-bench Lite instance wrapper implementing BaseSystemManager.

The manager owns a git worktree pointing at the instance's base commit.
`solve()` applies the pending patch, runs AST parse on modified .py files,
and performs an import smoke test. Sub-results (`ast_ok`, `imports_ok`) are
exposed for the GRPO dense reward function.

Shadow copies are independent worktrees in a temp directory — mutations
to the shadow never leak to the original.
"""
from __future__ import annotations

import ast
import copy
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# Top-level directories we skip when picking an import smoke-test target.
# Matches SWE-bench's standard layout: sklearn/ imports but doc/ doesn't.
_DOC_LIKE_DIRS = frozenset({
    "doc", "docs", "test", "tests", "examples", "example",
    "benchmarks", "scripts", "build", "dist",
})

from silr.core.interfaces import BaseSystemManager


@dataclass
class Instance:
    """SWE-bench Lite instance metadata."""
    instance_id: str
    repo: str                           # path to local clone (checked out at base_commit)
    base_commit: str
    problem_statement: str
    fail_to_pass: list[str]             # pytest node ids expected to become green
    pass_to_pass: list[str]             # pytest node ids that must stay green
    conda_env: Optional[str] = None     # e.g. "swe-django" (for rollout verifier)


class RepoManager(BaseSystemManager):
    """Wrap a single SWE-bench Lite instance as a SiLR system manager."""

    def __init__(self, instance: Instance, work_dir: Optional[Path] = None):
        self._instance = instance
        self._time: float = 0.0
        self._pending_patch: str = ""
        self._ast_ok: Optional[bool] = None
        self._imports_ok: Optional[bool] = None
        self._localized: list[str] = []   # stage-1 output kept here for observation
        # Each manager owns its own worktree copy so shadow/original are isolated.
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix=f"silr-swe-{instance.instance_id}-"))
        self._work_dir = work_dir
        # Register so checkers can look up fail_to_pass / pass_to_pass lists.
        from .checkers import register_instance
        register_instance(instance.instance_id, instance)
        # First-time init: copy source repo into work_dir if empty.
        if not (self._work_dir / ".git").exists():
            shutil.copytree(instance.repo, self._work_dir, dirs_exist_ok=True)

    # ── BaseSystemManager ──────────────────────────────────────────
    @property
    def sim_time(self) -> float:
        return self._time

    @property
    def base_mva(self) -> float:
        return 1.0

    @property
    def system_state(self) -> dict:
        return {
            "instance_id": self._instance.instance_id,
            "problem_statement": self._instance.problem_statement,
            "has_patch": bool(self._pending_patch),
            "localized": list(self._localized),
            "ast_ok": self._ast_ok,
            "imports_ok": self._imports_ok,
            "work_dir": str(self._work_dir),
        }

    def create_shadow_copy(self) -> "RepoManager":
        shadow_dir = Path(tempfile.mkdtemp(prefix=f"silr-swe-shadow-{self._instance.instance_id}-"))
        shutil.copytree(self._work_dir, shadow_dir, dirs_exist_ok=True)
        # deepcopy so list fields (fail_to_pass, pass_to_pass) are not aliased.
        shadow_inst = copy.deepcopy(self._instance)
        shadow = RepoManager.__new__(RepoManager)
        shadow._instance = shadow_inst
        shadow._time = self._time
        shadow._pending_patch = self._pending_patch
        shadow._ast_ok = self._ast_ok
        shadow._imports_ok = self._imports_ok
        shadow._localized = list(self._localized)
        shadow._work_dir = shadow_dir
        return shadow

    def solve(self) -> bool:
        """Apply pending patch, then AST-parse modified files and import smoke test.

        Returns True only if BOTH AST parse and import smoke test succeed.
        Sub-results are cached on `ast_ok` / `imports_ok` for reward shaping.
        """
        self._time += 1.0
        if not self._pending_patch:
            # No-op solve: nothing applied, treat as passing (vacuous)
            self._ast_ok = True
            self._imports_ok = True
            return True

        # Apply via `git apply` in the work_dir
        applied = self._apply_patch(self._pending_patch)
        if not applied:
            self._ast_ok = False
            self._imports_ok = False
            return False

        # Enumerate modified files. `None` means git diff failed/hung — we can't
        # judge the patch's effect, so treat as failure rather than vacuous pass.
        modified = self._modified_py_files()
        if modified is None:
            self._ast_ok = False
            self._imports_ok = False
            return False

        self._ast_ok = self._check_ast(modified)
        self._imports_ok = self._ast_ok and self._check_imports()
        return bool(self._ast_ok and self._imports_ok)

    # ── Manager-specific operations ─────────────────────────────────
    @property
    def pending_patch(self) -> str:
        return self._pending_patch

    @property
    def ast_ok(self) -> Optional[bool]:
        return self._ast_ok

    @property
    def imports_ok(self) -> Optional[bool]:
        return self._imports_ok

    @property
    def localized(self) -> list[str]:
        return list(self._localized)

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @property
    def instance(self) -> Instance:
        return self._instance

    def set_localized(self, file_line_list: list[str]) -> None:
        self._localized = list(file_line_list)

    def set_patch(self, patch_text: str) -> None:
        self._pending_patch = patch_text

    def cleanup(self) -> None:
        """Remove the temp work_dir. Safe to call multiple times. Never raises.

        GRPO creates millions of shadow copies; training code must call
        cleanup() on each shadow after verification to avoid /tmp exhaustion.
        """
        try:
            shutil.rmtree(self._work_dir, ignore_errors=True)
        except Exception:
            pass

    # ── Internal helpers ────────────────────────────────────────────
    def _apply_patch(self, patch: str) -> bool:
        # Atomic apply (no --reject): either all hunks apply or nothing does.
        # --reject would leave the worktree in a half-applied state with .rej
        # files, poisoning subsequent solve() calls on the same shadow.
        try:
            # surrogateescape: LLM-generated patches may contain lone surrogates
            # from tokenizer quirks; encode() would normally crash. We pass
            # them through as raw bytes — git apply will either accept or
            # reject them as it would any other byte sequence.
            patch_bytes = patch.encode("utf-8", errors="surrogateescape")
        except (UnicodeEncodeError, AttributeError):
            return False
        try:
            result = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=self._work_dir,
                input=patch_bytes,
                capture_output=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0

    def _modified_py_files(self) -> Optional[list[Path]]:
        """Return list of modified .py paths, or None if git failed/hung.

        None vs [] distinction matters: [] = legitimately nothing changed,
        None = we couldn't tell. solve() treats None as patch failure so a
        broken repo doesn't silently produce a vacuous PASS.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=self._work_dir,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        return [self._work_dir / p for p in result.stdout.splitlines() if p.endswith(".py")]

    def _check_ast(self, modified: list[Path]) -> bool:
        """AST-parse each modified .py file. Deleted files are legitimate.

        A patch can delete a .py file entirely; those appear in `git diff`
        but read_text() would raise FileNotFoundError. Skip them rather than
        crash — a deleted file has no AST to check.
        """
        for f in modified:
            if not f.exists():
                continue
            try:
                ast.parse(f.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError, OSError):
                return False
        return True

    def _check_imports(self) -> bool:
        """Lightweight import smoke: `python -c "import <top_pkg>"` in the repo.

        Picks the top-level package whose name matches the instance's upstream
        repo (e.g. `django__django-*` → `django`). Falls back to the first
        non-doc-like top-level package in sorted order. Filesystem iteration
        order alone is unreliable: a repo with both `doc/__init__.py` and
        `sklearn/__init__.py` could silently test the wrong one.
        """
        # Canonical pkg name derived from instance_id: `django__django-123` → `django`.
        expected = self._instance.instance_id.split("__", 1)[0].lower()
        top_pkgs = sorted(
            p.name for p in self._work_dir.iterdir()
            if p.is_dir()
            and (p / "__init__.py").exists()
            and p.name.lower() not in _DOC_LIKE_DIRS
        )
        if not top_pkgs:
            return True
        # Prefer the package matching the repo name; otherwise first sorted.
        pkg = next((p for p in top_pkgs if p.lower() == expected), top_pkgs[0])
        result = subprocess.run(
            [sys.executable, "-c", f"import {pkg}"],
            cwd=self._work_dir,
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
