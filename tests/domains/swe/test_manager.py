"""Tests for RepoManager.

Uses a throwaway local git repo fixture so tests don't touch the real
SWE-bench Lite cache. The manager protocol requires nothing more than
a `.git` directory and an initial commit.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from domains.swe.manager import RepoManager, Instance
from domains.swe.observation import SWEObserver


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """Create a git repo with one passing test and one failing test."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("")
    (repo / "src" / "ops.py").write_text("def add(a, b):\n    return a - b\n")  # bug
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("")
    (repo / "tests" / "test_ops.py").write_text(
        "from src.ops import add\n"
        "def test_add(): assert add(1, 2) == 3\n"
        "def test_identity(): assert add(0, 0) == 0\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=repo, check=True,
    )
    return repo


def test_repo_manager_initial_state(tiny_repo: Path) -> None:
    instance = Instance(
        instance_id="tiny-1",
        repo=str(tiny_repo),
        base_commit="HEAD",
        problem_statement="add(1,2) should be 3",
        fail_to_pass=["tests/test_ops.py::test_add"],
        pass_to_pass=["tests/test_ops.py::test_identity"],
        conda_env=None,
    )
    mgr = RepoManager(instance)
    assert mgr.sim_time == 0.0
    assert mgr.base_mva == 1.0
    assert mgr.ast_ok is None       # solve() hasn't been called yet
    assert mgr.imports_ok is None
    assert mgr.pending_patch == ""
    state = mgr.system_state
    assert state["instance_id"] == "tiny-1"
    assert state["has_patch"] is False


def test_solve_with_good_patch(tiny_repo: Path) -> None:
    instance = Instance(
        instance_id="tiny-good",
        repo=str(tiny_repo),
        base_commit="HEAD",
        problem_statement="fix add",
        fail_to_pass=["tests/test_ops.py::test_add"],
        pass_to_pass=["tests/test_ops.py::test_identity"],
    )
    mgr = RepoManager(instance)
    good_patch = (
        "diff --git a/src/ops.py b/src/ops.py\n"
        "--- a/src/ops.py\n"
        "+++ b/src/ops.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n"
        "-    return a - b\n"
        "+    return a + b\n"
    )
    mgr.set_patch(good_patch)
    assert mgr.solve() is True
    assert mgr.ast_ok is True
    assert mgr.imports_ok is True


def test_solve_with_syntax_break(tiny_repo: Path) -> None:
    instance = Instance(
        instance_id="tiny-bad",
        repo=str(tiny_repo),
        base_commit="HEAD",
        problem_statement="bogus",
        fail_to_pass=[],
        pass_to_pass=[],
    )
    mgr = RepoManager(instance)
    bad_patch = (
        "diff --git a/src/ops.py b/src/ops.py\n"
        "--- a/src/ops.py\n"
        "+++ b/src/ops.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n"
        "-    return a - b\n"
        "+    return a +++ b   # syntax error\n"
    )
    mgr.set_patch(bad_patch)
    assert mgr.solve() is False
    assert mgr.ast_ok is False
    assert mgr.imports_ok is False


def test_shadow_copy_is_isolated(tiny_repo: Path) -> None:
    instance = Instance(
        instance_id="tiny-shadow",
        repo=str(tiny_repo),
        base_commit="HEAD",
        problem_statement="",
        fail_to_pass=[],
        pass_to_pass=[],
    )
    mgr = RepoManager(instance)
    shadow = mgr.create_shadow_copy()
    shadow.set_patch("xxx")
    assert mgr.pending_patch == ""
    assert shadow.pending_patch == "xxx"
    assert shadow.work_dir != mgr.work_dir


def test_shadow_does_not_leak_instance_list_mutation(tiny_repo: Path) -> None:
    """Regression test: dataclasses.replace is shallow; use deepcopy."""
    instance = Instance(
        instance_id="leak-test",
        repo=str(tiny_repo),
        base_commit="HEAD",
        problem_statement="",
        fail_to_pass=["a::b"],
        pass_to_pass=["c::d"],
    )
    mgr = RepoManager(instance)
    shadow = mgr.create_shadow_copy()
    shadow.instance.fail_to_pass.append("leaked::test")
    shadow.instance.pass_to_pass.clear()
    # Original must remain unaffected by shadow's list mutations.
    assert mgr.instance.fail_to_pass == ["a::b"]
    assert mgr.instance.pass_to_pass == ["c::d"]


def test_solve_handles_deleted_file(tiny_repo: Path) -> None:
    """A patch that deletes a .py file must not crash _check_ast."""
    instance = Instance(
        instance_id="tiny-del",
        repo=str(tiny_repo),
        base_commit="HEAD",
        problem_statement="delete unused file",
        fail_to_pass=[],
        pass_to_pass=[],
    )
    mgr = RepoManager(instance)
    # Patch that removes src/ops.py entirely (deletion form of unified diff).
    delete_patch = (
        "diff --git a/src/ops.py b/src/ops.py\n"
        "deleted file mode 100644\n"
        "--- a/src/ops.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-def add(a, b):\n"
        "-    return a - b\n"
    )
    mgr.set_patch(delete_patch)
    # Should not raise FileNotFoundError; ast_ok True because deleted files
    # are skipped (no AST to verify).
    result = mgr.solve()
    assert mgr.ast_ok is True
    # imports_ok may be True or False depending on whether deleting ops.py
    # breaks `import src`; we just assert solve() returned cleanly.
    assert result in (True, False)


def test_solve_handles_unicode_surrogate_in_patch(tiny_repo: Path) -> None:
    """LLM-generated patches with lone surrogate bytes must not crash."""
    instance = Instance(
        instance_id="tiny-surrogate",
        repo=str(tiny_repo),
        base_commit="HEAD",
        problem_statement="",
        fail_to_pass=[],
        pass_to_pass=[],
    )
    mgr = RepoManager(instance)
    # A lone surrogate (U+D800) — invalid UTF-8 but LLMs can emit it.
    nasty = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+\ud800\n"
    mgr.set_patch(nasty)
    # Must return a bool, not raise UnicodeEncodeError.
    result = mgr.solve()
    assert isinstance(result, bool)
    # Broken patch should not claim success.
    assert result is False


def test_observer_returns_problem_and_tree(tiny_repo: Path) -> None:
    inst = Instance(
        instance_id="obs-1",
        repo=str(tiny_repo),
        base_commit="HEAD",
        problem_statement="add is broken",
        fail_to_pass=[],
        pass_to_pass=[],
    )
    mgr = RepoManager(inst)
    obs = SWEObserver(mgr).observe()
    assert obs["problem_statement"] == "add is broken"
    assert "src" in obs["file_tree"]
    assert "tests" in obs["file_tree"]
