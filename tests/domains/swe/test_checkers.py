from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from domains.swe.manager import RepoManager, Instance
from domains.swe.checkers import RegressionChecker, TargetTestChecker


@pytest.fixture
def fixed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("")
    (repo / "src" / "ops.py").write_text("def add(a, b): return a + b\n")  # already fixed
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


def test_regression_checker_all_green(fixed_repo: Path) -> None:
    inst = Instance(
        instance_id="reg-1",
        repo=str(fixed_repo),
        base_commit="HEAD",
        problem_statement="",
        fail_to_pass=[],
        pass_to_pass=["tests/test_ops.py::test_identity"],
    )
    mgr = RepoManager(inst)
    checker = RegressionChecker()
    result = checker.check(mgr.system_state, mgr.base_mva)
    assert result.passed is True
    assert result.summary["n_failed"] == 0


def test_target_checker_all_green(fixed_repo: Path) -> None:
    inst = Instance(
        instance_id="tgt-1",
        repo=str(fixed_repo),
        base_commit="HEAD",
        problem_statement="",
        fail_to_pass=["tests/test_ops.py::test_add"],
        pass_to_pass=[],
    )
    mgr = RepoManager(inst)
    checker = TargetTestChecker()
    result = checker.check(mgr.system_state, mgr.base_mva)
    assert result.passed is True
    assert result.summary["n_target"] == 1
    assert result.summary["n_green"] == 1


def test_target_checker_counts_real_failures(tmp_path: Path) -> None:
    """Regression test: dropping -x ensures all tests run so n_green is accurate.

    Fixture has 3 target tests; 2 pass, 1 fails. Without -x pytest runs all
    3; the parser must then report 2 passed + 1 failed. With -x (old
    behaviour) it would stop at the first failure and miscount un-run
    tests as passes.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("")
    (repo / "src" / "ops.py").write_text("def add(a, b): return a + b\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("")
    (repo / "tests" / "test_mixed.py").write_text(
        "from src.ops import add\n"
        "def test_a(): assert add(1, 2) == 3\n"
        "def test_b(): assert add(0, 0) == 0\n"
        "def test_wrong(): assert add(1, 1) == 99\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=repo, check=True,
    )
    inst = Instance(
        instance_id="count-test",
        repo=str(repo),
        base_commit="HEAD",
        problem_statement="",
        fail_to_pass=[
            "tests/test_mixed.py::test_a",
            "tests/test_mixed.py::test_b",
            "tests/test_mixed.py::test_wrong",
        ],
        pass_to_pass=[],
    )
    mgr = RepoManager(inst)
    result = TargetTestChecker().check(mgr.system_state, mgr.base_mva)
    assert result.passed is False
    assert result.summary["n_target"] == 3
    assert result.summary["n_green"] == 2
    assert result.summary["n_red"] == 1


def test_regression_checker_missing_instance_fails_safely(fixed_repo: Path) -> None:
    """Unregistered instance → passed=False with error summary, no crash."""
    from domains.swe.checkers import _INSTANCE_REGISTRY
    inst = Instance(
        instance_id="unreg-1",
        repo=str(fixed_repo),
        base_commit="HEAD",
        problem_statement="",
        fail_to_pass=[],
        pass_to_pass=["tests/test_ops.py::test_identity"],
    )
    mgr = RepoManager(inst)
    _INSTANCE_REGISTRY.pop("unreg-1", None)
    result = RegressionChecker().check(mgr.system_state, mgr.base_mva)
    assert result.passed is False
    assert "error" in result.summary
