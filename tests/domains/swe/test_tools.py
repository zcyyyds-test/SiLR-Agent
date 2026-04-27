from __future__ import annotations

from pathlib import Path

import pytest

from tests.domains.swe.test_manager import tiny_repo   # reuse fixture
from domains.swe.manager import RepoManager, Instance
from domains.swe.tools import LocalizeTool, PatchTool


@pytest.fixture
def mgr(tiny_repo: Path) -> RepoManager:
    inst = Instance(
        instance_id="tool-1",
        repo=str(tiny_repo),
        base_commit="HEAD",
        problem_statement="fix add",
        fail_to_pass=["tests/test_ops.py::test_add"],
        pass_to_pass=["tests/test_ops.py::test_identity"],
    )
    return RepoManager(inst)


def test_localize_stores_lines(mgr: RepoManager) -> None:
    tool = LocalizeTool(mgr)
    result = tool.execute(locations=["src/ops.py:2"])
    assert result["status"] == "success"
    assert mgr.localized == ["src/ops.py:2"]


def test_localize_rejects_empty(mgr: RepoManager) -> None:
    tool = LocalizeTool(mgr)
    result = tool.execute(locations=[])
    assert result["status"] == "error"
    assert "locations" in result["error"].lower()


def test_patch_stores_diff(mgr: RepoManager) -> None:
    tool = PatchTool(mgr)
    diff = (
        "diff --git a/src/ops.py b/src/ops.py\n"
        "--- a/src/ops.py\n"
        "+++ b/src/ops.py\n"
        "@@ -1 +1 @@\n"
        "-def add(a, b): return a - b\n"
        "+def add(a, b): return a + b\n"
    )
    result = tool.execute(patch=diff)
    assert result["status"] == "success"
    assert mgr.pending_patch == diff


def test_patch_rejects_non_diff(mgr: RepoManager) -> None:
    tool = PatchTool(mgr)
    result = tool.execute(patch="I'm not a diff")
    assert result["status"] == "error"
