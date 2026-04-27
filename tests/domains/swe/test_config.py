from __future__ import annotations

from pathlib import Path

from tests.domains.swe.test_manager import tiny_repo
from domains.swe.manager import RepoManager, Instance
from domains.swe.config import build_swe_domain_config


def test_domain_config_accepts_good_patch(tiny_repo: Path) -> None:
    inst = Instance(
        instance_id="cfg-1",
        repo=str(tiny_repo),
        base_commit="HEAD",
        problem_statement="fix add",
        fail_to_pass=["tests/test_ops.py::test_add"],
        pass_to_pass=["tests/test_ops.py::test_identity"],
    )
    mgr = RepoManager(inst)
    cfg = build_swe_domain_config(with_observer=True)
    assert cfg.domain_name == "swe_code_repair"
    assert cfg.allowed_actions == frozenset(["localize", "patch"])
    tools = cfg.create_toolset(mgr)
    good_patch = (
        "diff --git a/src/ops.py b/src/ops.py\n"
        "--- a/src/ops.py\n"
        "+++ b/src/ops.py\n"
        "@@ -1 +1 @@\n"
        "-def add(a, b):\n    return a - b\n"
        "+def add(a, b):\n    return a + b\n"
    )
    tools["localize"].execute(locations=["src/ops.py:2"])
    tools["patch"].execute(patch=good_patch)
    assert mgr.solve() is True
    assert mgr.ast_ok is True
    assert mgr.imports_ok is True
