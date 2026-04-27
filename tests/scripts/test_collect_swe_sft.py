from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.collect_swe_sft import filter_swegym, swegym_record_to_messages


def test_filter_keeps_only_lite_repos(tmp_path: Path) -> None:
    records = [
        {"instance_id": "django__django-1", "repo": "django/django",
         "problem_statement": "foo", "patch": "diff --git a/x b/x"},
        {"instance_id": "other/other-1", "repo": "other/other",
         "problem_statement": "bar", "patch": "diff --git a/y b/y"},
    ]
    src = tmp_path / "swegym.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in records))
    out = tmp_path / "sft.jsonl"
    kept = filter_swegym(src, out, lite_repos={"django/django"})
    assert kept == 1
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["messages"][-1]["role"] == "assistant"


def test_message_shape_has_localize_then_patch() -> None:
    rec = {
        "instance_id": "x",
        "repo": "django/django",
        "problem_statement": "thing",
        "patch": "diff --git a/z b/z\n--- a/z\n+++ b/z\n@@\n-x\n+y\n",
    }
    msgs = swegym_record_to_messages(rec)
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert roles[-1] == "assistant"
    assert "patch" in msgs[-1]["content"]


def test_oversized_trajectory_is_dropped(tmp_path: Path) -> None:
    """Records whose total message bytes exceed 12KB should be filtered out."""
    huge_patch = "diff --git a/big b/big\n" + ("x" * 15_000)
    records = [
        {"instance_id": "django__big-1", "repo": "django/django",
         "problem_statement": "huge", "patch": huge_patch},
    ]
    src = tmp_path / "swegym.jsonl"
    src.write_text(json.dumps(records[0]))
    out = tmp_path / "sft.jsonl"
    kept = filter_swegym(src, out, lite_repos={"django/django"})
    assert kept == 0
