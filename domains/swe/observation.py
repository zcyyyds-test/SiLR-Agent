"""SWEObserver: compact view the agent sees between tool calls."""
from __future__ import annotations

from pathlib import Path


class SWEObserver:
    def __init__(self, manager):
        self._mgr = manager

    def observe(self) -> dict:
        state = self._mgr.system_state
        file_tree = self._build_tree(Path(state["work_dir"]), max_depth=2)
        return {
            "instance_id": state["instance_id"],
            "problem_statement": state["problem_statement"],
            "file_tree": file_tree,
            "localized": state["localized"],
            "has_patch": state["has_patch"],
            "is_stable": False,   # SWE domain is always "not yet done"
        }

    def _build_tree(self, root: Path, max_depth: int, _depth: int = 0) -> dict:
        if _depth >= max_depth:
            return {}
        out: dict = {}
        try:
            for p in sorted(root.iterdir()):
                if p.name.startswith(".") or p.name == "__pycache__":
                    continue
                if p.is_dir():
                    out[p.name] = self._build_tree(p, max_depth, _depth + 1)
                else:
                    out[p.name] = "<file>"
        except OSError:
            pass
        return out
