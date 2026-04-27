"""SWE domain tools (Agentless two-stage).

LocalizeTool  — agent outputs a list of `file:line` strings identifying
                where the bug likely lives. Stored on the manager.
PatchTool     — agent outputs a unified diff; stored as the pending patch
                (execution is deferred to `solve()`).

PatchTool doubles as the submit action — once called, the verifier will
run solve() + checkers and issue a PASS/FAIL verdict, ending the episode.
"""
from __future__ import annotations

from silr.tools.base import BaseTool
from silr.exceptions import ValidationError


def _looks_like_unified_diff(text: str) -> bool:
    """Minimal shape check — avoid model outputting prose as 'patch'."""
    return (
        "diff --git " in text
        and "---" in text
        and "+++" in text
        and "@@" in text
    )


class LocalizeTool(BaseTool):
    name = "localize"
    description = (
        "Identify candidate buggy locations. "
        "Input: locations (list of 'path/to/file.py:line_no' strings). "
        "Output: confirmation."
    )

    def _validate_params(self, locations: list[str] | None = None, **kwargs) -> None:
        if not locations:
            raise ValidationError("locations must be a non-empty list of 'file:line' strings")
        for loc in locations:
            if ":" not in str(loc):
                raise ValidationError(f"bad location format: {loc!r}")

    def _run(self, locations: list[str], **kwargs) -> dict:
        self.manager.set_localized(locations)
        return {"stored": len(locations)}


class PatchTool(BaseTool):
    name = "patch"
    description = (
        "Submit a unified diff that fixes the bug. "
        "Input: patch (str, unified diff format). "
        "This is the terminal action — after this the verifier runs."
    )

    def _validate_params(self, patch: str | None = None, **kwargs) -> None:
        if not patch:
            raise ValidationError("patch is required")
        if not _looks_like_unified_diff(patch):
            raise ValidationError("patch does not look like a unified diff")

    def _run(self, patch: str, **kwargs) -> dict:
        # Use errors="replace" so lone surrogates (rare LLM outputs) don't
        # crash here after the patch was already stored. Manager's
        # _apply_patch uses surrogateescape and makes the final judgment;
        # this byte count is diagnostic only.
        n_bytes = len(patch.encode("utf-8", errors="replace"))
        self.manager.set_patch(patch)
        return {"patch_bytes": n_bytes}


def create_swe_toolset(manager) -> dict:
    return {
        "localize": LocalizeTool(manager),
        "patch": PatchTool(manager),
    }
