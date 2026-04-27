"""Post-process predictions-*.jsonl: unescape model_patch when it contains
literal `\\n` (2-char) instead of real newlines.

Root cause: eval_swe_inference.py used a markdown-oriented regex
`(diff --git .*?)(?:```|\\Z)` that captured the entire JSON-escaped diff
when the model emitted a tool-call envelope like
`{"tool_name":"patch","params":{"patch":"diff --git ...\\n..."}}`.
The captured string carries literal backslash-n pairs that git apply
silently rejects, collapsing resolve-rate to ~0 for SFT-trained tracks.

Usage:
    python scripts/fix_patch_escapes.py outputs/swe_eval/predictions-*.jsonl

Writes a backup `<file>.broken-<ts>` next to each file before overwriting.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def needs_unescape(s: str) -> bool:
    return "\\n" in s and "\n" not in s


def unescape_patch(s: str) -> str:
    """Recover real diff from a regex-captured JSON-string fragment.

    The original `_DIFF_BLOCK_RE` greedy-matched until end-of-string, so the
    captured text often has a trailing JSON envelope (`"}` or `"}}`) that
    came from `{"params":{"patch":"<DIFF>"}}`. Strip the tail, then re-parse
    the inner via json.loads which handles every JSON string escape (\\n,
    \\t, \\", \\\\, \\u00ff, ...).
    """
    inner = s
    for tail in ('"}}}', '"}}', '"}', '"]', '"]}'):
        if inner.endswith(tail):
            inner = inner[: -len(tail)]
            break
    try:
        decoded = json.loads('"' + inner + '"')
        if "diff --git" in decoded:
            return _ensure_trailing_newline(decoded.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return _ensure_trailing_newline(
            s.encode("utf-8", errors="replace").decode("unicode_escape").strip()
        )
    except (UnicodeDecodeError, UnicodeEncodeError):
        return _ensure_trailing_newline(s)


def _ensure_trailing_newline(s: str) -> str:
    """git apply requires a trailing newline; LLMs frequently omit it."""
    return s + "\n" if s and not s.endswith("\n") else s


def fix_file(path: Path) -> tuple[int, int]:
    fixed = total = 0
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        total += 1
        p = rec.get("model_patch", "") or ""
        if p and needs_unescape(p):
            rec["model_patch"] = unescape_patch(p)
            fixed += 1
        records.append(rec)
    if fixed:
        ts = int(time.time())
        backup = path.with_suffix(path.suffix + f".broken-{ts}")
        shutil.copy(path, backup)
        with path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
    return fixed, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="predictions-*.jsonl files")
    args = ap.parse_args()
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"SKIP {path} (missing)")
            continue
        fixed, total = fix_file(path)
        print(f"{path.name}: fixed {fixed}/{total} records"
              + (" (backup written)" if fixed else " (no changes)"))


if __name__ == "__main__":
    main()
