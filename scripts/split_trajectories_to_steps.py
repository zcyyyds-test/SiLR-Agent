"""Split trajectory SFT records into per-step samples.

Why: Each trajectory record is [system, user_0, assistant_0, user_1,
assistant_1, ..., user_{k-1}, assistant_{k-1}] — typically 27 messages
for 13-step fragmentation_surge, with ~17kchar observation per user
turn. That tokenizes to ~100k tokens per trajectory. SFTTrainer's
max_length=4096 (or even 16384) truncates far before assistant_0, so
no assistant target token ever appears in the training sequence.

Fix: flatten every trajectory into K independent single-turn samples,
each [system, user_t, assistant_t]. One turn is ~8k tokens; set
max_seq_len >= 16384 so the target survives truncation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _compress_json_string(s: str) -> str:
    """If s parses as JSON, re-serialize without whitespace.

    Observer exports pretty-printed JSON with 2-space indent — ~30% of
    those characters are whitespace. Round-tripping drops them at zero
    semantic cost and trims ~1-2k tokens per user turn.
    """
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def split_trajectory(record: dict, *, compress_user: bool = False) -> list[dict]:
    """Return one sample per assistant message in the trajectory.

    Each sample preserves the scenario_id / seed so downstream tools
    can trace it back to the originating trajectory.
    """
    msgs = record["messages"]
    system_msgs = [m for m in msgs if m["role"] == "system"]
    out: list[dict] = []
    # Walk the tail: user[i] must precede assistant[i].
    last_user: dict | None = None
    step_idx = 0
    for m in msgs:
        if m["role"] == "user":
            content = m["content"]
            if compress_user:
                content = _compress_json_string(content)
            last_user = {"role": "user", "content": content}
        elif m["role"] == "assistant":
            if last_user is None:
                continue  # assistant without preceding user (shouldn't happen)
            out.append({
                "scenario_id": record.get("scenario_id"),
                "seed": record.get("seed"),
                "step": step_idx,
                "messages": system_msgs + [last_user, m],
            })
            step_idx += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Input JSONL (one trajectory per line) or JSON array.")
    p.add_argument("--output", required=True,
                   help="Output JSON array (what train_sft.py expects).")
    p.add_argument("--compress-user", action="store_true",
                   help="Round-trip user JSON without whitespace to trim "
                        "~25%% of tokens per observation.")
    args = p.parse_args()

    src = Path(args.input)
    raw = src.read_text(encoding="utf-8")
    try:
        records = json.loads(raw)
        assert isinstance(records, list)
    except (json.JSONDecodeError, AssertionError):
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]

    all_samples: list[dict] = []
    per_scenario: dict[str, int] = {}
    for rec in records:
        samples = split_trajectory(rec, compress_user=args.compress_user)
        all_samples.extend(samples)
        sid = rec.get("scenario_id", "?")
        per_scenario[sid] = per_scenario.get(sid, 0) + len(samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=None)

    total_trajectories = len(records)
    total_samples = len(all_samples)
    print(f"Trajectories: {total_trajectories}")
    print(f"Per-step samples: {total_samples}")
    print(f"Expansion: {total_samples / max(total_trajectories, 1):.2f}x")
    print(f"Scenarios covered: {len(per_scenario)}")
    print(f"Samples per scenario (first 5):")
    for sid, n in list(sorted(per_scenario.items()))[:5]:
        print(f"  {sid}: {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
