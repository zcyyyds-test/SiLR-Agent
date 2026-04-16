"""Remove duplicate SFT conversations (byte-identical action sequences).

Dedup key: (scenario_id, tuple of all assistant action JSONs). Keeps first
occurrence of each unique trajectory.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def action_sequence(messages):
    seq = []
    for m in messages:
        if m["role"] != "assistant":
            continue
        c = m["content"]
        matches = re.findall(
            r'\{(?:[^{}]|\{[^{}]*\})*"tool_name"(?:[^{}]|\{[^{}]*\})*\}',
            c,
            re.DOTALL,
        )
        if matches:
            try:
                obj = json.loads(matches[-1])
                # canonical string
                seq.append(json.dumps(obj, sort_keys=True))
            except json.JSONDecodeError:
                seq.append(matches[-1])
    return tuple(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src = json.load(open(args.input))
    seen = {}
    dedup = []
    per_scen_before = Counter()
    per_scen_after = Counter()
    for conv in src:
        per_scen_before[conv["scenario_id"]] += 1
        key = (conv["scenario_id"], action_sequence(conv["messages"]))
        if key in seen:
            continue
        seen[key] = True
        dedup.append(conv)
        per_scen_after[conv["scenario_id"]] += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dedup, f, indent=2, ensure_ascii=False)

    print(f"Input: {len(src)} conversations")
    print(f"Output: {len(dedup)} unique trajectories")
    print(f"Removed: {len(src) - len(dedup)} duplicates ({(len(src)-len(dedup))/len(src)*100:.1f}%)")
    print("\nPer-scenario:")
    for sid in sorted(per_scen_before.keys()):
        if per_scen_before[sid] != per_scen_after[sid]:
            print(f"  {sid}: {per_scen_before[sid]} -> {per_scen_after[sid]}")


if __name__ == "__main__":
    main()
