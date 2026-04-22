"""Probe SFT inference vs training data.

Loads SFT model + adapter, reads first training sample, and for each
assistant turn generates with (system + user_t) and compares to the
recorded assistant_t. If they diverge the model isn't executing what
it was trained to say — usually a template/padding mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--sft-jsonl", required=True)
    p.add_argument("--n-turns", type=int, default=3)
    p.add_argument("--with-history", action="store_true",
                   help="Feed full prior history (train-faithful) "
                        "instead of system+latest_user (SingleTurnClient).")
    args = p.parse_args()

    from scripts.eval_sft import LocalQwenClient

    client = LocalQwenClient(
        model_path=args.model,
        adapter_path=args.adapter,
        max_new_tokens=256,
    )

    with open(args.sft_jsonl) as f:
        rec = json.loads(f.readline())

    print(f"Scenario: {rec['scenario_id']} seed={rec['seed']}")
    msgs = rec["messages"]
    print(f"Total messages: {len(msgs)}")
    print()

    # For each assistant in first n-turns, generate with full or single-turn
    asst_indices = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    for idx in asst_indices[: args.n_turns]:
        history = msgs[:idx]  # system + user/assistant pairs up to this point
        if not args.with_history:
            system_msgs = [m for m in history if m["role"] == "system"]
            last_user = None
            for m in history:
                if m["role"] == "user":
                    last_user = m
            history = system_msgs + ([last_user] if last_user else [])
        truth = msgs[idx]["content"]

        print(f"=== Turn at msg {idx} (history len={len(history)}) ===")
        resp = client.chat(history, temperature=0.0)
        print(f"TRUTH : {truth[:200]}")
        print(f"PRED  : {resp.content[:200]}")
        print(f"MATCH : {truth.strip() == resp.content.strip()}")
        print()


if __name__ == "__main__":
    main()
