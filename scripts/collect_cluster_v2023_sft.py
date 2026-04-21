"""Replay Best-fit expert on each scenario → write SFT JSONL.

Each JSONL line:
    {
        "scenario_id": ...,
        "seed": ...,
        "messages": [
            {"role": "system", "content": "<system prompt>"},
            {"role": "user", "content": "<observation JSON>"},
            {"role": "assistant", "content": "<tool call JSON>"},
            ... multi-turn ...
        ]
    }

Use per-scenario multiple seeds to get trajectory diversity via
Best-fit tiebreak (see BestFitExpert.__init__).

The JSONL format is NOT directly consumable by scripts/train_sft.py
— that script expects a single JSON array. Convert with one-line
Python after enrichment (see plan Task 23 Step 7).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.cluster_v2023.expert import BestFitExpert
from domains.cluster_v2023.observation import ClusterV2023Observer
from domains.cluster_v2023.prompts import build_system_prompt, build_tool_schemas
from domains.cluster_v2023.scenarios.loader import ScenarioLoader

logger = logging.getLogger(__name__)


def _trajectory(scenario: dict, seed: int) -> dict:
    mgr = ScenarioLoader.build_manager(scenario)
    mgr.solve()
    observer = ClusterV2023Observer(
        mgr, f_threshold=scenario.get("f_threshold", 10.0)
    )
    sys_prompt = build_system_prompt(mgr, build_tool_schemas(mgr))
    messages = [{"role": "system", "content": sys_prompt}]

    expert = BestFitExpert(seed=seed)
    actions = expert.plan(mgr, max_steps=15)

    for action in actions:
        obs = observer.observe()
        # SFT datasets typically need compact JSON in user role
        messages.append({"role": "user", "content": obs.compressed_json})
        messages.append({"role": "assistant",
                         "content": json.dumps(action)})
        BestFitExpert.apply(mgr, action)
        mgr.solve()

    return {"scenario_id": scenario["scenario_id"],
            "seed": seed,
            "messages": messages}


def collect(*, scenario_dir: Path, out_path: Path,
            seeds: list[int]) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for p in ScenarioLoader.list_scenarios(scenario_dir):
            scenario = ScenarioLoader.load(p)
            for seed in seeds:
                traj = _trajectory(scenario, seed)
                f.write(json.dumps(traj) + "\n")
                n += 1
    return n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[0, 1, 2, 3, 4, 5, 6, 7])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    n = collect(scenario_dir=Path(args.scenario_dir),
                out_path=Path(args.out), seeds=args.seeds)
    logger.info("wrote %d trajectories to %s", n, args.out)


if __name__ == "__main__":
    main()
