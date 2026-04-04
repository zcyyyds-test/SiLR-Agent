"""Collect SFT training data using a teacher model on cluster scheduling scenarios.

Usage:
    python scripts/collect_sft_data.py \
        --model gemini-3-flash-preview \
        --base-url https://api.openai.com/v1 \
        --api-key $OPENAI_API_KEY \
        --repeats 10 \
        --output outputs/sft_collection

Runs all cluster scenarios N times each, records successful trajectories,
and exports SFT data + DPO pairs.
"""

import argparse
import json
import logging
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.cluster import ClusterManager, build_cluster_domain_config, ClusterScenarioLoader
from silr.agent.config import AgentConfig
from silr.agent.llm.openai_client import OpenAIClient
from silr.eval.runner import EvalRunner


def setup_logging(name: str, log_dir: str = "."):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Collect SFT data for cluster domain")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", default="outputs/sft_collection")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Specific scenario IDs to run (default: all)")
    args = parser.parse_args()

    setup_logging("collect_sft", args.output)
    logger = logging.getLogger(__name__)

    logger.info(f"Config: model={args.model}, repeats={args.repeats}, "
                f"max_steps={args.max_steps}, output={args.output}")

    # Setup
    llm_client = OpenAIClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    domain_config = build_cluster_domain_config()
    loader = ClusterScenarioLoader()
    agent_config = AgentConfig(
        max_steps=args.max_steps,
        max_proposals_per_step=3,
        consecutive_fail_limit=2,
        temperature=0.7,  # Non-zero for diversity across repeats
    )

    scenarios = loader.load_all() if args.scenarios is None else [
        loader.load(sid) for sid in args.scenarios
    ]
    logger.info(f"Scenarios: {len(scenarios)}, repeats: {args.repeats}, "
                f"total episodes: {len(scenarios) * args.repeats}")

    # Run collection
    runner = EvalRunner(
        llm_client=llm_client,
        domain_config=domain_config,
        manager_factory=ClusterManager,
        scenario_loader=loader,
        config=agent_config,
        record_trajectories=True,
    )

    all_results = []
    t0 = time.perf_counter()

    for rep in range(args.repeats):
        logger.info(f"=== Repeat {rep + 1}/{args.repeats} ===")
        for scenario in scenarios:
            try:
                result = runner.run_scenario(scenario)
                all_results.append({
                    "repeat": rep,
                    "scenario_id": scenario.id,
                    "recovered": result.recovered,
                    "total_steps": result.total_steps,
                    "total_proposals": result.total_proposals,
                    "total_rejections": result.total_rejections,
                    "failsafe_triggered": result.failsafe_triggered,
                })
            except Exception as e:
                logger.error(f"Episode failed: {scenario.id} rep={rep}: {e}")
                all_results.append({
                    "repeat": rep,
                    "scenario_id": scenario.id,
                    "error": str(e),
                })

    elapsed = time.perf_counter() - t0

    # Save trajectories
    recorder = runner.trajectory_recorder
    if recorder:
        recorder.save(args.output)

    # Save episode summary
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "episodes.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    total = len(all_results)
    recovered = sum(1 for r in all_results if r.get("recovered"))
    errors = sum(1 for r in all_results if "error" in r)
    logger.info(f"\n{'='*60}")
    logger.info(f"Collection complete in {elapsed:.1f}s")
    logger.info(f"Total episodes: {total}")
    logger.info(f"Recovered: {recovered}/{total} ({recovered/total*100:.1f}%)")
    logger.info(f"Errors: {errors}")

    if recorder:
        sft = recorder.export_sft_data()
        dpo = recorder.export_dpo_pairs()
        logger.info(f"SFT samples: {len(sft)}")
        logger.info(f"DPO pairs: {len(dpo)}")

    # Per-scenario breakdown
    from collections import Counter
    scenario_stats: dict[str, dict] = {}
    for r in all_results:
        sid = r.get("scenario_id", "unknown")
        if sid not in scenario_stats:
            scenario_stats[sid] = {"total": 0, "recovered": 0, "errors": 0}
        scenario_stats[sid]["total"] += 1
        if r.get("recovered"):
            scenario_stats[sid]["recovered"] += 1
        if "error" in r:
            scenario_stats[sid]["errors"] += 1

    logger.info(f"\nPer-scenario breakdown:")
    for sid in sorted(scenario_stats):
        s = scenario_stats[sid]
        rate = s["recovered"] / s["total"] * 100 if s["total"] else 0
        logger.info(f"  {sid}: {s['recovered']}/{s['total']} ({rate:.0f}%)")


if __name__ == "__main__":
    main()
