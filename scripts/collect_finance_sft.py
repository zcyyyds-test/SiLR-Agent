"""Collect SFT training data using a teacher model on finance scenarios.

Usage:
    python scripts/collect_finance_sft.py \
        --model gemini-3-flash-preview \
        --base-url https://api.lemonapi.org/v1 \
        --api-key $LEMON_API_KEY \
        --repeats 10 \
        --output outputs/finance_sft_collection
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.finance import FinanceManager, build_finance_domain_config, FinanceScenarioLoader
from silr.agent.config import AgentConfig
from silr.agent.llm.openai_client import OpenAIClient
from silr.eval.runner import EvalRunner


def _build_llm_client(provider: str, args):
    """Pick the LLM client based on --provider."""
    if provider == "openai":
        return OpenAIClient(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    if provider == "kimi":
        from silr.agent.llm.kimi_anthropic_client import KimiAnthropicClient
        return KimiAnthropicClient(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    raise ValueError(f"Unknown provider: {provider}")


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
    parser = argparse.ArgumentParser(description="Collect SFT data for finance domain")
    parser.add_argument("--provider", default="openai", choices=["openai", "kimi"],
                        help="LLM provider. kimi uses the Anthropic-compat endpoint.")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--base-url", default=None,
                        help="Provider-specific base URL override")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", default="outputs/finance_sft_collection")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Specific scenario IDs (default: all)")
    args = parser.parse_args()

    setup_logging("collect_finance_sft", args.output)
    logger = logging.getLogger(__name__)

    logger.info(f"Config: model={args.model}, repeats={args.repeats}, "
                f"max_steps={args.max_steps}, output={args.output}")

    llm_client = _build_llm_client(args.provider, args)
    logger.info(f"Provider: {args.provider}, bare-text mode: {not llm_client.supports_tool_use()}")
    domain_config = build_finance_domain_config()
    loader = FinanceScenarioLoader()
    agent_config = AgentConfig(
        max_steps=args.max_steps,
        max_proposals_per_step=3,
        consecutive_fail_limit=2,
        temperature=0.7,
    )

    scenarios = loader.load_all() if args.scenarios is None else [
        loader.load(sid) for sid in args.scenarios
    ]
    logger.info(f"Scenarios: {len(scenarios)}, repeats: {args.repeats}, "
                f"total episodes: {len(scenarios) * args.repeats}")

    runner = EvalRunner(
        llm_client=llm_client,
        domain_config=domain_config,
        manager_factory=FinanceManager,
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
                    "difficulty": scenario.difficulty,
                    "recovered": result.recovered,
                    "total_steps": result.total_steps,
                    "total_proposals": result.total_proposals,
                    "total_rejections": result.total_rejections,
                    "failsafe_triggered": result.failsafe_triggered,
                })
                status = "RECOVERED" if result.recovered else "FAILED"
                logger.info(f"  {scenario.id} [{scenario.difficulty}]: {status} "
                            f"({result.total_steps} steps)")
            except Exception as e:
                logger.error(f"  Episode failed: {scenario.id} rep={rep}: {e}")
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

    # Summary
    total = len(all_results)
    recovered = sum(1 for r in all_results if r.get("recovered"))
    errors = sum(1 for r in all_results if "error" in r)
    logger.info(f"\n{'='*60}")
    logger.info(f"Collection complete in {elapsed:.1f}s")
    logger.info(f"Total episodes: {total}")
    logger.info(f"Recovered: {recovered}/{total} ({recovered/max(total,1)*100:.1f}%)")
    logger.info(f"Errors: {errors}")

    if recorder:
        sft = recorder.export_sft_data()
        dpo = recorder.export_dpo_pairs()
        logger.info(f"SFT samples: {len(sft)}")
        logger.info(f"DPO pairs: {len(dpo)}")

    # Per-difficulty
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in all_results if r.get("difficulty") == diff and "error" not in r]
        if diff_results:
            n = len(diff_results)
            rec = sum(1 for r in diff_results if r["recovered"])
            logger.info(f"  {diff}: {rec}/{n} ({rec/n*100:.1f}%)")

    # Per-scenario
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
