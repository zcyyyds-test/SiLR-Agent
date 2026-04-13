"""Evaluate SFT-trained model on portfolio compliance scenarios.

Loads Qwen3-14B + LoRA adapter and runs all finance scenarios.

Usage:
    python scripts/eval_finance.py \
        --base-model Qwen/Qwen3-14B \
        --adapter outputs/finance_sft/final \
        --output outputs/eval_finance \
        --repeats 3
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silr.agent.llm.base import BaseLLMClient, LLMResponse
from silr.agent.config import AgentConfig
from silr.eval.runner import EvalRunner
from domains.finance import FinanceManager, build_finance_domain_config, FinanceScenarioLoader

logger = logging.getLogger(__name__)


class LocalQwenClient(BaseLLMClient):
    """Local Qwen3 + LoRA client for evaluation."""

    def __init__(self, model_path: str, adapter_path: str | None,
                 max_new_tokens: int = 512):
        logger.info(f"Loading base model: {model_path}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="right",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

        if adapter_path:
            logger.info(f"Loading LoRA adapter: {adapter_path}")
            self._model = PeftModel.from_pretrained(self._model, adapter_path)

        self._model.eval()
        self._max_new_tokens = max_new_tokens
        logger.info("Model loaded successfully")

    def chat(
        self,
        messages: list,
        tools: list | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> LLMResponse:
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": self._max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "repetition_penalty": 1.1,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9

            outputs = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][prompt_len:]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return LLMResponse(
            content=content,
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": prompt_len, "completion_tokens": len(new_tokens)},
        )

    def supports_tool_use(self) -> bool:
        return False


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "eval.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on finance scenarios")
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--adapter", default="outputs/finance_sft/final")
    parser.add_argument("--output", default="outputs/eval_finance")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--no-adapter", action="store_true",
                        help="Eval base model without adapter")
    args = parser.parse_args()

    setup_logging(args.output)
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Finance Domain Evaluation")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Adapter: {'none' if args.no_adapter else args.adapter}")
    logger.info(f"Repeats: {args.repeats}, Max steps: {args.max_steps}")

    adapter = None if args.no_adapter else args.adapter
    client = LocalQwenClient(args.base_model, adapter)

    domain_config = build_finance_domain_config()
    loader = FinanceScenarioLoader()
    agent_config = AgentConfig(
        max_steps=args.max_steps,
        max_proposals_per_step=3,
        consecutive_fail_limit=2,
        temperature=0.0,
    )

    runner = EvalRunner(
        llm_client=client,
        domain_config=domain_config,
        manager_factory=FinanceManager,
        scenario_loader=loader,
        config=agent_config,
        record_trajectories=True,
    )

    scenarios = loader.load_all()
    all_results = []

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
                    "total_rejections": result.total_rejections,
                    "failsafe_triggered": result.failsafe_triggered,
                })
                status = "RECOVERED" if result.recovered else "FAILED"
                logger.info(f"  {scenario.id} [{scenario.difficulty}]: {status} "
                            f"({result.total_steps} steps, "
                            f"{result.total_rejections} rejections)")
            except Exception as e:
                logger.error(f"  {scenario.id}: ERROR - {e}")
                all_results.append({
                    "repeat": rep,
                    "scenario_id": scenario.id,
                    "difficulty": scenario.difficulty,
                    "error": str(e),
                })

    elapsed = time.perf_counter() - t0

    # Save raw results
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation complete in {elapsed:.1f}s")

    valid = [r for r in all_results if "error" not in r]
    total = len(valid)
    recovered = sum(1 for r in valid if r["recovered"])
    errors = len(all_results) - total
    logger.info(f"Total: {total}, Recovered: {recovered} ({recovered/max(total,1)*100:.1f}%), Errors: {errors}")

    # Per-difficulty
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in valid if r.get("difficulty") == diff]
        if diff_results:
            n = len(diff_results)
            rec = sum(1 for r in diff_results if r["recovered"])
            avg_steps = sum(r["total_steps"] for r in diff_results) / n
            logger.info(f"  {diff}: {rec}/{n} ({rec/n*100:.1f}%), avg steps: {avg_steps:.1f}")

    # Per-scenario
    scenario_stats = {}
    for r in valid:
        sid = r["scenario_id"]
        if sid not in scenario_stats:
            scenario_stats[sid] = {"total": 0, "recovered": 0, "steps": []}
        scenario_stats[sid]["total"] += 1
        if r["recovered"]:
            scenario_stats[sid]["recovered"] += 1
        scenario_stats[sid]["steps"].append(r["total_steps"])

    logger.info("\nPer-scenario:")
    for sid in sorted(scenario_stats):
        s = scenario_stats[sid]
        rate = s["recovered"] / s["total"] * 100 if s["total"] else 0
        avg_s = sum(s["steps"]) / len(s["steps"])
        logger.info(f"  {sid}: {s['recovered']}/{s['total']} ({rate:.0f}%), avg steps: {avg_s:.1f}")

    # Save metrics
    metrics = {
        "total_episodes": total,
        "recovered": recovered,
        "recovery_rate": recovered / max(total, 1),
        "errors": errors,
        "elapsed_seconds": elapsed,
        "per_scenario": {
            sid: {
                "total": s["total"],
                "recovered": s["recovered"],
                "recovery_rate": s["recovered"] / s["total"] if s["total"] else 0,
                "avg_steps": sum(s["steps"]) / len(s["steps"]),
            }
            for sid, s in scenario_stats.items()
        },
    }
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()
