"""Evaluate SFT-trained model on cluster scheduling scenarios.

Loads Qwen3-14B + LoRA adapter and runs all scenarios.
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
from domains.cluster import ClusterManager, build_cluster_domain_config, ClusterScenarioLoader

logger = logging.getLogger(__name__)


class LocalQwenClient(BaseLLMClient):
    """Local Qwen3 + LoRA client for evaluation."""

    def __init__(self, model_path: str, adapter_path: str, max_new_tokens: int = 512):
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
            # No merge_and_unload — 4-bit quantized weights can't merge directly

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
        # Apply chat template. Qwen3 defaults enable_thinking=True which
        # makes the model emit a <think>...</think> block before the
        # JSON tool call — often truncated at max_new_tokens, leaving
        # no parseable action. Disable for SFT-trained models that output
        # tool JSON directly. Unknown kwarg silently ignored by older
        # tokenizers (Qwen2.5, Llama, etc.) so this is safe across bases.
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
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
                "use_cache": True,  # within-generate KV cache (normal)
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9

            outputs = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][prompt_len:]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        n_new_tokens = len(new_tokens)  # capture before del below

        # Free KV / activations between calls. Observed without this on
        # cluster_v2023 SFT eval: allocations escalated 21 → 152 → 237
        # → 340 GiB across 4 consecutive generates and crashed OOM.
        # Cheap (~10ms) and prevents the leak.
        del outputs, new_tokens, inputs
        torch.cuda.empty_cache()

        # Don't parse JSON here — let ActionParser handle it uniformly
        # (its Layer 2/3 regex is more robust than manual brace counting)
        return LLMResponse(
            content=content,
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": prompt_len, "completion_tokens": n_new_tokens},
        )

    def supports_tool_use(self) -> bool:
        return False  # We parse JSON from content, not native tool_calls


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
    parser = argparse.ArgumentParser(description="Evaluate SFT model")
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--adapter", default="outputs/sft_model/final")
    parser.add_argument("--output", default="outputs/eval_sft")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--no-adapter", action="store_true", help="Eval base model without adapter")
    args = parser.parse_args()

    setup_logging(args.output)
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("SFT Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Adapter: {'none' if args.no_adapter else args.adapter}")
    logger.info(f"Repeats: {args.repeats}, Max steps: {args.max_steps}")

    # Load model
    adapter = None if args.no_adapter else args.adapter
    client = LocalQwenClient(args.base_model, adapter)

    # Setup eval
    domain_config = build_cluster_domain_config()
    loader = ClusterScenarioLoader()
    agent_config = AgentConfig(
        max_steps=args.max_steps,
        max_proposals_per_step=3,
        consecutive_fail_limit=2,
        temperature=0.0,  # Greedy for eval
    )

    runner = EvalRunner(
        llm_client=client,
        domain_config=domain_config,
        manager_factory=ClusterManager,
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
                    "recovered": result.recovered,
                    "total_steps": result.total_steps,
                    "total_rejections": result.total_rejections,
                    "failsafe_triggered": result.failsafe_triggered,
                })
                status = "RECOVERED" if result.recovered else "FAILED"
                logger.info(f"  {scenario.id}: {status} ({result.total_steps} steps, "
                            f"{result.total_rejections} rejections)")
            except Exception as e:
                logger.error(f"  {scenario.id}: ERROR - {e}")
                all_results.append({
                    "repeat": rep,
                    "scenario_id": scenario.id,
                    "error": str(e),
                })

    elapsed = time.perf_counter() - t0

    # Save results
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    from collections import Counter
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation complete in {elapsed:.1f}s")

    total = len([r for r in all_results if "error" not in r])
    recovered = sum(1 for r in all_results if r.get("recovered"))
    errors = sum(1 for r in all_results if "error" in r)
    logger.info(f"Total: {total}, Recovered: {recovered} ({recovered/max(total,1)*100:.1f}%), Errors: {errors}")

    # Per-scenario
    scenario_stats = {}
    for r in all_results:
        sid = r.get("scenario_id", "unknown")
        if sid not in scenario_stats:
            scenario_stats[sid] = {"total": 0, "recovered": 0}
        if "error" not in r:
            scenario_stats[sid]["total"] += 1
            if r.get("recovered"):
                scenario_stats[sid]["recovered"] += 1

    logger.info("\nPer-scenario:")
    for sid in sorted(scenario_stats):
        s = scenario_stats[sid]
        rate = s["recovered"] / s["total"] * 100 if s["total"] else 0
        logger.info(f"  {sid}: {s['recovered']}/{s['total']} ({rate:.0f}%)")

    # Save metrics
    metrics = {
        "total_episodes": total,
        "recovered": recovered,
        "recovery_rate": recovered / max(total, 1),
        "errors": errors,
        "elapsed_seconds": elapsed,
        "per_scenario": scenario_stats,
    }
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()
