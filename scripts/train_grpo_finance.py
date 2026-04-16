"""Step-Level GRPO training for finance (portfolio compliance) domain.

Online rollout → SiLR verification reward → step-level advantage → PPO update.
Uses the Finance SFT LoRA (e.g. DEDUP-ep3) as starting point.

Reward shape mirrors the cluster pipeline: accept=+0.45, reject=-0.5,
recovery bonus=+1.0 at final step. Finance's verifier has `checkers=[]`
(constraints are observer-based), so most StepOutcome=SUCCESS come from
tool-level validation passing — rejections happen on per-trade limit breach,
insufficient cash/shares, or unknown symbol. Sparse but real signal.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silr.agent.types import StepOutcome
from silr.training.grpo_trainer import StepSample, compute_advantages
from silr.agent.llm.base import BaseLLMClient, LLMResponse
from silr.agent.config import AgentConfig
from silr.eval.runner import EvalRunner
from domains.finance import FinanceManager, build_finance_domain_config, FinanceScenarioLoader

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train_grpo.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


class LocalModelClient(BaseLLMClient):
    def __init__(self, model, tokenizer, max_new_tokens=256):
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens

    def chat(self, messages, tools=None, temperature=0.7, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

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

    def supports_tool_use(self):
        return False


def collect_rollouts(
    model,
    tokenizer,
    scenarios,
    loader,
    domain_config,
    agent_config,
    rollouts_per_scenario,
    step_cost,
):
    client = LocalModelClient(model, tokenizer)
    all_samples = []
    stats = {"total_episodes": 0, "recovered": 0, "failed_scenario_ids": set()}
    per_scenario_recovered = defaultdict(int)
    per_scenario_total = defaultdict(int)

    for scenario in scenarios:
        for rollout_idx in range(rollouts_per_scenario):
            runner = EvalRunner(
                llm_client=client,
                domain_config=domain_config,
                manager_factory=FinanceManager,
                scenario_loader=loader,
                config=agent_config,
                record_trajectories=True,
            )

            result = runner.run_scenario(scenario)
            stats["total_episodes"] += 1
            per_scenario_total[scenario.id] += 1
            if result.recovered:
                stats["recovered"] += 1
                per_scenario_recovered[scenario.id] += 1

            for step_idx, step in enumerate(result.steps):
                accepted = step.outcome == StepOutcome.SUCCESS
                if accepted:
                    reward = 0.5 - step_cost
                else:
                    reward = -0.5

                if step_idx == len(result.steps) - 1 and result.recovered:
                    reward += 1.0

                obs_text = step.observation.compressed_json if step.observation else ""
                if step.applied_action:
                    thought_part = f"Thought: {step.thought}\n" if step.thought else ""
                    action_text = thought_part + json.dumps(step.applied_action)
                else:
                    action_text = step.thought or ""

                sample = StepSample(
                    obs_text=obs_text,
                    action_text=action_text,
                    reward=reward,
                    group_key=(scenario.id,),
                )
                all_samples.append(sample)

    for sid in per_scenario_total:
        if per_scenario_recovered[sid] < per_scenario_total[sid]:
            stats["failed_scenario_ids"].add(sid)

    return all_samples, stats


def _find_action_start(tokenizer, full_ids, messages):
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=False)["input_ids"]
    return prompt_ids.shape[1]


def _action_log_prob(model, tokenizer, messages, max_length=4096):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoding = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=max_length).to(model.device)
    input_ids = encoding["input_ids"]

    action_start = _find_action_start(tokenizer, input_ids, messages)
    action_start = min(action_start, input_ids.shape[1] - 1)

    labels = input_ids.clone()
    labels[0, :action_start] = -100

    outputs = model(**encoding, labels=labels)
    n_action_tokens = max((labels[0] != -100).sum().item(), 1)
    log_prob_sum = -outputs.loss * n_action_tokens

    return log_prob_sum, n_action_tokens


def compute_log_probs(model, tokenizer, samples, max_length=4096):
    model.eval()
    for sample in samples:
        messages = [
            {"role": "user", "content": sample.obs_text},
            {"role": "assistant", "content": sample.action_text},
        ]
        with torch.no_grad():
            log_prob_sum, _ = _action_log_prob(model, tokenizer, messages, max_length)
            sample.log_prob = log_prob_sum.item()


def grpo_policy_update(model, tokenizer, optimizer, samples, clip_eps, kl_coeff,
                       batch_size, max_length=4096):
    model.train()
    total_loss = 0.0
    n_batches = 0
    log_ratio_vals = []

    active = [s for s in samples if abs(s.advantage) > 1e-6]
    if not active:
        logger.warning("No active samples (all advantages are zero)")
        return 0.0

    for i in range(0, len(active), batch_size):
        batch = active[i:i + batch_size]
        batch_loss_sum = 0.0
        n_in_batch = 0

        for sample in batch:
            messages = [
                {"role": "user", "content": sample.obs_text},
                {"role": "assistant", "content": sample.action_text},
            ]
            new_log_prob, _ = _action_log_prob(model, tokenizer, messages, max_length)

            log_ratio = new_log_prob - sample.log_prob
            log_ratio_vals.append(log_ratio.item())
            log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
            ratio = torch.exp(log_ratio)
            adv = torch.tensor(sample.advantage, device=model.device)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            unclipped = ratio * adv
            policy_loss = -torch.min(unclipped, clipped)

            kl = (ratio - 1) - torch.log(ratio)
            loss = (policy_loss + kl_coeff * kl) / len(batch)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"  Skipping sample at batch {i}: loss is {loss.item()}")
                continue
            loss.backward()
            batch_loss_sum += loss.item()
            n_in_batch += 1

        if n_in_batch == 0:
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += batch_loss_sum
        n_batches += 1

    if log_ratio_vals:
        import statistics
        lr_min = min(log_ratio_vals)
        lr_max = max(log_ratio_vals)
        lr_mean = statistics.mean(log_ratio_vals)
        lr_clamped = sum(1 for v in log_ratio_vals if abs(v) >= 4.9)
        logger.info(f"  Log-ratio stats: min={lr_min:.3f} max={lr_max:.3f} "
                    f"mean={lr_mean:.3f} clamped={lr_clamped}/{len(log_ratio_vals)}")

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Step-Level GRPO Training — Finance")
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--sft-adapter", required=True,
                        help="Path to SFT LoRA adapter (e.g. outputs/finance_sft_lora_gemini_dedup/final)")
    parser.add_argument("--output", default="outputs/finance_grpo_model")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rollouts-per-scenario", type=int, default=3)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coeff", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--step-cost", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    setup_logging(args.output)
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Step-Level GRPO Training — Finance")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"SFT adapter: {args.sft_adapter}")
    logger.info(f"Iterations: {args.iterations}, Rollouts/scenario: {args.rollouts_per_scenario}")
    logger.info(f"Clip eps: {args.clip_eps}, KL coeff: {args.kl_coeff}, LR: {args.lr}")
    logger.info(f"Step cost: {args.step_cost}, Temperature: {args.temperature}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("Loading base model + SFT adapter...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    logger.info("SFT adapter loaded (trainable)")

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    domain_config = build_finance_domain_config(with_observer=True)
    loader = FinanceScenarioLoader()
    scenarios = loader.load_all()  # 30 training scenarios (held-out excluded)
    logger.info(f"Loaded {len(scenarios)} training scenarios")

    agent_config = AgentConfig(
        max_steps=args.max_steps,
        max_proposals_per_step=3,
        consecutive_fail_limit=2,
        temperature=args.temperature,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    iteration_metrics = []
    failed_ids = None
    for iteration in range(args.iterations):
        logger.info(f"\n{'='*40} Iteration {iteration+1}/{args.iterations} {'='*40}")
        iter_t0 = time.perf_counter()

        if failed_ids is not None and len(failed_ids) > 0:
            active_scenarios = [s for s in scenarios if s.id in failed_ids]
            logger.info(f"Curriculum: {len(active_scenarios)}/{len(scenarios)} active scenarios")
        elif failed_ids is not None and len(failed_ids) == 0:
            logger.info("All scenarios recovered! Stopping early.")
            break
        else:
            active_scenarios = scenarios

        logger.info("Phase 1: Collecting rollouts...")
        samples, rollout_stats = collect_rollouts(
            model=model,
            tokenizer=tokenizer,
            scenarios=active_scenarios,
            loader=loader,
            domain_config=domain_config,
            agent_config=agent_config,
            rollouts_per_scenario=args.rollouts_per_scenario,
            step_cost=args.step_cost,
        )
        recovery_rate = rollout_stats["recovered"] / max(rollout_stats["total_episodes"], 1)
        failed_ids = rollout_stats.get("failed_scenario_ids", set())
        logger.info(f"  Episodes: {rollout_stats['total_episodes']}, "
                    f"Recovered: {rollout_stats['recovered']} ({recovery_rate*100:.1f}%)")
        logger.info(f"  Step samples: {len(samples)}, Failed scenarios: {len(failed_ids)}")

        logger.info("Phase 2: Computing advantages...")
        compute_log_probs(model, tokenizer, samples)
        compute_advantages(samples)

        from collections import Counter
        group_sizes = Counter()
        reward_dist = Counter()
        for s in samples:
            group_sizes[s.group_key] += 1
            reward_dist[round(s.reward, 2)] += 1
        size_vals = sorted(group_sizes.values())
        logger.info(f"  Groups: {len(group_sizes)}, sizes: min={size_vals[0]} max={size_vals[-1]} "
                    f"avg={sum(size_vals)/len(size_vals):.1f}")
        logger.info(f"  Reward distribution: {dict(sorted(reward_dist.items()))}")

        pos_adv = sum(1 for s in samples if s.advantage > 0)
        neg_adv = sum(1 for s in samples if s.advantage < 0)
        logger.info(f"  Advantages: {pos_adv} positive, {neg_adv} negative, "
                    f"{len(samples) - pos_adv - neg_adv} zero")

        logger.info("Phase 3: Policy gradient update...")
        avg_loss = grpo_policy_update(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            samples=samples,
            clip_eps=args.clip_eps,
            kl_coeff=args.kl_coeff,
            batch_size=args.batch_size,
        )
        iter_elapsed = time.perf_counter() - iter_t0
        logger.info(f"  Loss: {avg_loss:.4f}, Time: {iter_elapsed:.1f}s")

        metrics = {
            "iteration": iteration + 1,
            "recovery_rate": recovery_rate,
            "n_samples": len(samples),
            "avg_loss": avg_loss,
            "pos_advantages": pos_adv,
            "neg_advantages": neg_adv,
            "elapsed_seconds": iter_elapsed,
        }
        iteration_metrics.append(metrics)

        # Save every iteration (GRPO is expensive, checkpoint-per-iter is cheap)
        ckpt_dir = os.path.join(args.output, f"iter_{iteration+1}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"  Saved checkpoint: {ckpt_dir}")

    elapsed = time.perf_counter() - t0
    final_dir = os.path.join(args.output, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(iteration_metrics, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"GRPO Training complete in {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    logger.info(f"Final model: {final_dir}")
    for m in iteration_metrics:
        logger.info(f"  Iter {m['iteration']}: recovery={m['recovery_rate']*100:.1f}%, "
                    f"loss={m['avg_loss']:.4f}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
