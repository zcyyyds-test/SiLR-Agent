"""GRPO v2 — Finance (portfolio compliance) domain.

Supports two reward modes:
  v1:    accept/reject + recovery bonus (cluster-style sparse reward)
  dense: violation-count delta + recovery bonus (for observer-only domains)

Key fixes over v1: no curriculum narrowing (always rollout all scenarios),
higher KL coef (0.08-0.1), configurable reward shaping.
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
    def __init__(self, model, tokenizer, max_new_tokens=512):
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
            content=content, tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": prompt_len, "completion_tokens": len(new_tokens)},
        )

    def supports_tool_use(self):
        return False


def _viol_count(obs):
    if obs is None or obs.violations is None:
        return 0
    return len(obs.violations)


def _v1_reward(accepted, is_last, recovered, step_cost):
    """Original cluster-style reward used in v1."""
    r = 0.5 - step_cost if accepted else -0.5
    if is_last and recovered:
        r += 1.0
    return r


def _dense_reward(prev_viol, curr_viol, accepted, is_last, recovered,
                  step_cost, violation_weight, recovery_bonus, reject_penalty):
    """Violation-count delta + recovery bonus, no trivial accept reward."""
    r = -step_cost
    if accepted:
        r += violation_weight * (prev_viol - curr_viol)
    else:
        r += reject_penalty
    if is_last and recovered:
        r += recovery_bonus
    return r


def collect_rollouts(
    model, tokenizer, scenarios, loader, domain_config, agent_config,
    rollouts_per_scenario, reward_mode,
    step_cost, violation_weight, recovery_bonus, reject_penalty,
):
    client = LocalModelClient(model, tokenizer)
    all_samples = []
    stats = {"total_episodes": 0, "recovered": 0, "failed_scenario_ids": set(),
             "n_violation_delta_pos": 0, "n_violation_delta_neg": 0, "n_violation_delta_zero": 0,
             "n_accepted": 0, "n_rejected": 0}
    per_recov = defaultdict(int)
    per_total = defaultdict(int)

    for scenario in scenarios:
        for _ in range(rollouts_per_scenario):
            runner = EvalRunner(
                llm_client=client, domain_config=domain_config,
                manager_factory=FinanceManager, scenario_loader=loader,
                config=agent_config, record_trajectories=True,
            )
            result = runner.run_scenario(scenario)
            stats["total_episodes"] += 1
            per_total[scenario.id] += 1
            if result.recovered:
                stats["recovered"] += 1
                per_recov[scenario.id] += 1

            n_steps = len(result.steps)
            for i, step in enumerate(result.steps):
                accepted = step.outcome == StepOutcome.SUCCESS
                stats["n_accepted" if accepted else "n_rejected"] += 1
                is_last = (i == n_steps - 1)

                prev_v = _viol_count(step.observation)
                if i + 1 < n_steps:
                    curr_v = _viol_count(result.steps[i + 1].observation)
                else:
                    curr_v = _viol_count(result.final_observation)
                delta = prev_v - curr_v
                if delta > 0:
                    stats["n_violation_delta_pos"] += 1
                elif delta < 0:
                    stats["n_violation_delta_neg"] += 1
                else:
                    stats["n_violation_delta_zero"] += 1

                if reward_mode == "v1":
                    reward = _v1_reward(accepted, is_last, result.recovered, step_cost)
                elif reward_mode == "dense":
                    reward = _dense_reward(
                        prev_v, curr_v, accepted, is_last, result.recovered,
                        step_cost, violation_weight, recovery_bonus, reject_penalty,
                    )
                else:
                    raise ValueError(f"Unknown reward_mode: {reward_mode}")

                obs_text = step.observation.compressed_json if step.observation else ""
                if step.applied_action:
                    thought_part = f"Thought: {step.thought}\n" if step.thought else ""
                    action_text = thought_part + json.dumps(step.applied_action)
                else:
                    action_text = step.thought or ""
                all_samples.append(StepSample(
                    obs_text=obs_text, action_text=action_text,
                    reward=reward, group_key=(scenario.id,),
                ))

    for sid in per_total:
        if per_recov[sid] < per_total[sid]:
            stats["failed_scenario_ids"].add(sid)
    return all_samples, stats


def _find_action_start(tokenizer, _, messages):
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
            lp, _ = _action_log_prob(model, tokenizer, messages, max_length)
            sample.log_prob = lp.item()


def grpo_policy_update(model, tokenizer, optimizer, samples, clip_eps, kl_coeff,
                       batch_size, max_length=4096):
    model.train()
    total_loss = 0.0
    n_batches = 0
    log_ratios = []
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
            new_lp, _ = _action_log_prob(model, tokenizer, messages, max_length)
            log_ratio = new_lp - sample.log_prob
            log_ratios.append(log_ratio.item())
            log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
            ratio = torch.exp(log_ratio)
            adv = torch.tensor(sample.advantage, device=model.device)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            unclipped = ratio * adv
            policy_loss = -torch.min(unclipped, clipped)
            kl = (ratio - 1) - torch.log(ratio)
            loss = (policy_loss + kl_coeff * kl) / len(batch)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"  Skip sample: loss={loss.item()}")
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
    if log_ratios:
        import statistics
        logger.info(f"  Log-ratio stats: min={min(log_ratios):.3f} max={max(log_ratios):.3f} "
                    f"mean={statistics.mean(log_ratios):.3f} "
                    f"clamped={sum(1 for v in log_ratios if abs(v) >= 4.9)}/{len(log_ratios)}")
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="GRPO v2 — Finance (no curriculum, tunable reward)")
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--output", default="outputs/finance_grpo_v2")
    parser.add_argument("--reward-mode", choices=["v1", "dense"], default="v1",
                        help="v1 = accept/reject/recovery-bonus (Codex plan); "
                             "dense = violation delta + recovery bonus (Kimi plan)")
    parser.add_argument("--iterations", type=int, default=2)  # Codex: 2
    parser.add_argument("--rollouts-per-scenario", type=int, default=4)  # Codex: 4
    parser.add_argument("--clip-eps", type=float, default=0.1)  # Codex: 0.1
    parser.add_argument("--kl-coeff", type=float, default=0.08)  # Codex: 0.08
    parser.add_argument("--lr", type=float, default=2e-6)  # Codex: 2e-6
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--step-cost", type=float, default=0.05)  # same as v1
    parser.add_argument("--temperature", type=float, default=0.3)  # Codex: 0.3
    parser.add_argument("--max-new-tokens", type=int, default=512)  # up from 256
    # Dense reward params (only used if --reward-mode dense)
    parser.add_argument("--violation-weight", type=float, default=0.5)
    parser.add_argument("--recovery-bonus", type=float, default=5.0)
    parser.add_argument("--reject-penalty", type=float, default=-0.1)
    args = parser.parse_args()

    setup_logging(args.output)
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info(f"GRPO v2 — Finance (reward_mode={args.reward_mode})")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"SFT adapter: {args.sft_adapter}")
    logger.info(f"Iterations: {args.iterations}, Rollouts/scenario: {args.rollouts_per_scenario}")
    logger.info(f"Clip: {args.clip_eps}, KL: {args.kl_coeff}, LR: {args.lr}")
    logger.info(f"Temp: {args.temperature}, max_new_tokens: {args.max_new_tokens}")
    if args.reward_mode == "dense":
        logger.info(f"Dense reward: viol_w={args.violation_weight}, "
                    f"recovery={args.recovery_bonus}, reject={args.reject_penalty}")
    logger.info("NO CURRICULUM — always rollout all 30 scenarios")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config,
        device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    logger.info("SFT adapter loaded (trainable)")
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable: {trainable:,}/{total:,} ({trainable/total*100:.2f}%)")

    domain_config = build_finance_domain_config(with_observer=True)
    loader = FinanceScenarioLoader()
    scenarios = loader.load_all()
    logger.info(f"Loaded {len(scenarios)} training scenarios")

    agent_config = AgentConfig(
        max_steps=args.max_steps, max_proposals_per_step=3,
        consecutive_fail_limit=2, temperature=args.temperature,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    iteration_metrics = []
    for iteration in range(args.iterations):
        logger.info(f"\n{'='*40} Iteration {iteration+1}/{args.iterations} {'='*40}")
        iter_t0 = time.perf_counter()
        logger.info(f"Phase 1: Collecting {len(scenarios)} scenarios x {args.rollouts_per_scenario} rollouts...")
        samples, rollout_stats = collect_rollouts(
            model=model, tokenizer=tokenizer, scenarios=scenarios,
            loader=loader, domain_config=domain_config, agent_config=agent_config,
            rollouts_per_scenario=args.rollouts_per_scenario,
            reward_mode=args.reward_mode,
            step_cost=args.step_cost,
            violation_weight=args.violation_weight,
            recovery_bonus=args.recovery_bonus,
            reject_penalty=args.reject_penalty,
        )
        recovery_rate = rollout_stats["recovered"] / max(rollout_stats["total_episodes"], 1)
        failed_ids = rollout_stats["failed_scenario_ids"]
        logger.info(f"  Episodes: {rollout_stats['total_episodes']}, "
                    f"Recovered: {rollout_stats['recovered']} ({recovery_rate*100:.1f}%)")
        logger.info(f"  Step samples: {len(samples)}, Failed scenarios: {len(failed_ids)}")
        logger.info(f"  Viol delta: pos={rollout_stats['n_violation_delta_pos']} "
                    f"neg={rollout_stats['n_violation_delta_neg']} "
                    f"zero={rollout_stats['n_violation_delta_zero']}")
        logger.info(f"  Accept/Reject: {rollout_stats['n_accepted']}/{rollout_stats['n_rejected']}")

        logger.info("Phase 2: Advantages...")
        compute_log_probs(model, tokenizer, samples)
        compute_advantages(samples)

        from collections import Counter
        group_sizes = Counter()
        reward_dist = Counter()
        for s in samples:
            group_sizes[s.group_key] += 1
            reward_dist[round(s.reward, 2)] += 1
        size_vals = sorted(group_sizes.values())
        logger.info(f"  Groups: {len(group_sizes)}, min={size_vals[0]} max={size_vals[-1]}")
        logger.info(f"  Reward dist (top 10): {dict(sorted(reward_dist.items())[:10])}")
        pos_adv = sum(1 for s in samples if s.advantage > 0)
        neg_adv = sum(1 for s in samples if s.advantage < 0)
        logger.info(f"  Advantages: {pos_adv}+ {neg_adv}- "
                    f"{len(samples)-pos_adv-neg_adv}zero")

        logger.info("Phase 3: Policy update...")
        avg_loss = grpo_policy_update(
            model=model, tokenizer=tokenizer, optimizer=optimizer,
            samples=samples, clip_eps=args.clip_eps, kl_coeff=args.kl_coeff,
            batch_size=args.batch_size,
        )
        iter_elapsed = time.perf_counter() - iter_t0
        logger.info(f"  Loss: {avg_loss:.4f}, Time: {iter_elapsed:.1f}s")

        iteration_metrics.append({
            "iteration": iteration + 1, "recovery_rate": recovery_rate,
            "n_samples": len(samples), "avg_loss": avg_loss,
            "pos_advantages": pos_adv, "neg_advantages": neg_adv,
            "n_accepted": rollout_stats["n_accepted"],
            "n_rejected": rollout_stats["n_rejected"],
            "elapsed_seconds": iter_elapsed,
        })

        ckpt_dir = os.path.join(args.output, f"iter_{iteration+1}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"  Saved: {ckpt_dir}")

    elapsed = time.perf_counter() - t0
    final_dir = os.path.join(args.output, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(iteration_metrics, f, indent=2)
    logger.info(f"\n{'='*60}")
    logger.info(f"Complete in {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    for m in iteration_metrics:
        logger.info(f"  Iter {m['iteration']}: recovery={m['recovery_rate']*100:.1f}%, "
                    f"loss={m['avg_loss']:.4f}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
