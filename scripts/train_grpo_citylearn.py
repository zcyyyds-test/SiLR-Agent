"""Step-level GRPO training for CityLearn (multi-type N=4) — verifier graded verdict AS the reward.

This is the support-pillar-2 main experiment (AAMAS-27 GAAI): the SiLR verifier
is no longer only a runtime gate, it also supplies the GRPO **process reward**.
Three reward arms (panel 2026-06-03, gate SEPARABLE at ρ(D,E)=0.736):

    --arm C : binary           compute_binary_reward   (admit +0.5 / reject -0.5)
    --arm D : structured       compute_grpo_reward     (product-order Φ-descent — CLAIM)
    --arm E : scalar           compute_scalar_reward   (count-projection — reference)

All three share the same scaffolding (step_cost, terminal recovery bonus,
group-relative advantage, PPO update); they differ ONLY in how each accepted
step's verdict + per-branch geometry Φ=(S,σ) maps to a scalar — that is the
whole experiment. The reward is read from the accepted action's verification
result, ``step.verification_results[-1]``, which now persists baseline/post
branch geometry (see silr/verifier/types.py).

Rollouts use the ReActAgent path (manual CityLearnManager + setup_episode +
SiLRVerifier + ReActAgent), mirroring the proven scripts/anm_eval_sweep.py and
scripts/anm_reward_separability_smoke.py construction — NOT EvalRunner, whose
ANM compatibility is unverified. The rollout LLM is the in-process HF model
(LocalModelClient, bare-text mode) so log-probs and gradients flow through it.

GRPO degeneration lessons baked in (decisions §5.1): smoke first (small
--iterations/--rollouts/--scenarios), watch group std (null-advantage), use
greedy eval (not this sampling rollout) to judge recovery.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.citylearn import (
    CityLearnScenarioLoader,
    CityLearnManager,
    build_citylearn_domain_config,
)
from domains.citylearn.scenarios import SCENARIOS
from silr.agent import AgentConfig, ReActAgent
from silr.agent.llm.base import BaseLLMClient, LLMResponse
from silr.agent.types import StepOutcome
from silr.training.grpo_trainer import StepSample, compute_advantages
from silr.training.reward import (
    RewardConfig,
    compute_binary_reward,
    compute_grpo_reward,
    compute_scalar_reward,
    compute_severity_scalar_reward,
)
from silr.verifier import SiLRVerifier

logger = logging.getLogger(__name__)

# Hardened N=4 multi-type band (cl_mined_*), frozen in scenarios_mined.json.
DEFAULT_SCENARIOS = [s.id for s in SCENARIOS if s.id.startswith("cl_mined_")]

_ARM_FN = {"C": compute_binary_reward, "D": compute_grpo_reward, "E": compute_scalar_reward, "E2": compute_severity_scalar_reward}


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train_grpo_citylearn.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


class LocalModelClient(BaseLLMClient):
    """In-process HF model client for GRPO rollouts (bare-text, no tool parser)."""

    def __init__(self, model, tokenizer, max_new_tokens=2048, enable_thinking=False):
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        self._enable_thinking = enable_thinking

    def chat(self, messages, tools=None, temperature=0.7, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        # Default thinking=OFF for in-process HF rollouts: Qwen3 thinking-mode
        # CoT overruns the token budget and truncates before the action JSON,
        # causing parse failures + multi-minute generations (observed in the
        # thinking-on smoke). Thinking-off yields short, fast, parseable actions.
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = {
                "max_new_tokens": self._max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "repetition_penalty": 1.1,
            }
            if temperature > 0:
                gen["temperature"] = temperature
                gen["top_p"] = 0.9
            out = self._model.generate(**inputs, **gen)
        new_tokens = out[0][prompt_len:]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return LLMResponse(
            content=content, tool_calls=[], finish_reason="stop",
            usage={"prompt_tokens": prompt_len, "completion_tokens": len(new_tokens)},
        )

    def supports_tool_use(self):
        return False


def _run_citylearn_episode(client, scenario, agent_config, rollout_seed):
    """Proven CityLearn rollout path (mirrors anm_eval_sweep / separability smoke).

    ``rollout_seed`` MUST be unique per rollout: LocalModelClient.chat() calls
    torch.manual_seed(seed) on every generation, so a shared seed makes all
    rollouts in a group identical → zero group variance → zero advantage → dead
    gradient (the null-advantage failure, introduced silently via a fixed seed).
    """
    cfg = build_citylearn_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = CityLearnScenarioLoader()
    sc = loader.load(scenario.id) if hasattr(scenario, "id") else loader.load(scenario)
    mgr = CityLearnManager(
        fixed_t=sc.fixed_t, initial_soc=sc.initial_soc,
        initial_actions=sc.initial_actions, peak_import_kw=sc.peak_import_kw,
    )
    loader.setup_episode(mgr, sc)
    verifier = SiLRVerifier(mgr, domain_config=cfg)
    agent = ReActAgent(
        manager=mgr, verifier=verifier, llm_client=client, domain_config=cfg,
        config=dataclasses.replace(agent_config, seed=rollout_seed),
    )
    return agent.run_episode(scenario_id=sc.id)


def collect_rollouts(model, tokenizer, scenarios, agent_config, rollouts_per_scenario,
                     arm, reward_config, step_cost, recovery_bonus, max_new_tokens,
                     enable_thinking=False, seed_base=10_000):
    """Online rollouts → per-step verdict-derived reward → StepSample list."""
    client = LocalModelClient(model, tokenizer, max_new_tokens=max_new_tokens,
                              enable_thinking=enable_thinking)
    arm_fn = _ARM_FN[arm]
    all_samples = []
    stats = {"total_episodes": 0, "recovered": 0, "failed_scenario_ids": set(),
             "n_accepted": 0, "n_rejected": 0, "n_unparsed": 0, "verdicts": Counter()}
    per_recov = defaultdict(int)
    per_total = defaultdict(int)

    rollout_ctr = 0
    for scenario in scenarios:
        for _ in range(rollouts_per_scenario):
            # Unique seed per rollout → distinct sampling → non-zero group variance.
            result = _run_citylearn_episode(client, scenario, agent_config, seed_base + rollout_ctr)
            rollout_ctr += 1
            stats["total_episodes"] += 1
            per_total[scenario.id] += 1
            if result.recovered:
                stats["recovered"] += 1
                per_recov[scenario.id] += 1

            n_steps = len(result.steps)
            for i, step in enumerate(result.steps):
                accepted = step.outcome == StepOutcome.SUCCESS
                stats["n_accepted" if accepted else "n_rejected"] += 1
                # Reward from the verdict of the last verification on this step:
                # accepted step -> the admitted PASS/SAFE_PROGRESS vr; rejected
                # step -> the last FAIL vr. Both carry Φ geometry now.
                if step.verification_results:
                    vr = step.verification_results[-1]
                    stats["verdicts"][vr.verdict.value] += 1
                    reward = arm_fn(vr, reward_config) - step_cost
                else:
                    # No adjudication = unparseable output / failsafe. Penalise as
                    # hard as a rejection (finance v2 uses -0.5): otherwise a parse
                    # failure (-step_cost ≈ -0.05) is CHEAPER than a real violation
                    # (-0.35..-1.05) and the policy degenerates to emitting garbage
                    # (the cluster-Iter2 natural-language collapse path).
                    stats["n_unparsed"] += 1
                    reward = -0.5
                if i == n_steps - 1 and result.recovered:
                    reward += recovery_bonus

                obs_text = step.observation.compressed_json if step.observation else ""
                if step.applied_action:
                    thought = f"Thought: {step.thought}\n" if step.thought else ""
                    action_text = thought + json.dumps(step.applied_action)
                else:
                    action_text = step.thought or ""
                all_samples.append(StepSample(
                    obs_text=obs_text, action_text=action_text,
                    reward=reward, group_key=(scenario.id,),
                    traj_id=rollout_ctr - 1,  # this rollout's id (ctr already ++'d)
                ))

    for sid in per_total:
        if per_recov[sid] < per_total[sid]:
            stats["failed_scenario_ids"].add(sid)
    return all_samples, stats


# --- GRPO mechanics: verbatim from the proven finance v2 trainer ---------------

def _find_action_start(tokenizer, _, messages):
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    return tokenizer(prompt_text, return_tensors="pt", truncation=False)["input_ids"].shape[1]


def _action_log_prob(model, tokenizer, messages, max_length=4096):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    action_start = min(_find_action_start(tokenizer, enc["input_ids"], messages),
                       enc["input_ids"].shape[1] - 1)
    labels = enc["input_ids"].clone()
    labels[0, :action_start] = -100
    out = model(**enc, labels=labels)
    n_action_tokens = max((labels[0] != -100).sum().item(), 1)
    return -out.loss * n_action_tokens, n_action_tokens


def compute_log_probs(model, tokenizer, samples, max_length=4096):
    model.eval()
    for s in samples:
        msgs = [{"role": "user", "content": s.obs_text},
                {"role": "assistant", "content": s.action_text}]
        with torch.no_grad():
            lp, _ = _action_log_prob(model, tokenizer, msgs, max_length)
            s.log_prob = lp.item()


def grpo_policy_update(model, tokenizer, optimizer, samples, clip_eps, kl_coeff,
                       batch_size, max_length=4096):
    model.train()
    total_loss = 0.0
    n_batches = 0
    log_ratios = []
    active = [s for s in samples if abs(s.advantage) > 1e-6]
    if not active:
        logger.warning("No active samples (all advantages zero)")
        return 0.0
    for i in range(0, len(active), batch_size):
        batch = active[i:i + batch_size]
        batch_loss_sum = 0.0
        n_in_batch = 0
        for s in batch:
            msgs = [{"role": "user", "content": s.obs_text},
                    {"role": "assistant", "content": s.action_text}]
            new_lp, _ = _action_log_prob(model, tokenizer, msgs, max_length)
            log_ratio = new_lp - s.log_prob
            log_ratios.append(log_ratio.item())
            log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
            ratio = torch.exp(log_ratio)
            adv = torch.tensor(s.advantage, device=model.device)
            policy_loss = -torch.min(ratio * adv,
                                     torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
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
        logger.info(f"  Log-ratio: min={min(log_ratios):.3f} max={max(log_ratios):.3f} "
                    f"mean={statistics.mean(log_ratios):.3f} "
                    f"clamped={sum(1 for v in log_ratios if abs(v) >= 4.9)}/{len(log_ratios)}")
    return total_loss / max(n_batches, 1)


def main():
    p = argparse.ArgumentParser(description="Step-level GRPO — CityLearn verifier-as-reward")
    p.add_argument("--arm", choices=["C", "D", "E", "E2"], required=True,
                   help="C=binary, D=structured product-order (claim), E=scalar projection")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--quant", choices=["bf16", "4bit"], default="bf16",
                   help="bf16 (default, no bitsandbytes; fits H100) or 4bit (needs bitsandbytes)")
    p.add_argument("--sft-adapter", default="", help="optional; empty = fresh LoRA on base")
    p.add_argument("--output", default="outputs/citylearn_grpo")
    p.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--rollouts-per-scenario", type=int, default=6)
    p.add_argument("--clip-eps", type=float, default=0.1)
    p.add_argument("--kl-coeff", type=float, default=0.08)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-proposals", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7, help="rollout exploration temp")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--thinking", action="store_true",
                   help="enable Qwen3 thinking-mode CoT in rollouts (default off: "
                        "thinking truncates the action under the token budget on the "
                        "in-process HF path → parse failures + slow generation)")
    p.add_argument("--step-cost", type=float, default=0.05)
    p.add_argument("--recovery-bonus", type=float, default=1.0)
    p.add_argument("--curriculum", action="store_true",
                   help="after iter 1, only roll out scenarios not yet fully recovered")
    p.add_argument("--seed", type=int, default=0,
                   help="replication seed: sets torch init (LoRA) + offsets rollout "
                        "sampling seeds so multi-seed runs are genuinely distinct")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    setup_logging(args.output)
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info(f"GRPO — CityLearn verifier-as-reward | ARM {args.arm}")
    logger.info("=" * 60)
    logger.info(f"Base: {args.base_model} | SFT adapter: {args.sft_adapter or '(none, fresh LoRA)'}")
    logger.info(f"Scenarios: {len(args.scenarios)} | iters={args.iterations} "
                f"rollouts/scen={args.rollouts_per_scenario} | curriculum={args.curriculum}")
    logger.info(f"clip={args.clip_eps} kl={args.kl_coeff} lr={args.lr} temp={args.temperature} "
                f"max_new_tokens={args.max_new_tokens}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kw = dict(device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if args.quant == "4bit":
        # Optional 4-bit (needs bitsandbytes). Default is bf16: Qwen3-8B (~16GB)
        # fits an H100 (95GB) full-precision, so LoRA trains in bf16 with no
        # bitsandbytes dependency — faster and dependency-free.
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    logger.info(f"Loading base model ({args.quant})...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kw)
    if args.sft_adapter:
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
        logger.info("SFT adapter loaded (trainable)")
    else:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        logger.info("Fresh LoRA initialised on base model")
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable: {trainable:,}/{total:,} ({trainable/total*100:.2f}%)")

    loader = CityLearnScenarioLoader()
    scenarios = [loader.load(sid) for sid in args.scenarios]
    agent_config = AgentConfig(
        max_steps=args.max_steps, max_proposals_per_step=args.max_proposals,
        consecutive_fail_limit=2, temperature=args.temperature, enable_verification=True)
    reward_config = RewardConfig()
    optimizer = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                                  lr=args.lr, weight_decay=0.01)

    metrics = []
    failed_ids = None
    for it in range(args.iterations):
        logger.info(f"\n{'='*40} Iteration {it+1}/{args.iterations} {'='*40}")
        iter_t0 = time.perf_counter()
        if args.curriculum and failed_ids is not None:
            if not failed_ids:
                logger.info("All scenarios recovered — stopping early.")
                break
            active = [s for s in scenarios if s.id in failed_ids]
            logger.info(f"Curriculum: {len(active)}/{len(scenarios)} active scenarios")
        else:
            active = scenarios

        logger.info(f"Phase 1: rollouts ({len(active)} scen x {args.rollouts_per_scenario}) ARM {args.arm}...")
        samples, rs = collect_rollouts(
            model, tokenizer, active, agent_config, args.rollouts_per_scenario,
            args.arm, reward_config, args.step_cost, args.recovery_bonus, args.max_new_tokens,
            enable_thinking=args.thinking,
            # distinct rollout seeds across iterations AND replication seeds
            seed_base=args.seed * 1_000_000 + 10_000 + it * 1_000)
        recovery = rs["recovered"] / max(rs["total_episodes"], 1)
        failed_ids = rs["failed_scenario_ids"]
        logger.info(f"  Episodes {rs['total_episodes']} | recovered {rs['recovered']} ({recovery*100:.1f}%)")
        logger.info(f"  Step samples {len(samples)} | accept/reject {rs['n_accepted']}/{rs['n_rejected']} "
                    f"| unparsed {rs['n_unparsed']}")
        logger.info(f"  Verdicts: {dict(rs['verdicts'])}")

        logger.info("Phase 2: advantages...")
        compute_log_probs(model, tokenizer, samples)
        if os.environ.get("SILR_TRAJ_ADV") == "1":
            from silr.training.grpo_trainer import compute_advantages_trajectory
            logger.info("  [trajectory-return advantage] (SILR_TRAJ_ADV=1)")
            compute_advantages_trajectory(samples)
        else:
            compute_advantages(samples)

        group_sizes = Counter()
        group_std = {}
        by_group = defaultdict(list)
        for s in samples:
            group_sizes[s.group_key] += 1
            by_group[s.group_key].append(s.reward)
        import statistics as _st
        zero_var_groups = sum(1 for g, rs_ in by_group.items()
                              if len(rs_) > 1 and _st.pstdev(rs_) == 0.0)
        pos = sum(1 for s in samples if s.advantage > 0)
        neg = sum(1 for s in samples if s.advantage < 0)
        sizes = sorted(group_sizes.values())
        logger.info(f"  Groups {len(group_sizes)} (min {sizes[0]} max {sizes[-1]}) | "
                    f"zero-variance groups {zero_var_groups}/{len(by_group)} "
                    f"(null-advantage risk)")
        logger.info(f"  Advantages: {pos}+ {neg}- {len(samples)-pos-neg}zero")

        logger.info("Phase 3: policy update...")
        loss = grpo_policy_update(model, tokenizer, optimizer, samples,
                                  args.clip_eps, args.kl_coeff, args.batch_size)
        dt = time.perf_counter() - iter_t0
        logger.info(f"  Loss {loss:.4f} | {dt:.1f}s")
        metrics.append({"iteration": it + 1, "arm": args.arm, "recovery_rate": recovery,
                        "n_samples": len(samples), "avg_loss": loss,
                        "pos_adv": pos, "neg_adv": neg, "zero_var_groups": zero_var_groups,
                        "elapsed_s": dt})

        # Save EVERY iteration: arm D collapsed at iter_5 while iter_2-4 were the
        # operating point (over-optimisation), so per-iter checkpoints are needed
        # to pick each arm's best for the matched comparison.
        ck = os.path.join(args.output, f"iter_{it+1}")
        model.save_pretrained(ck)
        tokenizer.save_pretrained(ck)
        logger.info(f"  Saved checkpoint {ck}")

    final = os.path.join(args.output, "final")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\n{'='*60}\nGRPO CityLearn (arm {args.arm}) done in "
                f"{(time.perf_counter()-t0)/3600:.2f}h | final={final}")
    for m in metrics:
        logger.info(f"  Iter {m['iteration']}: recovery={m['recovery_rate']*100:.1f}% "
                    f"loss={m['avg_loss']:.4f} null-var-groups={m['zero_var_groups']}")


if __name__ == "__main__":
    main()
