"""GRPO trainer — SWE (code repair) domain.

Copy of the finance v5 (`train_grpo_finance_v2.py`) training loop with the
rollout / reward / verification pieces swapped for SWE-specific logic:

  * Reward: 4-tier additive dense reward
      0.10 AST parse OK
      0.15 imports resolve after patch (additive)
      0.25 PASS_TO_PASS stays green (additive)
      1.00 FAIL_TO_PASS turns green (primary objective, additive)
    Maximum reward = 1.50.

  * Rollout: two-stage Agentless call (localize → patch) with
    do_sample=True, temperature=0.7 for GRPO exploration.

  * Verify: apply candidate patch on a shadow RepoManager, run AST +
    imports + RegressionChecker + TargetTestChecker. Shadow is cleaned
    up after every verification to avoid /tmp exhaustion.

  * Grouping: `group_key = (instance_id,)` so advantages normalise
    across the `rollouts-per-instance` candidate patches for the same
    bug.

Everything else (log_prob computation, advantage normalisation with the
[-3, +3] clamp inside `compute_advantages`, per-batch backward step,
checkpoint/logging layout) is byte-for-byte the same as finance v2 —
those are production-validated and not the variables we want to touch
between domains.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silr.training.grpo_trainer import StepSample, compute_advantages
from domains.swe.manager import RepoManager
from domains.swe.scenarios import SWEBenchLoader
from domains.swe.config import build_swe_domain_config
from domains.swe.checkers import RegressionChecker, TargetTestChecker

logger = logging.getLogger(__name__)


# ── Reward ──────────────────────────────────────────────────────────────

def compute_swe_reward(ast_ok, imports_ok, regression_pass, target_pass) -> float:
    """4-tier dense reward for SWE GRPO.

    0.10 — AST parse succeeds
    0.15 — imports resolve after patch applied (additive)
    0.25 — PASS_TO_PASS all still green (additive)
    1.00 — FAIL_TO_PASS all green (primary objective, additive)

    Values are strictly monotone: a patch that advances one tier cannot
    lose reward from a lower tier. This gives dense gradient even when
    the primary FAIL_TO_PASS signal is still sparse early in training.
    """
    reward = 0.0
    if ast_ok:
        reward += 0.10
    if ast_ok and imports_ok:
        reward += 0.15
    if ast_ok and imports_ok and regression_pass:
        reward += 0.25
    if ast_ok and imports_ok and regression_pass and target_pass:
        reward += 1.00
    return round(reward, 6)


# ── Logging ─────────────────────────────────────────────────────────────

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


# ── Rollout ─────────────────────────────────────────────────────────────

_LOC_BLOCK_RE = re.compile(r"\{[^{}]*locations[^{}]*\}", re.S)
_DIFF_BLOCK_RE = re.compile(r"(diff --git .*?)(?:```|\Z)", re.S)


def _extract_patch(text: str) -> str:
    """See scripts/eval_swe_inference.py::_extract_patch — same logic.

    Critical for GRPO: SFT-bootstrapped policy emits tool-call JSON, and
    rollout patches arriving as literal `\\n` make every git apply fail,
    collapsing reward signal to zero. Must JSON-unescape before verify.
    """
    if '"patch"' in text:
        first_brace = text.find("{")
        if first_brace >= 0:
            depth = 0
            in_str = False
            esc = False
            end = -1
            for i in range(first_brace, len(text)):
                ch = text[i]
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end > first_brace:
                try:
                    obj = json.loads(text[first_brace : end + 1])
                except (json.JSONDecodeError, ValueError):
                    obj = None
                if obj is not None:
                    def _walk(o):
                        if isinstance(o, dict):
                            v = o.get("patch")
                            if isinstance(v, str) and "diff --git" in v:
                                return v
                            for vv in o.values():
                                r = _walk(vv)
                                if r:
                                    return r
                        elif isinstance(o, list):
                            for vv in o:
                                r = _walk(vv)
                                if r:
                                    return r
                        return None
                    p = _walk(obj)
                    if p:
                        return p.strip()
    m = _DIFF_BLOCK_RE.search(text)
    if m:
        result = m.group(1).strip()
        if "\\n" in result and "\n" not in result:
            try:
                result = result.encode("utf-8", errors="replace").decode(
                    "unicode_escape"
                )
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
        return result
    return ""


def sample_patch(model, tokenizer, cfg, mgr, temperature: float,
                 max_new_tokens_loc: int = 512,
                 max_new_tokens_patch: int = 2048) -> tuple[str, list[dict], list[dict]]:
    """Two-stage Agentless rollout (localize → patch) with sampling.

    Mirrors `scripts/eval_swe_inference.py::run_one_instance` but flips
    generate to `do_sample=True` so GRPO actually gets exploration —
    greedy rollouts collapse the advantage signal (all patches identical
    per instance) and KL against old log_prob would be zero.

    Returns (patch_text, loc_messages, patch_messages) where the message
    lists include the assistant turn and are ready for log-prob scoring.
    """
    obs = cfg.create_observer(mgr) if cfg.create_observer else None
    sys_prompt = cfg.build_system_prompt(mgr, cfg.build_tool_schemas(mgr))
    user_msg = json.dumps(obs.observe() if obs else {}, indent=2)

    # Stage 1: localize
    loc_prompt_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
        {"role": "user", "content": "Call the `localize` tool with candidate locations."},
    ]
    text = tokenizer.apply_chat_template(
        loc_prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens_loc,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    loc_new_tokens = out[0][prompt_len:]
    loc_text = tokenizer.decode(loc_new_tokens, skip_special_tokens=True).strip()
    m = _LOC_BLOCK_RE.search(loc_text)
    loc_args: dict = {"locations": ["UNKNOWN:0"]}
    if m:
        try:
            loc_args = json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    tools = cfg.create_toolset(mgr)
    try:
        tools["localize"].execute(**loc_args)
    except Exception:
        # Bad localize args should not crash the rollout; the patch stage
        # will still run with whatever `localized` was set to (possibly []).
        pass

    loc_messages = loc_prompt_messages + [{"role": "assistant", "content": loc_text}]

    # Stage 2: patch
    patch_prompt_messages = list(loc_messages) + [
        {"role": "user",
         "content": "Now call the `patch` tool with the full unified diff."},
    ]
    text = tokenizer.apply_chat_template(
        patch_prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens_patch,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    patch_new_tokens = out[0][prompt_len:]
    patch_text = tokenizer.decode(patch_new_tokens, skip_special_tokens=True).strip()
    patch_str = _extract_patch(patch_text)

    patch_messages = patch_prompt_messages + [{"role": "assistant", "content": patch_text}]

    return patch_str, loc_messages, patch_messages


# ── Verify ──────────────────────────────────────────────────────────────

def verify_patch(mgr: RepoManager, patch: str) -> tuple[bool, bool, bool, bool]:
    """Apply `patch` to a shadow of `mgr`, run the 4 SWE checks.

    Always calls `shadow.cleanup()` before returning so GRPO training
    (which verifies O(rollouts_per_instance * n_instances * iterations)
    patches) does not fill /tmp with stale worktrees.
    """
    shadow = mgr.create_shadow_copy()
    try:
        shadow.set_patch(patch)
        solved = shadow.solve()
        ast_ok = bool(shadow.ast_ok)
        imports_ok = bool(shadow.imports_ok)
        regression_pass = False
        target_pass = False
        if solved:
            reg = RegressionChecker().check(shadow.system_state, shadow.base_mva)
            tgt = TargetTestChecker().check(shadow.system_state, shadow.base_mva)
            regression_pass = bool(reg.passed)
            target_pass = bool(tgt.passed)
        return ast_ok, imports_ok, regression_pass, target_pass
    finally:
        shadow.cleanup()


# ── Collect ─────────────────────────────────────────────────────────────

def collect_rollouts(model, tokenizer, cfg, loader: SWEBenchLoader,
                     instance_ids: list[str], rollouts_per_instance: int,
                     temperature: float):
    """For every instance, sample `rollouts_per_instance` candidate patches.

    Each patch is verified to produce the 4-bit check vector, which is
    reduced to a scalar reward via `compute_swe_reward`. Both stages
    (localize + patch) are packaged as `StepSample`s sharing the same
    group key and reward, so log-prob updates cover the whole trajectory.
    """
    all_samples: list[StepSample] = []
    stats = {
        "total_episodes": 0,
        "target_pass": 0,
        "regression_pass": 0,
        "imports_pass": 0,
        "ast_pass": 0,
        "patch_nonempty": 0,
    }
    per_solved = defaultdict(int)
    per_total = defaultdict(int)

    for iid in instance_ids:
        for _ in range(rollouts_per_instance):
            try:
                inst = loader.load(iid)
                mgr = RepoManager(inst)
            except Exception as e:
                logger.warning("load instance %s failed: %s", iid, e)
                continue
            try:
                patch_str, loc_messages, patch_messages = sample_patch(
                    model, tokenizer, cfg, mgr, temperature=temperature,
                )
                stats["total_episodes"] += 1
                per_total[iid] += 1
                if patch_str:
                    stats["patch_nonempty"] += 1
                ast_ok, imports_ok, regression_pass, target_pass = verify_patch(
                    mgr, patch_str,
                )
                stats["ast_pass"] += int(ast_ok)
                stats["imports_pass"] += int(imports_ok)
                stats["regression_pass"] += int(regression_pass)
                stats["target_pass"] += int(target_pass)
                if target_pass:
                    per_solved[iid] += 1

                reward = compute_swe_reward(
                    ast_ok, imports_ok, regression_pass, target_pass,
                )

                # Package both agent turns (localize + patch) as samples so
                # policy gradient covers the full two-stage trajectory. Both
                # share the same reward and group_key.
                for messages in (loc_messages, patch_messages):
                    obs_text = messages[-2]["content"] if len(messages) >= 2 else ""
                    action_text = messages[-1]["content"]
                    all_samples.append(StepSample(
                        obs_text=obs_text,
                        action_text=action_text,
                        reward=reward,
                        group_key=(iid,),
                    ))
            except Exception as e:
                logger.exception("rollout for %s failed: %s", iid, e)
            finally:
                mgr.cleanup()

    stats["per_solved"] = dict(per_solved)
    stats["per_total"] = dict(per_total)
    return all_samples, stats


# ── Log-prob / policy update (verbatim from finance v2) ─────────────────

class _TrajectoryCarrier:
    """Local stand-in so we can reuse finance v2's log_prob helpers verbatim.

    Finance's helpers operate on message lists; we already build message
    lists in `sample_patch`. No adaptation needed beyond this module-level
    comment.
    """


def _find_action_start(tokenizer, _input_ids, messages):
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


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GRPO — SWE code repair (4-tier dense reward, finance v5 loop)")
    parser.add_argument("--model-path", required=True,
                        help="Base model checkpoint (e.g. Qwen3-14B).")
    parser.add_argument("--sft-adapter", required=True,
                        help="SFT LoRA adapter to start GRPO from.")
    parser.add_argument("--manifest", required=True,
                        help="SWE-bench Lite manifest JSONL.")
    parser.add_argument("--repo-cache", required=True,
                        help="Directory with mirror clones of each repo.")
    parser.add_argument("--output-dir", default="outputs/swe_grpo_model")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--rollouts-per-instance", type=int, default=4)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--kl-coeff", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=16)
    # adv-clip is honoured by silr.training.grpo_trainer.compute_advantages
    # (it clamps to [-3.0, 3.0] internally). Exposed as an arg for
    # parity with the task spec / finance v5 config; changing it has no
    # effect unless you also patch compute_advantages.
    parser.add_argument("--adv-clip", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = all instances; else cap instance count (smoke tests).")
    args = parser.parse_args()

    setup_logging(args.output_dir)
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("GRPO — SWE (4-tier dense reward)")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.model_path}")
    logger.info(f"SFT adapter: {args.sft_adapter}")
    logger.info(f"Iterations: {args.iters}, Rollouts/instance: {args.rollouts_per_instance}")
    logger.info(f"Clip: {args.clip_eps}, KL: {args.kl_coeff}, LR: {args.lr}")
    logger.info(f"Adv clip: {args.adv_clip} (applied inside compute_advantages)")
    logger.info(f"Temperature: {args.temperature}, batch size: {args.batch_size}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb_config,
        device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    logger.info("SFT adapter loaded (trainable)")
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable: {trainable:,}/{total:,} ({trainable/total*100:.2f}%)")

    cfg = build_swe_domain_config(with_observer=True)
    loader = SWEBenchLoader(args.manifest, args.repo_cache)
    instance_ids = loader.list_instance_ids()
    if args.limit:
        instance_ids = instance_ids[: args.limit]
    logger.info(f"Loaded {len(instance_ids)} SWE instances")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    iteration_metrics = []
    for iteration in range(args.iters):
        logger.info(f"\n{'='*40} Iteration {iteration+1}/{args.iters} {'='*40}")
        iter_t0 = time.perf_counter()
        logger.info(f"Phase 1: Rollouts — {len(instance_ids)} instances x "
                    f"{args.rollouts_per_instance}...")
        samples, rollout_stats = collect_rollouts(
            model=model, tokenizer=tokenizer, cfg=cfg, loader=loader,
            instance_ids=instance_ids,
            rollouts_per_instance=args.rollouts_per_instance,
            temperature=args.temperature,
        )
        n_eps = max(rollout_stats["total_episodes"], 1)
        solve_rate = rollout_stats["target_pass"] / n_eps
        logger.info(f"  Episodes: {rollout_stats['total_episodes']}, "
                    f"patches non-empty: {rollout_stats['patch_nonempty']}, "
                    f"target solved: {rollout_stats['target_pass']} "
                    f"({solve_rate*100:.1f}%)")
        logger.info(f"  Tier hits — ast:{rollout_stats['ast_pass']} "
                    f"imports:{rollout_stats['imports_pass']} "
                    f"regression:{rollout_stats['regression_pass']} "
                    f"target:{rollout_stats['target_pass']}")
        logger.info(f"  Step samples: {len(samples)}")

        logger.info("Phase 2: Advantages...")
        compute_log_probs(model, tokenizer, samples)
        compute_advantages(samples)

        group_sizes = Counter()
        reward_dist = Counter()
        for s in samples:
            group_sizes[s.group_key] += 1
            reward_dist[round(s.reward, 2)] += 1
        if group_sizes:
            size_vals = sorted(group_sizes.values())
            logger.info(f"  Groups: {len(group_sizes)}, "
                        f"min={size_vals[0]} max={size_vals[-1]}")
        logger.info(f"  Reward dist: {dict(sorted(reward_dist.items()))}")
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
            "iteration": iteration + 1,
            "solve_rate": solve_rate,
            "n_samples": len(samples),
            "avg_loss": avg_loss,
            "pos_advantages": pos_adv,
            "neg_advantages": neg_adv,
            "ast_pass": rollout_stats["ast_pass"],
            "imports_pass": rollout_stats["imports_pass"],
            "regression_pass": rollout_stats["regression_pass"],
            "target_pass": rollout_stats["target_pass"],
            "elapsed_seconds": iter_elapsed,
        })

        ckpt_dir = os.path.join(args.output_dir, f"iter_{iteration+1}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"  Saved: {ckpt_dir}")

    elapsed = time.perf_counter() - t0
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(iteration_metrics, f, indent=2)
    logger.info(f"\n{'='*60}")
    logger.info(f"Complete in {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    for m in iteration_metrics:
        logger.info(f"  Iter {m['iteration']}: solve={m['solve_rate']*100:.1f}%, "
                    f"loss={m['avg_loss']:.4f}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
