"""Greedy eval of a GRPO-trained CityLearn adapter (or untrained base) -- decisive measure.

Training-time sampling rollouts (temp 0.7) are noisy by design; the paper number
is GREEDY (temp 0). This evaluates a model (base, optionally + LoRA adapter) on
the hardened N=4 multi-type band under two regimes:

    gated   : verifier ON (progress_mag) — recovery WITH the runtime gate.
    ungated : verifier OFF — recovery WITHOUT the gate. This is the KEY
              internalisation probe (panel/agy): if a GRPO-trained arm recovers
              ungated and leaves lower residual penalty (less material worsening)
              than the untrained base, the model has INTERNALISED the recovery
              geometry into its policy, not merely leaned on the runtime gate.

Run base (arm A baseline) and each trained adapter (C/D/E final) through this;
compare gated recovery (did training help at all) and ungated recovery + final
penalty (did it internalise the geometry). Thinking-off + temp 0, consistent
with training. In-process HF (mirrors train_grpo_citylearn rollout path).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.citylearn import (
    CityLearnScenarioLoader,
    CityLearnManager,
    build_citylearn_domain_config,
)
from silr.agent import AgentConfig, ReActAgent
from silr.verifier import SiLRVerifier
from scripts.train_grpo_citylearn import LocalModelClient, DEFAULT_SCENARIOS

logger = logging.getLogger("grpo_eval")


def eval_episode(client, scenario_id, gated, max_steps, max_proposals, observe_trace=False):
    cfg = build_citylearn_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = CityLearnScenarioLoader()
    sc = loader.load(scenario_id)
    mgr = CityLearnManager(
        fixed_t=sc.fixed_t, initial_soc=sc.initial_soc,
        initial_actions=sc.initial_actions, peak_import_kw=sc.peak_import_kw,
    )
    loader.setup_episode(mgr, sc)
    default_penalty = mgr.last_penalty
    verifier = SiLRVerifier(mgr, domain_config=cfg)
    agent = ReActAgent(
        manager=mgr, verifier=verifier, llm_client=client, domain_config=cfg,
        config=AgentConfig(max_steps=max_steps, max_proposals_per_step=max_proposals,
                           temperature=0.0, enable_verification=gated,
                           observe_verification=observe_trace, seed=0))
    result = agent.run_episode(scenario_id=sc.id)
    # Per-step geometry trace for mechanism metrics (pre-registered): does the
    # policy use the product-order geometry (eliminate the WORST branch, avoid
    # magnitude drift) rather than merely reduce count? Captured from the
    # accepted action's persisted Φ = (S, σ).
    from silr.agent.types import StepOutcome
    def _fam_summary(branches):
        out = {}
        for k, v in branches.items():
            fam = k[0] if isinstance(k, (list, tuple)) else str(k)
            c, s = out.get(fam, (0, 0.0))
            out[fam] = (c + 1, round(s + float(v), 4))
        return out

    step_trace = []
    first_admissible_step = None
    for i, s in enumerate(result.steps):
        if not s.verification_results:
            continue
        vr = s.verification_results[-1]
        pre = vr.baseline_branches or {}
        post = vr.post_branches or {}
        accepted = s.outcome == StepOutcome.SUCCESS
        if accepted and first_admissible_step is None:
            first_admissible_step = i
        item = {
            "step": i,
            "verdict": vr.verdict.value,
            "accepted": accepted,
            "n_pre": len(pre),
            "n_post": len(post),
            "max_sigma_pre": max(pre.values()) if pre else 0.0,
            "max_sigma_post": max(post.values()) if post else 0.0,
            "sum_sigma_pre": sum(pre.values()) if pre else 0.0,
            "sum_sigma_post": sum(post.values()) if post else 0.0,
            # worst-branch reduction: did this step shrink the single most severe branch?
            "worst_branch_reduced": (max(pre.values()) - max(post.values())) if (pre and post)
            else (max(pre.values()) if pre else 0.0),
            # per-family residual (k[0] = constraint_type) so we can see WHICH family
            # a policy leaves violated -- the sigma-het mechanism check (does the scalar
            # arm clear the big-sigma power family but leave the small-sigma soc family?)
            "post_fam": _fam_summary(post),
            "pre_fam": _fam_summary(pre),
        }
        step_trace.append(item)
    return {
        "scenario": scenario_id,
        "gated": gated,
        "observed": observe_trace,
        "recovered": bool(result.recovered),
        "default_penalty": default_penalty,
        "final_penalty": mgr.last_penalty,
        "proposals": result.total_proposals,
        "rejections": result.total_rejections,
        "steps": result.total_steps,
        "first_admissible_step": first_admissible_step,
        "step_trace": step_trace,
    }


def _agg(records):
    n = len(records)
    rec = sum(1 for r in records if r["recovered"])
    pen = [r["final_penalty"] for r in records]
    prop = sum(r["proposals"] for r in records)
    rej = sum(r["rejections"] for r in records)
    return {
        "n": n,
        "recovered": rec,
        "recovery_rate": round(rec / n, 3) if n else None,
        "mean_final_penalty": round(sum(pen) / n, 3) if n else None,
        "max_final_penalty": round(max(pen), 3) if pen else None,
        "reject_ratio": round(rej / prop, 3) if prop else None,
    }


def main():
    p = argparse.ArgumentParser(description="Greedy gated+ungated eval of base/adapter")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--adapter", default="", help="LoRA adapter dir; empty = untrained base (arm A)")
    p.add_argument("--label", required=True, help="e.g. baseA / armD")
    p.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-proposals", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--output", required=True)
    p.add_argument("--log-file", default=None)
    p.add_argument("--observer-trace", action="store_true",
                   help="Single ungated pass that ALSO logs Φ passively (no gating). "
                        "Measures the policy's intrinsic product-order geometry use "
                        "(pre-reg H1b worst-branch elimination) which the verifier-off "
                        "primary DV cannot record. Writes its own JSON.")
    args = p.parse_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)

    logger.info("Greedy eval | label=%s | adapter=%s | %d scenarios",
                args.label, args.adapter or "(none)", len(args.scenarios))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map={"": 0}, trust_remote_code=True,
        torch_dtype=torch.bfloat16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        logger.info("Adapter loaded: %s", args.adapter)
    model.eval()
    client = LocalModelClient(model, tokenizer, max_new_tokens=args.max_new_tokens,
                              enable_thinking=False)

    records = []
    if args.observer_trace:
        # Dedicated single pass: ungated execution + passive Φ trace. Recovery
        # numbers match the plain ungated regime (greedy temp 0, verify() has no
        # side effect); the new payload is the per-step Φ geometry under no gating.
        for sid in args.scenarios:
            t0 = time.time()
            r = eval_episode(client, sid, False, args.max_steps, args.max_proposals,
                             observe_trace=True)
            records.append(r)
            logger.info("  [observer] %s recovered=%s pen=%.2f (def %.2f) steps_traced=%d (%.1fs)",
                        sid, r["recovered"], r["final_penalty"], r["default_penalty"],
                        len(r["step_trace"]), time.time() - t0)
    else:
        for regime_gated in (True, False):
            for sid in args.scenarios:
                t0 = time.time()
                r = eval_episode(client, sid, regime_gated, args.max_steps, args.max_proposals)
                records.append(r)
                logger.info("  [%s] %s recovered=%s pen=%.2f (def %.2f) rej/prop=%d/%d (%.1fs)",
                            "gated" if regime_gated else "ungated", sid, r["recovered"],
                            r["final_penalty"], r["default_penalty"], r["rejections"],
                            r["proposals"], time.time() - t0)

    gated = [r for r in records if r["gated"] and not r.get("observed")]
    ungated = [r for r in records if not r["gated"] and not r.get("observed")]
    observed = [r for r in records if r.get("observed")]
    summary = {"label": args.label, "adapter": args.adapter,
               "gated": _agg(gated), "ungated": _agg(ungated)}
    if observed:
        summary["observed"] = _agg(observed)
    out = {"summary": summary, "records": records}
    tmp = args.output + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.output)

    logger.info("=" * 64)
    if observed:
        o = summary["observed"]
        logger.info("LABEL %s | OBSERVER(ungated+Φ trace) recovery %s/%s pen=%s",
                    args.label, o["recovered"], o["n"], o["mean_final_penalty"])
    else:
        logger.info("LABEL %s | GATED recovery %s/%s pen=%.2f | UNGATED recovery %s/%s pen=%.2f",
                    args.label, summary["gated"]["recovered"], summary["gated"]["n"],
                    summary["gated"]["mean_final_penalty"],
                    summary["ungated"]["recovered"], summary["ungated"]["n"],
                    summary["ungated"]["mean_final_penalty"])
    logger.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
