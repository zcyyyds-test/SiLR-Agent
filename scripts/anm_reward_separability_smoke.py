"""Offline reward-separability smoke — the GRPO go/no-go gate (panel 2026-06-03).

Before investing GPU-weeks in 3-arm GRPO training, falsify the cheapest way the
whole "structured reward extends structure-vs-scalar to the training signal"
thesis could be dead: **arm D (geometry) and arm E (count projection) might rank
the same SAFE_PROGRESS steps identically on single-type ANM**, in which case the
structured reward provides no signal a scalar reward lacks — no clever encoding
saves a domain with no geometry to preserve (DS + MiMo cross-family flag).

What this does (no training): run greedy rollouts on the sub-saturated band
scenarios under the progress_mag gate, and for every accepted SAFE_PROGRESS step
compute the three per-step rewards from the persisted per-branch geometry
Φ = (S, σ):
    r_C = compute_binary_reward   (arm C — verdict only)
    r_D = compute_grpo_reward     (arm D — severity-weighted product-order descent)
    r_E = compute_scalar_reward   (arm E — count-delta projection)
then report Spearman ρ(D,E) and ρ(D,C) over the pooled SAFE_PROGRESS steps, plus
per-scenario severity heterogeneity (max σ_i / min σ_i — if ≈1 the branches are
uniform and D necessarily collapses to E).

GATE (printed + in JSON):
    ρ(D,E) ≳ 0.9  OR  median severity-heterogeneity ≈ 1
        → arm D does not separate from arm E → DO NOT spend GPU-weeks; escalate
          to a severity-skewed subset / multi-type scenarios / reconsider.
    else → separation present → proceed to train_grpo_anm.py.

Runs on TSUBAME against a local vLLM endpoint, mirroring anm_eval_sweep.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.anm import (
    ANMScenarioLoader,
    GymANMManager,
    build_anm_domain_config,
)
from silr.agent import AgentConfig, ReActAgent
from silr.agent.llm.openai_client import OpenAIClient
from silr.agent.types import StepOutcome
from silr.verifier import SiLRVerifier
from silr.verifier.types import Verdict
from silr.training.reward import (
    compute_binary_reward,
    compute_grpo_reward,
    compute_scalar_reward,
)

logger = logging.getLogger("reward_sep")

# The 9 sub-saturated band scenarios from Step 0 (Qwen3-8B, recovery < 3/3).
# Default training/diagnostic set; override with --scenarios.
DEFAULT_SCENARIOS = [
    "mined_multi_action_6_l1p0g1p0_s16_socnear_min",   # 0/3 — hardest
    "mined_multi_action_1_l0p25g1p0_s5_socnear_min",   # 1/3
    "mined_multi_action_5_l0p25g1p0_s16_socnear_max",  # 1/3
    "mined_multi_action_2_l1p0g1p0_s5_socnear_min",    # 2/3
    "mined_multi_action_4_l1p0g1p0_s12",               # 2/3
    "mined_multi_action_5_l0p25g1p0_s16",              # 2/3
    "mined_multi_action_5_l0p25g1p0_s16_socnear_min",  # 2/3
    "mined_multi_action_7_l0p25g1p0_s19_socnear_max",  # 2/3
    "mined_multi_action_8_l1p0g1p0_s19",               # 2/3
]


def _severity_heterogeneity(branches: dict | None) -> float | None:
    """max σ_i / min σ_i over a per-branch map; None if <2 branches or no positive σ."""
    if not branches:
        return None
    vals = [v for v in branches.values() if v > 0]
    if len(vals) < 2:
        return None
    return max(vals) / (min(vals) + 1e-12)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation, stdlib only. None if <3 points or zero variance."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(vs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # average rank for ties (1-based)
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def run_episode_dump(client, scenario_id, max_steps, max_proposals, temperature, rep_seed):
    """Run one progress_mag episode; return per-accepted-step reward records."""
    cfg = build_anm_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = ANMScenarioLoader()
    scenario = loader.load(scenario_id)
    mgr_seed = scenario.source_seed if scenario.source_seed is not None else 42
    mgr = GymANMManager(seed=mgr_seed)
    loader.setup_episode(mgr, scenario)
    verifier = SiLRVerifier(mgr, domain_config=cfg)
    agent = ReActAgent(
        manager=mgr,
        verifier=verifier,
        llm_client=client,
        domain_config=cfg,
        config=AgentConfig(
            max_steps=max_steps,
            max_proposals_per_step=max_proposals,
            temperature=temperature,
            enable_verification=True,
            seed=rep_seed,
        ),
    )
    result = agent.run_episode(scenario_id=scenario_id)

    records = []
    for s in result.steps:
        if s.outcome != StepOutcome.SUCCESS or not s.verification_results:
            continue
        vr = s.verification_results[-1]  # the accepted action's verdict
        rec = {
            "scenario": scenario_id,
            "rep_seed": rep_seed,
            "step": s.step_number,
            "verdict": vr.verdict.value,
            "n_pre": len(vr.baseline_branches or {}),
            "n_post": len(vr.post_branches or {}),
            "severity_heterogeneity_pre": _severity_heterogeneity(vr.baseline_branches),
            "r_C": compute_binary_reward(vr),
            "r_D": compute_grpo_reward(vr),
            "r_E": compute_scalar_reward(vr),
        }
        records.append(rec)
    return records, bool(result.recovered)


def main():
    ap = argparse.ArgumentParser(description="Offline reward-separability smoke (GRPO gate)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8005/v1")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rep-start-seed", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-proposals", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--request-timeout-s", type=float, default=360.0)
    ap.add_argument("--max-retries", type=int, default=0)
    ap.add_argument("--output", default="reward_separability_smoke.json")
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--gate-rho", type=float, default=0.9,
                    help="ρ(D,E) at/above which arm D is judged non-separable")
    args = ap.parse_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)

    client = OpenAIClient(
        model=args.model, api_key=args.api_key, base_url=args.base_url,
        timeout_s=args.request_timeout_s, max_retries=args.max_retries,
    )

    logger.info("Reward-separability smoke | model=%s | %d scenarios x %d reps | gate ρ(D,E)≥%.2f",
                args.model, len(args.scenarios), args.reps, args.gate_rho)

    all_records = []
    per_scenario_rec = defaultdict(lambda: {"rec": 0, "n": 0})
    for sc in args.scenarios:
        for r in range(args.reps):
            seed = args.rep_start_seed + r
            t0 = time.time()
            try:
                recs, recovered = run_episode_dump(
                    client, sc, args.max_steps, args.max_proposals, args.temperature, seed)
            except Exception as e:  # keep the smoke alive across one bad episode
                logger.warning("  %s rep %d FAILED: %s", sc, r + 1, e)
                continue
            all_records.extend(recs)
            per_scenario_rec[sc]["n"] += 1
            per_scenario_rec[sc]["rec"] += int(recovered)
            n_sp = sum(1 for x in recs if x["verdict"] == "SAFE_PROGRESS")
            logger.info("  %s rep %d/%d: recovered=%s steps=%d SAFE_PROGRESS=%d (%.1fs)",
                        sc, r + 1, args.reps, recovered, len(recs), n_sp, time.time() - t0)

    # Pool SAFE_PROGRESS steps — the only verdict where arms D and E differ.
    sp = [x for x in all_records if x["verdict"] == "SAFE_PROGRESS"]
    rD = [x["r_D"] for x in sp]
    rE = [x["r_E"] for x in sp]
    rC = [x["r_C"] for x in sp]
    rho_DE = _spearman(rD, rE)
    rho_DC = _spearman(rD, rC)
    het = [x["severity_heterogeneity_pre"] for x in sp
           if x["severity_heterogeneity_pre"] is not None]
    het_sorted = sorted(het)
    het_median = het_sorted[len(het_sorted) // 2] if het_sorted else None

    # Gate verdict.
    collapsed = (
        (rho_DE is not None and rho_DE >= args.gate_rho)
        or (het_median is not None and het_median < 1.05)
        or len(sp) < 10
    )
    gate = "COLLAPSED — escalate (severity-skewed/multi-type), do NOT train" if collapsed \
        else "SEPARABLE — proceed to train_grpo_anm.py"

    summary = {
        "n_safe_progress_steps": len(sp),
        "n_total_accepted_steps": len(all_records),
        "spearman_rho_D_E": rho_DE,
        "spearman_rho_D_C": rho_DC,
        "severity_heterogeneity_pre": {
            "median": het_median,
            "min": het_sorted[0] if het_sorted else None,
            "max": het_sorted[-1] if het_sorted else None,
            "n": len(het),
        },
        "gate_rho_threshold": args.gate_rho,
        "gate_verdict": gate,
        "per_scenario_recovery": {k: f"{v['rec']}/{v['n']}" for k, v in per_scenario_rec.items()},
    }

    out = {"config": vars(args), "summary": summary, "records": all_records}
    tmp = args.output + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.output)

    logger.info("=" * 64)
    logger.info("SAFE_PROGRESS steps: %d | ρ(D,E)=%s | ρ(D,C)=%s | σ-heterogeneity median=%s",
                len(sp), _fmt(rho_DE), _fmt(rho_DC), _fmt(het_median))
    logger.info("GATE: %s", gate)
    logger.info("Output: %s", args.output)


def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    main()
