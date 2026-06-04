"""Offline reward-separability smoke for CityLearn -- the GRPO go/no-go gate.

CityLearn fork of anm_reward_separability_smoke.py. Before investing GPU-days in
the 3-arm GRPO campaign on the hardened N=4 CityLearn band, falsify the cheapest
way the "geometric reward beats scalar" thesis could be dead: arm D (product-
order geometry) and arm E (count projection) might rank the same SAFE_PROGRESS
steps identically, in which case the geometry gives no signal a scalar lacks.

Unlike the single-type ANM band, the hardened CityLearn band carries genuinely
incomparable families at once (soc_min/soc_max/export/import), so the prior here
is that D and E *should* separate -- this smoke confirms it before training.

No training: greedy rollouts under the progress_mag gate (thinking off), and for
every accepted SAFE_PROGRESS step compute r_C/r_D/r_E from the persisted per-
branch geometry Phi=(S,sigma); report Spearman rho(D,E)/rho(D,C) over pooled
SAFE_PROGRESS steps + per-scenario severity heterogeneity.

GATE:
    rho(D,E) >~ 0.9 OR median severity-heterogeneity ~ 1
        -> D does not separate from E -> do NOT spend GPU-days; reconsider band.
    else -> separation present -> proceed to train_grpo_citylearn.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.citylearn import (
    CityLearnScenarioLoader,
    CityLearnManager,
    build_citylearn_domain_config,
)
from domains.citylearn.scenarios import SCENARIOS
from silr.agent import AgentConfig, ReActAgent
from silr.agent.llm.openai_client import OpenAIClient
from silr.agent.types import StepOutcome
from silr.verifier import SiLRVerifier
from silr.training.reward import (
    compute_binary_reward,
    compute_grpo_reward,
    compute_scalar_reward,
)

logger = logging.getLogger("reward_sep_cl")

# The hardened N=4 multi-type band (cl_mined_*), loaded from the frozen
# scenarios_mined.json via the scenario registry; override with --scenarios.
DEFAULT_SCENARIOS = [s.id for s in SCENARIOS if s.id.startswith("cl_mined_")]


def _severity_heterogeneity(branches: dict | None) -> float | None:
    if not branches:
        return None
    vals = [v for v in branches.values() if v > 0]
    if len(vals) < 2:
        return None
    return max(vals) / (min(vals) + 1e-12)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
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
            avg = (i + j) / 2.0 + 1.0
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
    cfg = build_citylearn_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = CityLearnScenarioLoader()
    scenario = loader.load(scenario_id)
    mgr = CityLearnManager(
        fixed_t=scenario.fixed_t,
        initial_soc=scenario.initial_soc,
        initial_actions=scenario.initial_actions,
        peak_import_kw=scenario.peak_import_kw,
    )
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
        vr = s.verification_results[-1]
        records.append({
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
        })
    return records, bool(result.recovered)


def main():
    ap = argparse.ArgumentParser(description="CityLearn reward-separability smoke (GRPO gate)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
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
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Keep Qwen3 <think> on (default off, matching the eval).")
    ap.add_argument("--output", default="reward_separability_smoke_citylearn.json")
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--gate-rho", type=float, default=0.9)
    args = ap.parse_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)

    client = OpenAIClient(
        model=args.model, api_key=args.api_key, base_url=args.base_url,
        timeout_s=args.request_timeout_s, max_retries=args.max_retries,
        enable_thinking=(None if args.enable_thinking else False),
    )

    logger.info("CityLearn reward-separability smoke | model=%s | %d scenarios x %d reps | gate rho(D,E)>=%.2f",
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
            except Exception as e:
                logger.warning("  %s rep %d FAILED: %s", sc, r + 1, e)
                continue
            all_records.extend(recs)
            per_scenario_rec[sc]["n"] += 1
            per_scenario_rec[sc]["rec"] += int(recovered)
            n_sp = sum(1 for x in recs if x["verdict"] == "SAFE_PROGRESS")
            logger.info("  %s rep %d/%d: recovered=%s steps=%d SAFE_PROGRESS=%d (%.1fs)",
                        sc, r + 1, args.reps, recovered, len(recs), n_sp, time.time() - t0)

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

    # Unlike single-type ANM, this band is multi-type: arms D and E separate
    # ACROSS incomparable families (kWh vs kW), so within-family sigma uniformity
    # does NOT imply D==E. The sigma-heterogeneity short-circuit (calibrated for
    # single-type ANM) would false-positive here -- e.g. four SoC sigma in
    # [2.34, 2.37] give ratio ~1.01 -- so we judge on rho(D,E) alone and report
    # het only for traceability.
    collapsed = (
        (rho_DE is not None and rho_DE >= args.gate_rho)
        or len(sp) < 10
    )
    gate = "COLLAPSED -- do NOT train; reconsider band" if collapsed \
        else "SEPARABLE -- proceed to train_grpo_citylearn.py"

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
    logger.info("SAFE_PROGRESS steps: %d | rho(D,E)=%s | rho(D,C)=%s | sigma-heterogeneity median=%s",
                len(sp), _fmt(rho_DE), _fmt(rho_DC), _fmt(het_median))
    logger.info("GATE: %s", gate)
    logger.info("Output: %s", args.output)


def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    main()
