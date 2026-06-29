"""Multi-seed sweep over (scenario × gating-policy) for paper-grade ablation.

Why this script exists (paper context):
  - one ablation episode is an anecdote; with vLLM's temperature=0 inference
    still non-deterministic across runs (batch / kernel race effects), the
    only honest reporting is mean ± std over N reps.
  - this sweep produces the core Table-1 of the paper: recovery rate,
    proposal count, rejection count, and verdict mix (PASS / SAFE_PROGRESS /
    FAIL) for each (scenario, policy) cell, with N independent reps.
  - the three policies — verifier OFF (no admission gating), verifier
    terminal (zero-violation-only admission, the original SiLR semantics),
    verifier progress (recoverability-preserving admission), rollback
    (support-only post-hoc baseline), and the scalar-progress falsification baseline — span the ablation space defined
    by the panel.

Run on AMD ``silr-anm`` env, with vLLM serving Qwen3-14B on port 8001:

    PYTHONPATH=. python scripts/anm_eval_sweep.py \\
        --base-url http://localhost:8001/v1 --model qwen3-14b \\
        --reps 5 --max-steps 6 --output eval_sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import statistics
import time
from collections import Counter
from typing import Any

from domains.anm import (
    ANMScenarioLoader,
    GymANMManager,
    build_anm_domain_config,
)
from silr.agent import AgentConfig, ReActAgent
from silr.agent.llm.openai_client import OpenAIClient
from silr.verifier import SiLRVerifier
from scripts.anm_artifact_provenance import (
    code_fingerprint,
    sanitized_config,
    scenario_manifest,
)


SCENARIOS = ("easy_lightload", "medium_seed42_default", "hard_renewable_surge")
# (verify_on, gating_policy, label)
POLICIES = (
    (False, "terminal", "OFF"),        # gating_policy ignored when verify off
    (True,  "terminal", "terminal"),
    (True,  "progress", "progress"),
    (True,  "rollback", "rollback"),
    (True,  "progress_mag", "progress_mag"),
    (True,  "scalar_progress", "scalar_progress"),
)


def _verification_summary(verification_results: list[Any]) -> dict[str, Any]:
    """Compact per-step verifier summary for mechanism plots.

    Keep this local to the evaluation artifact so the runtime verifier remains
    domain-agnostic and unchanged. We summarize the last proposal because that
    is the proposal that determined the step outcome in the current ReAct loop.
    """
    if not verification_results:
        return {}
    vr = verification_results[-1]
    checks = getattr(vr, "check_results", []) or []
    violations = [
        v for cr in checks if not getattr(cr, "passed", False)
        for v in getattr(cr, "violations", [])
    ]
    severity_score = 0.0
    for v in violations:
        try:
            value = float(getattr(v, "value"))
            limit = float(getattr(v, "limit"))
        except (TypeError, ValueError):
            severity_score += 1.0
            continue
        if not (math.isfinite(value) and math.isfinite(limit)):
            severity_score += 1e6
            continue
        severity_score += abs(value - limit)
    return {
        "last_verdict": getattr(vr.verdict, "value", str(vr.verdict)),
        "last_fail_reason": vr.fail_reason,
        "last_violation_count": len(violations),
        "last_violation_types": sorted({
            getattr(cr, "checker_name", "")
            for cr in checks
            if not getattr(cr, "passed", False)
        }),
        "last_severity_score": severity_score,
    }


def setup_run_logger(log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("anm_eval_sweep")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def run_episode(
    client: OpenAIClient,
    scenario_id: str,
    gating_policy: str,
    enable_verification: bool,
    max_steps: int,
    max_proposals: int,
    temperature: float,
    rep_seed: int,
    stall_progress_budget: int | None = None,
    with_admission_criteria: bool = False,
) -> dict[str, Any]:
    """Run one episode and return summary stats (paper-grade numbers only)."""
    cfg = build_anm_domain_config(
        with_observer=True,
        gating_policy=gating_policy,
        with_admission_criteria=with_admission_criteria,
        stall_budget=stall_progress_budget,
    )
    loader = ANMScenarioLoader()
    scenario = loader.load(scenario_id)
    mgr_seed = scenario.source_seed if scenario.source_seed is not None else 42
    mgr = GymANMManager(seed=mgr_seed)
    converged = loader.setup_episode(mgr, scenario)
    default_penalty = mgr.last_penalty

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
            enable_verification=enable_verification,
            seed=rep_seed,  # passed to LLM call; ineffective for non-det vLLM but cheap
            stall_progress_budget=stall_progress_budget,
        ),
    )

    t0 = time.time()
    result = agent.run_episode(scenario_id=scenario_id)
    wallclock = time.time() - t0

    verdict_tally: Counter[str] = Counter()
    for s in result.steps:
        for vr in s.verification_results:
            verdict_tally[vr.verdict.value] += 1
    step_trace = []
    for s in result.steps:
        item = {
            "step": s.step_number,
            "outcome": s.outcome.value,
            "pre_penalty": s.pre_penalty,
            "post_penalty": s.post_penalty,
            "applied": s.applied_action is not None,
            "applied_action": s.applied_action,
            "verdicts": [
                getattr(vr.verdict, "value", str(vr.verdict))
                for vr in s.verification_results
            ],
        }
        item.update(_verification_summary(s.verification_results))
        step_trace.append(item)

    return {
        "scenario": scenario_id,
        "policy": gating_policy if enable_verification else "OFF",
        "stall_budget": stall_progress_budget,
        "stall_terminated": bool(getattr(result, "stall_terminated", False)),
        "rep_seed": rep_seed,
        "default_penalty": default_penalty,
        "default_converged": converged,
        "recovered": bool(result.recovered),
        "total_steps": result.total_steps,
        "total_proposals": result.total_proposals,
        "total_rejections": result.total_rejections,
        "final_penalty": mgr.last_penalty,
        "verdict_tally": dict(verdict_tally),
        "step_trace": step_trace,
        "wallclock_s": round(wallclock, 1),
    }


def _agg(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(values) == 1:
        return {"mean": round(values[0], 3), "std": 0.0, "n": 1}
    return {
        "mean": round(statistics.mean(values), 3),
        "std": round(statistics.stdev(values), 3),
        "n": len(values),
    }


def aggregate_cell(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    if n == 0:
        return {}
    rec = sum(1 for e in episodes if e["recovered"])
    pen = [e["final_penalty"] for e in episodes]
    prop = [e["total_proposals"] for e in episodes]
    rej = [e["total_rejections"] for e in episodes]
    wc = [e["wallclock_s"] for e in episodes]

    # Aggregate verdict mix (sum across reps, normalized by total verdicts).
    total_verdicts: Counter[str] = Counter()
    for e in episodes:
        for k, v in e["verdict_tally"].items():
            total_verdicts[k] += v
    grand = sum(total_verdicts.values()) or 1

    return {
        "n_reps": n,
        "recovery_rate": round(rec / n, 3),
        "final_penalty": _agg(pen),
        "proposals_per_episode": _agg(prop),
        "rejections_per_episode": _agg(rej),
        "wallclock_s": _agg(wc),
        "verdict_share": {
            k: round(v / grand, 3) for k, v in sorted(total_verdicts.items())
        },
        "verdict_total": dict(total_verdicts),
    }


def build_output(
    args: argparse.Namespace,
    selected_policies: list[tuple[bool, str, str]],
    all_episodes: list[dict[str, Any]],
    cells: dict[str, list[dict[str, Any]]],
    complete: bool,
    scenario_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = len(args.scenarios) * len(selected_policies) * args.reps
    return {
        "status": {
            "complete": complete,
            "completed_episodes": len(all_episodes),
            "expected_episodes": expected,
        },
        "config": sanitized_config(vars(args)),
        "scenario_manifest": (
            scenario_metadata
            if scenario_metadata is not None
            else scenario_manifest(args.scenarios)
        ),
        "code_fingerprint": code_fingerprint(extra_paths=("scripts/anm_eval_sweep.py",)),
        "policies": [p[2] for p in selected_policies],
        "episodes": all_episodes,
        "aggregates": {k: aggregate_cell(v) for k, v in cells.items()},
    }


def write_output_atomic(path: str, payload: dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp_path, path)


def _json_default(obj: Any):
    """Keep experiment artifacts writable when tools emit numpy scalars."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--model", default="qwen3-14b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--reps", type=int, default=5,
                        help="Independent reps per (scenario, policy) cell.")
    parser.add_argument("--rep-start-seed", type=int, default=1000,
                        help="First per-rep seed. Use 1003+ for focused "
                             "expansions that extend an existing 3-rep cell.")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-proposals", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout-s", type=float, default=60.0,
                        help="OpenAI-compatible read/write/pool timeout.")
    parser.add_argument("--connect-timeout-s", type=float, default=10.0,
                        help="OpenAI-compatible connect timeout.")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="OpenAI-compatible client retries.")
    parser.add_argument("--output", default="eval_sweep.json",
                        help="Write per-episode + aggregated results to this JSON.")
    parser.add_argument("--log-file", default=None,
                        help="Optional Python logging file for long remote runs.")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                        help="Subset of scenarios to run.")
    parser.add_argument("--policies", nargs="+",
                        default=[p[2] for p in POLICIES],
                        help="Subset of policy labels to run.")
    parser.add_argument("--stall-budget", type=int, default=None,
                        help="Anti-stall liveness budget (L4). None = disabled. "
                             "Used for per-layer ablation studies.")
    parser.add_argument("--with-admission-criteria", action="store_true",
                        help="Inject L2+L3 admission criteria into observation. "
                             "Tests progress-certificate forward-communication "
                             "of the apply-gate predicates to the LLM.")
    args = parser.parse_args()

    requested_policies = set(args.policies)
    selected_policies = [p for p in POLICIES if p[2] in requested_policies]
    unknown_policies = sorted(requested_policies - {p[2] for p in POLICIES})
    if unknown_policies:
        parser.error(f"Unknown policy label(s): {unknown_policies}")
    if not selected_policies:
        parser.error("At least one policy must be selected.")
    try:
        scenario_metadata = scenario_manifest(args.scenarios)
    except KeyError as exc:
        parser.error(str(exc))

    client = OpenAIClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout_s=args.request_timeout_s,
        connect_timeout_s=args.connect_timeout_s,
        max_retries=args.max_retries,
    )

    logger = setup_run_logger(args.log_file)

    logger.info("=== ANM gating-policy sweep ===")
    logger.info("  model=%s  endpoint=%s", args.model, args.base_url)
    logger.info(
        "  reps=%s  max_steps=%s  max_proposals=%s  temperature=%s",
        args.reps,
        args.max_steps,
        args.max_proposals,
        args.temperature,
    )
    logger.info("  rep_start_seed=%s", args.rep_start_seed)
    logger.info(
        "  request_timeout_s=%s  connect_timeout_s=%s  max_retries=%s",
        args.request_timeout_s,
        args.connect_timeout_s,
        args.max_retries,
    )
    logger.info("  scenarios=%s", args.scenarios)
    logger.info("  policies=%s", [p[2] for p in selected_policies])
    logger.info("")

    all_episodes: list[dict[str, Any]] = []
    cells: dict[str, list[dict[str, Any]]] = {}

    for scenario in args.scenarios:
        for verify_on, gating, label in selected_policies:
            cell_key = f"{scenario}__{label}"
            cells[cell_key] = []
            for rep in range(args.reps):
                t0 = time.time()
                ep = run_episode(
                    client=client,
                    scenario_id=scenario,
                    gating_policy=gating,
                    enable_verification=verify_on,
                    max_steps=args.max_steps,
                    max_proposals=args.max_proposals,
                    temperature=args.temperature,
                    rep_seed=args.rep_start_seed + rep,
                    stall_progress_budget=args.stall_budget,
                    with_admission_criteria=args.with_admission_criteria,
                )
                all_episodes.append(ep)
                cells[cell_key].append(ep)
                dt = time.time() - t0
                logger.info(
                    "  %-25s %-9s rep %s/%s: recovered=%s prop=%s "
                    "rej=%s penalty=%.2f verdicts=%s (%.1fs)",
                    scenario,
                    label,
                    rep + 1,
                    args.reps,
                    ep["recovered"],
                    ep["total_proposals"],
                    ep["total_rejections"],
                    ep["final_penalty"],
                    ep["verdict_tally"],
                    dt,
                )
                write_output_atomic(
                    args.output,
                    build_output(
                        args,
                        selected_policies,
                        all_episodes,
                        cells,
                        complete=False,
                        scenario_metadata=scenario_metadata,
                    ),
                )
            logger.info("")

    # Aggregate
    aggregates = {k: aggregate_cell(v) for k, v in cells.items()}

    # Pretty table
    logger.info("\n=== AGGREGATE TABLE (mean ± std over reps) ===")
    logger.info(
        "%-25s %-9s %9s %13s %13s %11s   verdict share",
        "scenario",
        "policy",
        "rec rate",
        "prop",
        "reject",
        "final pen",
    )
    logger.info("-" * 120)
    for scenario in args.scenarios:
        for _, _, label in selected_policies:
            agg = aggregates.get(f"{scenario}__{label}")
            if not agg:
                continue
            p = agg["proposals_per_episode"]
            r = agg["rejections_per_episode"]
            pen = agg["final_penalty"]
            logger.info(
                "%-25s %-9s %9.2f %6.1f±%-5.1f %6.1f±%-5.1f %6.2f±%-3.2f   %s",
                scenario,
                label,
                agg["recovery_rate"],
                p["mean"],
                p["std"],
                r["mean"],
                r["std"],
                pen["mean"],
                pen["std"],
                agg["verdict_share"],
            )
        logger.info("")

    output = build_output(
        args,
        selected_policies,
        all_episodes,
        cells,
        complete=True,
        scenario_metadata=scenario_metadata,
    )
    write_output_atomic(args.output, output)
    logger.info("\nWrote %s episodes + aggregates to %s", len(all_episodes), args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("anm_eval_sweep").exception("Fatal error in ANM sweep")
        raise
