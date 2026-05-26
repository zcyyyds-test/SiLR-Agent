"""Probe whether scalar-progress plateau states are recoverable by MPC.

This is a focused diagnostic for the paper's "scalar projection trap" claim:
when the scalar gate admits the first SAFE_PROGRESS action and then stalls, is
the resulting plateau state physically unrecoverable, or did the gate simply
commit the ReAct search to a poor basin?

The script reruns only the first scalar-progress ReAct step, records the first
admitted state, then asks gym-anm's MPCAgentConstant to act from that same state.
It uses normal SiLR manager/tool/verifier paths so the diagnostic stays aligned
with the main evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

from domains.anm import ANMScenarioLoader, GymANMManager, build_anm_domain_config
from gym_anm import MPCAgentConstant
from scripts.anm_artifact_provenance import (
    code_fingerprint,
    sanitized_config,
    scenario_manifest,
)
from scripts.anm_mpc_baseline import _unpack_mpc_action
from silr.agent import AgentConfig, ReActAgent
from silr.agent.llm.openai_client import OpenAIClient
from silr.verifier import SiLRVerifier


DEFAULT_SCENARIOS = ("mined_multi_action_3_l0p25g1p0_s12",)


def setup_logger(log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("anm_mpc_from_scalar_plateau")
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


def _severity_score(check_results: list[Any]) -> float:
    score = 0.0
    for cr in check_results:
        if getattr(cr, "passed", False):
            continue
        for viol in getattr(cr, "violations", []) or []:
            try:
                value = float(getattr(viol, "value"))
                limit = float(getattr(viol, "limit"))
            except (TypeError, ValueError):
                score += 1.0
                continue
            if not (math.isfinite(value) and math.isfinite(limit)):
                score += 1e6
                continue
            score += abs(value - limit)
    return float(score)


def _checker_snapshot(mgr: GymANMManager, cfg: Any) -> dict[str, Any]:
    """Serialize current checker state for plateau/MPC comparisons."""
    total = 0
    breakdown: dict[str, int] = {}
    violation_details: list[dict[str, Any]] = []
    checks = []
    for checker in cfg.checkers:
        cr = checker.check(mgr.system_state, mgr.base_mva)
        checks.append(cr)
        n = 0 if cr.passed else len(cr.violations)
        breakdown[checker.name] = n
        total += n
        for v in cr.violations:
            violation_details.append(
                {
                    "checker": checker.name,
                    "constraint_type": v.constraint_type,
                    "device_type": v.device_type,
                    "device_id": v.device_id,
                    "metric": v.metric,
                    "value": v.value,
                    "limit": v.limit,
                    "unit": v.unit,
                    "severity": v.severity,
                    "detail": v.detail,
                }
            )
    return {
        "penalty": float(mgr.last_penalty),
        "reward": float(mgr.last_reward),
        "violations": int(total),
        "breakdown": breakdown,
        "severity_score": round(_severity_score(checks), 6),
        "violation_details": violation_details,
    }


def _step_to_json(step: Any) -> dict[str, Any]:
    verdicts = [
        getattr(vr.verdict, "value", str(vr.verdict))
        for vr in step.verification_results
    ]
    return {
        "step": step.step_number,
        "outcome": step.outcome.value,
        "pre_penalty": step.pre_penalty,
        "post_penalty": step.post_penalty,
        "thought": step.thought,
        "proposed_actions": step.proposed_actions,
        "applied_action": step.applied_action,
        "verdicts": verdicts,
        "fail_reasons": [
            vr.fail_reason for vr in step.verification_results if vr.fail_reason
        ],
        "error": step.error,
    }


def _find_first_admission(result: Any) -> tuple[Any | None, Any | None]:
    for step in result.steps:
        if step.applied_action is None or not step.verification_results:
            continue
        vr = step.verification_results[-1]
        verdict = getattr(vr.verdict, "value", str(vr.verdict))
        if verdict in ("SAFE_PROGRESS", "PASS"):
            return step, vr
    return None, None


def run_scalar_first_admission(
    client: OpenAIClient,
    scenario_id: str,
    rep_seed: int,
    max_proposals: int,
    temperature: float,
    request_label: str,
) -> dict[str, Any]:
    cfg = build_anm_domain_config(
        with_observer=True,
        gating_policy="scalar_progress",
    )
    check_cfg = build_anm_domain_config(with_observer=False)
    loader = ANMScenarioLoader()
    scenario = loader.load(scenario_id)
    mgr_seed = scenario.source_seed if scenario.source_seed is not None else 42
    mgr = GymANMManager(seed=mgr_seed)
    default_converged = loader.setup_episode(mgr, scenario)
    default_snapshot = _checker_snapshot(mgr, check_cfg)

    verifier = SiLRVerifier(mgr, domain_config=cfg)
    agent = ReActAgent(
        manager=mgr,
        verifier=verifier,
        llm_client=client,
        domain_config=cfg,
        config=AgentConfig(
            max_steps=1,
            max_proposals_per_step=max_proposals,
            temperature=temperature,
            enable_verification=True,
            seed=rep_seed,
        ),
    )

    t0 = time.time()
    result = agent.run_episode(scenario_id=scenario_id)
    scalar_wallclock_s = time.time() - t0
    step, vr = _find_first_admission(result)
    scalar_trace = [_step_to_json(s) for s in result.steps]
    verdict_tally: Counter[str] = Counter()
    for s in result.steps:
        for item in s.verification_results:
            verdict_tally[getattr(item.verdict, "value", str(item.verdict))] += 1

    if step is None or vr is None:
        return {
            "scenario": scenario_id,
            "rep_seed": rep_seed,
            "request_label": request_label,
            "status": "no_admission",
            "default_converged": bool(default_converged),
            "default": default_snapshot,
            "scalar_wallclock_s": round(scalar_wallclock_s, 3),
            "scalar_total_proposals": result.total_proposals,
            "scalar_total_rejections": result.total_rejections,
            "scalar_verdict_tally": dict(verdict_tally),
            "scalar_trace": scalar_trace,
        }

    plateau_snapshot = _checker_snapshot(mgr, check_cfg)
    plateau_state = {
        "P_load": dict(mgr._P_load),
        "P_pot": dict(mgr._P_pot),
        "P_set": dict(mgr._P_set),
        "Q_set": dict(mgr._Q_set),
    }

    mpc_t0 = time.time()
    env = mgr._env
    mpc_agent = MPCAgentConstant(
        simulator=env.simulator,
        action_space=env.action_space,
        gamma=env.gamma,
        safety_margin=0.9,
        planning_steps=8,
    )
    mpc_action = np.asarray(mpc_agent.act(env), dtype=float)
    P_set, Q_set = _unpack_mpc_action(mgr, mpc_action)
    mgr._P_set = P_set
    mgr._Q_set = Q_set
    mpc_converged = mgr.solve()
    mpc_wallclock_s = time.time() - mpc_t0
    mpc_snapshot = _checker_snapshot(mgr, check_cfg)

    admitted_verdict = getattr(vr.verdict, "value", str(vr.verdict))
    return {
        "scenario": scenario_id,
        "rep_seed": rep_seed,
        "request_label": request_label,
        "status": "complete",
        "default_converged": bool(default_converged),
        "default": default_snapshot,
        "admitted_step": step.step_number,
        "admitted_verdict": admitted_verdict,
        "admitted_fail_reason": vr.fail_reason,
        "admitted_action": step.applied_action,
        "plateau": plateau_snapshot,
        "plateau_state": plateau_state,
        "scalar_wallclock_s": round(scalar_wallclock_s, 3),
        "scalar_total_proposals": result.total_proposals,
        "scalar_total_rejections": result.total_rejections,
        "scalar_verdict_tally": dict(verdict_tally),
        "scalar_trace": scalar_trace,
        "mpc": {
            "converged": bool(mpc_converged),
            "recovered": bool(mpc_converged and mpc_snapshot["violations"] == 0),
            "action": [round(float(x), 6) for x in mpc_action.tolist()],
            "P_set": P_set,
            "Q_set": Q_set,
            **mpc_snapshot,
            "wallclock_s": round(mpc_wallclock_s, 3),
        },
    }


def _write_atomic(path: str, payload: dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def build_output(args: argparse.Namespace, episodes: list[dict[str, Any]], complete: bool) -> dict[str, Any]:
    expected = len(args.scenarios) * len(args.seeds)
    return {
        "status": {
            "complete": complete,
            "completed_episodes": len(episodes),
            "expected_episodes": expected,
        },
        "config": sanitized_config(vars(args)),
        "scenario_manifest": scenario_manifest(args.scenarios),
        "code_fingerprint": code_fingerprint(
            extra_paths=(
                "scripts/anm_mpc_from_scalar_plateau.py",
                "scripts/anm_mpc_baseline.py",
                "scripts/anm_eval_sweep.py",
            )
        ),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--model", default="qwen3-14b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001, 1002])
    parser.add_argument("--max-proposals", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout-s", type=float, default=240.0)
    parser.add_argument("--connect-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--output", default="eval_mpc_from_scalar_plateau_gpu0.json")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger = setup_logger(args.log_file)
    logger.info("=== MPC from scalar-progress plateau diagnostic ===")
    logger.info("  model=%s endpoint=%s", args.model, args.base_url)
    logger.info("  scenarios=%s", args.scenarios)
    logger.info("  seeds=%s", args.seeds)
    logger.info(
        "  max_proposals=%s temperature=%s request_timeout_s=%s max_retries=%s",
        args.max_proposals,
        args.temperature,
        args.request_timeout_s,
        args.max_retries,
    )
    logger.info(
        "  scalar_relative_slack=%s",
        os.environ.get("SILR_SCALAR_PROGRESS_RELATIVE_SLACK", "<unset>"),
    )

    client = OpenAIClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout_s=args.request_timeout_s,
        connect_timeout_s=args.connect_timeout_s,
        max_retries=args.max_retries,
    )

    episodes: list[dict[str, Any]] = []
    for scenario_id in args.scenarios:
        for seed in args.seeds:
            label = f"{scenario_id}__seed{seed}"
            logger.info("")
            logger.info(">>> %s", label)
            try:
                ep = run_scalar_first_admission(
                    client=client,
                    scenario_id=scenario_id,
                    rep_seed=seed,
                    max_proposals=args.max_proposals,
                    temperature=args.temperature,
                    request_label=label,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("episode failed: %s", label)
                ep = {
                    "scenario": scenario_id,
                    "rep_seed": seed,
                    "request_label": label,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            episodes.append(ep)
            if ep.get("status") == "complete":
                logger.info(
                    "  scalar: %s pen %.3f -> %.3f action=%s",
                    ep["admitted_verdict"],
                    ep["default"]["penalty"],
                    ep["plateau"]["penalty"],
                    ep["admitted_action"],
                )
                logger.info(
                    "  mpc   : recovered=%s pen %.3f viol=%s",
                    ep["mpc"]["recovered"],
                    ep["mpc"]["penalty"],
                    ep["mpc"]["violations"],
                )
            else:
                logger.info(
                    "  status=%s proposals=%s rejections=%s",
                    ep.get("status"),
                    ep.get("scalar_total_proposals"),
                    ep.get("scalar_total_rejections"),
                )
            _write_atomic(args.output, build_output(args, episodes, complete=False))

    _write_atomic(args.output, build_output(args, episodes, complete=True))
    logger.info("")
    logger.info("Wrote %s", args.output)
    logger.info("MPC-FROM-SCALAR-PLATEAU DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()
