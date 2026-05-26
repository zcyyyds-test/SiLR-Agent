"""Cross-domain smoke: run a single cluster-scheduling scenario through SiLR.

Goal (paper §5.7): demonstrate that the SiLR apply-gate machinery extends
beyond power dispatch. Uses cluster domain's existing manager + checkers +
tools, runs 1 scenario × {OFF, terminal, progress} × N=1 with Qwen3-14B,
records recovery / proposals / verdict-share.

This is a **smoke test, not a deep eval**: 1 rep × 1 scenario × 3 policies
suffices to demonstrate framework generality. Honest scope note in §5.7.

Run on AMD:
    PYTHONPATH=. python scripts/anm_cluster_smoke.py
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from domains.cluster import (
    ClusterManager,
    ClusterScenarioLoader,
    build_cluster_domain_config,
)
from silr.agent import AgentConfig, ReActAgent
from silr.agent.llm.openai_client import OpenAIClient
from silr.verifier import SiLRVerifier


def run_one(client, scenario_id, gating, verify_on, max_steps=6, max_proposals=3, rep_seed=1000):
    cfg = build_cluster_domain_config(with_observer=True, gating_policy=gating)
    loader = ClusterScenarioLoader()
    scenario = loader.load(scenario_id)
    mgr = ClusterManager()
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
            temperature=0.0,
            enable_verification=verify_on,
            seed=rep_seed,
        ),
    )

    t0 = time.time()
    result = agent.run_episode(scenario_id=scenario_id)
    dt = time.time() - t0

    tally: Counter[str] = Counter()
    for s in result.steps:
        for vr in s.verification_results:
            tally[vr.verdict.value] += 1

    return {
        "scenario": scenario_id,
        "policy": gating if verify_on else "OFF",
        "recovered": bool(result.recovered),
        "total_steps": result.total_steps,
        "total_proposals": result.total_proposals,
        "total_rejections": result.total_rejections,
        "verdict_tally": dict(tally),
        "wallclock_s": round(dt, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="qwen3-14b")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--scenario", default="single_node_failure")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--max-proposals", type=int, default=3)
    ap.add_argument("--output", default="eval_cluster_smoke_v1.json")
    args = ap.parse_args()

    client = OpenAIClient(model=args.model, api_key=args.api_key, base_url=args.base_url)

    print(f"=== Cluster cross-domain smoke ===")
    print(f"  scenario: {args.scenario}")
    print(f"  model: {args.model}\n")

    policies = [
        (False, "terminal", "OFF"),
        (True, "terminal", "terminal"),
        (True, "progress", "progress"),
    ]
    episodes = []
    for verify_on, gating, label in policies:
        print(f"  running policy={label} ...")
        try:
            ep = run_one(client, args.scenario, gating, verify_on,
                         max_steps=args.max_steps, max_proposals=args.max_proposals)
            episodes.append(ep)
            print(f"    recovered={ep['recovered']} props={ep['total_proposals']} "
                  f"rej={ep['total_rejections']} verdicts={ep['verdict_tally']} "
                  f"({ep['wallclock_s']}s)")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            episodes.append({"scenario": args.scenario, "policy": label,
                             "error": f"{type(e).__name__}: {e}"})

    Path(args.output).write_text(json.dumps({"episodes": episodes}, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
