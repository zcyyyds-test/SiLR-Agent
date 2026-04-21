"""Eval SiLR ReActAgent on cluster_v2023 held-out scenarios.

Uses LocalQwenClient (Qwen3 base + optional LoRA adapter) — same pattern
as scripts/eval_sft.py. Model is loaded ONCE and reused across all
scenarios (loading 14B × N scenarios would burn hours).

Outputs JSON with per-scenario recovery + fragmentation + reject rate.

IMPORTANT: All torch / peft / transformers imports are LAZY (inside
main() / _load_client()). This keeps the module import-cheap so unit
tests can exercise helpers (_scenarios_in, _aggregate) without needing
a GPU-enabled Python environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.cluster_v2023.checkers import FragmentationChecker
from domains.cluster_v2023.config import build_cluster_v2023_domain_config
from domains.cluster_v2023.scenarios.loader import ScenarioLoader

logger = logging.getLogger(__name__)


def _scenarios_in(dir_path: Path) -> list[Path]:
    return ScenarioLoader.list_scenarios(dir_path)


def _aggregate(episodes: list[dict]) -> dict:
    n = max(len(episodes), 1)
    return {
        "n_episodes": len(episodes),
        "recovery_rate": sum(1 for e in episodes if e["recovered"]) / n,
        "mean_F_normalized": sum(e["fragmentation_F_normalized"]
                                 for e in episodes) / n,
        "reject_rate": (
            sum(e["total_rejections"] for e in episodes)
            / max(sum(e["total_proposals"] for e in episodes), 1)
        ),
    }


def _run_one(scenario_path: Path, *, llm_client, max_steps: int = 15) -> dict:
    # Lazy imports — only needed when actually running (GPU server).
    from silr.agent import ReActAgent
    from silr.agent.config import AgentConfig
    from silr.verifier import SiLRVerifier

    scenario = ScenarioLoader.load(scenario_path)
    mgr = ScenarioLoader.build_manager(scenario)
    mgr.solve()
    cfg = build_cluster_v2023_domain_config(
        f_threshold=scenario.get("f_threshold", 10.0))
    verifier = SiLRVerifier(mgr, domain_config=cfg)
    agent = ReActAgent(
        manager=mgr, verifier=verifier,
        llm_client=llm_client,
        domain_config=cfg,
        config=AgentConfig(max_steps=max_steps, temperature=0.0),
    )
    result = agent.run_episode(scenario_id=scenario["scenario_id"])

    frag = FragmentationChecker(
        f_threshold=scenario.get("f_threshold", 10.0)).check(
        mgr.system_state, 1.0)
    f_normalized = (frag.summary["F"]
                    / max(scenario.get("f_bestfit_baseline", 1.0), 1e-6))

    return {
        "scenario_id": scenario["scenario_id"],
        "fault_type": scenario["fault_type"],
        "recovered": result.recovered,
        "total_steps": result.total_steps,
        "total_rejections": result.total_rejections,
        "total_proposals": result.total_proposals,
        "failsafe_triggered": result.failsafe_triggered,
        "fragmentation_F_final": frag.summary["F"],
        "fragmentation_F_normalized": round(f_normalized, 3),
    }


def _load_client(model_path: str, adapter_path: str | None,
                 max_new_tokens: int):
    """Lazy-import LocalQwenClient — brings in torch / peft / transformers."""
    from scripts.eval_sft import LocalQwenClient
    return LocalQwenClient(
        model_path=model_path,
        adapter_path=adapter_path or "",
        max_new_tokens=max_new_tokens,
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", required=True)
    p.add_argument("--model", required=True,
                   help="Path to base Qwen model, e.g. D:/zcy/models/Qwen3-14B")
    p.add_argument("--adapter", default=None,
                   help="Path to LoRA adapter directory. Omit for zero-shot.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Loading model once: %s (adapter=%s)",
                args.model, args.adapter)
    llm_client = _load_client(args.model, args.adapter, args.max_new_tokens)

    episodes: list[dict] = []
    for path in _scenarios_in(Path(args.scenario_dir)):
        for rep in range(args.repeats):
            r = _run_one(path, llm_client=llm_client)
            r["repeat"] = rep
            episodes.append(r)
            logger.info("%s rep=%d recovered=%s F_norm=%.2f",
                        r["scenario_id"], rep, r["recovered"],
                        r["fragmentation_F_normalized"])

    agg = _aggregate(episodes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"aggregate": agg, "episodes": episodes}, f, indent=2)
    logger.info("summary: %s", agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
