"""Real-LLM smoke on the ANM domain — first signal that Qwen3-14B (or any
OpenAI-compatible model) can drive the ReAct + SiLR-verify loop on a stressed
ANM6-Easy snapshot.

What it does:
  1. Health-check the LLM endpoint (one /chat/completions round trip).
  2. Setup the ``medium_seed42_default`` scenario (branch 2-4 overload at default).
  3. Run a bounded ReAct episode with the real LLM and full SiLR verification.
  4. Print every step's tool call, verdict, applied action.
  5. Tally token usage; compare final penalty to the MPC baseline (penalty 0).

Why this matters (paper context):
  - confirms the wiring (tool schemas / device-id maps / observer JSON) the
    MockClient smoke already exercised holds up with a non-deterministic LLM;
  - first data point on whether a zero-shot 14B-class model can produce
    grid-feasible set-points without SFT/GRPO — that floor decides whether
    we need training before the paper eval, or can stick to inference-only.

Defaults assume a local vLLM serving ``Qwen3-14B`` on port 8001, but ``--base-url``
+ ``--model`` make this work against any OpenAI-compatible endpoint.

Run on AMD (vllm-serve env or any env with ``openai`` installed):

    PYTHONPATH=. python scripts/anm_real_llm_smoke.py
    PYTHONPATH=. python scripts/anm_real_llm_smoke.py \
        --base-url http://localhost:8001/v1 --model /d/zcy/models/Qwen3-14B
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict

from domains.anm import (
    ANMScenarioLoader,
    GymANMManager,
    build_anm_domain_config,
)
from silr.agent import AgentConfig, ReActAgent
from silr.agent.llm.openai_client import OpenAIClient
from silr.verifier import SiLRVerifier


def _health_check(client: OpenAIClient, timeout_s: float = 30.0) -> None:
    """One-shot ping the endpoint; raise with a useful message if it fails."""
    t0 = time.time()
    try:
        resp = client.chat(
            messages=[{"role": "user", "content": 'Reply with the single word "ok".'}],
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"LLM health-check failed after {time.time()-t0:.1f}s: {type(e).__name__}: {e}\n"
            "Is vLLM serving on the expected --base-url? Try: "
            "curl <base-url>/models"
        ) from e
    dt = time.time() - t0
    body = (resp.content or "").strip()
    print(f"  health-check: {dt:.2f}s, response={body!r}, "
          f"usage={resp.usage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8001/v1",
                        help="OpenAI-compatible base URL (default: local vLLM on 8001)")
    parser.add_argument("--model", default="/d/zcy/models/Qwen3-14B",
                        help="Model name / path as registered with the server")
    parser.add_argument("--api-key", default="EMPTY",
                        help="API key (vLLM ignores it; APIs need a real one)")
    parser.add_argument("--scenario", default="medium_seed42_default",
                        help="ANM scenario id (see domains/anm/scenarios.py)")
    parser.add_argument("--max-steps", type=int, default=4,
                        help="ReAct max steps (small for a smoke run)")
    parser.add_argument("--max-proposals", type=int, default=3,
                        help="Max LLM proposals per step before giving up")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-verify", action="store_true",
                        help="Disable SiLR verification (NoVerify ablation mode)")
    parser.add_argument("--gating-policy", choices=("terminal", "progress", "progress_mag"),
                        default="progress",
                        help="Verifier gating policy. 'terminal' requires zero "
                             "violations after each admitted action (the "
                             "original semantics); 'progress' (default for ANM) "
                             "also admits recoverability-preserving steps.")
    args = parser.parse_args()

    print(f"=== Real-LLM smoke on ANM ===")
    print(f"  endpoint    : {args.base_url}")
    print(f"  model       : {args.model}")
    print(f"  scenario    : {args.scenario}")
    print(f"  max_steps   : {args.max_steps}")
    if args.no_verify:
        print(f"  verify mode : OFF (ablation)")
    else:
        print(f"  verify mode : ON  ({args.gating_policy} gating)")
    print()

    client = OpenAIClient(model=args.model, api_key=args.api_key, base_url=args.base_url)
    _health_check(client)
    print()

    cfg = build_anm_domain_config(
        with_observer=True,
        gating_policy=args.gating_policy,
    )
    loader = ANMScenarioLoader()
    scenario = loader.load(args.scenario)
    mgr = GymANMManager(seed=scenario.source_seed if scenario.source_seed is not None else 42)
    converged = loader.setup_episode(mgr, scenario)
    default_penalty = mgr.last_penalty
    print(f"scenario default state: converged={converged}, "
          f"penalty={default_penalty:.3f}")

    verifier = SiLRVerifier(mgr, domain_config=cfg)
    agent = ReActAgent(
        manager=mgr,
        verifier=verifier,
        llm_client=client,
        domain_config=cfg,
        config=AgentConfig(
            max_steps=args.max_steps,
            max_proposals_per_step=args.max_proposals,
            temperature=args.temperature,
            enable_verification=not args.no_verify,
        ),
    )

    t_episode = time.time()
    result = agent.run_episode(scenario_id=args.scenario)
    episode_secs = time.time() - t_episode
    final_penalty = mgr.last_penalty

    print("\n--- per-step trace ---")
    for s in result.steps:
        thought = (s.thought or "").strip().replace("\n", " ")
        if len(thought) > 110:
            thought = thought[:107] + "..."
        verdicts = [vr.verdict.value for vr in s.verification_results]
        proposed = s.proposed_actions[:2] if s.proposed_actions else []
        print(f"step {s.step_number} [{s.outcome.value}]")
        if thought:
            print(f"  thought   : {thought}")
        if proposed:
            for i, p in enumerate(proposed):
                print(f"  proposed{i}: {p}")
        if verdicts:
            print(f"  verdicts  : {verdicts}")
        if s.applied_action:
            print(f"  applied   : {s.applied_action}")
        if s.error:
            print(f"  error     : {s.error}")

    print("\n--- summary ---")
    print(f"  total steps      : {result.total_steps}")
    print(f"  total proposals  : {result.total_proposals}")
    print(f"  total rejections : {result.total_rejections}")
    print(f"  recovered        : {result.recovered}")
    if result.final_observation:
        print(f"  final is_stable  : {result.final_observation.is_stable}")
        print(f"  final violations : {len(result.final_observation.violations)}")
    print(f"  default penalty  : {default_penalty:.3f}")
    print(f"  final penalty    : {final_penalty:.3f} (MPC baseline: 0.000)")
    print(f"  episode wallclock: {episode_secs:.1f}s")

    # Structural assertion: the LLM-driven loop actually ran and the verifier
    # gate was reachable (we can't assert recovered=True for zero-shot 14B —
    # that's exactly what this smoke is measuring).
    assert result.total_steps >= 1, "ReAct loop produced no steps"
    # A scenario whose default state is already stable recovers at step 1
    # without ever calling the LLM — that's correct behaviour, not a failure.
    if default_penalty > 0:
        assert result.total_proposals >= 1, "LLM was never called for an action"

    # Verbose: per-step verification breakdown (paper data)
    print("\n--- verifier verdict tally ---")
    from collections import Counter
    tally = Counter()
    for s in result.steps:
        for vr in s.verification_results:
            tally[vr.verdict.value] += 1
    for k, v in sorted(tally.items()):
        print(f"  {k:<8} {v}")

    print("\nREAL-LLM SMOKE OK")


if __name__ == "__main__":
    main()
