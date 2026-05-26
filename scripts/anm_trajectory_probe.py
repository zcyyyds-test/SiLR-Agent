"""Deterministic replay of a control-action sequence on an ANM scenario, printing
per-step violation count / penalty / per-checker breakdown.

Purpose: answer the panel caveat (ds-agent / agy-agent) — does the no-verify
recovery trajectory improve *monotonically* (each action reduces or holds the
violation set), or does some intermediate step transiently *worsen* before
recovering? If any step worsens, a "progress gating" verifier (only apply
non-worsening actions) would still block hard recovery — so this probe decides
whether the dual-criterion redesign actually solves the deadlock.

No LLM needed — we replay the exact actions the no-verify Qwen3-14B run applied.

Run from repo root (AMD silr-anm env):
    PYTHONPATH=. python scripts/anm_trajectory_probe.py
"""

from __future__ import annotations

from domains.anm import ANMScenarioLoader, GymANMManager, build_anm_domain_config


def violation_snapshot(mgr, cfg):
    """Return (total_violations, per_checker_dict, penalty)."""
    breakdown = {}
    total = 0
    for checker in cfg.checkers:
        cr = checker.check(mgr.system_state, mgr.base_mva)
        n = 0 if cr.passed else len(cr.violations)
        breakdown[checker.name] = n
        total += n
    return total, breakdown, mgr.last_penalty


def replay(scenario_id, actions, cfg):
    """Apply `actions` (list of {tool_name, params}) sequentially, printing the
    violation snapshot after each, mirroring the no-verify ReAct loop:
    tool mutates setpoints → manager.solve() refreshes the power flow."""
    mgr = GymANMManager(seed=42)
    loader = ANMScenarioLoader()
    scenario = loader.load(scenario_id)
    loader.setup_episode(mgr, scenario)
    tools = cfg.create_toolset(mgr)

    t0, b0, p0 = violation_snapshot(mgr, cfg)
    print(f"=== {scenario_id} ===")
    print(f"step 0 (default): viol={t0} {b0} penalty={p0:.3f}")

    prev_total = t0
    monotonic = True
    for i, action in enumerate(actions, 1):
        tool = tools[action["tool_name"]]
        res = tool.execute(**action["params"])
        mgr.solve()  # refresh power flow under frozen conditions (loop does this)
        total, breakdown, penalty = violation_snapshot(mgr, cfg)
        worsened = total > prev_total
        if worsened:
            monotonic = False
        flag = "  <-- WORSENED" if worsened else ""
        print(f"step {i}: {action['tool_name']}({action['params']}) "
              f"-> viol={total} {breakdown} penalty={penalty:.3f}"
              f" status={res.get('status')}{flag}")
        prev_total = total

    print(f"\nmonotonic non-increasing violation count: {monotonic}")
    print(f"final: viol={prev_total}")
    return monotonic


def main():
    cfg = build_anm_domain_config()

    # The exact action sequence the no-verify Qwen3-14B run applied on
    # hard_renewable_surge (from decisions.md 2026-05-22 ablation trace).
    hard_actions = [
        {"tool_name": "set_generator_setpoint", "params": {"gen_id": 4, "p_mw": 30.0}},
        {"tool_name": "set_generator_setpoint", "params": {"gen_id": 4, "p_mw": 20.0}},
        {"tool_name": "set_generator_setpoint", "params": {"gen_id": 4, "p_mw": 10.0}},
        {"tool_name": "set_generator_setpoint", "params": {"gen_id": 4, "p_mw": 0.0}},
        {"tool_name": "set_generator_setpoint", "params": {"gen_id": 2, "p_mw": 14.0}},
    ]
    mono = replay("hard_renewable_surge", hard_actions, cfg)

    print("\n=== INTERPRETATION ===")
    if mono:
        print("Trajectory is monotonic — a progress-gating verifier (apply iff "
              "not-worse) would ADMIT every step → dual-criterion redesign solves "
              "the hard deadlock.")
    else:
        print("Trajectory is NON-monotonic — at least one step transiently "
              "worsened. A strict progress-gating verifier would BLOCK that step, "
              "so dual-criterion alone may NOT solve hard; need recoverability-set "
              "gating (lookahead) or compound-action tool.")


if __name__ == "__main__":
    main()
