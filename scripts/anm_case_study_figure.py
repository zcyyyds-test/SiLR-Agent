"""Case study figure: hard_renewable_surge under progress_mag gating.

Deterministic replay of the 6-step trajectory we observed (Qwen3-14B
+ progress_mag verifier, 5 SAFE_PROGRESS + 1 PASS, recovered). For each
step record:
  step, action, verdict, post_viol_count, post_penalty, severity_score.

Renders a 2-panel time-series figure:
  - top: violation count + final penalty (twin-axis)
  - bottom: severity score (single axis), with the L3 threshold band
            (1.05 × baseline + 0.001) overlaid as a sanity check.

PNG saved as `figures/hard_progress_mag_trajectory.png`.

Run on AMD silr-anm env (also needs matplotlib):
    PYTHONPATH=. python scripts/anm_case_study_figure.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

# Ensure non-interactive backend for headless servers.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from domains.anm import ANMScenarioLoader, GymANMManager, build_anm_domain_config
from silr.verifier import SiLRVerifier


SCENARIO = "hard_renewable_surge"
ACTIONS = [
    {"tool_name": "set_generator_setpoint", "params": {"gen_id": 4, "p_mw": 40.0}},
    {"tool_name": "set_generator_setpoint", "params": {"gen_id": 4, "p_mw": 35.0}},
    {"tool_name": "set_generator_setpoint", "params": {"gen_id": 4, "p_mw": 30.0}},
    {"tool_name": "set_generator_setpoint", "params": {"gen_id": 4, "p_mw": 20.0}},
    {"tool_name": "set_generator_setpoint", "params": {"gen_id": 4, "p_mw": 10.0}},
    {"tool_name": "set_generator_setpoint", "params": {"gen_id": 2, "p_mw": 15.0}},
]


def _severity_score(check_results) -> float:
    score = 0.0
    for cr in check_results:
        for v in cr.violations:
            try:
                val = float(v.value)
                lim = float(v.limit)
            except (TypeError, ValueError):
                score += 1.0
                continue
            if not (math.isfinite(val) and math.isfinite(lim)):
                score += 1e6
                continue
            score += abs(val - lim)
    return score


def snapshot(mgr, cfg):
    checks = [c.check(mgr.system_state, mgr.base_mva) for c in cfg.checkers]
    viol_count = sum(len(cr.violations) for cr in checks if not cr.passed)
    sev_score = _severity_score(checks)
    pen = mgr.last_penalty
    return viol_count, sev_score, pen


def run() -> dict:
    cfg = build_anm_domain_config(gating_policy="progress_mag")
    loader = ANMScenarioLoader()
    scenario = loader.load(SCENARIO)
    mgr = GymANMManager(seed=42)
    loader.setup_episode(mgr, scenario)
    verifier = SiLRVerifier(mgr, domain_config=cfg)

    # Step 0: pre-control default state
    v0, s0, p0 = snapshot(mgr, cfg)
    trace = [{
        "step": 0,
        "action": "<default>",
        "verdict": "—",
        "viol": v0,
        "severity": s0,
        "penalty": p0,
    }]

    tools = cfg.create_toolset(mgr)
    for i, act in enumerate(ACTIONS, 1):
        # Use the verifier to assign a verdict (semantics-preserving — we
        # need the verdict label for the figure even though we also apply
        # the action ourselves).
        result = verifier.verify(act)
        verdict = result.verdict.value
        # Apply on the live manager mirroring the ReAct loop:
        tool = tools[act["tool_name"]]
        tool.execute(**act["params"])
        mgr.solve()
        v, s, p = snapshot(mgr, cfg)
        trace.append({
            "step": i,
            "action": f"{act['tool_name'].split('_')[1]}(id={act['params'].get('gen_id', act['params'].get('storage_id'))}, "
                       f"p={act['params']['p_mw']})",
            "verdict": verdict,
            "viol": v,
            "severity": s,
            "penalty": p,
        })

    return {"trace": trace}


def plot(trace: list[dict], out_path: Path) -> None:
    steps = [r["step"] for r in trace]
    viols = [r["viol"] for r in trace]
    pens = [r["penalty"] for r in trace]
    sevs = [r["severity"] for r in trace]
    verdicts = [r["verdict"] for r in trace]

    verdict_color = {
        "—": "#666",
        "PASS": "#2ca02c",
        "SAFE_PROGRESS": "#1f77b4",
        "FAIL": "#d62728",
        "ERROR": "#9467bd",
    }

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                          gridspec_kw={"hspace": 0.15})

    # Top: violation count (left axis) + penalty (right axis)
    ax_top.plot(steps, viols, "o-", color="#1f77b4", linewidth=2,
                markersize=8, label="Violations")
    ax_top.set_ylabel("Violation count", color="#1f77b4")
    ax_top.tick_params(axis="y", labelcolor="#1f77b4")
    ax_top.set_ylim(-0.5, max(viols) + 1)
    ax_top.grid(True, alpha=0.3)

    ax_top_r = ax_top.twinx()
    ax_top_r.plot(steps, pens, "s--", color="#d62728", linewidth=2,
                  markersize=8, alpha=0.8, label="Penalty (gym-anm native)")
    ax_top_r.set_ylabel("Penalty", color="#d62728")
    ax_top_r.tick_params(axis="y", labelcolor="#d62728")

    # Annotate each step with its verdict color
    for r in trace:
        ax_top.scatter([r["step"]], [r["viol"]], s=150,
                       c=verdict_color.get(r["verdict"], "#aaa"),
                       edgecolors="black", linewidth=1.2, zorder=10)
        ax_top.annotate(r["verdict"], (r["step"], r["viol"]),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9, fontweight="bold",
                        color=verdict_color.get(r["verdict"], "#444"))

    ax_top.set_title("ANM6-Easy hard_renewable_surge under progress_mag gating\n"
                     "Qwen3-14B + SiLR L1-L4 recovery trajectory",
                     fontsize=11)

    # Bottom: severity score + L3 threshold band
    ax_bot.plot(steps, sevs, "o-", color="#ff7f0e", linewidth=2,
                markersize=8, label="Severity score Σ|v - limit|")
    for i in range(1, len(sevs)):
        # L3 threshold = max(1.05 × baseline, baseline + 1e-3)
        baseline = sevs[i - 1]
        thr = max(1.05 * baseline, baseline + 1e-3)
        ax_bot.fill_between([steps[i - 1], steps[i]], [0, 0], [thr, thr],
                            alpha=0.08, color="#2ca02c", step="post",
                            label="L3 admission ceiling" if i == 1 else None)

    ax_bot.set_xlabel("ReAct step")
    ax_bot.set_ylabel("Σ |value - limit|")
    ax_bot.legend(loc="upper right", fontsize=9)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_xticks(steps)

    # Action labels along the bottom
    for r in trace[1:]:
        ax_bot.annotate(r["action"], (r["step"], 0),
                        textcoords="offset points", xytext=(0, -22),
                        ha="center", fontsize=7, color="#444",
                        rotation=20, rotation_mode="anchor")
    ax_bot.set_ylim(-max(sevs) * 0.1, max(sevs) * 1.15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    data = run()
    print(json.dumps(data, indent=2))
    Path("figures/hard_progress_mag_trace.json").write_text(json.dumps(data, indent=2))
    plot(data["trace"], out_dir / "hard_progress_mag_trajectory.png")


if __name__ == "__main__":
    main()
