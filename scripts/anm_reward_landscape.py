"""Reward landscaping: does the SCALAR reward (arm E) prefer a different action
than the GEOMETRIC reward (arm D) at a trap state, and does greedily following
the scalar reward fall into the scalar-projection plateau?

This is the deterministic bridge that fuses pillar-1 (the scalar GATE traps) with
pillar-2 (the scalar REWARD mis-trains): on the SAME progress_mag verifier output,
we score every legal single-setpoint action with compute_grpo_reward (D, product-
order Φ descent) and compute_scalar_reward (E, count-delta projection) and compare
their argmax. Then we greedily roll out each reward's argmax-among-admissible
policy (NO LLM, NO GPU) and read the penalty trajectory:

  - if argmax_E != argmax_D and the E-greedy rollout plateaus while the D-greedy
    rollout recovers, the scalar-projection trap is a property of the reward
    function (not just the gate) -> fusion holds.
  - if argmax_E == argmax_D (ANM single-family count/sigma collinear), the trap
    does not exist at the reward level here -> do NOT force the fusion.

Runs on TSUBAME (CPU only; imports the ANM simulator + verifier). No vLLM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.anm import (  # noqa: E402
    ANMScenarioLoader,
    GymANMManager,
    build_anm_domain_config,
    create_anm_toolset,
)
from silr.verifier import SiLRVerifier  # noqa: E402
from silr.verifier import Verdict  # noqa: E402
from silr.training.reward import (  # noqa: E402
    compute_grpo_reward,
    compute_scalar_reward,
    compute_binary_reward,
)

ADMISSIBLE = (Verdict.PASS, Verdict.SAFE_PROGRESS)


def enumerate_actions(mgr, n_grid=9):
    """All single-setpoint p actions on the 9-point device grid (q=0)."""
    base = mgr.base_mva
    actions = []
    for gen_id in mgr._gen_ids:
        dev = mgr._sim.devices[gen_id]
        p_lo = float(dev.p_min) * base
        p_hi = min(float(dev.p_max) * base, float(mgr._P_pot.get(gen_id, dev.p_max * base)))
        for p in np.linspace(p_lo, p_hi, n_grid):
            actions.append({"tool_name": "set_generator_setpoint",
                            "params": {"gen_id": int(gen_id), "p_mw": float(p), "q_mvar": 0.0}})
    for sid in mgr._des_ids:
        dev = mgr._sim.devices[sid]
        p_lo = float(dev.p_min) * base
        p_hi = float(dev.p_max) * base
        for p in np.linspace(p_lo, p_hi, n_grid):
            actions.append({"tool_name": "set_storage_setpoint",
                            "params": {"storage_id": int(sid), "p_mw": float(p), "q_mvar": 0.0}})
    return actions


def score_actions(mgr, verifier):
    """Score every legal action at the CURRENT mgr state with rD, rE."""
    rows = []
    for action in enumerate_actions(mgr):
        try:
            vr = verifier.verify(action)
        except Exception as e:  # noqa: BLE001
            continue
        pre = vr.baseline_branches or {}
        post = vr.post_branches or {}
        rows.append({
            "action": action,
            "verdict": vr.verdict.value,
            "admissible": vr.verdict in ADMISSIBLE,
            "rD": compute_grpo_reward(vr),
            "rE": compute_scalar_reward(vr),
            "rC": compute_binary_reward(vr),
            "n_pre": len(pre), "n_post": len(post),
            "max_sigma_pre": max(pre.values()) if pre else 0.0,
            "max_sigma_post": max(post.values()) if post else 0.0,
            "sum_sigma_post": sum(post.values()) if post else 0.0,
        })
    return rows


def apply_action(mgr, action):
    """Execute the chosen action on the REAL manager (mutate + solve)."""
    tools = create_anm_toolset(mgr)
    tools.get(action["tool_name"]).execute(**action["params"])
    mgr.solve()
    return mgr.last_penalty


def _spearman(xs, ys):
    """Spearman rank correlation (stdlib)."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return round(num / (dx * dy), 3) if dx > 0 and dy > 0 else None


def value_landscape(scenario_id):
    """For every admissible action: its rD, rE, and TRUE one-step value (penalty
    reduction). Tests which reward is the better PRM (ranks actions by true value)
    and how degenerate the scalar reward is (distinct levels / GRPO-advantage var)."""
    import statistics as st
    cfg = build_anm_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = ANMScenarioLoader()
    sc = loader.load(scenario_id)
    mgr = GymANMManager(seed=0)
    loader.setup_episode(mgr, sc)
    verifier = SiLRVerifier(mgr, domain_config=cfg)
    base_pen = float(mgr.last_penalty)
    rows = [r for r in score_actions(mgr, verifier) if r["admissible"]]
    rD = [r["rD"] for r in rows]
    rE = [r["rE"] for r in rows]
    # TRUE value of each action = penalty reduction after applying it (fresh mgr)
    values = []
    for r in rows:
        m2 = GymANMManager(seed=0)
        loader.setup_episode(m2, sc)
        pen = apply_action(m2, r["action"])
        values.append(round(base_pen - float(pen), 4))  # higher = better
    # PRM quality: which reward ranks actions by true value?
    rho_D = _spearman(rD, values)
    rho_E = _spearman(rE, values)
    # scalar degeneracy: distinct reward levels + GRPO advantage variance
    nD = len(set(round(x, 4) for x in rD))
    nE = len(set(round(x, 4) for x in rE))
    varD = round(st.pvariance(rD), 5) if len(rD) > 1 else 0.0
    varE = round(st.pvariance(rE), 5) if len(rE) > 1 else 0.0
    # E's top tie-group: actions E rates best — can it tell good from bad?
    emax = max(rE)
    tie = [i for i in range(len(rows)) if abs(rE[i] - emax) < 1e-9]
    tie_val = [values[i] for i in tie]
    tie_rD = [rD[i] for i in tie]
    # GRPO-signal metrics: among ALL admissible action pairs, what fraction does
    # each reward leave INDISTINGUISHABLE (equal reward) while their TRUE value
    # differs meaningfully -> that pair gives GRPO zero advantage signal to pick
    # the better action. delta = 5% of the best true value.
    n = len(rows)
    vmax = max(values) if values else 0.0
    # threshold-robustness: confusion at delta = 1% / 5% / 10% of max true value
    conf_E = {}
    conf_D = {}
    for frac in (0.01, 0.05, 0.10):
        delta = frac * vmax
        pairs = cE = cD = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(values[i] - values[j]) > delta:
                    pairs += 1
                    if abs(rE[i] - rE[j]) < 1e-9:
                        cE += 1
                    if abs(rD[i] - rD[j]) < 1e-9:
                        cD += 1
        conf_E[f"{int(frac*100)}pct"] = round(cE / pairs, 3) if pairs else 0.0
        conf_D[f"{int(frac*100)}pct"] = round(cD / pairs, 3) if pairs else 0.0
    conf_rate_E = conf_E["5pct"]
    conf_rate_D = conf_D["5pct"]
    # tie-break regret: GRPO with no signal picks a random action in E's top tie
    # (expected = mean value); the geometric reward picks the best in that group.
    import statistics as _st
    regret = round(max(tie_val) - _st.mean(tie_val), 4) if tie_val else 0.0
    return {
        "base_penalty": round(base_pen, 4), "n_admissible": len(rows),
        "prm_quality_spearman": {"rD_vs_value": rho_D, "rE_vs_value": rho_E},
        "scalar_degeneracy": {"distinct_levels_D": nD, "distinct_levels_E": nE,
                              "grpo_adv_var_D": varD, "grpo_adv_var_E": varE},
        "E_top_tiegroup": {
            "n_tied": len(tie), "E_value": round(emax, 4),
            "true_value_range": [round(min(tie_val), 4), round(max(tie_val), 4)],
            "true_value_spread": round(max(tie_val) - min(tie_val), 4),
            "rD_within_tie_range": [round(min(tie_rD), 4), round(max(tie_rD), 4)],
            "rD_resolves_tie": (max(tie_rD) - min(tie_rD)) > 1e-6,
            "best_true_in_tie": round(max(tie_val), 4),
            "best_true_overall": round(max(values), 4)},
        "grpo_signal": {
            "scalar_confusion_rate": conf_rate_E,
            "geometric_confusion_rate": conf_rate_D,
            "scalar_confusion_by_threshold": conf_E,
            "geometric_confusion_by_threshold": conf_D,
            "tie_break_regret": regret},
    }


def greedy_rollout(scenario_id, reward_key, max_steps=8):
    """Greedily apply argmax-`reward_key` among ADMISSIBLE actions; track penalty."""
    cfg = build_anm_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = ANMScenarioLoader()
    sc = loader.load(scenario_id)
    mgr = GymANMManager(seed=0)
    loader.setup_episode(mgr, sc)
    verifier = SiLRVerifier(mgr, domain_config=cfg)
    traj = [round(float(mgr.last_penalty), 4)]
    picks = []
    for _ in range(max_steps):
        rows = score_actions(mgr, verifier)
        adm = [r for r in rows if r["admissible"]]
        if not adm:
            break  # plateau: no admissible action
        best = max(adm, key=lambda r: r[reward_key])
        picks.append({"verdict": best["verdict"],
                      "max_sigma_pre": round(best["max_sigma_pre"], 3),
                      "max_sigma_post": round(best["max_sigma_post"], 3),
                      "tool": best["action"]["tool_name"]})
        pen = apply_action(mgr, best["action"])
        traj.append(round(float(pen), 4))
        if pen < 1e-6:
            break  # recovered
        if len(traj) >= 3 and abs(traj[-1] - traj[-2]) < 1e-6:
            break  # plateau: penalty frozen
    return {"reward": reward_key, "penalty_traj": traj,
            "recovered": traj[-1] < 1e-6, "picks": picks}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+",
                   default=["mined_multi_action_3_l0p25g1p0_s12"])
    p.add_argument("--out", default="figures/anm_reward_landscape.json")
    args = p.parse_args()

    report = {}
    for sid in args.scenarios:
        cfg = build_anm_domain_config(with_observer=True, gating_policy="progress_mag")
        loader = ANMScenarioLoader()
        sc = loader.load(sid)
        mgr = GymANMManager(seed=0)
        loader.setup_episode(mgr, sc)
        verifier = SiLRVerifier(mgr, domain_config=cfg)
        base_pen = float(mgr.last_penalty)
        rows = score_actions(mgr, verifier)
        adm = [r for r in rows if r["admissible"]]
        if adm:
            argD = max(adm, key=lambda r: r["rD"])
            argE = max(adm, key=lambda r: r["rE"])
            diverge = (argD["action"] != argE["action"])
        else:
            argD = argE = None
            diverge = False
        # greedy rollouts under each reward
        roll_D = greedy_rollout(sid, "rD")
        roll_E = greedy_rollout(sid, "rE")
        report[sid] = {
            "base_penalty": round(base_pen, 4),
            "n_actions_scored": len(rows), "n_admissible": len(adm),
            "argmax_diverges": diverge,
            "argmax_D": None if not argD else {
                "rD": round(argD["rD"], 4), "rE": round(argD["rE"], 4),
                "max_sigma_pre": round(argD["max_sigma_pre"], 3),
                "max_sigma_post": round(argD["max_sigma_post"], 3),
                "n_post": argD["n_post"], "tool": argD["action"]["tool_name"]},
            "argmax_E": None if not argE else {
                "rD": round(argE["rD"], 4), "rE": round(argE["rE"], 4),
                "max_sigma_pre": round(argE["max_sigma_pre"], 3),
                "max_sigma_post": round(argE["max_sigma_post"], 3),
                "n_post": argE["n_post"], "tool": argE["action"]["tool_name"]},
            "greedy_D": roll_D, "greedy_E": roll_E,
        }
        print(f"\n=== {sid} (base_penalty={base_pen:.3f}, {len(adm)}/{len(rows)} admissible) ===")
        print(f"  argmax diverges: {diverge}")
        if argD and argE:
            print(f"  argmax_D: maxσ {argD['max_sigma_pre']:.2f}->{argD['max_sigma_post']:.2f} "
                  f"n_post={argD['n_post']} | rD={argD['rD']:.3f} rE={argD['rE']:.3f}")
            print(f"  argmax_E: maxσ {argE['max_sigma_pre']:.2f}->{argE['max_sigma_post']:.2f} "
                  f"n_post={argE['n_post']} | rD={argE['rD']:.3f} rE={argE['rE']:.3f}")
        print(f"  greedy-D penalty traj: {roll_D['penalty_traj']} recovered={roll_D['recovered']}")
        print(f"  greedy-E penalty traj: {roll_E['penalty_traj']} recovered={roll_E['recovered']}")
        vl = value_landscape(sid)
        report[sid]["value_landscape"] = vl
        q = vl["prm_quality_spearman"]; d = vl["scalar_degeneracy"]; t = vl["E_top_tiegroup"]
        print(f"  PRM quality (Spearman reward-vs-true-value): rD={q['rD_vs_value']}  rE={q['rE_vs_value']}")
        print(f"  scalar degeneracy: distinct levels D={d['distinct_levels_D']} E={d['distinct_levels_E']} "
              f"| GRPO-adv var D={d['grpo_adv_var_D']} E={d['grpo_adv_var_E']}")
        print(f"  E top tie-group: {t['n_tied']} actions tied at rE={t['E_value']}; "
              f"their TRUE value spans {t['true_value_range']} (spread {t['true_value_spread']}); "
              f"D resolves tie={t['rD_resolves_tie']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
