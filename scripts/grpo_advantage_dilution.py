"""Research: HOW does the geometric process reward land in GRPO training, and why
does its reward-layer fidelity advantage NOT translate to a policy-outcome
advantage (D approx E on recovery)?

Hypothesis: the per-step process reward (arm_fn, where D's geometry beats E) is
DILUTED in the group-relative advantage by the terminal recovery_bonus (+1.0,
IDENTICAL for D and E). The advantage a policy learns from is
    A_step = (reward_step - group_mean) / group_std
and group_std is inflated by the +1.0 terminal spikes on recovered rollouts, so
the per-step process signal (D-vs-E, ~0.1-0.3) becomes a small fraction of A.

We measure, on the REAL landscape per-action rewards, how well the FIRST-step
advantage tracks the first action's TRUE value (= the training signal that teaches
"pick the good first action") for D vs E, as recovery_bonus is swept 0 -> 1.0.
If D's corr(A_first, value) >> E's at bonus=0 but both collapse as bonus grows,
the terminal outcome bonus is drowning the geometric process signal -> the fix is
to weight the process reward (small/zero terminal bonus) so PRM fidelity drives
the update. Sensitivity-swept over the recovery<-value coupling so it is not rigged.
CPU only; reuses anm_reward_landscape primitives.
"""
from __future__ import annotations

import argparse
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.anm import ANMScenarioLoader, GymANMManager, build_anm_domain_config  # noqa: E402
from silr.verifier import SiLRVerifier  # noqa: E402
from scripts.anm_reward_landscape import score_actions  # noqa: E402

STEP_COST = 0.05


def _spear(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    import math
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def sim_scenario(actions, bonus, slope, rng, G=6, n_other_steps=2):
    """actions: list of (rD, rE, value). One GRPO group = G rollouts. Each rollout
    picks a random admissible FIRST action; recovers with prob coupled to its value;
    rollout = first step (process reward) + n_other_steps filler steps (small reward)
    + terminal bonus if recovered. Returns per-arm (A_first list, value list)."""
    vals = [a[2] for a in actions]
    vmean = st.mean(vals)
    vsd = st.pstdev(vals) or 1.0
    out = {}
    for arm_idx, key in ((0, "D"), (1, "E")):
        rewards = []   # (rollout_id, step_kind, reward, first_value)
        for g in range(G):
            a = actions[rng.randrange(len(actions))]
            p_rec = max(0.05, min(0.95, 0.65 + slope * (a[2] - vmean) / vsd))
            recovered = rng.random() < p_rec
            r_first = a[arm_idx] - STEP_COST
            rewards.append((g, "first", r_first, a[2]))
            for _ in range(n_other_steps):
                rewards.append((g, "filler", rng.gauss(0.15, 0.05) - STEP_COST, a[2]))
            if recovered:
                # bonus lands on the terminal (filler) step of the rollout
                gi = len(rewards) - 1
                rewards[gi] = (g, "term", rewards[gi][2] + bonus, a[2])
        allr = [r[2] for r in rewards]
        m = st.mean(allr); sd = st.pstdev(allr) or 1e-6
        A_first = [(r[2] - m) / sd for r in rewards if r[1] == "first"]
        v_first = [r[3] for r in rewards if r[1] == "first"]
        absA_first = st.mean([abs(x) for x in A_first]) if A_first else 0.0
        absA_term = st.mean([abs((r[2] - m) / sd) for r in rewards if r[1] == "term"]) or 0.0
        out[key] = (A_first, v_first, absA_first, absA_term)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", required=True)
    args = p.parse_args()
    cfg = build_anm_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = ANMScenarioLoader()
    scen_actions = []
    for sid in args.scenarios:
        sc = loader.load(sid)
        mgr = GymANMManager(seed=0)
        loader.setup_episode(mgr, sc)
        verifier = SiLRVerifier(mgr, domain_config=cfg)
        base = float(mgr.last_penalty)
        rows = [r for r in score_actions(mgr, verifier) if r["admissible"]]
        if len(rows) < 4:
            continue
        acts = []
        for r in rows:
            sh = mgr.create_shadow_copy()
            from domains.anm import create_anm_toolset
            create_anm_toolset(sh).get(r["action"]["tool_name"]).execute(**r["action"]["params"])
            sh.solve()
            acts.append((r["rD"], r["rE"], base - float(sh.last_penalty)))
        scen_actions.append(acts)
    print(f"=== GRPO advantage dilution: corr(first-step advantage, true value) "
          f"over {len(scen_actions)} scenarios, mean over 200 group draws ===")
    print(f"{'recovery_bonus':>14} | {'D corr(A,value)':>15} | {'E corr(A,value)':>15} | {'D-E gap':>8}")
    for bonus in (0.0, 0.1, 0.25, 0.5, 1.0):
        for slope in (0.5,):  # primary coupling; sweep below
            dC, eC = [], []
            rng = random.Random(20260607)
            for acts in scen_actions:
                for _ in range(200):
                    o = sim_scenario(acts, bonus, slope, rng)
                    dC.append(_spear(o["D"][0], o["D"][1]))
                    eC.append(_spear(o["E"][0], o["E"][1]))
            dm, em = st.mean(dC), st.mean(eC)
            print(f"{bonus:>14} | {dm:>15.3f} | {em:>15.3f} | {dm - em:>8.3f}")
    print("\n-- sensitivity: D-E gap at bonus=0 vs bonus=1.0 across recovery<-value slope --")
    for slope in (0.0, 0.3, 0.6, 1.0):
        rows = []
        for bonus in (0.0, 1.0):
            dC, eC = [], []
            rng = random.Random(20260607)
            for acts in scen_actions:
                for _ in range(120):
                    o = sim_scenario(acts, bonus, slope, rng)
                    dC.append(_spear(o["D"][0], o["D"][1]))
                    eC.append(_spear(o["E"][0], o["E"][1]))
            rows.append(st.mean(dC) - st.mean(eC))
        print(f"  slope={slope}: D-E gap  bonus0={rows[0]:+.3f}  bonus1={rows[1]:+.3f}  "
              f"(dilution = {rows[0]-rows[1]:+.3f})")


if __name__ == "__main__":
    main()
