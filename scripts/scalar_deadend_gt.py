"""Non-circular validation with a DISCRIMINATING independent GT (panel 2026-06-07:
the natural-scenario steps-to-recovery washed out because everything recovers in
2-3 steps). We construct budget-constrained B-branch recovery tasks where the
FIRST action's quality decides whether the episode recovers WITHIN BUDGET under an
oracle. The independent ground truth = "recovers within budget" (a TASK OUTCOME,
NOT the one-step penalty the geometric reward is derived from). We then ask which
reward (count E vs geometric D) better separates the recovery-leading first actions
from the dead-end-leading ones -- across a parameter SWEEP (budget H, branch count
B, severity spread), so the result is robust, not a single rigged point.

Mechanism (not circular): clearing the WORST branch first leaves an easier residual
-> recovers within a tight budget; clearing a trivial branch wastes the budget.
The geometric (severity-weighted) reward prefers clearing the worst; count is
branch-blind. If geometric's AUC/confusion vs the recovery GT beats count ROBUSTLY
across the sweep, the fidelity ladder is validated against an independent outcome.
CPU only.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silr.verifier.types import VerificationResult, Verdict  # noqa: E402
from silr.training.reward import compute_grpo_reward, compute_scalar_reward  # noqa: E402


def vr(pre, post):
    return VerificationResult(verdict=Verdict.SAFE_PROGRESS, action={},
                              baseline_branches=pre, post_branches=post)


def first_actions(sigmas):
    """clear / halve each branch -> the resulting branch dict."""
    B = len(sigmas)
    acts = []
    for k in range(B):
        post_clear = {("f", j): sigmas[j] for j in range(B) if j != k}
        post_half = {("f", j): (sigmas[j] / 2 if j == k else sigmas[j]) for j in range(B)}
        acts.append(post_clear)
        acts.append(post_half)
    return acts


def recovers_within(sigmas_post, budget, clear_cost=1.0):
    """Oracle: each remaining branch needs ceil(sigma/clear_step) clears; recovers
    if total residual clears <= remaining budget. Independent of the SiLR penalty —
    pure step-count to clear the residual support set."""
    clear_step = 5.0  # one action reduces a branch's severity by this much
    import math
    need = sum(math.ceil(s / clear_step) for s in sigmas_post.values() if s > 1e-9)
    return need <= budget - 1  # -1: the first action already spent one step


def post_to_sigmas(post):
    return {k: v for k, v in post.items()}


def auc(scores, labels):
    """AUC of `scores` separating label=1 (recovers) from label=0 (dead-end)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def run(B, budget, spread, rng):
    sigmas = sorted((rng.uniform(1.0, spread) for _ in range(B)), reverse=True)
    pre = {("f", k): sigmas[k] for k in range(B)}
    acts = first_actions(sigmas)
    rE, rD, labels = [], [], []
    for post in acts:
        rE.append(compute_scalar_reward(vr(pre, post)))
        rD.append(compute_grpo_reward(vr(pre, post)))
        labels.append(1 if recovers_within(post_to_sigmas(post), budget) else 0)
    return rE, rD, labels


def main():
    import random
    rng = random.Random(20260607)
    print("independent GT = 'first action leads to recovery WITHIN BUDGET' (task outcome, "
          "not one-step penalty). AUC of reward separating recover vs dead-end:")
    print(f"{'B':>3} {'budget':>6} {'spread':>6} | {'count E AUC':>11} | {'geom D AUC':>10} | {'frac recover':>12}")
    aucE, aucD = [], []
    for B in (3, 4, 5, 6):
        for budget in (2, 3, 4):
            for spread in (20.0, 50.0):
                eAs, dAs, fr = [], [], []
                for _ in range(40):
                    rE, rD, lab = run(B, budget, spread, rng)
                    if 0 < sum(lab) < len(lab):  # need both classes
                        a, b = auc(rE, lab), auc(rD, lab)
                        if a is not None and b is not None:
                            eAs.append(a); dAs.append(b); fr.append(sum(lab) / len(lab))
                if eAs:
                    me, md = st.mean(eAs), st.mean(dAs)
                    aucE.append(me); aucD.append(md)
                    print(f"{B:>3} {budget:>6} {spread:>6.0f} | {me:11.3f} | {md:10.3f} | {st.mean(fr):12.2f}")
    print(f"\n=== OVERALL (AUC vs independent recovery-within-budget GT) ===")
    print(f"  count E mean AUC = {st.mean(aucE):.3f}   geometric D mean AUC = {st.mean(aucD):.3f}")
    win = sum(1 for e, d in zip(aucE, aucD) if d > e + 1e-9)
    print(f"  geometric > count in {win}/{len(aucE)} parameter cells "
          f"(AUC 0.5=chance, 1.0=perfect separation)")
    print("  robust geom>count across the sweep => the ladder predicts an INDEPENDENT "
          "task outcome, not just the penalty-derived value (non-circular).")


if __name__ == "__main__":
    main()
