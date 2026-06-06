"""Constructed complexity-scaling test: does the count-vs-geometric PRM fidelity
gap grow with the number of simultaneous violation branches B? Natural ANM trap
scenarios only span B=3-4, so we construct violation states with B = 2..10
(severity-heterogeneous branches) and score the REAL reward functions over a
canonical action set (clear each branch / halve each branch). The TRUE one-step
value of an action is its Sigma-sigma reduction. We measure, per B, the confusion
rate (action-pairs with different true value left at EQUAL reward = zero GRPO
signal) for count E vs geometric D.

Hypothesis (ds, 2026-06-07): count collapses more actions as B grows ->
gap(B) increases -> geometric is asymptotically necessary, not just 'better'.
This isolates the scaling the natural data cannot show. CPU only, no vLLM.
"""
from __future__ import annotations

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


def confusion(rvals, values, frac=0.05):
    n = len(rvals)
    vmax = max(values) if values else 0.0
    delta = frac * vmax
    pairs = conf = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(values[i] - values[j]) > delta:
                pairs += 1
                if abs(rvals[i] - rvals[j]) < 1e-9:
                    conf += 1
    return conf / pairs if pairs else 0.0


def run_B(B, sigmas):
    """One family, B branches with the given severities. Actions: clear each
    branch, halve each branch (2B actions)."""
    pre = {("loading", k): sigmas[k] for k in range(B)}
    actions = []
    for k in range(B):
        post_clear = {("loading", j): sigmas[j] for j in range(B) if j != k}
        post_half = dict(pre); post_half[("loading", k)] = sigmas[k] / 2.0
        actions.append(post_clear)
        actions.append(post_half)
    rE, rD, vals = [], [], []
    sum_pre = sum(sigmas)
    for post in actions:
        rE.append(compute_scalar_reward(vr(pre, post)))
        rD.append(compute_grpo_reward(vr(pre, post)))
        vals.append(round(sum_pre - sum(post.values()), 6))  # true one-step value
    return confusion(rE, vals), confusion(rD, vals)


def main():
    import random
    rng = random.Random(20260607)
    print(f"{'B':>3} | {'count E confusion':>17} | {'geom D confusion':>16} | {'gap (E-D)':>9}")
    print("-" * 56)
    gaps = []
    for B in range(2, 11):
        # severity-heterogeneous branches: sigma_k spread over ~2 orders (like real
        # multi-family sigma-het); average over a few random draws to be robust.
        cE_l, cD_l = [], []
        for _ in range(20):
            sigmas = [round(rng.uniform(1.0, 50.0), 3) for _ in range(B)]
            cE, cD = run_B(B, sigmas)
            cE_l.append(cE); cD_l.append(cD)
        cE = sum(cE_l) / len(cE_l); cD = sum(cD_l) / len(cD_l)
        gaps.append((B, cE - cD))
        print(f"{B:>3} | {cE:17.3f} | {cD:16.3f} | {cE - cD:9.3f}")
    # trend
    Bs = [g[0] for g in gaps]; gs = [g[1] for g in gaps]
    n = len(Bs); mb = sum(Bs) / n; mg = sum(gs) / n
    cov = sum((Bs[i] - mb) * (gs[i] - mg) for i in range(n)) / n
    import statistics as st
    sb = st.pstdev(Bs); sg = st.pstdev(gs)
    r = cov / (sb * sg) if sb > 0 and sg > 0 else 0
    slope = cov / (sb * sb) if sb > 0 else 0
    print(f"\ntrend: Pearson r(B, gap) = {r:.3f}, slope = {slope:.4f}/branch "
          f"(gap B=2 -> {gs[0]:.3f}, B=10 -> {gs[-1]:.3f})")
    print("monotone increasing gap => geometric is asymptotically necessary as the "
          "number of simultaneous violations grows; count's confusion grows because it "
          "collapses more distinct actions onto the same count-delta level.")


if __name__ == "__main__":
    main()
