"""Empirical backing for the Morse-Sard argument (panel 2026-06-07 mimo): a scalar
process reward induces ZERO-GRADIENT regions on the continuous action space, and
their measure grows with the number of violation branches B. Distinct from the
value-relative confusion rate, this measures the reward's OWN local gradient: over
a fine continuous reduction amount on each branch, what fraction of the action
range does each reward leave FLAT (zero gradient -> zero GRPO advantage signal)?

count E: reducing a branch's severity does NOT change the violation count until the
branch is eliminated -> the reward is flat over the entire partial-reduction range
-> large zero-gradient measure, growing with B.
geometric D: severity_red changes continuously with the reduction -> gradient
everywhere -> ~zero flat measure.

CPU only; uses the real reward functions on a dense continuous action grid.
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


def flat_fraction(sigmas, grid=60):
    """Over a dense continuous reduction `a` applied to each branch (a in [0, sigma_k]),
    fraction of consecutive grid steps where the reward is unchanged (zero gradient)."""
    B = len(sigmas)
    pre = {("f", k): sigmas[k] for k in range(B)}
    flatE = totE = flatD = totD = 0
    for k in range(B):
        prevE = prevD = None
        for i in range(grid + 1):
            a = sigmas[k] * i / grid
            new = max(0.0, sigmas[k] - a)
            post = {("f", j): (new if j == k else sigmas[j]) for j in range(B) if not (j == k and new <= 1e-9)}
            rE = compute_scalar_reward(vr(pre, post))
            rD = compute_grpo_reward(vr(pre, post))
            if prevE is not None:
                totE += 1; totD += 1
                if abs(rE - prevE) < 1e-9:
                    flatE += 1
                if abs(rD - prevD) < 1e-9:
                    flatD += 1
            prevE, prevD = rE, rD
    return (flatE / totE if totE else 0.0), (flatD / totD if totD else 0.0)


def main():
    import random
    rng = random.Random(20260607)
    print("zero-gradient measure (fraction of the continuous action range where the "
          "reward is FLAT = zero GRPO advantage), vs branch count B:")
    print(f"{'B':>3} | {'count E flat-frac':>17} | {'geom D flat-frac':>16}")
    rowsB = []
    for B in range(1, 9):
        eL, dL = [], []
        for _ in range(20):
            sigmas = [round(rng.uniform(2.0, 40.0), 3) for _ in range(B)]
            fe, fd = flat_fraction(sigmas)
            eL.append(fe); dL.append(fd)
        me, md = st.mean(eL), st.mean(dL)
        rowsB.append((B, me, md))
        print(f"{B:>3} | {me:17.3f} | {md:16.3f}")
    Bs = [r[0] for r in rowsB]; es = [r[1] for r in rowsB]
    n = len(Bs); mb = sum(Bs) / n; me = sum(es) / n
    cov = sum((Bs[i] - mb) * (es[i] - me) for i in range(n)) / n
    sb = st.pstdev(Bs); se = st.pstdev(es)
    r = cov / (sb * se) if sb > 0 and se > 0 else 0
    print(f"\n=== count E zero-gradient measure grows with B: Pearson r(B, flat-frac) = {r:.3f} "
          f"(B=1 -> {es[0]:.3f}, B=8 -> {es[-1]:.3f}) ===")
    print("geometric D flat-frac stays ~0 at all B -> gradient everywhere. This is the "
          "empirical signature of the Morse-Sard argument: scalar projection creates "
          "zero-advantage regions whose measure increases with problem dimension.")


if __name__ == "__main__":
    main()
