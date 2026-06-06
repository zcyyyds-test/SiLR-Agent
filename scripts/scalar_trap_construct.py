"""Constructed sigma-heterogeneity demonstration: the per-family product-order
reward (arm D) vs the severity-weighted SCALAR (cross-family Sigma sigma, the
control E2). This isolates the one comparison the natural domains cannot
(rD ~= rE2 in single-family ANM and on one-step CityLearn value): when two
physically-incomparable families have very different sigma magnitudes, the
cross-family scalar is HIJACKED by the large-sigma family and deprioritises
clearing the small-sigma family -- even when the small family is the binding
constraint. The per-family reward weights families equally.

We score, with the REAL reward functions, the choice between:
  Z = eliminate the LARGE-sigma family (big Sigma-sigma drop)
  Y = eliminate the SMALL-sigma family (the bottleneck, small Sigma-sigma drop)
across a sweep of the sigma-heterogeneity ratio rho = sigma_large/sigma_small.
A reward that prefers Y over Z when the small family is the bottleneck makes the
RIGHT call; one that prefers Z (chasing magnitude) falls into the scalar trap.

CPU only; no vLLM. Run on TSUBAME (imports silr.training.reward).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silr.verifier.types import VerificationResult, Verdict  # noqa: E402
from silr.training.reward import compute_grpo_reward, compute_scalar_reward  # noqa: E402
from scripts.anm_reward_landscape import severity_scalar_reward  # noqa: E402


def vr(pre, post):
    return VerificationResult(verdict=Verdict.SAFE_PROGRESS, action={},
                              baseline_branches=pre, post_branches=post)


def main():
    sigma_small = 1.0
    print("sigma-het sweep: family L (large, key ('feeder',0)) vs family S (small, "
          "key ('battery',0), sigma=1.0). Z=clear L, Y=clear S (the bottleneck).")
    print(f"{'rho=sL/sS':>9} | {'D(Y)':>6} {'D(Z)':>6} {'D pick':>7} | "
          f"{'E2(Y)':>6} {'E2(Z)':>6} {'E2 pick':>8} | {'count(Y)':>8} {'count(Z)':>8}")
    rows = []
    for rho in (1, 2, 5, 10, 30, 100, 286):
        sL = sigma_small * rho
        pre = {("feeder", 0): sL, ("battery", 0): sigma_small}
        postY = {("feeder", 0): sL}                 # cleared small family S
        postZ = {("battery", 0): sigma_small}        # cleared large family L
        dY, dZ = compute_grpo_reward(vr(pre, postY)), compute_grpo_reward(vr(pre, postZ))
        eY, eZ = severity_scalar_reward(vr(pre, postY)), severity_scalar_reward(vr(pre, postZ))
        cY, cZ = compute_scalar_reward(vr(pre, postY)), compute_scalar_reward(vr(pre, postZ))
        EPS = 1e-4  # ignore 1e-8 per-family-normalizer numerical noise
        dpick = "Y=S" if dY > dZ + EPS else ("Z=L" if dZ > dY + EPS else "tie(bal)")
        epick = "Y=S" if eY > eZ + EPS else ("Z=L" if eZ > eY + EPS else "tie")
        ratio = eZ / eY if eY > 1e-9 else float("inf")  # E2 bias toward the large family
        print(f"{rho:9d} | {dY:6.3f} {dZ:6.3f} {dpick:>8} | "
              f"{eY:6.3f} {eZ:6.3f} {epick:>8} (E2 bias L/S={ratio:6.1f}x) | {cY:6.3f} {cZ:6.3f}")
        rows.append((rho, dpick, epick))
    print("\nReading: with the SMALL family S as the binding constraint, the correct")
    print("call is Y (clear S). The per-family geometric reward D is family-balanced")
    print("(never prefers L just because it is larger); the cross-family severity")
    print("scalar E2 increasingly prefers Z (clear the large family) as sigma-het")
    print("grows -> the scalar-projection trap. Count is sigma-blind (Y==Z).")
    trap = [r for r in rows if r[2] == "Z=L"]
    print(f"\nE2 falls into the trap (prefers clearing the LARGE family) at rho in "
          f"{[r[0] for r in trap]}; D never does (D picks: {sorted(set(r[1] for r in rows))}).")


if __name__ == "__main__":
    main()
