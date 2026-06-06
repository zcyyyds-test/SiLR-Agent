"""Reward-hacking ladder: which verifier-as-reward can be GAMED? Reward hacking is
the central failure mode of PRMs (a policy farms reward without real progress). We
score three canonical step types with the REAL reward functions:

  progress : eliminates a violation branch (genuine Phi-descent)
  stall    : admissible (SAFE_PROGRESS) but NO progress (post == pre)   <- denial
  drift    : count-preserving magnitude WORSENING (same #branches, larger sigma)

A reward that pays positively for `stall` is farmable by a do-nothing policy; one
that is indifferent (0) to `drift` lets a policy reallocate/worsen severity for
free; only a reward that PENALISES drift defends the product order.

CPU only; uses the real compute_{binary,scalar,grpo}_reward.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silr.verifier.types import VerificationResult, Verdict  # noqa: E402
from silr.training.reward import (  # noqa: E402
    compute_binary_reward, compute_scalar_reward, compute_grpo_reward)
from scripts.anm_reward_landscape import severity_scalar_reward  # noqa: E402


def vr(pre, post):
    return VerificationResult(verdict=Verdict.SAFE_PROGRESS, action={},
                              baseline_branches=pre, post_branches=post)


def main():
    # pre-state: two violated branches in one family (severities 10, 2)
    pre = {("loading", 0): 10.0, ("loading", 1): 2.0}
    steps = {
        "progress (eliminate worst branch)": {("loading", 1): 2.0},          # cleared branch 0
        "stall (no change, admitted)":        dict(pre),                       # post == pre
        "drift (count same, worse sigma)":    {("loading", 0): 13.0, ("loading", 1): 2.0},  # worsen
    }
    print(f"{'step type':36s} | {'binary C':>9} | {'count E':>8} | {'sev-scalar E2':>13} | {'geom D':>7}")
    print("-" * 92)
    for name, post in steps.items():
        rC = compute_binary_reward(vr(pre, post))
        rE = compute_scalar_reward(vr(pre, post))
        rE2 = severity_scalar_reward(vr(pre, post))
        rD = compute_grpo_reward(vr(pre, post))
        print(f"{name:36s} | {rC:9.3f} | {rE:8.3f} | {rE2:13.3f} | {rD:7.3f}")
    print("\nReading the reward-hacking ladder:")
    print("  binary C   : pays +0.5 for STALL and DRIFT -> farmable by a do-nothing /")
    print("               denial-of-recovery policy (cannot tell progress from stalling).")
    print("  count E    : 0 for stall (good), but 0 for DRIFT too -> indifferent, lets a")
    print("               policy worsen severity for free as long as #branches is constant.")
    print("  severity E2: >0 only for real Sigma-sigma drop; NEGATIVE for drift (good) but")
    print("               still cross-family magnitude-hijackable (see scalar_trap_construct).")
    print("  geometric D: ~0 for stall AND penalises drift (drift term) -> defends the")
    print("               product order against both hacks. Anti-hacking is monotone in")
    print("               how much constraint geometry the reward preserves.")

    # Behavioural consequence (analytic, no tuning): cumulative episode reward of a
    # DENIAL-OF-RECOVERY stalling policy vs a policy that actually recovers.
    r_stall = {"binary C": compute_binary_reward(vr(pre, dict(pre))),
               "count E": compute_scalar_reward(vr(pre, dict(pre))),
               "geom D": compute_grpo_reward(vr(pre, dict(pre)))}
    r_prog = {"binary C": compute_binary_reward(vr(pre, {("loading", 1): 2.0})),
              "count E": compute_scalar_reward(vr(pre, {("loading", 1): 2.0})),
              "geom D": compute_grpo_reward(vr(pre, {("loading", 1): 2.0}))}
    H_STALL, K_RECOVER = 8, 3   # stall the whole episode vs recover in 3 steps
    print(f"\n=== behavioural: cumulative reward, STALL {H_STALL} steps (recovery=0) vs "
          f"RECOVER in {K_RECOVER} steps (recovery=1) ===")
    print(f"{'reward':10s} | {'stall-policy cum':>16s} | {'recover-policy cum':>18s} | rewards stalling more?")
    for k in ("binary C", "count E", "geom D"):
        cs = H_STALL * r_stall[k]
        cr = K_RECOVER * r_prog[k]
        print(f"{k:10s} | {cs:16.2f} | {cr:18.2f} | {'YES — denial-of-recovery wins' if cs > cr + 1e-9 else 'no'}")
    print("Under binary, stalling 8 steps (4.0) beats recovering in 3 (1.5): the reward "
          "INCENTIVISES denial-of-recovery. Geometric gives stalling ~0 -> recovery wins.")


if __name__ == "__main__":
    main()
