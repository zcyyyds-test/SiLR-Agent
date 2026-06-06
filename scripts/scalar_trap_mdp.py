"""Behavioural sigma-het trap: a minimal, deterministic 2-family recovery MDP where
following the cross-family severity SCALAR reward (E2) sacrifices a small-sigma
bottleneck family for a large-sigma magnitude win and hits an IRREVERSIBLE floor,
while the per-family geometric reward (D) protects it and recovers. No training,
no GPU: we roll out the greedy-argmax policy of each REAL reward function.

Physical motivation (CityLearn-like): clearing the feeder-import family with a
hard battery discharge (fix_L_hard) drives that battery's SoC down by `couple`;
past the SoC hard floor the battery is unrecoverable for the episode. The severity
scalar sees only the large net Sigma-sigma drop and takes it; the per-family
reward's drift penalty + equal family weighting refuses to trade the small family.

This isolates the mechanism the natural one-step landscape washes out (rD~=rE2):
the product-order / per-family reward yields a RECOVERING policy where the scalar
yields a FAILING one, exactly in the sigma-heterogeneous multi-family regime.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silr.verifier.types import VerificationResult, Verdict  # noqa: E402
from silr.training.reward import compute_grpo_reward  # noqa: E402
from scripts.anm_reward_landscape import severity_scalar_reward  # noqa: E402

L = ("feeder", 0)      # large-sigma family
S = ("battery", 0)     # small-sigma bottleneck family
COUPLE = 3.0           # fix_L_hard worsens S by this (shared resource)
S_FLOOR = 3.0          # if sigma_S exceeds this it is IRREVERSIBLE this episode
H = 3                  # step budget


def branches(sL, sS, s_lost):
    d = {}
    if sL > 1e-9:
        d[L] = sL
    if sS > 1e-9 or s_lost:
        d[S] = max(sS, S_FLOOR + 1.0) if s_lost else sS
    return d


def actions(sL, sS, s_lost):
    """Return list of (name, next_state) deterministic transitions."""
    out = []
    if sL > 1e-9:
        # hard feeder fix: clears L fully but discharges the battery -> worsens S
        nsS = sS + COUPLE
        out.append(("fix_L_hard", (0.0, nsS, s_lost or nsS > S_FLOOR)))
        # soft feeder fix: gentle, no coupling, partial
        out.append(("fix_L_soft", (max(0.0, sL - 2.0), sS, s_lost)))
    if sS > 1e-9 and not s_lost:
        out.append(("fix_S", (sL, 0.0, s_lost)))
    return out


def reward(reward_fn, pre_state, post_state):
    sL, sS, lost = pre_state
    nL, nS, nlost = post_state
    pre = branches(sL, sS, lost)
    post = branches(nL, nS, nlost)
    if not post:  # fully recovered
        return 1.0
    vr = VerificationResult(verdict=Verdict.SAFE_PROGRESS, action={},
                            baseline_branches=pre, post_branches=post)
    return reward_fn(vr)


def rollout(reward_fn, name):
    st = (6.0, 1.0, False)  # sigma_L=6, sigma_S=1 (sigma-het 6), S not yet lost
    traj = [st]
    for _ in range(H):
        sL, sS, lost = st
        if sL <= 1e-9 and sS <= 1e-9 and not lost:
            break  # recovered
        acts = actions(sL, sS, lost)
        if not acts:
            break
        scored = [(reward(reward_fn, st, ns), nm, ns) for nm, ns in acts]
        scored.sort(key=lambda x: -x[0])
        _, pick, ns = scored[0]
        st = ns
        traj.append((pick, ns))
    final = traj[-1] if not isinstance(traj[-1], tuple) or len(traj[-1]) == 3 else traj[-1][1]
    fL, fS, flost = (final if len(final) == 3 else final[1])
    recovered = fL <= 1e-9 and fS <= 1e-9 and not flost
    print(f"[{name}] trajectory:")
    for t in traj:
        if isinstance(t[0], str):
            nm, (a, b, c) = t[0], t[1]
            print(f"    -> {nm:11s} -> sigma_L={a:.1f} sigma_S={b:.1f} S_lost={c}")
        else:
            print(f"    start  sigma_L={t[0]:.1f} sigma_S={t[1]:.1f}")
    print(f"  RESULT: {'RECOVERED' if recovered else 'FAILED (S irreversibly lost)' if flost else 'FAILED (budget)'}\n")
    return recovered


def main():
    print(f"sigma-het trap MDP: sigma_L=6, sigma_S=1, couple={COUPLE}, S_floor={S_FLOOR}, "
          f"budget={H}. fix_L_hard clears the LARGE family but pushes the small "
          f"family past its irreversible floor.\n")
    d_ok = rollout(compute_grpo_reward, "geometric D (per-family product-order)")
    e_ok = rollout(severity_scalar_reward, "severity SCALAR E2 (cross-family Sigma-sigma)")
    print("=" * 64)
    print(f"geometric D: {'RECOVERED' if d_ok else 'FAILED'} | "
          f"severity scalar E2: {'RECOVERED' if e_ok else 'FAILED'}")
    print("The scalar reward's greedy policy takes the large-family magnitude win "
          "(fix_L_hard) and irreversibly loses the small bottleneck family; the "
          "per-family geometric reward refuses the trade and recovers.")


if __name__ == "__main__":
    main()
