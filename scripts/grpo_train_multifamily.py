"""Controlled GRPO training in a MULTI-FAMILY sigma-heterogeneous recovery MDP --
the regime where the per-family geometric reward is STRUCTURALLY different from a
scalar projection, so it should train a better policy.

Task (sigma-het hijack, trained version):
  * Family L: 1 branch, LARGE sigma (~20), NON-urgent (never dead-ends, clearable
    any time).
  * Family S: 1 branch, SMALL sigma (~2), URGENT (dead-ends after `grace` steps if
    not cleared -> permanent, no recovery).
  * Each step: clear one surviving branch. Recovery = family S cleared before it
    dead-ends.
  * Policy: softmax over branches, logit = theta * sigma_k (+ bias). The ONLY
    feature is severity sigma -- so a reward that rewards high-sigma clears pushes
    theta>0 (toward L), while recovery needs clearing the LOW-sigma urgent S
    (theta<0).

Three reward arms (per accepted clear):
  * E2 severity-scalar : sigma_k / sum(sigma_all)   -> strongly rewards clearing L
  * E  count           : 1 / n_alive                -> sigma-blind
  * D  per-family geom  : (sigma_k / sum(sigma_in_its_family)) / n_families
                         -> EQUAL per-family, sigma-MAGNITUDE-NEUTRAL across families
The severity-scalar's per-step signal fights recovery (chases big L); the per-family
reward is magnitude-neutral so the recovery return can teach "clear urgent S first".
CPU only, inline rewards (no silr import) so it runs locally and sweeps fast.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics as st


def softmax(zs):
    m = max(zs); es = [math.exp(z - m) for z in zs]; s = sum(es)
    return [e / s for e in es]


_USE_REAL = False
_REAL = {}


def _load_real():
    from silr.verifier.types import VerificationResult, Verdict  # noqa
    from silr.training.reward import compute_grpo_reward, compute_scalar_reward  # noqa
    _REAL.update(VR=VerificationResult, V=Verdict, D=compute_grpo_reward, E=compute_scalar_reward)


def step_reward(arm, k, fam, alive_sigma, fam_of):
    """reward for clearing branch k given the pre-clear surviving {idx: sigma}."""
    if _USE_REAL:
        pre = {(fam_of[j], j): alive_sigma[j] for j in alive_sigma}
        post = {(fam_of[j], j): alive_sigma[j] for j in alive_sigma if j != k}
        vr = _REAL["VR"](verdict=_REAL["V"].SAFE_PROGRESS, action={},
                         baseline_branches=pre, post_branches=post)
        if arm == "D":
            return _REAL["D"](vr)
        if arm == "E":
            return _REAL["E"](vr)
        return (sum(pre.values()) - sum(post.values())) / (sum(pre.values()) + 1e-9)  # E2
    tot = sum(alive_sigma.values()) + 1e-9
    if arm == "E2":
        return alive_sigma[k] / tot
    if arm == "E":
        return 1.0 / len(alive_sigma)
    # D: per-family normalized, averaged over families with equal weight
    fams = {}
    for j, s in alive_sigma.items():
        fams.setdefault(fam_of[j], 0.0)
        fams[fam_of[j]] += s
    return (alive_sigma[k] / (fams[fam[k]] + 1e-9)) / len(fams)


def rollout(theta, rng, arm, grace, bonus, step_cost, n_L=1, n_S=1, H=4, sigL=20.0):
    # build branches: L family (large, non-urgent), S family (small, urgent)
    sigma = {}
    fam = {}
    bid = 0
    for _ in range(n_L):
        sigma[bid] = rng.uniform(0.8 * sigL, 1.2 * sigL); fam[bid] = "L"; bid += 1
    for _ in range(n_S):
        sigma[bid] = rng.uniform(1.0, 3.0); fam[bid] = "S"; bid += 1
    alive = {k: True for k in sigma}
    s_age = 0
    s_dead = False
    s_cleared = False
    traj = []
    for _ in range(H):
        idx = [k for k in sigma if alive[k]]
        if not idx:
            break
        feats = [sigma[k] for k in idx]
        logits = [theta[0] * f + theta[1] for f in feats]
        probs = softmax(logits)
        r = rng.random(); c = 0.0; choose = idx[-1]
        for j, p in enumerate(probs):
            c += p
            if r <= c:
                choose = idx[j]; break
        alive_sigma = {k: sigma[k] for k in idx}
        sr = step_reward(arm, choose, fam, alive_sigma, fam) - step_cost
        alive[choose] = False
        if fam[choose] == "S":
            s_cleared = True
        traj.append((feats, idx.index(choose), sr))
        # urgency clock on S
        if not s_cleared:
            s_age += 1
            if s_age > grace:
                s_dead = True
                break
    recovered = s_cleared and not s_dead
    if recovered and bonus and traj:
        traj[-1] = (traj[-1][0], traj[-1][1], traj[-1][2] + bonus)
    return recovered, traj


def train(arm, grace, bonus, iters=80, G=24, lr=0.05, step_cost=0.02, seed=0,
          n_L=1, n_S=1, H=4, sigL=20.0):
    rng = random.Random(seed)
    theta = [0.0, 0.0]
    curve = []
    for _ in range(iters):
        eps = [rollout(theta, rng, arm, grace, bonus, step_cost, n_L, n_S, H, sigL) for _ in range(G)]
        curve.append(sum(1 for r, _ in eps if r) / G)
        rets = [sum(s[2] for s in tj) for _, tj in eps]
        mu = st.mean(rets); sd = st.pstdev(rets) or 1e-6
        adv = [(R - mu) / sd for R in rets]
        grad = [0.0, 0.0]
        for (rec, tj), A in zip(eps, adv):
            for feats, ci, _ in tj:
                logits = [theta[0] * f + theta[1] for f in feats]
                probs = softmax(logits)
                fbar0 = sum(p * f for p, f in zip(probs, feats))
                grad[0] += A * (feats[ci] - fbar0)
                grad[1] += A * (1 - 1)  # bias feature constant -> handled implicitly
        theta[0] += lr * grad[0] / G
    final = st.mean([sum(1 for _ in range(300) if rollout(theta, rng, arm, grace, bonus, step_cost, n_L, n_S, H, sigL)[0]) / 300])
    return curve, theta, final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--grace", type=int, default=1)
    p.add_argument("--bonus", type=float, default=1.0)
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--nS", type=int, default=1)
    p.add_argument("--nL", type=int, default=1)
    p.add_argument("--real", action="store_true", help="use real compute_grpo_reward/compute_scalar_reward")
    p.add_argument("--sigL", type=float, default=20.0)
    args = p.parse_args()
    global _USE_REAL
    if args.real:
        _USE_REAL = True
        _load_real()
        print("[using REAL silr reward functions]")
    print("=== multi-family sigma-het training: per-family geom (D) vs severity-scalar "
          "(E2) vs count (E) ===")
    print(f"task: L(sigma~20,non-urgent) + {args.nS}xS(sigma~2,urgent grace={args.grace}), "
          f"H={args.H}, bonus={args.bonus}, feature=sigma only")
    res = {}
    th = {}
    for arm in ("D", "E2", "E"):
        fr, tt = [], []
        for s in range(args.seeds):
            _, theta, final = train(arm, args.grace, args.bonus, seed=2000 + s,
                                     n_S=args.nS, n_L=args.nL, H=args.H, sigL=args.sigL)
            fr.append(final); tt.append(theta[0])
        res[arm] = fr; th[arm] = tt
    print(f"\nfinal recovery (clear urgent S in time) over {args.seeds} seeds:")
    for arm, name in (("D", "per-family geometric"), ("E2", "severity-scalar"), ("E", "count")):
        print(f"  {arm:3s} {name:22s}: recovery {st.mean(res[arm]):.3f}  | learned theta_sigma "
              f"{st.mean(th[arm]):+.3f} ({'prefers BIG-L' if st.mean(th[arm])>0 else 'prefers small-urgent-S'})")
    print(f"\n  D - E2 = {st.mean(res['D']) - st.mean(res['E2']):+.3f}   "
          f"D - E = {st.mean(res['D']) - st.mean(res['E']):+.3f}")


if __name__ == "__main__":
    main()
