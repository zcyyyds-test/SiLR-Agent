"""Stronger training demo (ds panel 2026-06-07): cross-family COUPLING + irreversible
floor, so the SCALAR (count) reward trains a FAILING policy while the geometric
per-family + drift-aware reward trains a recovering one.

Task:
  * Family L: nL branches, LARGE sigma (~6-9), grow x growth/step if untouched.
  * Family S: nS branches, SMALL sigma (~0.5-1.5), IRREVERSIBLE floor: if any
    sigma_S >= floor it is permanently lost -> no recovery.
  * Actions each step (pick one):
      - hard_clear L_k : remove L_k, but ALL surviving S get sigma_S += couple
                         (clearing the big family drains the shared resource).
      - clear S_k      : remove S_k (safe).
  * Recovery = all S cleared before any S floors.

Reward per accepted action uses the REAL functions on (pre, post) branch dicts:
  * count E  (compute_scalar_reward): sees only |S| drop -> hard_clear L scores +1/n,
    BLIND to the sigma_S drift it caused -> farms hard-clear -> S floors -> FAIL.
  * geom D   (compute_grpo_reward): hard_clear L = support_elim(L) MINUS drift(S)
    (sigma_S increased on surviving S) -> net low/negative -> avoids it, clears S
    first -> RECOVER. The drift term is exactly what the scalar projection lacks.
Sweep the coupling to find the trap sweet-spot. CPU; --real uses silr functions.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics as st

_USE_REAL = False
_REAL = {}


def _load_real():
    from silr.verifier.types import VerificationResult, Verdict
    from silr.training.reward import compute_grpo_reward, compute_scalar_reward
    _REAL.update(VR=VerificationResult, V=Verdict, D=compute_grpo_reward, E=compute_scalar_reward)


def softmax(zs):
    m = max(zs); es = [math.exp(z - m) for z in zs]; s = sum(es)
    return [e / s for e in es]


def reward(arm, pre, post):
    """pre/post: dict {(family, id): sigma}. Real or inline."""
    if _USE_REAL:
        vr = _REAL["VR"](verdict=_REAL["V"].SAFE_PROGRESS, action={},
                         baseline_branches=pre, post_branches=post)
        return _REAL["D"](vr) if arm == "D" else _REAL["E"](vr)
    pre_k, post_k = set(pre), set(post)
    elim = pre_k - post_k
    surv = pre_k & post_k
    if arm == "E":  # count
        return (len(pre) - len(post)) / (len(pre) + 1e-9)
    # D per-family support_elim + severity_red - drift
    fams = {}
    for k in pre_k:
        fams.setdefault(k[0], []).append(k)
    se, sr, drifts = [], [], []
    for keys in fams.values():
        tot = sum(pre[k] for k in keys) + 1e-8
        se.append(sum(pre[k] for k in keys if k in elim) / tot)
        sr.append(sum(max(0.0, pre[k] - post[k]) for k in keys if k in surv) / tot)
        for k in keys:
            if k in surv:
                drifts.append(max(0.0, post[k] - pre[k]) / (pre[k] + 1e-8))
    return 0.6 * (sum(se) / len(se)) + 0.3 * (sum(sr) / len(sr)) - 0.3 * min(max(drifts, default=0.0), 1.0)


def rollout(theta, rng, arm, couple, floor, growth, nL, nS, H, step_cost):
    sig = {}
    for i in range(nL):
        sig[("L", i)] = rng.uniform(6.0, 9.0)
    for i in range(nS):
        sig[("S", i)] = rng.uniform(0.5, 1.5)
    alive = {k: True for k in sig}
    floored = False
    traj = []
    for _ in range(H):
        acts = []  # (action_key, kind)
        for k in sig:
            if alive[k]:
                acts.append(k)
        if not acts:
            break
        # features: [sigma, is_L]
        feats = [[sig[k], 1.0 if k[0] == "L" else 0.0] for k in acts]
        logits = [theta[0] * f[0] + theta[1] * f[1] + theta[2] for f in feats]
        probs = softmax(logits)
        r = rng.random(); c = 0.0; ci = len(acts) - 1
        for j, p in enumerate(probs):
            c += p
            if r <= c:
                ci = j; break
        choose = acts[ci]
        pre = {k: sig[k] for k in sig if alive[k]}
        post = dict(pre); del post[choose]
        if choose[0] == "L":
            # hard clear L drains shared resource: surviving S sigma += couple
            for k in list(post):
                if k[0] == "S":
                    post[k] = post[k] + couple
        sr = reward(arm, pre, post) - step_cost
        traj.append((feats, ci, sr))
        # commit
        alive[choose] = False
        if choose[0] == "L":
            for k in sig:
                if alive[k] and k[0] == "S":
                    sig[k] += couple
        # L branches grow if still alive
        for k in sig:
            if alive[k] and k[0] == "L":
                sig[k] *= growth
        # S floor check
        for k in sig:
            if alive[k] and k[0] == "S" and sig[k] >= floor:
                alive[k] = False; floored = True
    s_left = any(alive[k] for k in sig if k[0] == "S")
    recovered = (not floored) and (not s_left)
    return recovered, traj


def train(arm, couple, floor, growth, nL, nS, H, iters=120, G=32, lr=0.03,
          step_cost=0.02, seed=0):
    rng = random.Random(seed)
    theta = [0.0, 0.0, 0.0]
    for _ in range(iters):
        eps = [rollout(theta, rng, arm, couple, floor, growth, nL, nS, H, step_cost) for _ in range(G)]
        rets = [sum(s[2] for s in tj) for _, tj in eps]
        mu = st.mean(rets); sd = st.pstdev(rets) or 1e-6
        adv = [(R - mu) / sd for R in rets]
        grad = [0.0, 0.0, 0.0]
        for (rec, tj), A in zip(eps, adv):
            for feats, ci, _ in tj:
                logits = [theta[0] * f[0] + theta[1] * f[1] + theta[2] for f in feats]
                probs = softmax(logits)
                for d in range(2):
                    grad[d] += A * (feats[ci][d] - sum(p * f[d] for p, f in zip(probs, feats)))
        for d in range(2):
            theta[d] += lr * grad[d] / G
    final = sum(1 for _ in range(400) if rollout(theta, rng, arm, couple, floor, growth, nL, nS, H, step_cost)[0]) / 400
    return final, theta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--couple", type=float, default=1.5)
    p.add_argument("--floor", type=float, default=3.0)
    p.add_argument("--growth", type=float, default=1.25)
    p.add_argument("--nL", type=int, default=3)
    p.add_argument("--nS", type=int, default=2)
    p.add_argument("--H", type=int, default=6)
    p.add_argument("--real", action="store_true")
    args = p.parse_args()
    global _USE_REAL
    if args.real:
        _USE_REAL = True; _load_real(); print("[REAL silr reward functions]")
    print(f"=== coupled sigma-het training: geom D vs count E (bonus=0) ===")
    print(f"L({args.nL}x sigma~7, growth{args.growth}) + S({args.nS}x sigma~1, floor{args.floor}), "
          f"hard-clear-L drains S by couple={args.couple}, H={args.H}")
    res = {}
    for arm in ("D", "E"):
        fr = [train(arm, args.couple, args.floor, args.growth, args.nL, args.nS, args.H, seed=3000 + s)[0]
              for s in range(args.seeds)]
        res[arm] = fr
    print(f"\nfinal recovery over {args.seeds} seeds:")
    print(f"  D geometric (per-family + drift) : {st.mean(res['D']):.3f}")
    print(f"  E count (scalar projection)      : {st.mean(res['E']):.3f}")
    print(f"  D - E = {st.mean(res['D']) - st.mean(res['E']):+.3f}")


if __name__ == "__main__":
    main()
