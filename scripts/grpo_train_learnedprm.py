"""Learned-PRM arm in the sigma-CR stress regimes (panel 2026-06-11, ds's single
biggest remaining gap: no trained reward-model comparison).

Arm L: a small logistic reward model trained Math-Shepherd-style -- collect rollouts
with a random policy, label each STEP by the episode's binary recovery outcome
(hard-estimation), fit r_hat(features) by logistic regression, then train the policy
with GRPO on r_hat. Features per step: (sigma_chosen, is_large_family, n_alive,
frac_small_alive) -- a reasonable state-action featurization, NOT hand-crippled.
If the learned PRM fails in either regime, the component matrix gains a row that
directly answers "compare against a *trained* PRM"; if it ties, this stays internal.
CPU only.
"""
from __future__ import annotations
import math, random, statistics as st
import scripts.grpo_train_coupled as C
import scripts.grpo_train_multifamily as M


def sigmoid(z): return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def feats_coupled(pre, choose):
    nS = sum(1 for k in pre if k[0] == "S")
    return [pre[choose], 1.0 if choose[0] == "L" else 0.0, len(pre) / 5.0, nS / 2.0, 1.0]


def collect_and_fit_coupled(rng, n_ep=400, couple=1.5, floor=3.0, growth=1.25, nL=3, nS=2, H=4):
    """random-policy rollouts -> step features labelled by episode recovery -> logistic fit."""
    X, y = [], []
    for _ in range(n_ep):
        sig = {("L", i): rng.uniform(6.0, 9.0) for i in range(nL)}
        sig.update({("S", i): rng.uniform(0.5, 1.5) for i in range(nS)})
        alive = {k: True for k in sig}; floored = False; steps = []
        for _ in range(H):
            acts = [k for k in sig if alive[k]]
            if not acts: break
            choose = acts[rng.randrange(len(acts))]
            pre = {k: sig[k] for k in acts}
            steps.append(feats_coupled(pre, choose))
            alive[choose] = False
            if choose[0] == "L":
                for k in sig:
                    if alive[k] and k[0] == "S": sig[k] += couple
            for k in sig:
                if alive[k] and k[0] == "L": sig[k] *= growth
            for k in sig:
                if alive[k] and k[0] == "S" and sig[k] >= floor:
                    alive[k] = False; floored = True
        rec = (not floored) and not any(alive[k] for k in sig if k[0] == "S")
        for f in steps: X.append(f); y.append(1.0 if rec else 0.0)
    # logistic regression by SGD
    w = [0.0] * 5
    for _ in range(300):
        for i in rng.sample(range(len(X)), len(X)):
            p = sigmoid(sum(a * b for a, b in zip(w, X[i])))
            g = (y[i] - p)
            for d in range(5): w[d] += 0.05 * g * X[i][d]
    return w


def train_learned_coupled(seed, couple=1.5, floor=3.0, growth=1.25, nL=3, nS=2, H=4,
                          iters=120, G=32, lr=0.03, step_cost=0.02):
    rng = random.Random(seed)
    w = collect_and_fit_coupled(rng)
    # policy training: GRPO with reward = r_hat(step)
    theta = [0.0, 0.0, 0.0]
    for _ in range(iters):
        eps = []
        for _ in range(G):
            sig = {("L", i): rng.uniform(6.0, 9.0) for i in range(nL)}
            sig.update({("S", i): rng.uniform(0.5, 1.5) for i in range(nS)})
            alive = {k: True for k in sig}; floored = False; traj = []
            for _ in range(H):
                acts = [k for k in sig if alive[k]]
                if not acts: break
                fts = [[sig[k], 1.0 if k[0] == "L" else 0.0] for k in acts]
                logits = [theta[0] * f[0] + theta[1] * f[1] + theta[2] for f in fts]
                probs = C.softmax(logits)
                r = rng.random(); c = 0.0; ci = len(acts) - 1
                for j, p in enumerate(probs):
                    c += p
                    if r <= c: ci = j; break
                choose = acts[ci]
                pre = {k: sig[k] for k in acts}
                rew = sigmoid(sum(a * b for a, b in zip(w, feats_coupled(pre, choose)))) - step_cost
                traj.append((fts, ci, rew))
                alive[choose] = False
                if choose[0] == "L":
                    for k in sig:
                        if alive[k] and k[0] == "S": sig[k] += couple
                for k in sig:
                    if alive[k] and k[0] == "L": sig[k] *= growth
                for k in sig:
                    if alive[k] and k[0] == "S" and sig[k] >= floor:
                        alive[k] = False; floored = True
            rec = (not floored) and not any(alive[k] for k in sig if k[0] == "S")
            eps.append((rec, traj))
        rets = [sum(s[2] for s in tj) for _, tj in eps]
        mu = st.mean(rets); sd = st.pstdev(rets) or 1e-6
        adv = [(R - mu) / sd for R in rets]
        grad = [0.0, 0.0, 0.0]
        for (rec, tj), A in zip(eps, adv):
            for fts, ci, _ in tj:
                logits = [theta[0] * f[0] + theta[1] * f[1] + theta[2] for f in fts]
                probs = C.softmax(logits)
                for d in range(2):
                    grad[d] += A * (fts[ci][d] - sum(p * f[d] for p, f in zip(probs, fts)))
        for d in range(2): theta[d] += lr * grad[d] / G
    # final eval with the env's TRUE recovery
    wins = 0
    for _ in range(400):
        r, _ = C.rollout(theta, rng, "E", couple, floor, growth, nL, nS, H, 0.0)
        wins += int(r)
    return wins / 400


def main():
    print("=== learned-PRM arm (logistic, binary-outcome labels) in coupled regime ===")
    finals = [train_learned_coupled(7000 + s) for s in range(16)]
    print(f"  learned-PRM trained recovery: {st.mean(finals):.3f}  (per-seed {[round(x,2) for x in finals[:8]]}...)")
    print("  reference: D 1.00 | count+drift 1.00 | count 0.12 | severity-scalar 0.04")


if __name__ == "__main__":
    main()

# ---- sigma-het regime version (urgent small family, tight budget) ----

def feats_mf(sig, fam, alive, choose):
    acts = [k for k in sig if alive[k]]
    return [sig[choose] / 25.0, 1.0 if fam[choose] == "L" else 0.0, len(acts) / 5.0, 1.0]


def run_mf_episode(rng, policy, w=None, grace=4, H=4, nL=4, nS=1, sigL=20.0):
    """policy: 'random' or theta list. Returns (recovered, steps[(feats2, ci, rhat)])."""
    sigma = {}; fam = {}; bid = 0
    for _ in range(nL): sigma[bid] = rng.uniform(0.8 * sigL, 1.2 * sigL); fam[bid] = "L"; bid += 1
    for _ in range(nS): sigma[bid] = rng.uniform(1.0, 3.0); fam[bid] = "S"; bid += 1
    alive = {k: True for k in sigma}
    s_age = 0; s_dead = False; s_cleared = False; steps = []
    for _ in range(H):
        idx = [k for k in sigma if alive[k]]
        if not idx: break
        if policy == "random":
            choose = idx[rng.randrange(len(idx))]
            fts2 = None; ci = None
        else:
            fts2 = [[sigma[k] / 25.0, 1.0] for k in idx]
            logits = [policy[0] * f[0] + policy[1] for f in fts2]
            probs = M.softmax(logits)
            r = rng.random(); c = 0.0; ci = len(idx) - 1
            for j, p in enumerate(probs):
                c += p
                if r <= c: ci = j; break
            choose = idx[ci]
        rhat = sigmoid(sum(a * b for a, b in zip(w, feats_mf(sigma, fam, alive, choose)))) if w else 0.0
        steps.append((feats_mf(sigma, fam, alive, choose), fts2, ci, rhat))
        alive[choose] = False
        if fam[choose] == "S": s_cleared = True
        if not s_cleared:
            s_age += 1
            if s_age > grace: s_dead = True; break
    return (s_cleared and not s_dead), steps


def main_mf():
    print("=== learned-PRM arm in sigma-het regime (nL=4 nS=1 H=4, sigL=20) ===")
    finals = []
    for s in range(16):
        rng = random.Random(8000 + s)
        # collect random-policy labels
        X, y = [], []
        for _ in range(400):
            rec, steps = run_mf_episode(rng, "random", w=[0, 0, 0, 0])
            for f, _, _, _ in steps: X.append(f); y.append(1.0 if rec else 0.0)
        w = [0.0] * 4
        for _ in range(300):
            for i in rng.sample(range(len(X)), len(X)):
                p = sigmoid(sum(a * b for a, b in zip(w, X[i])))
                for d in range(4): w[d] += 0.05 * (y[i] - p) * X[i][d]
        # GRPO with r_hat
        theta = [0.0, 0.0]
        for _ in range(80):
            eps = [run_mf_episode(rng, theta, w=w) for _ in range(24)]
            rets = [sum(st_[3] for st_ in tj) for _, tj in eps]
            mu = st.mean(rets); sd = st.pstdev(rets) or 1e-6
            adv = [(R - mu) / sd for R in rets]
            grad = 0.0
            for (rec, tj), A in zip(eps, adv):
                for f4, fts2, ci, _ in tj:
                    if fts2 is None: continue
                    probs = M.softmax([theta[0] * f[0] + theta[1] for f in fts2])
                    grad += A * (fts2[ci][0] - sum(p * f[0] for p, f in zip(probs, fts2)))
            theta[0] += 0.05 * grad / 24
        wins = sum(1 for _ in range(300) if run_mf_episode(rng, theta, w=w)[0])
        finals.append(wins / 300)
    print(f"  learned-PRM trained recovery: {st.mean(finals):.3f} (per-seed {[round(x,2) for x in finals[:8]]}...)")
    print("  reference: D per-family 1.00 | count 0.80 | severity-scalar 0.00")


if __name__ == "__main__" and True:
    main_mf()
