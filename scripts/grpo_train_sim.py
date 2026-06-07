"""Controlled GRPO training simulation: does the geometric (severity-weighted)
process reward TRAIN a better policy than the scalar (count) reward, in a task
where first-action quality gates recovery?

Tabular recovery MDP (CPU, fast, no LLM, no eval nondeterminism):
  * State: B violation branches with severities sigma_k. Budget H steps.
  * Action: pick ONE branch to clear this step.
  * Dynamics: the WORST surviving branch DECAYS each step it is left unaddressed
    (sigma *= growth). If any branch's sigma crosses DEADEND, it is irreversible
    -> episode cannot recover. Recovery = all branches cleared before any dead-ends.
    => clearing the highest-severity branch FIRST is what avoids dead-ends.
  * Policy: softmax over branches, logit_k = theta . [sigma_k, 1]; theta learned.
  * Reward per step: arm D = severity-weighted elimination (compute_grpo_reward-like)
    vs arm E = count elimination (1/n). + terminal recovery bonus.
  * Training: GRPO -- group of G rollouts/update, advantage=(R-mean)/std, REINFORCE
    step on theta. Measure recovery rate vs iteration for D vs E.

If D's policy converges to higher recovery than E's, the geometric process reward
trains a better policy when the task rewards severity-aware ordering -- the
architecture enhancing the training signal, demonstrated end-to-end internally.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics as st


def softmax(zs):
    m = max(zs)
    es = [math.exp(z - m) for z in zs]
    s = sum(es)
    return [e / s for e in es]


def rollout(theta, B, H, growth, deadend, rng, arm, bonus, step_cost):
    """One episode. CRITICAL branches (high sigma) grow and DEAD-END if not cleared
    in time; DISTRACTOR branches (low sigma) never dead-end and are free count points.
    The episode runs the full H steps even after a dead-end (a dead-ended critical
    just flags non-recovery) -- so a count-maximising policy can keep farming clears
    (incl. distractors) while letting a critical dead-end. Recovery = NO critical
    dead-ended. count-return is maximised by total clears (severity-blind); geometric-
    return is maximised by clearing the high-sigma criticals first.
    Returns (recovered, traj)."""
    n_crit = B // 2
    sig = [rng.uniform(6.0, 10.0) if k < n_crit else rng.uniform(0.4, 1.2) for k in range(B)]
    crit = [k < n_crit for k in range(B)]
    alive = [True] * B
    deaded = [False] * B
    traj = []
    for _ in range(H):
        idx = [k for k in range(B) if alive[k]]
        if not idx:
            break
        feats = [[sig[k], 1.0] for k in idx]
        logits = [theta[0] * f[0] + theta[1] * f[1] for f in feats]
        probs = softmax(logits)
        r = rng.random(); c = 0.0; choose = idx[-1]
        for j, p in enumerate(probs):
            c += p
            if r <= c:
                choose = idx[j]; break
        sig_total = sum(sig[k] for k in idx) + 1e-8
        step_r = (sig[choose] / sig_total) if arm == "D" else (1.0 / len(idx))
        step_r -= step_cost
        alive[choose] = False
        traj.append((feats, idx.index(choose), step_r))
        # surviving CRITICAL branches grow; dead-end if over threshold (episode continues)
        for k in range(B):
            if alive[k] and crit[k]:
                sig[k] *= growth
                if sig[k] >= deadend:
                    deaded[k] = True
                    alive[k] = False   # removed (lost), no longer clearable
    recovered = not any(deaded)
    if recovered and bonus and traj:
        traj[-1] = (traj[-1][0], traj[-1][1], traj[-1][2] + bonus)
    return recovered, traj


def train(arm, B=4, H=4, growth=1.8, deadend=14.0, iters=60, G=24, lr=0.04,
          bonus=1.0, step_cost=0.05, seed=0):
    rng = random.Random(seed)
    theta = [0.0, 0.0]
    rec_curve = []
    for _ in range(iters):
        # one GRPO update from a group of G rollouts
        eps = [rollout(theta, B, H, growth, deadend, rng, arm, bonus, step_cost) for _ in range(G)]
        rec_curve.append(sum(1 for r, _ in eps if r) / G)
        returns = [sum(s[2] for s in tj) for _, tj in eps]
        mu = st.mean(returns); sd = st.pstdev(returns) or 1e-6
        adv = [(R - mu) / sd for R in returns]
        grad = [0.0, 0.0]
        for (rec, tj), A in zip(eps, adv):
            for feats, ci, _ in tj:
                logits = [theta[0] * f[0] + theta[1] * f[1] for f in feats]
                probs = softmax(logits)
                for d in (0, 1):
                    gi = feats[ci][d] - sum(p * f[d] for p, f in zip(probs, feats))
                    grad[d] += A * gi
        theta[0] += lr * grad[0] / G
        theta[1] += lr * grad[1] / G
    # final recovery over a fresh eval batch
    final = st.mean([sum(1 for _ in range(200) if rollout(theta, B, H, growth, deadend, rng, arm, bonus, step_cost)[0]) / 200
                     for _ in range(1)])
    return rec_curve, theta, final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--bonus", type=float, default=0.0)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--H", type=int, default=6)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.04)
    p.add_argument("--growth", type=float, default=1.5)
    p.add_argument("--deadend", type=float, default=16.0)
    args = p.parse_args()
    print(f"=== controlled GRPO training: geometric (D) vs scalar (E) process reward ===")
    print(f"task: B={args.B} ({args.B//2} critical hi-sigma + {args.B-args.B//2} distractor), "
          f"H={args.H}, growth x{args.growth}, dead-end>={args.deadend}, bonus={args.bonus}")
    curves = {"D": [], "E": []}
    finals = {"D": [], "E": []}
    for arm in ("D", "E"):
        for s in range(args.seeds):
            curve, theta, final = train(arm, B=args.B, H=args.H, growth=args.growth,
                                        deadend=args.deadend, iters=args.iters, lr=args.lr,
                                        bonus=args.bonus, seed=1000 + s)
            curves[arm].append(curve); finals[arm].append(final)
    def avg_at(arm, it):
        return st.mean([c[min(it, len(c) - 1)] for c in curves[arm]])
    print(f"\nlearning curve (mean recovery over {args.seeds} seeds) -- CONVERGENCE SPEED:")
    print(f"{'iter':>6} | {'D (geom)':>9} | {'E (scalar)':>10} | {'D-E':>7}")
    for it in (2, 5, 10, 20, 40, args.iters - 1):
        d, e = avg_at("D", it), avg_at("E", it)
        print(f"{it:>6} | {d:>9.3f} | {e:>10.3f} | {d - e:>+7.3f}")
    # area under the learning curve (sample efficiency) + iters to reach 90% of final
    def auc(arm):
        return st.mean([sum(c) / len(c) for c in curves[arm]])
    def iters_to(arm, frac):
        tot = []
        for c in curves[arm]:
            tgt = frac * c[-1]
            tot.append(next((i for i, v in enumerate(c) if v >= tgt), len(c)))
        return st.mean(tot)
    print(f"\nsample efficiency (area under recovery curve): D {auc('D'):.3f} vs E {auc('E'):.3f}  "
          f"(D-E {auc('D')-auc('E'):+.3f})")
    print(f"iters to reach 90% of own final: D {iters_to('D',0.9):.1f} vs E {iters_to('E',0.9):.1f}")
    print(f"final recovery: D {st.mean(finals['D']):.3f} vs E {st.mean(finals['E']):.3f}")


if __name__ == "__main__":
    main()
