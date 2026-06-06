"""Mechanism comparison from the observer-trace eval (--observer-trace).

The pre-registration (docs/experiment_plan_pillar2.md H1b) bets that arm D's
policy INTRINSICALLY uses the product-order geometry under no gating: it
eliminates the worst (most-severe) branch faster and avoids count-preserving
magnitude drift. The primary ungated DV cannot see this (verifier off => no Phi).
The observer eval runs ungated but logs Phi passively; this script reads those
*_obs.json files and tests D vs E on the per-trajectory mechanism metrics:

  worst_red  : sum over applied steps of (max_sigma_pre - max_sigma_post)
               -- higher D = eliminates the worst branch more (geometry use)
  maxsig_red : same idea on the running max sigma
  drift      : # applied steps that preserved violation count but RAISED
               total sigma -- lower D = less count-preserving magnitude drift

Held-out scenarios, paired over (scenario x seed). Stdlib only.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys

TRAIN = set(
    "cl_mined_000_t11_smax-exp cl_mined_005_t12_smax-exp cl_mined_010_t13_smax-exp "
    "cl_mined_002_t11_smax-smin-exp cl_mined_007_t12_smax-smin-exp cl_mined_012_t13_smax-smin-exp "
    "cl_mined_003_t11_smin-exp cl_mined_008_t12_smin-exp cl_mined_013_t13_smin-exp "
    "cl_mined_004_t11_smax-smin cl_mined_009_t12_smax-smin cl_mined_014_t13_smax-smin".split()
)


def exact_two_sided(k_small, n):
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, k_small + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def episode_metrics(rec):
    """Per-trajectory mechanism summary from the observer step_trace."""
    tr = rec.get("step_trace", [])
    worst = sum((t.get("worst_branch_reduced", 0) or 0) for t in tr)
    maxsig = sum((t.get("max_sigma_pre", 0) or 0) - (t.get("max_sigma_post", 0) or 0) for t in tr)
    drift = sum(1 for t in tr
                if t.get("n_post") == t.get("n_pre")
                and (t.get("sum_sigma_post", 0) or 0) > (t.get("sum_sigma_pre", 0) or 0) + 1e-9)
    # NB: `worst`/`maxsig` summed over the whole trajectory TELESCOPE under no
    # gating (step t's post-state is step t+1's pre-state), so they collapse to
    # ~(initial - final) max-sigma -- an ENDPOINT/recovery proxy, not a path
    # signal (panel 2026-06-06-2230). The non-telescoping DVs below are the
    # honest test of H1b (does the policy intrinsically work the worst branch):
    #   active_improve_frac : fraction of steps that strictly reduce max-sigma
    #   severity_auc        : mean max-sigma carried across the trajectory (risk load)
    #   worst_mid           : worst_branch_reduced over intermediate steps only
    #                         (post non-empty, admitted verdict) -- recovery-terminal
    #                         steps (post={} -> +max(pre)) removed to de-conflate.
    n = len(tr)
    active_improve_frac = (sum(1 for t in tr
                               if (t.get("max_sigma_pre", 0) or 0) - (t.get("max_sigma_post", 0) or 0) > 1e-9)
                           / n) if n else 0.0
    severity_auc = (sum((t.get("max_sigma_pre", 0) or 0) for t in tr) / n) if n else 0.0
    mid = [t for t in tr
           if (t.get("n_post") or 0) > 0 and t.get("verdict") in ("PASS", "SAFE_PROGRESS")]
    worst_mid = sum((t.get("worst_branch_reduced", 0) or 0) for t in mid)
    return {"worst_red": worst, "maxsig_red": maxsig, "drift": drift, "n_steps": n,
            "active_improve_frac": active_improve_frac, "severity_auc": severity_auc,
            "worst_mid": worst_mid}


def load(arm):
    out = {}
    for s in (0, 1, 2):
        fs = glob.glob("eval_citylearn_grpo_%s_s%d_obs.json" % (arm, s))
        if not fs:
            continue
        d = json.load(open(fs[0]))
        for r in d["records"]:
            if not r.get("observed") or r["scenario"] in TRAIN:
                continue
            out.setdefault(s, {})[r["scenario"]] = episode_metrics(r)
    return out


def main():
    arm1 = sys.argv[1] if len(sys.argv) > 1 else "D"
    arm2 = sys.argv[2] if len(sys.argv) > 2 else "E"
    os.chdir("/work/6/us06396/SILR-WISE26")
    A, B = load(arm1), load(arm2)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        print("no observer JSONs yet (%s/%s)" % (arm1, arm2)); return
    scens = sorted({sid for s in seeds for sid in A.get(s, {})} &
                   {sid for s in seeds for sid in B.get(s, {})})
    print("observer mechanism %s-vs-%s  held-out scenarios=%d seeds=%s"
          % (arm1, arm2, len(scens), seeds))
    # trace sanity: non-empty step_trace?
    nsteps = [A[s][sid]["n_steps"] for s in seeds for sid in scens if sid in A.get(s, {})]
    print("trace sanity: %d episodes, steps/episode min/mean/max = %d/%.1f/%d"
          % (len(nsteps), min(nsteps), st.mean(nsteps), max(nsteps)))

    print("  (worst_red/maxsig_red TELESCOPE = endpoint proxy; trust the non-telescoping DVs below)")
    for metric, better in [("worst_red", "higher (telescoped)"), ("maxsig_red", "higher (telescoped)"),
                           ("drift", "lower"),
                           ("worst_mid", "higher (de-conflated, intermediate steps)"),
                           ("active_improve_frac", "higher (non-telescoping)"),
                           ("severity_auc", "lower (non-telescoping risk load)")]:
        diffs = []
        for s in seeds:
            for sid in scens:
                a, b = A[s].get(sid), B[s].get(sid)
                if a is None or b is None:
                    continue
                diffs.append(a[metric] - b[metric])
        pos = sum(1 for x in diffs if x > 1e-9)
        neg = sum(1 for x in diffs if x < -1e-9)
        am = st.mean([A[s][sid][metric] for s in seeds for sid in scens if sid in A.get(s, {})])
        bm = st.mean([B[s][sid][metric] for s in seeds for sid in scens if sid in B.get(s, {})])
        p = exact_two_sided(min(pos, neg), pos + neg)
        print("[%-10s %s=better] %s=%.3f %s=%.3f  mean(%s-%s)=%+.4f  pos=%d neg=%d tie=%d  sign p=%.4f"
              % (metric, better, arm1, am, arm2, bm, arm1, arm2,
                 st.mean(diffs) if diffs else 0, pos, neg, len(diffs) - pos - neg, p))


if __name__ == "__main__":
    main()
