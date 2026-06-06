"""Pre-registered paired analysis for the CityLearn multi-type D-vs-E contrast.

The pre-registration (docs/experiment_plan_pillar2.md §statistics) specifies a
*paired* test over (scenario x training-seed), NOT the flat pooled-Wilson read in
citylearn_grpo_aggregate.py. This script executes that plan on the held-out band:

  1. seed-matched McNemar (exact)  -- the (scenario,seed) paired binary test
  2. scenario-level sign test (exact) -- pairs each scenario's D-rate vs E-rate
  3. cluster bootstrap over scenarios -- 95% CI of the D-E recovery-rate diff
  4. continuous penalty-reduction paired test -- (default-final)/default, the
     higher-information outcome DV (pre-reg secondary)

UNGATED records only (the primary internalisation DV). Binary recovery has a hard
ceiling on N=12 scenarios (few discordant pairs); the continuous DV and the
observer-trace mechanism eval (--observer-trace in citylearn_grpo_eval.py) are the
levers that the binary outcome cannot reach. Deterministic bootstrap (fixed seed)
so the reported CI is reproducible. Stdlib only.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import statistics as st
import sys

TRAIN = set(
    "cl_mined_000_t11_smax-exp cl_mined_005_t12_smax-exp cl_mined_010_t13_smax-exp "
    "cl_mined_002_t11_smax-smin-exp cl_mined_007_t12_smax-smin-exp cl_mined_012_t13_smax-smin-exp "
    "cl_mined_003_t11_smin-exp cl_mined_008_t12_smin-exp cl_mined_013_t13_smin-exp "
    "cl_mined_004_t11_smax-smin cl_mined_009_t12_smax-smin cl_mined_014_t13_smax-smin".split()
)


def exact_two_sided(k_small, n):
    """Two-sided exact binomial tail at p=0.5 (McNemar / sign test)."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, k_small + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_arm(arm, suffix):
    """Return {seed: {scenario: record}} for ungated held-out records."""
    out = {}
    for s in (0, 1, 2):
        fs = glob.glob("eval_citylearn_grpo_%s_s%d_%s.json" % (arm, s, suffix))
        if not fs:
            continue
        d = json.load(open(fs[0]))
        for r in d["records"]:
            if r["gated"] or r.get("observed"):
                continue
            if r["scenario"] in TRAIN:
                continue  # held-out only
            dp, fp = r.get("default_penalty"), r.get("final_penalty")
            red = (dp - fp) / dp if (dp not in (None, 0) and fp is not None) else None
            out.setdefault(s, {})[r["scenario"]] = {"rec": int(r["recovered"]), "red": red}
    return out


def main():
    arm1 = sys.argv[1] if len(sys.argv) > 1 else "D"
    arm2 = sys.argv[2] if len(sys.argv) > 2 else "E"
    suffix = sys.argv[3] if len(sys.argv) > 3 else "i2fix"
    os.chdir("/work/6/us06396/SILR-WISE26")
    A, B = load_arm(arm1, suffix), load_arm(arm2, suffix)
    seeds = sorted(set(A) & set(B))
    scens = sorted({sid for s in seeds for sid in A.get(s, {})} &
                   {sid for s in seeds for sid in B.get(s, {})})
    print("paired %s-vs-%s  suffix=%s  held-out scenarios=%d seeds=%s"
          % (arm1, arm2, suffix, len(scens), seeds))

    # 1. seed-matched McNemar
    b = c = 0
    for s in seeds:
        for sid in scens:
            ra, rb = A[s].get(sid, {}).get("rec"), B[s].get(sid, {}).get("rec")
            if ra is None or rb is None:
                continue
            if ra == 1 and rb == 0:
                b += 1
            elif ra == 0 and rb == 1:
                c += 1
    print("\n[1] seed-matched McNemar: %s>%s b=%d, %s>%s c=%d, discordant=%d, exact p=%.4f"
          % (arm1, arm2, b, arm2, arm1, c, b + c, exact_two_sided(min(b, c), b + c)))

    # 2. scenario-level sign test (mean recovery rate over seeds)
    dwin = ewin = tie = 0
    for sid in scens:
        ma = st.mean([A[s][sid]["rec"] for s in seeds if sid in A.get(s, {})])
        mb = st.mean([B[s][sid]["rec"] for s in seeds if sid in B.get(s, {})])
        if ma > mb:
            dwin += 1
        elif mb > ma:
            ewin += 1
        else:
            tie += 1
    print("[2] scenario sign test: %s-win=%d %s-win=%d tie=%d, exact p=%.4f"
          % (arm1, dwin, arm2, ewin, tie, exact_two_sided(min(dwin, ewin), dwin + ewin)))

    # 3. cluster bootstrap over scenarios (resample scenarios, keep their seeds)
    def rate(arm_data):
        rc = n = 0
        for s in seeds:
            for sid in boot:
                r = arm_data[s].get(sid)
                if r is not None:
                    rc += r["rec"]; n += 1
        return rc / n if n else 0.0
    rng = random.Random(20260606)
    diffs = []
    for _ in range(10000):
        boot = [rng.choice(scens) for _ in scens]
        diffs.append(rate(A) - rate(B))
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    pt = sum(A[s][sid]["rec"] for s in seeds for sid in scens if sid in A[s]) / (len(seeds) * len(scens)) \
        - sum(B[s][sid]["rec"] for s in seeds for sid in scens if sid in B[s]) / (len(seeds) * len(scens))
    print("[3] cluster-bootstrap %s-%s recovery diff = %+.3f  95%% CI [%+.3f, %+.3f]  (sig if CI excludes 0)"
          % (arm1, arm2, pt, lo, hi))

    # 4. continuous penalty-reduction paired (higher-information outcome DV)
    cd = []
    for s in seeds:
        for sid in scens:
            ra, rb = A[s].get(sid, {}).get("red"), B[s].get(sid, {}).get("red")
            if ra is not None and rb is not None:
                cd.append(ra - rb)
    if cd:
        pos = sum(1 for x in cd if x > 1e-9)
        neg = sum(1 for x in cd if x < -1e-9)
        t = st.mean(cd) / (st.stdev(cd) / math.sqrt(len(cd))) if len(cd) > 1 and st.stdev(cd) > 0 else 0.0
        print("[4] continuous pen-reduction paired: mean(%s-%s)=%+.4f  n=%d  pos=%d neg=%d  paired t=%+.2f  sign p=%.4f"
              % (arm1, arm2, st.mean(cd), len(cd), pos, neg, t, exact_two_sided(min(pos, neg), pos + neg)))


if __name__ == "__main__":
    main()
