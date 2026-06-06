"""Aggregate the CityLearn GRPO eval JSONs into the headline multi-type result.

Splits each adapter's recovery into in-distribution (the 12 trained scenarios)
vs held-out (the 12 not trained on), for gated and UNGATED regimes. Ungated is
the primary DV (internalization). Pools the 3 seeds per arm with a Wilson 95% CI.
"""
from __future__ import annotations

import glob
import json
import math
import os

TRAIN = set(
    "cl_mined_000_t11_smax-exp cl_mined_005_t12_smax-exp cl_mined_010_t13_smax-exp "
    "cl_mined_002_t11_smax-smin-exp cl_mined_007_t12_smax-smin-exp cl_mined_012_t13_smax-smin-exp "
    "cl_mined_003_t11_smin-exp cl_mined_008_t12_smin-exp cl_mined_013_t13_smin-exp "
    "cl_mined_004_t11_smax-smin cl_mined_009_t12_smax-smin cl_mined_014_t13_smax-smin".split()
)


def wilson(k, n):
    if n == 0:
        return (None, None, None)
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 3), round(c - h, 2), round(c + h, 2))


def main():
    import sys
    # Optional substring filter so the buggy-reward run (labels '*_i2') and the
    # fixed-reward run (labels '*_i2fix' / 'base_fix') can be aggregated apart.
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    os.chdir("/work/6/us06396/SILR-WISE26")
    rows = {}
    for f in glob.glob("eval_citylearn_grpo_*.json"):
        d = json.load(open(f))
        lab = d["summary"]["label"]
        if want and want not in lab:
            continue
        recs = d["records"]

        def rate(g, sub):
            rs = [r for r in recs if r["gated"] == g
                  and ((r["scenario"] in TRAIN) == (sub == "in"))]
            return (sum(1 for r in rs if r["recovered"]), len(rs))

        rows[lab] = dict(gin=rate(True, "in"), ghe=rate(True, "held"),
                         uin=rate(False, "in"), uhe=rate(False, "held"))

    def fmt(t):
        return "%d/%d" % t

    print("%-12s|  gated in/held |  UNGATED in/held" % "label")
    for lab in sorted(rows):
        r = rows[lab]
        print("%-12s| %5s %5s  | %5s %5s" % (lab, fmt(r["gin"]), fmt(r["ghe"]),
                                             fmt(r["uin"]), fmt(r["uhe"])))

    print("\n=== pooled per-arm UNGATED (3 seeds, Wilson 95%% CI) ===")
    base = rows.get("base") or rows.get("base_fix")
    if base:
        pi, lo, hi = wilson(*base["uin"])
        ph, lh, hh = wilson(*base["uhe"])
        print("base  ungated: in-dist %.3f [%.2f,%.2f] %s | held-out %.3f [%.2f,%.2f] %s"
              % (pi, lo, hi, fmt(base["uin"]), ph, lh, hh, fmt(base["uhe"])))
    for arm in ["C", "D", "E"]:
        ui = [0, 0]
        uh = [0, 0]
        for s in [0, 1, 2]:
            pref = "%s_s%d" % (arm, s)
            lab = next((k for k in rows if k.startswith(pref)), None)
            if lab:
                ui[0] += rows[lab]["uin"][0]; ui[1] += rows[lab]["uin"][1]
                uh[0] += rows[lab]["uhe"][0]; uh[1] += rows[lab]["uhe"][1]
        if ui[1]:
            pi, lo, hi = wilson(*ui)
            ph, lh, hh = wilson(*uh)
            print("arm %s ungated: in-dist %.3f [%.2f,%.2f] %s | held-out %.3f [%.2f,%.2f] %s"
                  % (arm, pi, lo, hi, fmt(tuple(ui)), ph, lh, hh, fmt(tuple(uh))))


if __name__ == "__main__":
    main()
