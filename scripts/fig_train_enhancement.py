"""Figure: the per-family geometric reward enhances GRPO training where a scalar
projection fails. Two panels:
  (a) learning curve in the coupled sigma-het trap (D -> 1.0 in ~5 iters; count
      degrades below random init).
  (b) trained recovery vs sigma-heterogeneity (severity-scalar collapses to 0 by
      CityLearn's 286; per-family geometric stays ~1.0).
Runs the experiments and writes figures/train_enhancement.{png,pdf}.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scripts.grpo_train_coupled as C  # noqa: E402
import scripts.grpo_train_multifamily as M  # noqa: E402


def coupled_curves(seeds=24):
    curves = {"D": [], "E": []}
    for arm in ("D", "E"):
        for s in range(seeds):
            _, _, cur = C.train(arm, couple=1.5, floor=3.0, growth=1.25, nL=3, nS=2, H=4, seed=3000 + s)
            curves[arm].append(cur)
    n = len(curves["D"][0])
    return list(range(n)), [st.mean([c[i] for c in curves["D"]]) for i in range(n)], \
        [st.mean([c[i] for c in curves["E"]]) for i in range(n)]


def het_sweep(seeds=16):
    ratios = [1, 2, 4, 8, 20, 50, 286]
    D, E2 = [], []
    for sl in ratios:
        d = st.mean([M.train("D", 4, 0.0, seed=2000 + s, n_S=1, n_L=4, H=4, sigL=float(sl))[2] for s in range(seeds)])
        e2 = st.mean([M.train("E2", 4, 0.0, seed=2000 + s, n_S=1, n_L=4, H=4, sigL=float(sl))[2] for s in range(seeds)])
        D.append(d); E2.append(e2)
    return ratios, D, E2


def main():
    its, dC, eC = coupled_curves()
    ratios, dH, e2H = het_sweep()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(its, dC, label="geometric (per-family + drift)", color="#1b7837", lw=2)
    ax1.plot(its, eC, label="count (scalar projection)", color="#c2453a", lw=2)
    ax1.axhline(dC[0], ls=":", color="gray", lw=1, label="random init")
    ax1.set_xlabel("GRPO iteration"); ax1.set_ylabel("recovery rate")
    ax1.set_title("(a) coupled $\\sigma$-het trap: training curve")
    ax1.set_ylim(-0.02, 1.05); ax1.legend(fontsize=8, loc="center right")
    ax2.semilogx(ratios, dH, "o-", label="geometric (per-family)", color="#1b7837", lw=2)
    ax2.semilogx(ratios, e2H, "s-", label="severity-scalar", color="#c2453a", lw=2)
    ax2.axvline(286, ls="--", color="gray", lw=1)
    ax2.text(286, 0.5, " CityLearn\n $\\sigma$-het=286", fontsize=8, va="center")
    ax2.set_xlabel("$\\sigma$-heterogeneity ($\\sigma_L/\\sigma_S$)"); ax2.set_ylabel("trained recovery rate")
    ax2.set_title("(b) scalar collapse vs $\\sigma$-heterogeneity")
    ax2.set_ylim(-0.02, 1.05); ax2.legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "train_enhancement.png", dpi=150)
    fig.savefig(out / "train_enhancement.pdf")
    print("wrote figures/train_enhancement.png/.pdf")
    print("panel a final:", round(dC[-1], 3), "vs", round(eC[-1], 3))
    print("panel b @286 :", round(dH[-1], 3), "vs", round(e2H[-1], 3))


if __name__ == "__main__":
    main()
