"""Deterministic-vs-learned axis: a learned PRM is an APPROXIMATION (finite data ->
noisy reward); the simulator-backed reward is EXACT (zero noise). We model a learned
PRM as the geometric reward plus Gaussian noise of magnitude sigma (a fraction of
the reward's own spread) and measure how its confusion vs the true value degrades
with sigma. The deterministic simulator sits at sigma=0 (confusion 0); the question
is how much approximation noise erases the geometric advantage -- i.e., what the
'exact, zero-shot' property is worth, and at what noise a learned PRM degrades to
the count projection's confusion (0.25).

Uses the real per-action geometric reward + true value from the landscape (computed
here), so it is non-circular w.r.t. noise (noise is added on top, independent of the
value). CPU only.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.anm import ANMScenarioLoader, GymANMManager, build_anm_domain_config  # noqa: E402
from silr.verifier import SiLRVerifier  # noqa: E402
from scripts.anm_reward_landscape import score_actions, apply_action  # noqa: E402


def confusion(rvals, values, frac=0.05):
    n = len(rvals)
    vmax = max(values) if values else 0.0
    delta = frac * vmax
    pr = cc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(values[i] - values[j]) > delta:
                pr += 1
                if abs(rvals[i] - rvals[j]) < 1e-9:
                    cc += 1
    return cc / pr if pr else 0.0


def confusion_noisy(rvals, values, sigma_frac, rng, frac=0.05):
    """confusion when the reward is corrupted by Gaussian noise of sigma_frac*std."""
    n = len(rvals)
    sd = st.pstdev(rvals) if len(rvals) > 1 else 0.0
    noisy = [r + rng.gauss(0, sigma_frac * sd) for r in rvals]
    vmax = max(values) if values else 0.0
    delta = frac * vmax
    pr = 0
    # fraction of different-value pairs the noisy reward ranks WRONG (lower reward for
    # the higher-value action). Exact ties count as 0.5 (no signal either way), not as
    # full errors -- fixes the codex 2026-06-07 note that '<=' inflated the sigma=0 rate.
    wrong = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(values[i] - values[j]) > delta:
                pr += 1
                hi, lo = (i, j) if values[i] > values[j] else (j, i)
                if noisy[hi] < noisy[lo]:
                    wrong += 1.0
                elif noisy[hi] == noisy[lo]:
                    wrong += 0.5
    return wrong / pr if pr else 0.0


def main():
    import random
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", required=True)
    args = p.parse_args()
    cfg = build_anm_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = ANMScenarioLoader()
    rng = random.Random(20260607)
    # collect per-scenario (geom rewards, count rewards, values)
    data = []
    for sid in args.scenarios:
        sc = loader.load(sid)
        mgr = GymANMManager(seed=0)
        loader.setup_episode(mgr, sc)
        verifier = SiLRVerifier(mgr, domain_config=cfg)
        base = float(mgr.last_penalty)
        rows = [r for r in score_actions(mgr, verifier) if r["admissible"]]
        if len(rows) < 3:
            continue
        vals = []
        for r in rows:
            sh = mgr.create_shadow_copy()
            from domains.anm import create_anm_toolset
            create_anm_toolset(sh).get(r["action"]["tool_name"]).execute(**r["action"]["params"])
            sh.solve()
            vals.append(base - float(sh.last_penalty))
        data.append(([r["rD"] for r in rows], [r["rE"] for r in rows], vals))
    # count's CLEAN mis-order rate (no noise) as the reference line
    count_wrong = st.mean([confusion_noisy(rE, v, 0.0, rng) for _, rE, v in data])
    print(f"reference: count E mis-order rate (no noise) = {count_wrong:.3f}")
    print(f"\n{'noise sigma (x reward std)':>26} | {'geom D mis-order rate':>22}")
    for sf in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5):
        rates = []
        for rD, _, v in data:
            rates.append(st.mean([confusion_noisy(rD, v, sf, rng) for _ in range(20)]))
        m = st.mean(rates)
        flag = "  <- exceeds count" if m > count_wrong else ""
        print(f"{sf:>26} | {m:22.3f}{flag}")
    print("\nThe simulator-backed geometric reward sits at sigma=0 (mis-order ~0). A learned "
          "PRM's approximation noise must exceed the level above to erase the advantage and "
          "fall to the count projection's reference -- quantifying the value of EXACTNESS.")


if __name__ == "__main__":
    main()
