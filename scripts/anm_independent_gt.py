"""Non-circular validation: re-score the PRM fidelity ladder against an INDEPENDENT
ground truth that is NOT the one-step penalty the geometric reward is derived from.
Reviewer concern (panel 2026-06-07 ds+qwen): the 'true value' = SiLR penalty drop,
and arm D is also severity/penalty-based -> Spearman could be endogenous/circular.

Independent GT here = STEPS-TO-RECOVERY: apply each candidate action, then run an
oracle (min-penalty) rollout to full recovery and count the steps. Fewer steps =
better action. This is a TASK-OUTCOME measure (how fast can you recover from the
resulting state), distinct from the instantaneous penalty magnitude. If count's
confusion still exceeds geometric's against this independent GT, the ladder is not
an artifact of the penalty-based value.

CPU only; reuses the ANM landscape primitives.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.anm import ANMScenarioLoader, GymANMManager, build_anm_domain_config  # noqa: E402
from silr.verifier import SiLRVerifier  # noqa: E402
from scripts.anm_reward_landscape import (  # noqa: E402
    score_actions, apply_action, enumerate_actions, severity_scalar_reward)


def steps_to_recovery(loader, sc, first_action, cap=8):
    """Apply first_action, then oracle-greedy (min penalty) until recovered; return
    step count (capped). Independent of the one-step penalty magnitude used as value."""
    mgr = GymANMManager(seed=0)
    loader.setup_episode(mgr, sc)
    apply_action(mgr, first_action)
    steps = 1
    while steps < cap and mgr.last_penalty >= 1e-6:
        best_pen = mgr.last_penalty
        best = None
        for a in enumerate_actions(mgr):
            sh = mgr.create_shadow_copy()
            from domains.anm import create_anm_toolset
            create_anm_toolset(sh).get(a["tool_name"]).execute(**a["params"])
            sh.solve()
            if sh.last_penalty < best_pen - 1e-9:
                best_pen = sh.last_penalty
                best = a
        if best is None:
            return cap + 1  # stuck -> never recovers within budget (worst)
        apply_action(mgr, best)
        steps += 1
    return steps if mgr.last_penalty < 1e-6 else cap + 1


def confusion(rvals, gt, frac=0.0):
    """gt = steps-to-recovery (lower better). Pairs with DIFFERENT gt that the reward
    leaves at EQUAL value = zero signal. frac=0 -> any integer step difference counts."""
    n = len(rvals)
    pairs = conf = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(gt[i] - gt[j]) > 0.5:  # different recovery-step class
                pairs += 1
                if abs(rvals[i] - rvals[j]) < 1e-9:
                    conf += 1
    return round(conf / pairs, 3) if pairs else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--out", default="figures/anm_independent_gt.json")
    args = p.parse_args()
    cfg = build_anm_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = ANMScenarioLoader()
    report = {}
    import statistics as st
    cE_all, cD_all = [], []
    for sid in args.scenarios:
        sc = loader.load(sid)
        mgr = GymANMManager(seed=0)
        loader.setup_episode(mgr, sc)
        verifier = SiLRVerifier(mgr, domain_config=cfg)
        rows = [r for r in score_actions(mgr, verifier) if r["admissible"]]
        if len(rows) < 3:
            continue
        gt = [steps_to_recovery(loader, sc, r["action"]) for r in rows]
        rE = [r["rE"] for r in rows]
        rD = [r["rD"] for r in rows]
        cE = confusion(rE, gt)
        cD = confusion(rD, gt)
        cE_all.append(cE); cD_all.append(cD)
        report[sid] = {"n_admissible": len(rows), "gt_steps_range": [min(gt), max(gt)],
                       "count_confusion_vs_indepGT": cE, "geom_confusion_vs_indepGT": cD}
        print(f"{sid[:40]:40s} steps-GT range {min(gt)}-{max(gt)} | count conf {cE:.3f} | geom conf {cD:.3f}")
    print(f"\n=== INDEPENDENT GT (steps-to-recovery, NOT one-step penalty) over {len(cE_all)} scenarios ===")
    print(f"  count E confusion = {st.mean(cE_all):.3f}   geometric D confusion = {st.mean(cD_all):.3f}")
    print("  if count >> geometric here, the fidelity ladder is NOT an artifact of the "
          "penalty-based value (non-circular).")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
