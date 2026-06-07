"""On-policy fidelity check (panel 2026-06-07 qwen): the fidelity ladder is computed
at the INITIAL trap state. A reviewer asks whether it holds at the states the
trained policy actually VISITS along its trajectory (the on-policy distribution),
not just the static initial state. We replay a greedy recovery trajectory and
re-measure count vs geometric confusion@5% at EACH visited (still-violated) state,
then aggregate. If the ladder direction (geometric << count) holds across the
on-policy states, it is not a static-distribution artifact. CPU only.
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
from scripts.anm_reward_landscape import score_actions, apply_action, enumerate_actions  # noqa: E402


def confusion_at_state(loader, sc, mgr, verifier, frac=0.05):
    """count vs geometric confusion at the CURRENT mgr state (true value = one-step
    penalty reduction over admissible actions)."""
    base = float(mgr.last_penalty)
    rows = [r for r in score_actions(mgr, verifier) if r["admissible"]]
    if len(rows) < 3:
        return None
    vals = []
    for r in rows:
        m2 = GymANMManager(seed=0)
        loader.setup_episode(m2, sc)
        # replay to current state by re-applying the same setpoints is hard; instead
        # use a shadow of the current mgr to score each action's one-step value.
        sh = mgr.create_shadow_copy()
        from domains.anm import create_anm_toolset
        create_anm_toolset(sh).get(r["action"]["tool_name"]).execute(**r["action"]["params"])
        sh.solve()
        vals.append(base - float(sh.last_penalty))
    vmax = max(vals) if vals else 0.0
    delta = frac * vmax

    def conf(key):
        rv = [r[key] for r in rows]
        pr = cc = 0
        for i in range(len(rv)):
            for j in range(i + 1, len(rv)):
                if abs(vals[i] - vals[j]) > delta:
                    pr += 1
                    if abs(rv[i] - rv[j]) < 1e-9:
                        cc += 1
        return cc / pr if pr else 0.0
    return conf("rE"), conf("rD")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--max-steps", type=int, default=4)
    args = p.parse_args()
    cfg = build_anm_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = ANMScenarioLoader()
    init_cE, init_cD, onp_cE, onp_cD = [], [], [], []
    n_states = 0
    for sid in args.scenarios:
        sc = loader.load(sid)
        mgr = GymANMManager(seed=0)
        loader.setup_episode(mgr, sc)
        verifier = SiLRVerifier(mgr, domain_config=cfg)
        # greedy-geometric trajectory; measure confusion at each visited violated state
        for step in range(args.max_steps):
            if mgr.last_penalty < 1e-6:
                break
            r = confusion_at_state(loader, sc, mgr, verifier)
            if r is not None:
                cE, cD = r
                if step == 0:
                    init_cE.append(cE); init_cD.append(cD)
                onp_cE.append(cE); onp_cD.append(cD); n_states += 1
            # advance greedily under geometric reward (the policy whose distribution we probe)
            rows = [x for x in score_actions(mgr, verifier) if x["admissible"]]
            if not rows:
                break
            best = max(rows, key=lambda x: x["rD"])
            apply_action(mgr, best["action"])
    print(f"=== on-policy fidelity ({len(args.scenarios)} scenarios, {n_states} visited "
          f"violated states along greedy trajectory) ===")
    print(f"  INITIAL states only: count confusion {st.mean(init_cE):.3f} | geom {st.mean(init_cD):.3f}")
    print(f"  ALL on-policy states: count confusion {st.mean(onp_cE):.3f} | geom {st.mean(onp_cD):.3f}")
    print("  ladder holds on-policy if geom << count at the visited states too "
          "(not just the static initial trap state).")


if __name__ == "__main__":
    main()
