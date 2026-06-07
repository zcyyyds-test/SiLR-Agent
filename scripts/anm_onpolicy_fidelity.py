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


def run_policy(loader, scenarios, cfg, policy, max_steps, rng):
    """policy in {'greedy_D','greedy_E','random'}: which action advances the trajectory.
    'random' = neutral (no arm bias) -> removes the selection-bias concern (codex
    2026-06-07: greedy-D visits D-favourable states)."""
    cE_all, cD_all = [], []
    n_states = 0
    for sid in scenarios:
        sc = loader.load(sid)
        mgr = GymANMManager(seed=0)
        loader.setup_episode(mgr, sc)
        verifier = SiLRVerifier(mgr, domain_config=cfg)
        for _ in range(max_steps):
            if mgr.last_penalty < 1e-6:
                break
            r = confusion_at_state(loader, sc, mgr, verifier)
            if r is not None:
                cE_all.append(r[0]); cD_all.append(r[1]); n_states += 1
            rows = [x for x in score_actions(mgr, verifier) if x["admissible"]]
            if not rows:
                break
            if policy == "greedy_D":
                nxt = max(rows, key=lambda x: x["rD"])
            elif policy == "greedy_E":
                nxt = max(rows, key=lambda x: x["rE"])
            else:  # random admissible -- neutral, no arm bias
                nxt = rows[rng.randrange(len(rows))]
            apply_action(mgr, nxt["action"])
    return cE_all, cD_all, n_states


def main():
    import random
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--max-steps", type=int, default=4)
    args = p.parse_args()
    cfg = build_anm_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = ANMScenarioLoader()
    print("=== on-policy fidelity: ladder at states visited under DIFFERENT trajectory "
          "policies (selection-bias control) ===")
    print(f"{'trajectory policy':>22} | {'#states':>7} | {'count confusion':>15} | {'geom confusion':>14}")
    for policy in ("greedy_D", "greedy_E", "random"):
        rng = random.Random(20260607)
        cE, cD, n = run_policy(loader, args.scenarios, cfg, policy, args.max_steps, rng)
        print(f"{policy:>22} | {n:>7} | {st.mean(cE):15.3f} | {st.mean(cD):14.3f}")
    print("If geom << count under ALL trajectory policies (incl. neutral 'random' and the "
          "count-favouring 'greedy_E'), the ladder is not a selection-bias artifact of "
          "probing only D's own distribution.")


if __name__ == "__main__":
    main()
