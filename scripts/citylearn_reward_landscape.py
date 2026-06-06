"""CityLearn (multi-family) reward landscaping — does the PRODUCT-ORDER per-family
geometric reward beat a severity-weighted SCALAR (cross-family Σσ) the way it beats
count? This is the experiment ANM (single-family) cannot run: with one family,
per-family normalization == sum normalization, so rD≈rE2 there. CityLearn has
physically-incomparable families (battery SoC kWh + feeder import/export kW,
σ-het ~286), where a cross-family Σσ scalar lets the largest-magnitude family
hijack the signal -> the per-family product-order reward should separate.

For each violating CityLearn state, enumerate single-building set-point actions,
score each with rD (per-family geometric), rE (count), rE2 (severity scalar =
cross-family Σσ reduction), and measure each action's TRUE value as a MULTI-STEP
recoverability (apply it, then oracle-greedy minimise penalty for H steps). One-
step value is biased toward rE2 (≈ penalty reduction); the per-family advantage is
about preserving recoverability across families, which only shows multi-step.

CPU only (ANM-style), SILR_CITYLEARN_N_BUILDINGS=4. No vLLM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.citylearn import (  # noqa: E402
    CityLearnScenarioLoader,
    CityLearnManager,
    build_citylearn_domain_config,
    create_citylearn_toolset,
)
from domains.citylearn import simulator as sim  # noqa: E402
from silr.verifier import SiLRVerifier, Verdict  # noqa: E402
from silr.training.reward import compute_grpo_reward, compute_scalar_reward  # noqa: E402

ADMISSIBLE = (Verdict.PASS, Verdict.SAFE_PROGRESS)
# import the shared helpers from the ANM script to stay identical
from scripts.anm_reward_landscape import _spearman, severity_scalar_reward  # noqa: E402


def _mgr(loader, sc):
    mgr = CityLearnManager(fixed_t=sc.fixed_t, initial_soc=sc.initial_soc,
                           initial_actions=sc.initial_actions,
                           peak_import_kw=sc.peak_import_kw)
    loader.setup_episode(mgr, sc)
    return mgr


def enumerate_actions(_mgr_unused):
    acts = []
    for b in range(sim.N_BUILDINGS):
        for p in sim.ACTIONS_PER_BUILDING:
            acts.append({"tool_name": "set_building_setpoint",
                         "params": {"building_index": int(b), "power_kw": float(p)}})
    return acts


def score_actions(mgr, verifier):
    rows = []
    for action in enumerate_actions(mgr):
        try:
            vr = verifier.verify(action)
        except Exception:  # noqa: BLE001
            continue
        rows.append({"action": action, "verdict": vr.verdict.value,
                     "admissible": vr.verdict in ADMISSIBLE,
                     "rD": compute_grpo_reward(vr), "rE": compute_scalar_reward(vr),
                     "rE2": severity_scalar_reward(vr)})
    return rows


def apply_action(mgr, action):
    create_citylearn_toolset(mgr).get(action["tool_name"]).execute(**action["params"])
    mgr.solve()
    return mgr.last_penalty


def multistep_value(loader, sc, first_action, base_pen, H=4):
    """base_pen - penalty after first_action then H oracle-greedy (min-penalty) steps."""
    mgr = _mgr(loader, sc)
    apply_action(mgr, first_action)
    for _ in range(H):
        if mgr.last_penalty < 1e-6:
            break
        # oracle-greedy: pick the single-building action that most reduces penalty,
        # evaluated on a shadow copy of the CURRENT state, then apply it for real.
        best_pen = mgr.last_penalty
        best_act = None
        for a in enumerate_actions(mgr):
            sh = mgr.create_shadow_copy()
            create_citylearn_toolset(sh).get(a["tool_name"]).execute(**a["params"])
            sh.solve()
            if sh.last_penalty < best_pen - 1e-9:
                best_pen = sh.last_penalty
                best_act = a
        if best_act is None:
            break
        apply_action(mgr, best_act)
    return round(base_pen - float(mgr.last_penalty), 4)


def greedy_rollout(loader, sc, cfg, reward_key, max_steps=8):
    """Greedily apply the argmax-`reward_key` admissible action each step; track
    penalty. Tests whether FOLLOWING a reward leads to recovery or a plateau (the
    behavioural product-order test: a cross-family Σσ scalar may clear the largest
    family while leaving a small-σ family that dooms recovery)."""
    mgr = _mgr(loader, sc)
    verifier = SiLRVerifier(mgr, domain_config=cfg)
    traj = [round(float(mgr.last_penalty), 4)]
    for _ in range(max_steps):
        rows = [r for r in score_actions(mgr, verifier) if r["admissible"]]
        if not rows:
            break
        best = max(rows, key=lambda r: r[reward_key])
        pen = apply_action(mgr, best["action"])
        traj.append(round(float(pen), 4))
        if pen < 1e-6:
            break
        if len(traj) >= 3 and abs(traj[-1] - traj[-2]) < 1e-6:
            break
    return {"penalty_traj": traj, "recovered": traj[-1] < 1e-6, "final_penalty": traj[-1]}


def confusion(rvals, values, vmax):
    n = len(rvals)
    out = {}
    for frac in (0.01, 0.05, 0.10):
        delta = frac * vmax
        pairs = conf = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(values[i] - values[j]) > delta:
                    pairs += 1
                    if abs(rvals[i] - rvals[j]) < 1e-9:
                        conf += 1
        out[f"{int(frac*100)}pct"] = round(conf / pairs, 3) if pairs else 0.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--out", default="figures/citylearn_reward_landscape.json")
    args = p.parse_args()
    cfg = build_citylearn_domain_config(with_observer=True, gating_policy="progress_mag")
    loader = CityLearnScenarioLoader()
    report = {}
    for sid in args.scenarios:
        try:
            sc = loader.load(sid)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {sid}: {e}")
            continue
        mgr = _mgr(loader, sc)
        verifier = SiLRVerifier(mgr, domain_config=cfg)
        base_pen = float(mgr.last_penalty)
        rows = [r for r in score_actions(mgr, verifier) if r["admissible"]]
        if len(rows) < 3:
            print(f"[skip] {sid}: only {len(rows)} admissible")
            continue
        rD = [r["rD"] for r in rows]; rE = [r["rE"] for r in rows]; rE2 = [r["rE2"] for r in rows]
        values = [multistep_value(loader, sc, r["action"], base_pen, args.horizon) for r in rows]
        vmax = max(values) if values else 0.0
        res = {
            "base_penalty": round(base_pen, 4), "n_admissible": len(rows),
            "multistep_horizon": args.horizon,
            "prm_fidelity_spearman": {
                "rD_perfamily_geom": _spearman(rD, values),
                "rE_count": _spearman(rE, values),
                "rE2_severity_scalar": _spearman(rE2, values)},
            "confusion": {
                "rD_perfamily_geom": confusion(rD, values, vmax),
                "rE_count": confusion(rE, values, vmax),
                "rE2_severity_scalar": confusion(rE2, values, vmax)},
        }
        # behavioural product-order test: greedy rollout under each reward
        res["greedy"] = {
            "rD_perfamily": greedy_rollout(loader, sc, cfg, "rD"),
            "rE2_severity_scalar": greedy_rollout(loader, sc, cfg, "rE2"),
            "rE_count": greedy_rollout(loader, sc, cfg, "rE")}
        report[sid] = res
        f = res["prm_fidelity_spearman"]; g = res["greedy"]
        print(f"=== {sid} (base_pen {base_pen:.2f}, {len(rows)} admissible) ===")
        print(f"  greedy-rollout recovery: perfamily-geom rD={g['rD_perfamily']['recovered']} "
              f"(traj {g['rD_perfamily']['penalty_traj']}) | severity-scalar rE2={g['rE2_severity_scalar']['recovered']} "
              f"(traj {g['rE2_severity_scalar']['penalty_traj']}) | count rE={g['rE_count']['recovered']} "
              f"(traj {g['rE_count']['penalty_traj']})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fo:
        json.dump(report, fo, indent=2)
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
