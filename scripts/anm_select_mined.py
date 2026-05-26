"""Pick a paper-quality 9-scenario extension from the v2 mining catalogue.

Inputs: ``mined_scenarios_v2.json`` (output of ``anm_scenario_mine_v2.py``).
Output: ``domains/anm/scenarios_mined.json`` (small JSON consumed by
``ANMScenarioLoader``) — 3 single_action / 3 multi_action / 3 mpc_unsolved
entries chosen for stress diversity across the (default_penalty, load_mul,
gen_mul) dimensions.

Selection policy (paper rationale):
  - single_action: low / mid / high default penalty quartiles → covers the
    trivial-recovery baseline well without overweighting one stress level.
  - multi_action: 3 entries with the most distinct (load_mul, gen_mul)
    combinations → demonstrates that the terminal-gating deadlock is a
    structural property, not a one-snapshot artifact.
  - mpc_unsolved: low / mid / high default-violation count → spans the
    unrecoverable boundary (paper limitation showcase + verifier honesty
    test under stress beyond MPC's reach).

Run:
    PYTHONPATH=. python scripts/anm_select_mined.py \\
        --input mined_scenarios_v2.json \\
        --output domains/anm/scenarios_mined.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _bucket(entries, key_fn, k):
    """Pick ``k`` entries spread across the key_fn-induced range."""
    sorted_ents = sorted(entries, key=key_fn)
    if len(sorted_ents) <= k:
        return sorted_ents
    # Pick at quantile positions 0, 1/(k-1), ..., 1.
    out = []
    for i in range(k):
        idx = round(i * (len(sorted_ents) - 1) / max(k - 1, 1))
        out.append(sorted_ents[idx])
    return out


def _diverse_by_conditions(entries, k):
    """Pick ``k`` entries with distinct (source_seed, load_mul, gen_mul) tuples.

    SoC perturbation is now serialized separately as ``initial_soc``; this
    helper still spreads by operating-point conditions so the selected
    multi-action cases do not collapse onto one load/generation snapshot.
    Returns fewer than ``k`` entries if not enough distinct combinations exist.
    """
    seen = set()
    diverse = []
    for e in entries:
        key = (e["source_seed"], e["load_mul"], e["gen_mul"])
        if key in seen:
            continue
        seen.add(key)
        diverse.append(e)
        if len(diverse) == k:
            return diverse
    return diverse


def _reconstruct_conditions(entry) -> tuple[dict, dict]:
    """Re-derive ``(P_load, P_pot)`` from the (seed, load_mul, gen_mul) tuple.
    v2 catalogue entries don't snapshot the conditions inline to keep the
    JSON small; we reconstruct deterministically by replaying the same
    seeded ``GymANMManager`` initialisation."""
    from domains.anm import GymANMManager  # noqa: WPS433 (local import — runtime-only)

    mgr = GymANMManager(seed=entry["source_seed"])
    P_load = {int(k): float(v) * float(entry["load_mul"])
              for k, v in mgr._P_load.items()}
    P_pot = {int(k): float(v) * float(entry["gen_mul"])
             for k, v in mgr._P_pot.items()}
    return P_load, P_pot


def _initial_soc(entry) -> dict[int, float] | None:
    """Materialize the SoC perturbation used by mining.

    The selected scenario JSON must contain the concrete SoC value, not just
    ``soc_pert`` prose, so replay through ``ANMScenarioLoader`` uses the same
    state that the mining classifier saw.
    """
    perturb = entry.get("soc_pert", "native")

    from domains.anm import GymANMManager  # noqa: WPS433 (runtime-only)

    mgr = GymANMManager(seed=entry["source_seed"])
    out: dict[int, float] = {}
    for sid in mgr._des_ids:
        dev = mgr._sim.devices[sid]
        if perturb == "native":
            val = float(dev.soc)
        elif perturb == "near_min":
            val = float(dev.soc_min) + 0.01 * (float(dev.soc_max) - float(dev.soc_min))
        elif perturb == "near_max":
            val = float(dev.soc_max) - 0.01 * (float(dev.soc_max) - float(dev.soc_min))
        else:
            raise ValueError(f"unknown soc perturbation: {perturb}")
        out[int(sid)] = val
    return out


def _make_scenario(entry, idx_in_class: int, class_name: str) -> dict:
    """Convert a mined entry to a scenarios-library JSON record."""
    sid = (
        f"mined_{class_name}_{idx_in_class+1}"
        f"_l{entry['load_mul']}g{entry['gen_mul']}"
        f"_s{entry['source_seed']}"
    ).replace(".", "p")
    desc_parts = [
        f"Mined v2: seed={entry['source_seed']}, load×{entry['load_mul']}, "
        f"gen×{entry['gen_mul']}, soc={entry['soc_pert']}.",
    ]
    d = entry["default"]
    desc_parts.append(
        f"Default state: {d['total']} violations "
        f"({d['by_checker']}), penalty {d['penalty']:.2f}."
    )
    if entry.get("mpc"):
        desc_parts.append(
            f"MPC: recovered={entry['mpc']['recovered']}, "
            f"post-violations={entry['mpc']['post_violations']}, "
            f"penalty={entry['mpc']['penalty']:.2f}."
        )
    if "single_action_solvable" in entry and entry["single_action_solvable"] is not None:
        desc_parts.append(f"Single-action solvable: {entry['single_action_solvable']}.")

    difficulty_map = {
        "single_action": "easy",
        "multi_action": "medium",
        "mpc_unsolved": "hard",
    }

    P_load, P_pot = _reconstruct_conditions(entry)

    record = {
        "id": sid,
        "description": " ".join(desc_parts),
        "difficulty": difficulty_map[class_name],
        "class": class_name,
        "P_load": P_load,
        "P_pot": P_pot,
        "source_seed": entry["source_seed"],
        "load_mul": entry["load_mul"],
        "gen_mul": entry["gen_mul"],
        "soc_pert": entry["soc_pert"],
        "default_penalty": entry["default"]["penalty"],
        "default_violation_count": entry["default"]["total"],
        "mpc_recovered": entry.get("mpc", {}).get("recovered") if entry.get("mpc") else None,
        "mpc_penalty": entry.get("mpc", {}).get("penalty") if entry.get("mpc") else None,
        "mpc_post_violations": (
            entry.get("mpc", {}).get("post_violations") if entry.get("mpc") else None
        ),
        "single_action_solvable": entry.get("single_action_solvable"),
    }
    initial_soc = _initial_soc(entry)
    record["initial_soc"] = initial_soc
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="mined_scenarios_v2.json")
    parser.add_argument("--output", default="domains/anm/scenarios_mined.json")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    by_class: dict[str, list] = {}
    for e in data["catalogue"]:
        by_class.setdefault(e["class"], []).append(e)
    print(f"Loaded {len(data['catalogue'])} candidates: "
          f"{ {k: len(v) for k, v in by_class.items()} }")

    selections = {}

    # single_action: low / mid / high default penalty
    if by_class.get("single_action"):
        picks = _bucket(by_class["single_action"],
                        key_fn=lambda e: e["default"]["penalty"], k=3)
        selections["single_action"] = picks

    # multi_action: diverse load_mul/gen_mul combinations (only 24 entries
    # in v2 so we want spread, not buckets)
    if by_class.get("multi_action"):
        picks = _diverse_by_conditions(by_class["multi_action"], k=3)
        selections["multi_action"] = picks

    # mpc_unsolved: low / mid / high default violation count
    if by_class.get("mpc_unsolved"):
        picks = _bucket(by_class["mpc_unsolved"],
                        key_fn=lambda e: e["default"]["total"], k=3)
        selections["mpc_unsolved"] = picks

    output_records = []
    for class_name, entries in selections.items():
        for i, entry in enumerate(entries):
            rec = _make_scenario(entry, i, class_name)
            output_records.append(rec)
            print(f"  [{class_name}] {rec['id']:<55} "
                  f"viol={rec['default_violation_count']} "
                  f"pen={rec['default_penalty']:.2f} "
                  f"mpc_rec={rec['mpc_recovered']} "
                  f"single={rec['single_action_solvable']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"scenarios": output_records}, f, indent=2)
    print(f"\nWrote {len(output_records)} selected scenarios to {out_path}")


if __name__ == "__main__":
    main()
