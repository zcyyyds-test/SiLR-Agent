"""Select a CityLearn multi-action band from the mined catalogue.

Reads ``mined_scenarios_citylearn.json`` (from ``citylearn_scenario_mine.py``)
and promotes a diverse band of ``multi_action`` scenarios to
``domains/citylearn/scenarios_mined.json``, which the scenario loader merges
into the curated library at import time (mirrors ANM's ``anm_select_mined.py``).

Selection priorities, in order:
  1. multi-action only (single-step-recoverable snapshots are excluded -- they
     do not exercise the multi-step recovery the gate is about);
  2. multi-TYPE first (>= 2 physically incomparable constraint families active)
     -- the regime where a scalar surrogate must collapse the product order and
     the geometric reward should separate most sharply;
  3. spread across hours and family combinations (dedup near-identical starts).

The downstream Step-0 LLM gate then trims this band to the *sub-saturated*
subset (Qwen3-8B recovery < reps) that actually carries a training signal.

Run from repo root:
    PYTHONPATH=. python scripts/citylearn_select_multi_action.py \\
        --catalogue mined_scenarios_citylearn.json --max-scenarios 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_FAMILY_TAG = {
    "soc_min": "smin",
    "soc_max": "smax",
    "import_limit": "imp",
    "export_limit": "exp",
}


def _family_tag(families: list[str]) -> str:
    return "-".join(_FAMILY_TAG.get(f, f) for f in families) or "none"


def _dedup_key(entry: dict[str, Any]) -> tuple:
    """Group near-identical starts: same hour + family signature + coarse SoC."""
    soc_bucket = tuple(round(x, 1) for x in entry["initial_soc"])
    return (entry["fixed_t"], tuple(sorted(entry["families"])), soc_bucket)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", default="mined_scenarios_citylearn.json")
    parser.add_argument("--output",
                        default="domains/citylearn/scenarios_mined.json")
    parser.add_argument("--max-scenarios", type=int, default=24)
    parser.add_argument("--multitype-only", action="store_true",
                        help="Select only multi-type (>=2 family) scenarios; "
                             "do not top up with single-family multi-action.")
    args = parser.parse_args()

    with open(args.catalogue) as f:
        data = json.load(f)
    multi = [e for e in data["catalogue"] if e.get("class") == "multi_action"]
    print(f"Loaded {len(data['catalogue'])} candidates; "
          f"{len(multi)} are multi_action.")

    # Priority: multi-type first, then more feasible recoveries (more headroom
    # for a monotone gated path), then deterministic tie-break.
    multi.sort(key=lambda e: (-e["n_families"], -e["n_feasible"],
                              e["fixed_t"], tuple(e["initial_actions"])))

    # Bucket by family signature so the band balances across *kinds* of
    # incomparability -- the rarer cross-family pairs (battery SoC x feeder
    # export) would otherwise be drowned out by the abundant SoC x SoC
    # antichains. Multi-type buckets (the amplification regime) are filled
    # first by round-robin; single-family multi-action scenarios only top up
    # any remaining slots for contrast.
    buckets: dict[tuple, list[dict[str, Any]]] = {}
    for e in multi:
        buckets.setdefault(tuple(sorted(e["families"])), []).append(e)
    multitype_sigs = sorted((k for k in buckets if len(k) > 1),
                            key=lambda k: (len(buckets[k]), k))
    single_sigs = sorted((k for k in buckets if len(k) == 1),
                         key=lambda k: (len(buckets[k]), k))

    selected: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    def _fill(sig_order: list[tuple]) -> None:
        progress = True
        while len(selected) < args.max_scenarios and progress:
            progress = False
            for sig in sig_order:
                while buckets[sig]:
                    e = buckets[sig].pop(0)
                    key = _dedup_key(e)
                    if key in seen:
                        continue
                    seen.add(key)
                    selected.append(e)
                    progress = True
                    break  # one per bucket per round
                if len(selected) >= args.max_scenarios:
                    return

    _fill(multitype_sigs)   # amplification regime first
    if not args.multitype_only:
        _fill(single_sigs)  # contrast top-up

    scenarios = []
    fam_tally: dict[str, int] = {}
    for i, e in enumerate(selected):
        tag = _family_tag(e["families"])
        fam_tally[tag] = fam_tally.get(tag, 0) + 1
        n_fam = e["n_families"]
        scenarios.append({
            "id": f"cl_mined_{i:03d}_t{e['fixed_t']}_{tag}",
            "fixed_t": e["fixed_t"],
            "initial_soc": e["initial_soc"],
            "initial_actions": e["initial_actions"],
            "peak_import_kw": 0.0,
            "difficulty": "hard" if n_fam > 1 else "medium",
            "source_seed": 0,
            "class": e["class"],
            "families": e["families"],
            "n_families": n_fam,
            "default_penalty": e["default_penalty"],
            "n_feasible": e["n_feasible"],
            "single_action_solvable": e["single_action_solvable"],
        })

    n_multitype = sum(1 for s in scenarios if s["n_families"] > 1)
    print(f"Selected {len(scenarios)} multi-action scenarios "
          f"({n_multitype} multi-type). Family tags: {fam_tally}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"scenarios": scenarios}, f, indent=2)
    print(f"  wrote {out}")
    print("  scenario ids:")
    for s in scenarios:
        print(f"    {s['id']}  fams={s['families']} "
              f"pen={s['default_penalty']} n_feasible={s['n_feasible']}")


if __name__ == "__main__":
    main()
