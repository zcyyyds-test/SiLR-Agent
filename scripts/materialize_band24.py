"""Materialize the full 24-scenario multi-action band (8 operating points x 3 SoC).

The existing scenarios_mined.json holds the 8 native-SoC operating points
(ids mined_multi_action_1..8). This appends the 16 non-native SoC variants
(near_min, near_max) with unique ids (= operating-point base id + _soc<pert>),
preserving the existing 8 ids untouched. Result: a 24-scenario multi-action band.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.anm_select_mined import _make_scenario

cat = json.load(open(ROOT/"mined_scenarios_v2.json"))["catalogue"]
ma = [r for r in cat if r.get("class") == "multi_action"]
# group by operating point, sorted exactly like _select_multi_action (so the
# native row's generated base id matches the existing mined_multi_action_{i}).
groups = {}
for r in ma:
    groups.setdefault((int(r["source_seed"]), float(r["load_mul"]), float(r["gen_mul"])), []).append(r)
order = sorted(groups)

reg_path = ROOT/"domains/anm/scenarios_mined.json"
reg = json.load(open(reg_path))
existing_ids = {s["id"] for s in reg["scenarios"]}
new_records = []
for idx, key in enumerate(order):
    for row in groups[key]:
        if row.get("soc_pert") == "native":
            continue  # already in registry as mined_multi_action_{idx+1}
        rec = _make_scenario(row, idx, "multi_action")  # base id w/o soc
        rec["id"] = f"{rec['id']}_soc{row['soc_pert']}"   # make unique
        if rec["id"] in existing_ids:
            continue
        new_records.append(rec)

n_ma = sum(1 for s in reg["scenarios"] if s.get("class") == "multi_action")
print("existing multi_action:", n_ma)
print("new SoC-variant records:", len(new_records), "(expect 16)")
for r in new_records:
    print("  +", r["id"])

if "--append" in sys.argv:
    add = [r for r in new_records if r["id"] not in existing_ids]
    reg["scenarios"].extend(add)
    json.dump(reg, open(reg_path, "w"), indent=2)
    total_ma = sum(1 for s in reg["scenarios"] if s.get("class") == "multi_action")
    print("APPENDED", len(add), "-> total scenarios", len(reg["scenarios"]), "multi_action", total_ma)
else:
    json.dump(new_records, open(ROOT/"_band24_new16.json", "w"), indent=2)
    print("dry-run only; re-run with --append to commit")
