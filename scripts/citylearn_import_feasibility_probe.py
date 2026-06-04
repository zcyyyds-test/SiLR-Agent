"""0-GPU probe: can the 3-building CityLearn district physically produce an
``import_limit`` violation? Decides how many constraint families the multi-type
amplification experiment can actually exercise (mirror of the ANM voltage probe).

Exhaustively sweeps every hour x every per-building SoC x every joint set-point
in the discrete action space and records the maximum district import seen. If
that maximum stays below the 16 kW import limit, the family is physically
unreachable -- the reachable set is {soc_min, soc_max, export_limit}, three
families, not four. This is the honest scoping fact behind the CityLearn
multi-type band (cf. the ANM voltage feasibility result).

Run from repo root (CPU-only):
    PYTHONPATH=. python scripts/citylearn_import_feasibility_probe.py
"""

from __future__ import annotations

import itertools
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.citylearn import simulator as sim
from domains.citylearn.manager import CityLearnManager
from domains.citylearn.checkers import (
    DistrictImportChecker,
    DistrictExportChecker,
    SoCChecker,
)

ACTIONS = sim.ACTIONS_PER_BUILDING
JOINT = list(itertools.product(ACTIONS, repeat=sim.N_BUILDINGS))
# Cover the full SoC band per building at fine resolution.
SOC_FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)

imp_chk = DistrictImportChecker()
exp_chk = DistrictExportChecker()
soc_chk = SoCChecker()

print(f"import limit  = {sim.DISTRICT_IMPORT_LIMIT_KW} kW")
print(f"export limit  = {sim.DISTRICT_EXPORT_LIMIT_KW} kW")
print(f"sweep: 24 hours x {len(SOC_FRACS)**sim.N_BUILDINGS} SoC configs x "
      f"{len(JOINT)} joint actions = "
      f"{24 * len(SOC_FRACS)**sim.N_BUILDINGS * len(JOINT)} states")

max_import = 0.0
max_export = 0.0
import_viol_reachable = False
reachable_families: set[str] = set()

for t in range(24):
    for fracs in itertools.product(SOC_FRACS, repeat=sim.N_BUILDINGS):
        soc = tuple(
            sim.SOC_MIN_KWH[b] + f * (sim.SOC_MAX_KWH[b] - sim.SOC_MIN_KWH[b])
            for b, f in enumerate(fracs)
        )
        for action in JOINT:
            mgr = CityLearnManager(fixed_t=t, initial_soc=soc, initial_actions=action)
            st = mgr.system_state
            max_import = max(max_import, st["district_import_kw"])
            max_export = max(max_export, st["district_export_kw"])
            for chk in (soc_chk, imp_chk, exp_chk):
                for v in chk.check(st, mgr.base_mva).violations:
                    reachable_families.add(v.constraint_type)
                    if v.constraint_type == "import_limit":
                        import_viol_reachable = True

print("\n=== SUMMARY ===")
print(f"max district import observed = {max_import:.3f} kW "
      f"(limit {sim.DISTRICT_IMPORT_LIMIT_KW}) -> "
      f"{'REACHABLE' if import_viol_reachable else 'UNREACHABLE'}")
print(f"max district export observed = {max_export:.3f} kW "
      f"(limit {sim.DISTRICT_EXPORT_LIMIT_KW})")
print(f"physically reachable families = {sorted(reachable_families)}")
print(f"=> CityLearn exercises {len(reachable_families)} constraint families "
      f"(import_limit is {'reachable' if import_viol_reachable else 'NOT reachable'}).")
