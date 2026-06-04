"""0-GPU probe: can ANM6-Easy physically produce (recoverable) voltage violations?
Decides whether multi-type (voltage+branch_loading) experiments use ANM or CityLearn.
Sweeps extreme load/gen scalings, solves power flow, runs the voltage checker."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from domains.anm import ANMScenarioLoader, GymANMManager, build_anm_domain_config
from domains.anm.checkers import ANMVoltageChecker, ANMBranchLoadingChecker

cfg = build_anm_domain_config(with_observer=True, gating_policy="progress_mag")
loader = ANMScenarioLoader()
sc = loader.load("mined_multi_action_1_l0p25g1p0_s5")
mgr = GymANMManager(seed=sc.source_seed or 42)
loader.setup_episode(mgr, sc)
P_load0 = dict(mgr._P_load); P_pot0 = dict(mgr._P_pot)
vchk = ANMVoltageChecker(); lchk = ANMBranchLoadingChecker()
print("baseline P_load=%s P_pot=%s" % (P_load0, P_pot0))
# bus v_min/v_max
sim = mgr.system_state
print("bus limits:", {b: (round(float(bus.v_min),3), round(float(bus.v_max),3)) for b,bus in sim.buses.items()})
worst_low = 9; worst_high = 0; any_v_viol = False
for lm, gm in itertools.product([0.05,0.2,0.5,1,2,4,8,16], [0.0,0.05,0.5,1,4,16]):
    Pl = {k: v*lm for k,v in P_load0.items()}
    Pp = {k: v*gm for k,v in P_pot0.items()}
    try:
        ok = mgr.set_conditions(Pl, Pp)
    except Exception as e:
        continue
    cr = vchk.check(mgr.system_state, mgr.base_mva)
    lr = lchk.check(mgr.system_state, mgr.base_mva)
    s = cr.summary
    if s.get("min_pu") is not None:
        worst_low = min(worst_low, s["min_pu"]); worst_high = max(worst_high, s["max_pu"])
    if cr.violations:
        any_v_viol = True
        print("  V-VIOL load*%.2f gen*%.2f: min=%s max=%s nV=%d | loading nV=%d converged=%s" % (
            lm, gm, s.get("min_pu"), s.get("max_pu"), s["n_violations"], lr.summary.get("n_violations"), ok))
print("=== SUMMARY ===")
print("voltage range observed across sweep: [%.4f, %.4f]" % (worst_low, worst_high))
print("ANY voltage violation reachable: %s" % any_v_viol)
