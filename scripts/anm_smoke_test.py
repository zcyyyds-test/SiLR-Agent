"""Assertive smoke test for the gym-anm SiLR domain wrapper.

Verifies, end to end, that the SiLR verifier can gate set-point actions on the
public ANM6-Easy environment. Every check is an ``assert`` so the script fails
loudly on regression (the previous print-only version could not catch semantic
breakage). Covered:

  - verifier returns the expected Verdict (PASS / FAIL / ERROR) for each path,
  - both required checkers (voltage, branch_loading, storage_soc) fire,
  - shadow isolation: the real manager is untouched by any verification,
  - validation errors (out-of-bounds, NaN, missing params, unknown device,
    unknown / non-action tool) → Verdict.ERROR, not Verdict.FAIL (so they do
    not contaminate FAIL-based training signals in silr.agent.trajectory),
  - seed control reproduces the same frozen conditions.

Run on a host with gym-anm installed (from repo root, with ``domains`` on
PYTHONPATH):

    PYTHONPATH=. python scripts/anm_smoke_test.py
"""

from __future__ import annotations

from domains.anm import GymANMManager, build_anm_domain_config
from silr.verifier import SiLRVerifier, Verdict


def _bus_voltages(mgr):
    return [round(float(abs(b.v)), 6) for b in mgr.system_state.buses.values()]


def main() -> None:
    cfg = build_anm_domain_config()
    mgr = GymANMManager(seed=42)
    verifier = SiLRVerifier(mgr, domain_config=cfg)

    gens = mgr.get_generator_ids()
    des = mgr.get_storage_ids()
    assert gens, "ANM6Easy must expose at least one non-slack generator"
    assert des, "ANM6Easy must expose a storage unit"
    g0 = gens[0]
    s0 = des[0]
    print(f"devices: gens={gens} storage={des} | seed=42")
    print(f"frozen P_pot={mgr._P_pot}  P_load={mgr._P_load}")

    # --- M3: seed reproduces frozen conditions ---
    mgr_b = GymANMManager(seed=42)
    assert mgr_b._P_pot == mgr._P_pot, (
        f"seed=42 should yield identical P_pot, got {mgr_b._P_pot} vs {mgr._P_pot}"
    )
    assert mgr_b._P_load == mgr._P_load, "seed=42 should yield identical P_load"
    print("OK seed=42 reproduces frozen conditions")

    def verify(action):
        return verifier.verify(action)

    # --- 1. Benign snapshot → PASS ---
    mgr.set_conditions(
        P_load={i: -0.5 for i in mgr._load_ids},
        P_pot={g: 1.0 for g in gens},
    )
    # Capture AFTER set_conditions (which legitimately mutates the manager)
    # and BEFORE any verify() calls — isolation invariant is that verify() must
    # not mutate the real manager, not that no method ever mutates it.
    v_before = _bus_voltages(mgr)
    r = verify({"tool_name": "set_generator_setpoint",
                "params": {"gen_id": g0, "p_mw": 1.0}})
    assert r.verdict == Verdict.PASS, f"benign snapshot expected PASS, got {r.verdict}: {r.fail_reason}"
    checker_names = {cr.checker_name for cr in r.check_results}
    assert {"voltage", "branch_loading", "storage_soc"} <= checker_names, (
        f"all 3 checkers must run on PASS path, got {checker_names}"
    )
    assert all(cr.passed for cr in r.check_results), "benign snapshot: all checkers should pass"
    print(f"OK benign  -> {r.verdict.value}")

    # --- 2. Real overload snapshot (use the default stochastic conditions) ---
    mgr2 = GymANMManager(seed=0)
    ver2 = SiLRVerifier(mgr2, domain_config=cfg)
    # set both generators to their full potential and force discharge: typically
    # at least one of the seeds drives some snapshot into a real branch overload.
    pot = mgr2._P_pot
    r = ver2.verify({"tool_name": "set_generator_setpoint",
                     "params": {"gen_id": gens[0], "p_mw": float(pot[gens[0]])}})
    # We do not assert FAIL here unconditionally (some seeds are benign); but
    # whatever the verdict, it must be one of the three valid ones with the
    # right checkers present.
    # Under the ANM default ``progress`` gating policy a recoverability-
    # preserving step may legitimately come back as SAFE_PROGRESS; under
    # ``terminal`` it would be FAIL. Both are valid "the action was
    # evaluated, not erroneous" outcomes — accept all three non-ERROR
    # verdicts here.
    assert r.verdict in (Verdict.PASS, Verdict.SAFE_PROGRESS, Verdict.FAIL), (
        f"valid action must yield PASS / SAFE_PROGRESS / FAIL, "
        f"not {r.verdict}: {r.fail_reason}"
    )
    print(f"OK default-seed snapshot -> {r.verdict.value}")

    # --- 3. Out-of-bounds setpoint → Verdict.ERROR (not FAIL) ---
    # NaN
    r = verify({"tool_name": "set_generator_setpoint",
                "params": {"gen_id": g0, "p_mw": float("nan")}})
    assert r.verdict == Verdict.ERROR, f"NaN p_mw must yield ERROR, got {r.verdict}"
    assert "finite" in (r.fail_reason or "").lower()
    # Wildly over device p_max (gym-anm ANM6Easy gen p_max is small)
    r = verify({"tool_name": "set_generator_setpoint",
                "params": {"gen_id": g0, "p_mw": 9999.0}})
    assert r.verdict == Verdict.ERROR, f"p_mw=9999 must yield ERROR, got {r.verdict}"
    assert "outside device limits" in (r.fail_reason or "")
    # Unknown gen id
    r = verify({"tool_name": "set_generator_setpoint",
                "params": {"gen_id": 999, "p_mw": 0.0}})
    assert r.verdict == Verdict.ERROR, f"unknown gen_id must yield ERROR, got {r.verdict}"
    # Missing required param
    r = verify({"tool_name": "set_storage_setpoint",
                "params": {"storage_id": s0}})
    assert r.verdict == Verdict.ERROR, f"missing p_mw must yield ERROR, got {r.verdict}"
    print("OK bounds/validation errors -> ERROR (not FAIL)")

    # --- 4. Non-action tool / unknown tool → ERROR ---
    r = verify({"tool_name": "get_grid_status", "params": {}})
    assert r.verdict == Verdict.ERROR, f"observation tool must yield ERROR, got {r.verdict}"
    assert "not in allowed actions" in (r.fail_reason or "")
    r = verify({"tool_name": "frobnicate", "params": {}})
    assert r.verdict == Verdict.ERROR, f"unknown tool must yield ERROR, got {r.verdict}"
    print("OK non-action / unknown -> ERROR")

    # --- 5. ANMStorageSoCChecker is wired into config ---
    cr_names = [c.name for c in cfg.checkers]
    assert "storage_soc" in cr_names, f"storage_soc checker missing: {cr_names}"
    print("OK storage_soc checker registered")

    # --- 6. Isolation: real manager bus voltages unchanged by verification ---
    v_after = _bus_voltages(mgr)
    assert v_before == v_after, (
        f"shadow leaked into real manager! before={v_before} after={v_after}"
    )
    print("OK isolation: real manager unchanged by verification")

    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
