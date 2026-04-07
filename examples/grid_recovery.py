"""Grid recovery example: demonstrate SiLR-verified power grid restoration.

This example shows a complete recovery episode on the IEEE 39-bus system.
It requires ANDES: pip install silr[grid]

Usage:
    python examples/grid_recovery.py
"""

from __future__ import annotations

import sys


def main():
    # Check ANDES availability
    try:
        import andes
    except ImportError:
        print("This example requires ANDES. Install with:")
        print("  pip install 'silr[grid]'")
        sys.exit(1)

    from domains.grid import SystemManager, build_grid_domain_config
    from domains.grid.scenarios import ScenarioLoader
    from silr.verifier import SiLRVerifier, Verdict

    print("=== SiLR Grid Recovery Demo ===\n")

    # 1. Load IEEE 39-bus system
    manager = SystemManager()
    raw = andes.get_case("ieee39/ieee39_full.xlsx")
    try:
        dyr = andes.get_case("ieee39/ieee39.dyr")
    except FileNotFoundError:
        dyr = None
    manager.load(raw, addfile=dyr)
    print(f"Loaded IEEE 39-bus: {manager.ss.Bus.n} buses, {manager.ss.Line.n} lines")

    # 2. Load and apply scenario
    loader = ScenarioLoader()
    scenario = loader.load("s01_single_line_trip")
    print(f"\nScenario: {scenario.name} ({scenario.difficulty})")
    print(f"  Prestress: load_scale={scenario.prestress.get('load_scale', 1.0)}")
    print(f"  Fault: {scenario.fault_sequence}")

    loader.setup_episode(manager, scenario)
    print("\nPost-fault system state applied.")

    # 3. Create verifier (PFlow-only for speed)
    domain_config = build_grid_domain_config(pflow_only=True)
    verifier = SiLRVerifier(manager, domain_config=domain_config)

    # 4. Verify a recovery action
    print("\n--- Verifying: adjust_gen(GENROU_1, delta_p_mw=-20) ---")
    result = verifier.verify(
        {"tool_name": "adjust_gen", "params": {"gen_id": "GENROU_1", "delta_p_mw": -20}},
    )
    print(f"Verdict: {result.verdict.value}")
    print(f"Solver converged: {result.solver_converged}")
    for cr in result.check_results:
        status = "PASS" if cr.passed else "FAIL"
        print(f"  {cr.checker_name}: {status}")
        for v in cr.violations:
            print(f"    - {v.detail}")

    if result.verdict == Verdict.PASS:
        # Apply verified action
        from domains.grid.tools import create_toolset
        tools = create_toolset(manager)
        tools["adjust_gen"].execute(gen_id="GENROU_1", delta_p_mw=-20)
        manager.solve()
        print("\n✓ Action applied and verified.")
    else:
        print("\n✗ Action rejected by SiLR verifier.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
