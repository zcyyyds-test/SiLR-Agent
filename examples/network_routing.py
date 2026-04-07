"""Network routing example: demonstrate SiLR verification on a toy network.

This example shows how the SiLR verifier protects a 5-node network domain
from actions that would violate safety constraints (link overload, connectivity).

No external dependencies required — runs with pure Python.
"""

from domains.network import NetworkManager, build_network_domain_config
from silr.verifier import SiLRVerifier, Verdict


def main():
    # 1. Create domain components
    manager = NetworkManager()
    domain_config = build_network_domain_config()

    print("=== SiLR Network Routing Demo ===\n")
    print(f"Topology: {len(manager.get_node_ids())} nodes, "
          f"{len(manager.get_link_ids())} links\n")

    # 2. Inject a fault: fail link 1-2
    print("--- Injecting fault: link 1-2 down ---")
    manager.fail_link(1, 2)
    manager.solve()

    # 3. Create verifier
    verifier = SiLRVerifier(manager, domain_config=domain_config)

    # 4. Verify a good action: restore the failed link
    print("\n--- Verifying action: restore_link(src=1, dst=2) ---")
    result = verifier.verify(
        {"tool_name": "restore_link", "params": {"src": 1, "dst": 2}},
    )
    print(f"Verdict: {result.verdict.value}")
    print(f"Solver converged: {result.solver_converged}")
    for cr in result.check_results:
        print(f"  {cr.checker_name}: {'PASS' if cr.passed else 'FAIL'} "
              f"({cr.summary})")

    if result.verdict == Verdict.PASS:
        # Apply the verified action
        manager.restore_link(1, 2)
        manager.solve()
        print("\n✓ Action applied successfully.")

    # 5. Try a reroute that might overload
    print("\n--- Verifying action: reroute_traffic(4→5, 50 Mbps) ---")
    result2 = verifier.verify(
        {"tool_name": "reroute_traffic", "params": {"src": 4, "dst": 5, "amount_mbps": 50}},
    )
    print(f"Verdict: {result2.verdict.value}")
    if result2.verdict == Verdict.FAIL:
        print("Violations found:")
        for cr in result2.check_results:
            for v in cr.violations:
                print(f"  - {v.detail}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
