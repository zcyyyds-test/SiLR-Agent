"""Demo: GPU cluster scheduling with SiLR verification.

Shows the domain working end-to-end:
1. Create cluster + apply node failure scenario
2. Verify a scheduling action via SiLR
3. Observer produces LLM-readable state
"""

from domains.cluster import (
    ClusterManager,
    build_cluster_domain_config,
    ClusterScenarioLoader,
)
from silr.verifier import SiLRVerifier


def main():
    # Setup
    mgr = ClusterManager()
    config = build_cluster_domain_config()
    loader = ClusterScenarioLoader()

    # Apply scenario: single node failure
    scenario = loader.load("node_failure_single")
    loader.setup_episode(mgr, scenario)
    print(f"Scenario: {scenario.description}")
    print(f"Queued jobs: {len(mgr.get_queued_jobs())}")

    # Create verifier
    verifier = SiLRVerifier(mgr, domain_config=config)

    # Try assigning a queued job
    queued = mgr.get_queued_jobs()
    schedulable = mgr.get_schedulable_nodes()
    if queued and schedulable:
        action = {
            "tool_name": "assign_job",
            "params": {"job_id": queued[0], "node_id": schedulable[0]},
        }
        result = verifier.verify(action)
        print(f"\nAction: assign {queued[0]} -> {schedulable[0]}")
        print(f"Verdict: {result.verdict.value}")
        if result.check_results:
            for cr in result.check_results:
                status = "PASS" if cr.passed else "FAIL"
                print(f"  {cr.checker_name}: {status}")

    # Observer output
    observer = config.create_observer(mgr)
    obs = observer.observe()
    print(f"\nSystem stable: {obs.is_stable}")
    print(f"Violations: {len(obs.violations)}")
    print(f"Compressed (first 200 chars): {obs.compressed_json[:200]}...")


if __name__ == "__main__":
    main()
