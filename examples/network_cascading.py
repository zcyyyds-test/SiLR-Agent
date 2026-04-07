"""Multi-agent coordinator on cascading network faults.

Demonstrates the CoordinatorAgent dispatching connectivity and
utilization specialists to handle simultaneous link failure + overload.

Runs with MockClient — no real LLM needed.
"""

from domains.network import (
    NetworkManager,
    NetworkScenarioLoader,
    build_network_domain_config,
    build_connectivity_specialist_config,
    build_utilization_specialist_config,
)
from silr.agent.coordinator import (
    CoordinatorAgent,
    CoordinatorConfig,
    SpecialistSpec,
)
from silr.agent.llm.mock_client import MockClient
from silr.agent.llm.base import LLMResponse
from silr.verifier import SiLRVerifier


def main():
    # 1. Set up scenario: link 1-2 down + link 2-5 near overload
    manager = NetworkManager()
    loader = NetworkScenarioLoader()
    scenario = loader.load("cascade_easy")
    loader.setup_episode(manager, scenario)

    full_config = build_network_domain_config()
    verifier = SiLRVerifier(manager, domain_config=full_config)

    # 2. Define specialists
    specialists = [
        SpecialistSpec(
            name="connectivity",
            domain_config=build_connectivity_specialist_config(),
        ),
        SpecialistSpec(
            name="utilization",
            domain_config=build_utilization_specialist_config(),
        ),
    ]

    # 3. Mock LLM responses
    coordinator_llm = MockClient(responses=[
        LLMResponse(content='{"specialist": "connectivity", "reason": "Link 1-2 is down"}'),
        LLMResponse(content='{"specialist": "utilization", "reason": "Link 2-5 near overload"}'),
        LLMResponse(content='{"action": "done", "reason": "All constraints satisfied"}'),
    ])

    specialist_llm = MockClient(responses=[
        # Connectivity specialist
        LLMResponse(content=(
            'Thought: Restore the failed link.\n'
            '```json\n{"tool_name": "restore_link", "params": {"src": 1, "dst": 2}}\n```'
        )),
        LLMResponse(content='Thought: Done.\n```json\n{"tool_name": "none", "params": {}}\n```'),
        # Utilization specialist
        LLMResponse(content=(
            'Thought: Reroute traffic away from overloaded link.\n'
            '```json\n{"tool_name": "reroute_traffic", "params": {"src": 2, "dst": 5, "amount_mbps": 10}}\n```'
        )),
        LLMResponse(content='Thought: Done.\n```json\n{"tool_name": "none", "params": {}}\n```'),
    ])

    # 4. Run coordinator
    coordinator = CoordinatorAgent(
        manager=manager,
        verifier=verifier,
        llm_client=coordinator_llm,
        specialists=specialists,
        full_domain_config=full_config,
        config=CoordinatorConfig(max_rounds=4, max_specialist_steps=3),
        specialist_llm_client=specialist_llm,
    )

    result = coordinator.run_episode(scenario_id="cascade_easy")

    # 5. Print results
    print(f"Scenario: {result.scenario_id}")
    print(f"Recovered: {result.recovered}")
    print(f"Rounds: {result.total_rounds}")
    print(f"Conflicts: {result.conflict_count}")
    print()
    for a in result.activations:
        print(
            f"  Round {a.round_number}: {a.specialist_name} "
            f"(improved: {a.constraints_improved}, worsened: {a.constraints_worsened})"
        )


if __name__ == "__main__":
    main()
