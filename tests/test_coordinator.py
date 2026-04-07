"""Test CoordinatorAgent with MockClient on the network domain."""

import pytest

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


def _make_coordinator(
    coordinator_responses,
    specialist_responses,
    scenario_id="cascade_easy",
    config=None,
):
    """Helper: set up coordinator with mock LLMs on a cascading scenario."""
    mgr = NetworkManager()
    loader = NetworkScenarioLoader()
    scenario = loader.load(scenario_id)
    loader.setup_episode(mgr, scenario)

    full_config = build_network_domain_config(with_observer=True)
    verifier = SiLRVerifier(mgr, domain_config=full_config)

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

    return CoordinatorAgent(
        manager=mgr,
        verifier=verifier,
        llm_client=MockClient(responses=coordinator_responses),
        specialists=specialists,
        full_domain_config=full_config,
        config=config or CoordinatorConfig(max_rounds=4, max_specialist_steps=3),
        specialist_llm_client=MockClient(responses=specialist_responses),
    )


class TestCoordinatorDispatch:
    def test_single_specialist_recovery(self):
        """One fault -> dispatch connectivity -> recovered."""
        # Setup: cascade_easy has link 1-2 down + link 2-5 near overload
        coordinator_responses = [
            # Round 1: dispatch connectivity
            LLMResponse(content='{"specialist": "connectivity", "reason": "link down"}'),
            # Round 2: (won't be called if system is stable, but need for utilization)
            LLMResponse(content='{"specialist": "utilization", "reason": "overload"}'),
            LLMResponse(content='{"action": "done", "reason": "stable"}'),
        ]
        specialist_responses = [
            # Connectivity specialist: restore link 1-2
            LLMResponse(content=(
                'Thought: Restore link 1-2.\n'
                '```json\n{"tool_name": "restore_link", "params": {"src": 1, "dst": 2}}\n```'
            )),
            LLMResponse(content=(
                'Thought: Done.\n'
                '```json\n{"tool_name": "none", "params": {}}\n```'
            )),
            # Utilization specialist: reroute traffic
            LLMResponse(content=(
                'Thought: Reroute traffic from 2-5.\n'
                '```json\n{"tool_name": "reroute_traffic", "params": {"src": 2, "dst": 5, "amount_mbps": 10}}\n```'
            )),
            LLMResponse(content=(
                'Thought: Done.\n'
                '```json\n{"tool_name": "none", "params": {}}\n```'
            )),
        ]

        agent = _make_coordinator(coordinator_responses, specialist_responses)
        result = agent.run_episode(scenario_id="test_single")

        assert result.total_rounds >= 1
        assert result.activations[0].specialist_name == "connectivity"

    def test_two_specialist_sequential(self):
        """Custom scenario: link failure + heavy demand causing persistent overload."""
        # Manually set up: fail link 1-2 + add heavy demand so pflow creates overload
        mgr = NetworkManager()
        mgr.fail_link(1, 2)
        # Add extra demand that stresses link 2-5 after rerouting
        mgr._demands[(1, 5)] = 40  # increase from 20 to 40
        mgr._demands[(4, 2)] = 25  # increase from 10 to 25
        mgr.solve()

        full_config = build_network_domain_config(with_observer=True)
        verifier = SiLRVerifier(mgr, domain_config=full_config)

        specialists = [
            SpecialistSpec(name="connectivity",
                           domain_config=build_connectivity_specialist_config()),
            SpecialistSpec(name="utilization",
                           domain_config=build_utilization_specialist_config()),
        ]

        coordinator_responses = [
            LLMResponse(content='{"specialist": "connectivity", "reason": "link down"}'),
            LLMResponse(content='{"specialist": "utilization", "reason": "overload"}'),
            LLMResponse(content='{"action": "done", "reason": "stable"}'),
        ]
        specialist_responses = [
            # Round 1: restore link 1-2
            LLMResponse(content=(
                'Thought: Restore.\n'
                '```json\n{"tool_name": "restore_link", "params": {"src": 1, "dst": 2}}\n```'
            )),
            LLMResponse(content='Thought: Done.\n```json\n{"tool_name": "none", "params": {}}\n```'),
            # Round 2: reroute traffic
            LLMResponse(content=(
                'Thought: Reroute.\n'
                '```json\n{"tool_name": "reroute_traffic", "params": {"src": 2, "dst": 5, "amount_mbps": 10}}\n```'
            )),
            LLMResponse(content='Thought: Done.\n```json\n{"tool_name": "none", "params": {}}\n```'),
        ]

        agent = CoordinatorAgent(
            manager=mgr,
            verifier=verifier,
            llm_client=MockClient(responses=coordinator_responses),
            specialists=specialists,
            full_domain_config=full_config,
            config=CoordinatorConfig(max_rounds=4, max_specialist_steps=3),
            specialist_llm_client=MockClient(responses=specialist_responses),
        )
        result = agent.run_episode(scenario_id="test_two")

        # Should need at least 1 round (connectivity fix)
        assert result.total_rounds >= 1
        assert result.activations[0].specialist_name == "connectivity"

    def test_already_stable(self):
        """No faults -> coordinator should return immediately."""
        mgr = NetworkManager()
        mgr.solve()

        full_config = build_network_domain_config(with_observer=True)
        verifier = SiLRVerifier(mgr, domain_config=full_config)

        agent = CoordinatorAgent(
            manager=mgr,
            verifier=verifier,
            llm_client=MockClient(responses=[]),
            specialists=[
                SpecialistSpec(name="conn", domain_config=build_connectivity_specialist_config()),
            ],
            full_domain_config=full_config,
        )
        result = agent.run_episode(scenario_id="stable")
        assert result.recovered is True
        assert result.total_rounds == 0

    def test_max_rounds_reached(self):
        """Coordinator exhausts max_rounds."""
        # Always dispatch connectivity, but specialist proposes invalid actions
        coordinator_responses = [
            LLMResponse(content='{"specialist": "connectivity", "reason": "try"}'),
        ] * 5
        specialist_responses = [
            LLMResponse(content='```json\n{"tool_name": "delete_node", "params": {"id": 1}}\n```'),
        ] * 20

        agent = _make_coordinator(
            coordinator_responses, specialist_responses,
            config=CoordinatorConfig(max_rounds=3, max_specialist_steps=2),
        )
        result = agent.run_episode(scenario_id="test_max")
        assert result.total_rounds <= 3

    def test_unknown_specialist_stops(self):
        """Coordinator requests nonexistent specialist -> error."""
        coordinator_responses = [
            LLMResponse(content='{"specialist": "thermal", "reason": "heat"}'),
        ]
        agent = _make_coordinator(coordinator_responses, [])
        result = agent.run_episode(scenario_id="test_unknown")
        assert result.error is not None
        assert "thermal" in result.error

    def test_coordinator_done_action(self):
        """Coordinator says done -> episode ends."""
        coordinator_responses = [
            LLMResponse(content='{"action": "done", "reason": "no improvement possible"}'),
        ]
        agent = _make_coordinator(coordinator_responses, [])
        result = agent.run_episode(scenario_id="test_done")
        assert result.total_rounds == 0


class TestConstraintChangeDetection:
    def test_detect_improvement(self):
        """After restoring a link, connectivity should improve."""
        coordinator_responses = [
            LLMResponse(content='{"specialist": "connectivity", "reason": "fix"}'),
            LLMResponse(content='{"action": "done", "reason": "check"}'),
        ]
        specialist_responses = [
            # cascade_medium: links 2-3 and 3-5 are down (node 3 isolated)
            # Restoring 2-3 reconnects node 3 via path 2-3
            LLMResponse(content=(
                'Thought: Restore link 2-3.\n'
                '```json\n{"tool_name": "restore_link", "params": {"src": 2, "dst": 3}}\n```'
            )),
            LLMResponse(content='Thought: Done.\n```json\n{"tool_name": "none", "params": {}}\n```'),
        ]

        agent = _make_coordinator(
            coordinator_responses, specialist_responses,
            scenario_id="cascade_medium",
        )
        result = agent.run_episode(scenario_id="test_improve")

        # First activation should show connectivity improved
        assert len(result.activations) >= 1
        first = result.activations[0]
        assert "connectivity" in first.constraints_improved
