"""Test ReActAgent with MockClient on the network domain."""

import pytest

from domains.network import NetworkManager, build_network_domain_config
from silr.agent.react_loop import ReActAgent
from silr.agent.config import AgentConfig
from silr.agent.types import StepOutcome
from silr.agent.llm.mock_client import MockClient
from silr.agent.llm.base import LLMResponse
from silr.verifier import SiLRVerifier


class TestReActAgentNetwork:
    def _make_agent(self, responses: list[LLMResponse]):
        manager = NetworkManager()
        manager.fail_link(1, 2)
        manager.solve()

        config = build_network_domain_config()
        verifier = SiLRVerifier(manager, domain_config=config)
        llm = MockClient(responses=responses)
        agent_config = AgentConfig(max_steps=3)

        return ReActAgent(
            manager=manager,
            verifier=verifier,
            llm_client=llm,
            config=agent_config,
            domain_config=config,
        )

    def test_successful_recovery(self):
        """Agent proposes restore_link → verified PASS → then 'none' to end."""
        responses = [
            LLMResponse(content=(
                'Thought: Link 1-2 is down, restore it.\n'
                '```json\n{"tool_name": "restore_link", "params": {"src": 1, "dst": 2}}\n```'
            )),
            LLMResponse(content=(
                'Thought: System looks stable now.\n'
                '```json\n{"tool_name": "none", "params": {}}\n```'
            )),
        ]
        agent = self._make_agent(responses)
        result = agent.run_episode(scenario_id="test")
        # Step 1: restore_link verified PASS and applied
        assert result.steps[0].outcome == StepOutcome.SUCCESS
        assert result.steps[0].applied_action["tool_name"] == "restore_link"
        assert result.steps[0].verification_results[0].verdict.value == "PASS"

    def test_none_action_terminates(self):
        """Agent says 'none' → episode ends."""
        responses = [
            LLMResponse(content=(
                'Thought: System looks fine.\n'
                '```json\n{"tool_name": "none", "params": {}}\n```'
            )),
        ]
        agent = self._make_agent(responses)
        result = agent.run_episode(scenario_id="test")
        assert result.total_steps >= 1

    def test_rejection_then_recovery(self):
        """First proposal rejected (unknown action) → second proposal passes."""
        responses = [
            # Step 1, proposal 1: invalid action → SiLR rejects
            LLMResponse(content=(
                'Thought: Try deleting the node.\n'
                '```json\n{"tool_name": "delete_node", "params": {"id": 1}}\n```'
            )),
            # Step 1, proposal 2: valid action → SiLR passes
            LLMResponse(content=(
                'Thought: Restore the failed link instead.\n'
                '```json\n{"tool_name": "restore_link", "params": {"src": 1, "dst": 2}}\n```'
            )),
            # Step 2: system stable
            LLMResponse(content=(
                'Thought: System recovered.\n'
                '```json\n{"tool_name": "none", "params": {}}\n```'
            )),
        ]
        agent = self._make_agent(responses)
        result = agent.run_episode(scenario_id="test")

        # First step: had a rejection then a success
        assert result.total_rejections >= 1
        assert result.steps[0].outcome == StepOutcome.SUCCESS
        assert result.steps[0].applied_action["tool_name"] == "restore_link"
        assert len(result.steps[0].verification_results) >= 2

    def test_max_steps_reached(self):
        """Agent exhausts max_steps without recovering."""
        # Always propose an invalid action
        bad_response = LLMResponse(content=(
            'Thought: Try something.\n'
            '```json\n{"tool_name": "delete_node", "params": {"id": 1}}\n```'
        ))
        responses = [bad_response] * 10  # more than max_steps * max_proposals
        agent = self._make_agent(responses)
        result = agent.run_episode(scenario_id="test")
        assert result.total_steps == 3  # max_steps from AgentConfig
