"""Test ActionParser with injectable allowed_actions."""

import pytest

from silr.agent.action_parser import ActionParser, ParseError
from silr.agent.llm.base import LLMResponse


@pytest.fixture
def parser():
    """Parser configured for the network domain."""
    return ActionParser(
        allowed_actions=frozenset(["restore_link", "reroute_traffic"]),
    )


@pytest.fixture
def grid_parser():
    """Parser configured like the grid domain (with aliases)."""
    return ActionParser(
        allowed_actions=frozenset(["adjust_gen", "shed_load", "trip_line", "close_line"]),
        aliases={"generator_adjust": "adjust_gen", "load_shed": "shed_load"},
        numeric_fields={"delta_p_mw", "amount_mw"},
    )


class TestActionParserJSON:
    def test_parse_json_block(self, parser):
        text = '''Thought: need to restore link
```json
{"tool_name": "restore_link", "params": {"src": 1, "dst": 2}}
```'''
        thought, action = parser.parse(LLMResponse(content=text))
        assert action["tool_name"] == "restore_link"
        assert action["params"]["src"] == 1

    def test_parse_none_action(self, parser):
        text = '''Thought: system is stable
```json
{"tool_name": "none", "params": {}}
```'''
        thought, action = parser.parse(LLMResponse(content=text))
        assert action["tool_name"] == "none"

    def test_parse_invalid_tool(self, parser):
        text = '''```json
{"tool_name": "delete_everything", "params": {}}
```'''
        thought, action = parser.parse(LLMResponse(content=text))
        # Should still parse but tool_name preserved as-is (verifier will reject)
        assert action["tool_name"] == "delete_everything"


class TestActionParserAliases:
    def test_alias_resolution(self, grid_parser):
        text = '''```json
{"tool_name": "generator_adjust", "params": {"gen_id": "GENROU_1", "delta_p_mw": 50}}
```'''
        thought, action = grid_parser.parse(LLMResponse(content=text))
        assert action["tool_name"] == "adjust_gen"

    def test_numeric_coercion(self, grid_parser):
        text = '''```json
{"tool_name": "adjust_gen", "params": {"gen_id": "GENROU_1", "delta_p_mw": "50"}}
```'''
        thought, action = grid_parser.parse(LLMResponse(content=text))
        assert isinstance(action["params"]["delta_p_mw"], float)
