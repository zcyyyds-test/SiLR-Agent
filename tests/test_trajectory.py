"""Test trajectory recording and SFT/DPO export."""

import pytest

from silr.agent.trajectory import TrajectoryRecorder, _clean_thought
from silr.agent.types import (
    EpisodeResult, Observation, StepOutcome, StepRecord,
)
from silr.verifier.types import Verdict, VerificationResult


def _obs(stable=False):
    return Observation(
        raw={}, compressed_json='{"status": "test"}',
        violations=[], is_stable=stable,
    )


def _vr(verdict):
    return VerificationResult(
        verdict=verdict,
        action={"tool_name": "test", "params": {}},
    )


def _step(num, outcome, action=None, thought="think", vrs=None):
    return StepRecord(
        step_number=num,
        observation=_obs(),
        thought=thought,
        proposed_actions=[action] if action else [],
        verification_results=vrs or [],
        applied_action=action,
        outcome=outcome,
    )


class TestCleanThought:
    def test_strips_thought_prefix(self):
        assert _clean_thought("Thought: hello") == "hello"

    def test_strips_bold_prefix(self):
        assert _clean_thought("**Thought**: hello") == "hello"

    def test_strips_double_prefix(self):
        assert _clean_thought("Thought: Thought: hello") == "hello"

    def test_empty_string(self):
        assert _clean_thought("") == ""

    def test_no_prefix(self):
        assert _clean_thought("just text") == "just text"


class TestTrajectoryRecorder:
    def test_record_and_count(self):
        rec = TrajectoryRecorder()
        ep = EpisodeResult(scenario_id="s1")
        rec.record_episode(ep)
        assert rec.episode_count == 1

    def test_sft_export_recovered_only(self):
        rec = TrajectoryRecorder()
        action = {"tool_name": "restore_link", "params": {"src": 1, "dst": 2}}

        good = EpisodeResult(scenario_id="good", recovered=True, total_steps=1)
        good.steps = [_step(1, StepOutcome.SUCCESS, action)]

        bad = EpisodeResult(scenario_id="bad", recovered=False, total_steps=1)
        bad.steps = [_step(1, StepOutcome.FAIL_VERIFY)]

        rec.record_episode(good)
        rec.record_episode(bad)

        sft = rec.export_sft_data()
        assert len(sft) == 1
        assert sft[0]["scenario_id"] == "good"

    def test_sft_export_quality_filter(self):
        rec = TrajectoryRecorder(quality_filter=lambda ep: True)
        action = {"tool_name": "test", "params": {}}
        ep = EpisodeResult(scenario_id="filtered", recovered=True, total_steps=1)
        ep.steps = [_step(1, StepOutcome.SUCCESS, action)]
        rec.record_episode(ep)

        sft = rec.export_sft_data()
        assert len(sft) == 0  # filtered out

    def test_sft_message_format(self):
        rec = TrajectoryRecorder()
        action = {"tool_name": "restore_link", "params": {"src": 1}}
        ep = EpisodeResult(scenario_id="s1", recovered=True, total_steps=1)
        ep.steps = [_step(1, StepOutcome.SUCCESS, action, thought="fix link")]
        rec.record_episode(ep)

        sft = rec.export_sft_data()
        messages = sft[0]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "Thought:" in messages[1]["content"]

    def test_dpo_export_needs_pass_and_fail(self):
        rec = TrajectoryRecorder()
        bad_action = {"tool_name": "bad", "params": {}}
        good_action = {"tool_name": "good", "params": {}}

        step = StepRecord(
            step_number=1,
            observation=_obs(),
            thought="try",
            proposed_actions=[bad_action, good_action],
            verification_results=[_vr(Verdict.FAIL), _vr(Verdict.PASS)],
            applied_action=good_action,
            outcome=StepOutcome.SUCCESS,
        )
        ep = EpisodeResult(scenario_id="s1", recovered=True, total_steps=1)
        ep.steps = [step]
        rec.record_episode(ep)

        pairs = rec.export_dpo_pairs()
        assert len(pairs) == 1
        assert "good" in pairs[0]["chosen"]
        assert "bad" in pairs[0]["rejected"]

    def test_dpo_export_no_rejections_empty(self):
        rec = TrajectoryRecorder()
        action = {"tool_name": "test", "params": {}}
        step = _step(1, StepOutcome.SUCCESS, action, vrs=[_vr(Verdict.PASS)])
        ep = EpisodeResult(scenario_id="s1", recovered=True, total_steps=1)
        ep.steps = [step]
        rec.record_episode(ep)

        pairs = rec.export_dpo_pairs()
        assert len(pairs) == 0

    def test_recovered_none_action_in_sft(self):
        """RECOVERED step without applied_action should export none action."""
        rec = TrajectoryRecorder()
        step = _step(1, StepOutcome.RECOVERED, action=None, thought="stable")
        ep = EpisodeResult(scenario_id="s1", recovered=True, total_steps=1)
        ep.steps = [step]
        rec.record_episode(ep)

        sft = rec.export_sft_data()
        assert len(sft) == 1
        assert '"none"' in sft[0]["messages"][1]["content"]
