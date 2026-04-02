"""Test multi-agent trajectory recording and training data export."""

from silr.agent.multi_trajectory import MultiAgentTrajectoryRecorder
from silr.agent.multi_types import (
    MultiAgentEpisodeResult,
    SpecialistActivation,
)
from silr.agent.types import (
    EpisodeResult, Observation, StepOutcome, StepRecord,
)


def _obs(stable=False):
    return Observation(
        raw={}, compressed_json='{"test": true}',
        violations=[], is_stable=stable,
    )


def _step(num, outcome=StepOutcome.SUCCESS, action=None):
    return StepRecord(
        step_number=num, observation=_obs(),
        outcome=outcome, applied_action=action,
        thought="test thought",
    )


def _activation(name, round_num, improved=None, worsened=None, recovered=True):
    action = {"tool_name": "test", "params": {}}
    ep = EpisodeResult(
        scenario_id="test", recovered=recovered, total_steps=1,
    )
    ep.steps = [_step(1, StepOutcome.SUCCESS, action)]
    return SpecialistActivation(
        specialist_name=name,
        round_number=round_num,
        coordinator_thought=f"dispatch {name}",
        episode_result=ep,
        pre_observation=_obs(),
        post_observation=_obs(),
        constraints_improved=improved or [],
        constraints_worsened=worsened or [],
    )


class TestMultiAgentTrajectoryRecorder:
    def test_record_and_count(self):
        rec = MultiAgentTrajectoryRecorder()
        ep = MultiAgentEpisodeResult(scenario_id="s1")
        rec.record_episode(ep)
        assert rec.episode_count == 1

    def test_coordinator_sft_recovered_only(self):
        rec = MultiAgentTrajectoryRecorder()

        good = MultiAgentEpisodeResult(
            scenario_id="good", recovered=True,
            activations=[_activation("conn", 1, improved=["connectivity"])],
        )
        bad = MultiAgentEpisodeResult(
            scenario_id="bad", recovered=False,
            activations=[_activation("conn", 1)],
        )
        rec.record_episode(good)
        rec.record_episode(bad)

        sft = rec.export_coordinator_sft()
        assert len(sft) == 1
        assert sft[0]["scenario_id"] == "good"

    def test_coordinator_sft_message_format(self):
        rec = MultiAgentTrajectoryRecorder()
        ep = MultiAgentEpisodeResult(
            scenario_id="s1", recovered=True,
            activations=[_activation("conn", 1, improved=["connectivity"])],
        )
        rec.record_episode(ep)

        sft = rec.export_coordinator_sft()
        messages = sft[0]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "dispatch conn" in messages[1]["content"]

    def test_coordinator_dpo_pairs(self):
        rec = MultiAgentTrajectoryRecorder()
        ep = MultiAgentEpisodeResult(
            scenario_id="s1", recovered=True,
            activations=[
                _activation("conn", 1, improved=["connectivity"]),
                _activation("util", 2, worsened=["link_utilization"]),
            ],
        )
        rec.record_episode(ep)

        pairs = rec.export_coordinator_dpo()
        assert len(pairs) == 1
        assert "conn" in pairs[0]["chosen"]
        assert "util" in pairs[0]["rejected"]

    def test_coordinator_dpo_no_bad_empty(self):
        rec = MultiAgentTrajectoryRecorder()
        ep = MultiAgentEpisodeResult(
            scenario_id="s1", recovered=True,
            activations=[_activation("conn", 1, improved=["connectivity"])],
        )
        rec.record_episode(ep)
        assert len(rec.export_coordinator_dpo()) == 0

    def test_specialist_sft_delegated(self):
        rec = MultiAgentTrajectoryRecorder()
        ep = MultiAgentEpisodeResult(
            scenario_id="s1", recovered=True,
            activations=[_activation("conn", 1, recovered=True)],
        )
        rec.record_episode(ep)

        specialist_sft = rec.export_specialist_sft()
        assert len(specialist_sft) == 1
