"""Test multi-agent episode data model."""

from silr.agent.multi_types import (
    MultiAgentEpisodeResult,
    SpecialistActivation,
)
from silr.agent.types import (
    EpisodeResult, Observation, StepOutcome, StepRecord,
)


def _obs(stable=False):
    return Observation(
        raw={}, compressed_json='{"status": "test"}',
        violations=[], is_stable=stable,
    )


def _step(num, outcome=StepOutcome.SUCCESS):
    return StepRecord(step_number=num, observation=_obs(), outcome=outcome)


def _activation(name, round_num, improved=None, worsened=None, steps=None):
    ep = EpisodeResult(scenario_id="test", total_steps=len(steps or []))
    ep.steps = steps or []
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


class TestMultiAgentEpisodeResult:
    def test_empty_result(self):
        r = MultiAgentEpisodeResult(scenario_id="s1")
        assert r.conflict_count == 0
        assert r.recovered is False

    def test_conflict_count(self):
        r = MultiAgentEpisodeResult(
            scenario_id="s1",
            activations=[
                _activation("a", 1, improved=["conn"]),
                _activation("b", 2, worsened=["util"]),
                _activation("a", 3, improved=["util"], worsened=["conn"]),
            ],
        )
        assert r.conflict_count == 2

    def test_to_single_agent_view(self):
        steps_a = [_step(1), _step(2)]
        steps_b = [_step(3)]
        r = MultiAgentEpisodeResult(
            scenario_id="s1",
            activations=[
                _activation("a", 1, steps=steps_a),
                _activation("b", 2, steps=steps_b),
            ],
            recovered=True,
            total_proposals=5,
            total_rejections=1,
            final_observation=_obs(stable=True),
        )
        flat = r.to_single_agent_view()
        assert flat.scenario_id == "s1"
        assert flat.total_steps == 3
        assert flat.recovered is True
        assert flat.total_proposals == 5
        assert flat.total_rejections == 1
        assert len(flat.steps) == 3

    def test_to_single_agent_view_empty(self):
        r = MultiAgentEpisodeResult(scenario_id="s1")
        flat = r.to_single_agent_view()
        assert flat.total_steps == 0
        assert flat.steps == []
