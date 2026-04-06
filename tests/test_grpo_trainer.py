"""Tests for step-level GRPO data structures and advantage computation."""

import pytest

from silr.training.grpo_trainer import (
    GRPOConfig,
    StepSample,
    compute_advantages,
)


# ── GRPOConfig ───────────────────────────────────────────────────────

class TestGRPOConfig:
    def test_defaults(self):
        cfg = GRPOConfig()
        assert cfg.num_iterations == 5
        assert cfg.rollouts_per_scenario == 8
        assert cfg.clip_eps == 0.2
        assert cfg.kl_coeff == 0.02
        assert cfg.lr == 5e-6
        assert cfg.batch_size == 4
        assert cfg.grpo_epochs == 1
        assert cfg.max_seq_len == 4096
        assert cfg.base_model == "Qwen/Qwen3-14B"
        assert cfg.sft_adapter_path == ""
        assert cfg.output_dir == "outputs/grpo"
        assert cfg.step_cost == 0.05

    def test_custom_values(self):
        cfg = GRPOConfig(
            num_iterations=10,
            clip_eps=0.3,
            lr=3e-4,
            base_model="my-model",
            output_dir="/tmp/grpo",
        )
        assert cfg.num_iterations == 10
        assert cfg.clip_eps == 0.3
        assert cfg.lr == 3e-4
        assert cfg.base_model == "my-model"
        assert cfg.output_dir == "/tmp/grpo"
        # Unchanged defaults still hold.
        assert cfg.rollouts_per_scenario == 8
        assert cfg.kl_coeff == 0.02


# ── StepSample ───────────────────────────────────────────────────────

class TestStepSample:
    def test_creation_with_defaults(self):
        s = StepSample(
            obs_text='{"bus": 1}',
            action_text="think: check voltage\nact: inspect_bus(1)",
            reward=0.5,
            group_key=("sc1", 0),
        )
        assert s.obs_text == '{"bus": 1}'
        assert s.reward == 0.5
        assert s.group_key == ("sc1", 0)
        assert s.advantage == 0.0
        assert s.log_prob == 0.0

    def test_explicit_advantage(self):
        s = StepSample(
            obs_text="{}",
            action_text="act: noop",
            reward=1.0,
            group_key=("sc2", 1),
            advantage=0.75,
            log_prob=-1.2,
        )
        assert s.advantage == 0.75
        assert s.log_prob == -1.2


# ── compute_advantages ───────────────────────────────────────────────

def _make_sample(reward: float, group_key: tuple = ("s", 0)) -> StepSample:
    return StepSample(
        obs_text="{}",
        action_text="act: x",
        reward=reward,
        group_key=group_key,
    )


class TestComputeAdvantages:
    def test_single_group_positive_negative(self):
        """Rewards above/below mean get positive/negative advantages."""
        samples = [_make_sample(r) for r in [1.0, 2.0, 3.0, 4.0]]
        compute_advantages(samples)

        # Mean = 2.5, so first two should be negative, last two positive.
        assert samples[0].advantage < 0
        assert samples[1].advantage < 0
        assert samples[2].advantage > 0
        assert samples[3].advantage > 0

    def test_single_group_zero_mean_advantage(self):
        """Advantages within a group should sum to approximately zero."""
        samples = [_make_sample(r) for r in [1.0, 2.0, 3.0, 4.0]]
        compute_advantages(samples)

        total = sum(s.advantage for s in samples)
        assert abs(total) < 1e-6

    def test_multiple_groups_independent(self):
        """Each group is normalised independently."""
        group_a = [_make_sample(r, ("a", 0)) for r in [10.0, 20.0]]
        group_b = [_make_sample(r, ("b", 0)) for r in [100.0, 200.0]]
        all_samples = group_a + group_b

        compute_advantages(all_samples)

        # Within group_a: rewards differ by the same ratio as group_b,
        # so normalised advantages should match (same z-scores).
        assert abs(group_a[0].advantage - group_b[0].advantage) < 1e-6
        assert abs(group_a[1].advantage - group_b[1].advantage) < 1e-6

    def test_single_sample_group(self):
        """A group with one sample should get advantage = 0."""
        samples = [_make_sample(42.0, ("lone", 0))]
        compute_advantages(samples)
        assert samples[0].advantage == 0.0

    def test_all_same_reward(self):
        """Zero-variance group: all advantages should be 0."""
        samples = [_make_sample(5.0) for _ in range(4)]
        compute_advantages(samples)
        for s in samples:
            assert s.advantage == 0.0

    def test_empty_input(self):
        """Calling with no samples should not raise."""
        compute_advantages([])

    def test_mixed_single_and_multi(self):
        """Single-sample and multi-sample groups coexist correctly."""
        multi = [_make_sample(r, ("m", 0)) for r in [1.0, 3.0]]
        single = [_make_sample(99.0, ("s", 0))]
        compute_advantages(multi + single)

        # Single group → advantage = 0.
        assert single[0].advantage == 0.0
        # Multi group → symmetric around 0.
        assert multi[0].advantage < 0
        assert multi[1].advantage > 0
        assert abs(multi[0].advantage + multi[1].advantage) < 1e-6
