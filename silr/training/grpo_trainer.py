"""Step-level GRPO data structures and advantage computation.

Provides :class:`GRPOConfig`, :class:`StepSample`, and
:func:`compute_advantages` for step-level Group Relative Policy
Optimization.  The full training loop (model loading, tokenization,
gradient updates) lives in training scripts since it depends on
PyTorch / TRL / PEFT — this module is pure-stdlib so it can be
tested and used without GPU dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import Sequence


@dataclass
class GRPOConfig:
    """Hyperparameters for step-level GRPO training."""

    num_iterations: int = 5
    rollouts_per_scenario: int = 8
    clip_eps: float = 0.2
    kl_coeff: float = 0.02
    lr: float = 5e-6
    batch_size: int = 4
    grpo_epochs: int = 1
    max_seq_len: int = 4096
    base_model: str = "Qwen/Qwen3-14B"
    sft_adapter_path: str = ""
    output_dir: str = "outputs/grpo"
    step_cost: float = 0.05


@dataclass
class StepSample:
    """One (observation, action) pair with its reward and group info.

    ``group_key`` identifies the normalisation group — typically
    ``(scenario_id,)`` — so that advantages are computed relative
    to other steps/rollouts within the same scenario.
    """

    obs_text: str
    action_text: str
    reward: float
    group_key: tuple  # (scenario_id,)
    advantage: float = 0.0
    log_prob: float = 0.0


def compute_advantages(samples: Sequence[StepSample]) -> None:
    """Normalise rewards within each group and set ``advantage`` in-place.

    For each group identified by :pyattr:`StepSample.group_key`:

    * ``mean_r`` = arithmetic mean of rewards in the group
    * ``std_r``  = **population** standard deviation (ddof=0)
    * ``advantage_i = (reward_i - mean_r) / (std_r + 1e-8)``

    Groups with a single sample or zero variance receive ``advantage = 0.0``
    so they do not bias the policy gradient.
    """
    # Partition samples by group_key.
    groups: dict[tuple, list[StepSample]] = defaultdict(list)
    for s in samples:
        groups[s.group_key].append(s)

    for members in groups.values():
        n = len(members)

        # Single-sample group — no meaningful relative signal.
        if n == 1:
            members[0].advantage = 0.0
            continue

        mean_r = sum(s.reward for s in members) / n
        # Population variance (not sample variance).
        var_r = sum((s.reward - mean_r) ** 2 for s in members) / n
        std_r = sqrt(var_r)

        # Zero-variance group — all rewards identical.
        if std_r == 0.0:
            for s in members:
                s.advantage = 0.0
            continue

        for s in members:
            raw = (s.reward - mean_r) / (std_r + 1e-8)
            s.advantage = max(-3.0, min(3.0, raw))
