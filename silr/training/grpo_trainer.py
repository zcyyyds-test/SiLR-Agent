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
    traj_id: int = -1  # rollout/episode id (for trajectory-return advantage)


def compute_advantages_trajectory(samples: Sequence[StepSample]) -> None:
    """Trajectory-RETURN advantage (panel 2026-06-08, codex root cause).

    Step-level scenario z-score advantage truncates delayed consequences: a count
    reward's "clear the big family now, floor the small one later" is invisible if
    each step is normalised on its own. This computes, per scenario group, the
    episode RETURN of each rollout (sum of its step rewards), z-scores the returns
    ACROSS rollouts, and assigns each rollout's return-advantage to ALL its steps --
    exactly what the tabular GRPO sims do (where geometric D wins decisively). This
    preserves the path-quality / delayed-consequence signal the geometric reward
    encodes. Enable with SILR_TRAJ_ADV=1.
    """
    # scenario group -> {traj_id: [steps]}
    scen: dict[tuple, dict[int, list[StepSample]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        scen[s.group_key][s.traj_id].append(s)

    for trajs in scen.values():
        returns = {tid: sum(st.reward for st in steps) for tid, steps in trajs.items()}
        n = len(returns)
        if n <= 1:
            for steps in trajs.values():
                for st in steps:
                    st.advantage = 0.0
            continue
        mean_R = sum(returns.values()) / n
        var_R = sum((R - mean_R) ** 2 for R in returns.values()) / n
        std_R = sqrt(var_R)
        if std_R == 0.0:
            for steps in trajs.values():
                for st in steps:
                    st.advantage = 0.0
            continue
        for tid, steps in trajs.items():
            adv = (returns[tid] - mean_R) / (std_R + 1e-8)
            adv = max(-3.0, min(3.0, adv))
            for st in steps:
                st.advantage = adv


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
