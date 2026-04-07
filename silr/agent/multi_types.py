"""Data types for multi-agent coordinator episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import EpisodeResult, Observation, StepOutcome, StepRecord


@dataclass
class SpecialistActivation:
    """Record of a single specialist agent activation within a coordinator round."""

    specialist_name: str
    round_number: int
    coordinator_thought: str
    episode_result: EpisodeResult
    pre_observation: Observation
    post_observation: Observation
    constraints_improved: list[str] = field(default_factory=list)
    constraints_worsened: list[str] = field(default_factory=list)


@dataclass
class MultiAgentEpisodeResult:
    """Result of a multi-agent coordinator episode.

    Contains the full history of specialist activations, coordinator
    decisions, and aggregate statistics. Can be flattened into a
    single-agent EpisodeResult for compatibility with existing metrics.
    """

    scenario_id: str
    activations: list[SpecialistActivation] = field(default_factory=list)
    recovered: bool = False
    total_rounds: int = 0
    total_specialist_steps: int = 0
    total_proposals: int = 0
    total_rejections: int = 0
    final_observation: Optional[Observation] = None
    error: Optional[str] = None

    @property
    def conflict_count(self) -> int:
        """Number of activations where a specialist worsened a constraint."""
        return sum(1 for a in self.activations if a.constraints_worsened)

    def to_single_agent_view(self) -> EpisodeResult:
        """Flatten into EpisodeResult for compatibility with existing metrics."""
        steps: list[StepRecord] = []
        for activation in self.activations:
            steps.extend(activation.episode_result.steps)

        return EpisodeResult(
            scenario_id=self.scenario_id,
            steps=steps,
            recovered=self.recovered,
            total_steps=len(steps),
            total_proposals=self.total_proposals,
            total_rejections=self.total_rejections,
            failsafe_triggered=any(
                a.episode_result.failsafe_triggered for a in self.activations
            ),
            final_observation=self.final_observation,
            error=self.error,
        )
