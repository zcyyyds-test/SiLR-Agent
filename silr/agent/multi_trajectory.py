"""Multi-agent trajectory recording for coordinator training data.

Produces three kinds of training data:
1. Coordinator SFT: observation → dispatch decision
2. Specialist SFT: reuses existing TrajectoryRecorder per specialist
3. Coordinator DPO: pairs of good/bad dispatch decisions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .multi_types import MultiAgentEpisodeResult, SpecialistActivation
from .trajectory import TrajectoryRecorder

logger = logging.getLogger(__name__)


class MultiAgentTrajectoryRecorder:
    """Records multi-agent episodes for SFT/DPO export."""

    def __init__(self):
        self._episodes: list[MultiAgentEpisodeResult] = []
        self._specialist_recorder = TrajectoryRecorder()

    def record_episode(self, result: MultiAgentEpisodeResult) -> None:
        self._episodes.append(result)
        for activation in result.activations:
            self._specialist_recorder.record_episode(activation.episode_result)
        logger.info(
            f"Recorded multi-agent episode '{result.scenario_id}': "
            f"{result.total_rounds} rounds, recovered={result.recovered}"
        )

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    def export_coordinator_sft(self) -> list[dict[str, Any]]:
        """Export coordinator dispatch decisions as SFT conversations.

        Each recovered episode produces a conversation where:
        - user messages contain system state + violation summary
        - assistant messages contain dispatch decisions
        """
        sft_data = []

        for ep in self._episodes:
            if not ep.recovered:
                continue

            messages = []
            for activation in ep.activations:
                messages.append({
                    "role": "user",
                    "content": activation.pre_observation.compressed_json,
                })
                dispatch = json.dumps({
                    "specialist": activation.specialist_name,
                    "reason": activation.coordinator_thought,
                }, ensure_ascii=False)
                messages.append({
                    "role": "assistant",
                    "content": f"Thought: {activation.coordinator_thought}\n\n{dispatch}",
                })

            if messages:
                sft_data.append({
                    "scenario_id": ep.scenario_id,
                    "messages": messages,
                    "recovered": ep.recovered,
                    "total_rounds": ep.total_rounds,
                })

        return sft_data

    def export_specialist_sft(self) -> list[dict[str, Any]]:
        """Delegate to existing TrajectoryRecorder for specialist data."""
        return self._specialist_recorder.export_sft_data()

    def export_coordinator_dpo(self) -> list[dict[str, Any]]:
        """Export DPO pairs from coordinator dispatch decisions.

        A dispatch that led to constraint improvement is preferred over
        one that led to worsening or no change.
        """
        pairs = []

        for ep in self._episodes:
            good_dispatches = []
            bad_dispatches = []

            for activation in ep.activations:
                entry = {
                    "observation": activation.pre_observation.compressed_json,
                    "specialist": activation.specialist_name,
                    "reason": activation.coordinator_thought,
                }
                if activation.constraints_improved and not activation.constraints_worsened:
                    good_dispatches.append(entry)
                elif activation.constraints_worsened:
                    bad_dispatches.append(entry)

            for good in good_dispatches:
                for bad in bad_dispatches:
                    pairs.append({
                        "scenario_id": ep.scenario_id,
                        "prompt": good["observation"],
                        "chosen": json.dumps({
                            "specialist": good["specialist"],
                            "reason": good["reason"],
                        }, ensure_ascii=False),
                        "rejected": json.dumps({
                            "specialist": bad["specialist"],
                            "reason": bad["reason"],
                        }, ensure_ascii=False),
                    })

        return pairs

    def save(self, path: str | Path) -> None:
        """Save all training data to JSON files."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        coordinator_sft = self.export_coordinator_sft()
        specialist_sft = self.export_specialist_sft()
        coordinator_dpo = self.export_coordinator_dpo()

        for name, data in [
            ("coordinator_sft", coordinator_sft),
            ("specialist_sft", specialist_sft),
            ("coordinator_dpo", coordinator_dpo),
        ]:
            with open(path / f"{name}.json", "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved {len(coordinator_sft)} coordinator SFT, "
            f"{len(specialist_sft)} specialist SFT, "
            f"{len(coordinator_dpo)} coordinator DPO to {path}"
        )
