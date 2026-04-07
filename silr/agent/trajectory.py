"""Trajectory recording for training data collection.

Collects episode data and exports in SFT / DPO formats.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .types import EpisodeResult, StepOutcome
from silr.verifier.types import Verdict

logger = logging.getLogger(__name__)


def _clean_thought(raw: str) -> str:
    """Strip duplicated Thought prefixes from LLM output.

    Some models output '**Thought**: ...' which gets wrapped into
    'Thought: **Thought**: ...' by our formatting. This collapses
    all variants back to plain text without any prefix.
    """
    cleaned = re.sub(
        r'^(\*?\*?Thought\*?\*?:\s*)+',
        '',
        raw.strip(),
    )
    return cleaned.strip()


class TrajectoryRecorder:
    """Records episode trajectories for later SFT/DPO export."""

    def __init__(self, quality_filter: "Callable[[EpisodeResult], bool] | None" = None):
        """
        Args:
            quality_filter: Optional callable that returns True if an episode
                should be excluded from SFT export. Domain-specific quality
                checks (e.g., detecting ineffective action streaks) can be
                injected here.
        """
        self._episodes: list[EpisodeResult] = []
        self._quality_filter = quality_filter

    def record_episode(self, result: EpisodeResult) -> None:
        """Add an episode result to the collection."""
        self._episodes.append(result)
        logger.info(
            f"Recorded episode '{result.scenario_id}': "
            f"{result.total_steps} steps, recovered={result.recovered}"
        )

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    def export_sft_data(self) -> list[dict[str, Any]]:
        """Export successful episodes as SFT training data.

        Format: list of conversation dicts with system/user/assistant messages.
        Only includes steps where the action was PASS (verified and applied).
        """
        sft_data = []

        for ep in self._episodes:
            if not ep.recovered:
                continue

            # Apply domain-specific quality filter if provided
            if self._quality_filter and self._quality_filter(ep):
                logger.info(
                    f"Filtered out '{ep.scenario_id}': quality filter rejected"
                )
                continue

            messages = []
            for step in ep.steps:
                if step.outcome not in (StepOutcome.SUCCESS, StepOutcome.RECOVERED):
                    continue

                messages.append({
                    "role": "user",
                    "content": step.observation.compressed_json,
                })

                thought = _clean_thought(step.thought or "")
                if step.applied_action:
                    action_str = json.dumps(step.applied_action, ensure_ascii=False)
                    content = f"Thought: {thought}\n\n{action_str}"
                else:
                    none_action = json.dumps(
                        {"tool_name": "none", "params": {}},
                        ensure_ascii=False,
                    )
                    stable_thought = thought if thought else "System is stable. No further action needed."
                    content = f"Thought: {stable_thought}\n\n{none_action}"

                messages.append({
                    "role": "assistant",
                    "content": content,
                })

            if messages:
                sft_data.append({
                    "scenario_id": ep.scenario_id,
                    "messages": messages,
                    "recovered": ep.recovered,
                    "total_steps": ep.total_steps,
                })

        return sft_data

    def export_dpo_pairs(self) -> list[dict[str, Any]]:
        """Export PASS/FAIL pairs for DPO preference training.

        Each pair: same observation, chosen=PASS action, rejected=FAIL action.
        """
        pairs = []

        for ep in self._episodes:
            for step in ep.steps:
                if step.outcome != StepOutcome.SUCCESS:
                    continue
                if not step.applied_action or not step.verification_results:
                    continue

                rejected_actions = []
                for i, vr in enumerate(step.verification_results):
                    if hasattr(vr, 'verdict') and vr.verdict == Verdict.FAIL:
                        if i < len(step.proposed_actions):
                            rejected_actions.append(step.proposed_actions[i])

                if not rejected_actions:
                    continue

                obs_json = step.observation.compressed_json
                chosen = json.dumps(step.applied_action, ensure_ascii=False)

                for rejected_action in rejected_actions:
                    rejected = json.dumps(rejected_action, ensure_ascii=False)
                    pairs.append({
                        "scenario_id": ep.scenario_id,
                        "step": step.step_number,
                        "prompt": obs_json,
                        "chosen": chosen,
                        "rejected": rejected,
                    })

        return pairs

    def save(self, path: str | Path) -> None:
        """Save all data to JSON files."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        sft = self.export_sft_data()
        dpo = self.export_dpo_pairs()

        with open(path / "sft_data.json", "w") as f:
            json.dump(sft, f, indent=2, ensure_ascii=False)

        with open(path / "dpo_pairs.json", "w") as f:
            json.dump(dpo, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved {len(sft)} SFT conversations, "
            f"{len(dpo)} DPO pairs to {path}"
        )
