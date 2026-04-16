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
    """Normalize LLM reasoning text for SFT export.

    - Strips one-or-more leading ``Thought:`` / ``**Thought**:`` prefixes.
    - Drops bodies that are actually the JSON action (no real reasoning).
    - Drops a trailing ``Action: tool_name(...)`` echo line since the
      canonical JSON is appended separately by the exporter.
    Returns an empty string when the body contains no usable reasoning,
    which the exporter can then treat as "no Thought" (honest) rather
    than pad with a JSON-echoing fake.
    """
    cleaned = re.sub(
        r'^(\*?\*?Thought\*?\*?:\s*)+',
        '',
        raw.strip(),
    ).strip()

    if not cleaned:
        return ""

    # Bodies that *are* the JSON action aren't reasoning; refuse them.
    if cleaned.startswith("{") and '"tool_name"' in cleaned[:80]:
        return ""
    if cleaned.startswith("```"):
        return ""

    # Strip trailing "Action: tool(...)" echo line left behind when the
    # teacher model emits both a call-style and a JSON-style action.
    cleaned = re.sub(
        r'\n+Action:\s*\w+\s*\([^\n]*\)\s*$',
        '',
        cleaned,
    ).strip()

    return cleaned


def _wrap_observation(step_number: int, compressed_json: str, violations: list) -> str:
    """Mirror ReActAgent._build_observation_message so training and inference see
    the same user-message shape."""
    parts = [f"## Step {step_number} — System Observation\n"]
    parts.append(compressed_json)
    if violations:
        parts.append(f"\n{len(violations)} active violation(s) detected.")
    else:
        parts.append("\nNo violations detected.")
    parts.append("\nPropose ONE action to improve the system state.")
    return "\n".join(parts)


_STABLE_THOUGHT = (
    "All active compliance constraints are now satisfied. No further trade is "
    "required — the portfolio is stable."
)


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
            last_was_none = False
            for step in ep.steps:
                if step.outcome not in (StepOutcome.SUCCESS, StepOutcome.RECOVERED):
                    continue

                messages.append({
                    "role": "user",
                    "content": _wrap_observation(
                        step.step_number,
                        step.observation.compressed_json,
                        step.observation.violations,
                    ),
                })

                thought = _clean_thought(step.thought or "")
                if step.applied_action:
                    action_str = json.dumps(step.applied_action, ensure_ascii=False)
                    thought_line = f"Thought: {thought}\n\n" if thought else ""
                    content = f"{thought_line}{action_str}"
                    last_was_none = (
                        step.applied_action.get("tool_name") == "none"
                    )
                else:
                    none_action = json.dumps(
                        {"tool_name": "none", "params": {}},
                        ensure_ascii=False,
                    )
                    stable_thought = thought or _STABLE_THOUGHT
                    content = f"Thought: {stable_thought}\n\n{none_action}"
                    last_was_none = True

                messages.append({
                    "role": "assistant",
                    "content": content,
                })

            # Terminal-state patch: if the episode recovered but the trajectory
            # doesn't end on a `none` turn, synthesize one from the final
            # observation so the model learns to recognize stability and stop.
            if (
                messages
                and ep.recovered
                and not last_was_none
                and ep.final_observation is not None
                and ep.final_observation.is_stable
            ):
                messages.append({
                    "role": "user",
                    "content": _wrap_observation(
                        len(ep.steps) + 1,
                        ep.final_observation.compressed_json,
                        ep.final_observation.violations,
                    ),
                })
                none_action = json.dumps(
                    {"tool_name": "none", "params": {}},
                    ensure_ascii=False,
                )
                messages.append({
                    "role": "assistant",
                    "content": f"Thought: {_STABLE_THOUGHT}\n\n{none_action}",
                })

            if messages:
                sft_data.append({
                    "scenario_id": ep.scenario_id,
                    "messages": messages,
                    "recovered": ep.recovered,
                    "total_steps": ep.total_steps,
                })

        return sft_data

    def export_dpo_pairs(self, min_numeric_gap: int = 5) -> list[dict[str, Any]]:
        """Export PASS/FAIL pairs for DPO preference training.

        Each pair: same observation, chosen=PASS action, rejected=FAIL action.

        Args:
            min_numeric_gap: When chosen and rejected differ only by a single
                numeric parameter (e.g. ``qty_delta``), drop the pair if the
                absolute gap is smaller than this. Tiny deltas like -70 vs -71
                are too low-signal to train on and mostly reflect clip-to-cap
                rounding, not strategic preference.
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
                    if self._is_trivial_numeric_diff(
                        step.applied_action, rejected_action, min_numeric_gap
                    ):
                        logger.info(
                            "Dropping low-signal DPO pair at %s step %d "
                            "(numeric gap < %d)",
                            ep.scenario_id, step.step_number, min_numeric_gap,
                        )
                        continue
                    rejected = json.dumps(rejected_action, ensure_ascii=False)
                    pairs.append({
                        "scenario_id": ep.scenario_id,
                        "step": step.step_number,
                        "prompt": obs_json,
                        "chosen": chosen,
                        "rejected": rejected,
                    })

        return pairs

    @staticmethod
    def _is_trivial_numeric_diff(
        chosen: dict[str, Any],
        rejected: dict[str, Any],
        min_gap: int,
    ) -> bool:
        """Return True if chosen and rejected differ only in a numeric field
        by less than ``min_gap``. Such pairs teach "clip to cap" not policy."""
        if chosen.get("tool_name") != rejected.get("tool_name"):
            return False
        cp = chosen.get("params") or {}
        rp = rejected.get("params") or {}
        if set(cp.keys()) != set(rp.keys()):
            return False

        diff_keys = [k for k in cp if cp[k] != rp[k]]
        if len(diff_keys) != 1:
            return False
        k = diff_keys[0]
        cv, rv = cp[k], rp[k]
        if isinstance(cv, (int, float)) and isinstance(rv, (int, float)):
            return abs(cv - rv) < min_gap
        return False

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
