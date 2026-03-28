"""Agent data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StepOutcome(Enum):
    """Outcome of a single agent step."""
    SUCCESS = "success"              # action verified + applied
    FAIL_VERIFY = "fail_verify"      # all proposals rejected by SiLR
    FAIL_PARSE = "fail_parse"        # LLM output unparseable after retries
    FAIL_EXECUTE = "fail_execute"    # action execution error on main system
    FAILSAFE = "failsafe"           # fell back to rule-based action
    RECOVERED = "recovered"          # system already stable, no action needed


@dataclass
class Observation:
    """Compressed system observation for LLM consumption."""
    raw: dict[str, Any]              # full tool results
    compressed_json: str             # compact JSON for prompt injection
    violations: list[dict]           # list of active violations
    is_stable: bool                  # True if system is in acceptable state


@dataclass
class StepRecord:
    """Record of a single ReAct step."""
    step_number: int
    observation: Observation
    thought: str = ""
    proposed_actions: list[dict] = field(default_factory=list)
    verification_results: list[Any] = field(default_factory=list)
    applied_action: Optional[dict] = None
    tool_result: Optional[dict] = None
    outcome: StepOutcome = StepOutcome.SUCCESS
    error: Optional[str] = None


@dataclass
class EpisodeResult:
    """Result of a complete scenario episode."""
    scenario_id: str
    steps: list[StepRecord] = field(default_factory=list)
    recovered: bool = False
    total_steps: int = 0
    total_proposals: int = 0
    total_rejections: int = 0
    failsafe_triggered: bool = False
    final_observation: Optional[Observation] = None
    error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Fraction of steps that resulted in a successful action."""
        if not self.steps:
            return 0.0
        ok = sum(1 for s in self.steps if s.outcome == StepOutcome.SUCCESS)
        return ok / len(self.steps)

    @property
    def rejection_rate(self) -> float:
        """Fraction of proposals that were rejected by SiLR."""
        if self.total_proposals == 0:
            return 0.0
        return self.total_rejections / self.total_proposals
