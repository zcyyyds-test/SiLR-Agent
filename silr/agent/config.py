"""Agent configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the ReAct agent."""

    max_steps: int = 8
    max_proposals_per_step: int = 3
    consecutive_fail_limit: int = 2       # fail-safe after N full steps of all-reject
    enable_verification: bool = True      # False = ablation NoVerify mode
    temperature: float = 0.0
    seed: int | None = 42
    verification_feedback_mode: str = "detailed"  # "summary" | "detailed" | "full"
