"""Agent configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the ReAct agent."""

    max_steps: int = 8
    max_proposals_per_step: int = 3
    consecutive_fail_limit: int = 2       # fail-safe after N full steps of all-reject
    enable_verification: bool = True      # False = ablation NoVerify mode
    observe_verification: bool = False    # NoVerify execution + passive Φ trace, no gating
    temperature: float = 0.0
    seed: int | None = 42
    verification_feedback_mode: str = "detailed"  # "summary" | "detailed" | "full"

    stall_progress_budget: int | None = None
    """Anti-stall liveness guard. When set, the loop tracks consecutive
    SAFE_PROGRESS-admitted actions whose post-action violation count
    did *not* strictly decrease from the running minimum. After the
    budget is exhausted the episode terminates with outcome
    ``StepOutcome.STALL`` and ``EpisodeResult.stall_terminated=True``.

    Defends against the denial-of-recovery / stall attacker described
    in §threat-model: a malicious prompt that nudges the LLM into
    making infinitesimal monotone setpoint changes that individually
    pass the progress-gating verifier but, in aggregate, never reach
    terminal recovery. ``None`` (default) preserves the original
    semantics — no liveness budget, episode runs until ``max_steps``."""
