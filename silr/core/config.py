"""DomainConfig: bundles all domain-specific components for injection.

Passed to SiLRVerifier and ReActAgent as a required parameter,
replacing any hardcoded domain defaults with the domain's own
tools, checkers, prompts, and observation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .interfaces import BaseSystemManager, BaseConstraintChecker


@dataclass
class DomainConfig:
    """Configuration bundle for a specific domain.

    Required fields define the verification pipeline.
    Optional fields customize the agent loop (prompts, observation, failsafe).
    """

    # --- Required ---
    domain_name: str
    """Human-readable domain identifier (e.g. "power_grid", "network")."""

    checkers: list[BaseConstraintChecker]
    """Constraint checkers for SiLR verification."""

    allowed_actions: frozenset[str]
    """Tool names the verifier accepts (others are rejected as ERROR)."""

    create_toolset: Callable[[BaseSystemManager], dict[str, Any]]
    """Factory: manager → {tool_name: tool_instance}."""

    # --- Optional: Agent customization ---
    build_system_prompt: Optional[Callable[[BaseSystemManager, list], str]] = None
    """Factory: (manager, tool_schemas) → system prompt string."""

    build_tool_schemas: Optional[Callable[[BaseSystemManager], list[dict]]] = None
    """Factory: manager → list of tool schema dicts for LLM."""

    get_valid_device_ids: Optional[Callable[[BaseSystemManager], dict[str, list]]] = None
    """Factory: manager → {device_type: [id_list]} for action parser validation."""

    param_aliases: Optional[dict[str, dict[str, str]]] = None
    """Per-tool parameter-name alias map for the action parser.

    Maps common typos back to the canonical param name used by the tool
    layer, e.g. ``{"adjust_position": {"delta_qty": "qty_delta"}}``.
    """

    create_observer: Optional[Callable[[BaseSystemManager], Any]] = None
    """Factory: manager → observer object with observe() method."""

    create_failsafe: Optional[Callable[[BaseSystemManager], Any]] = None
    """Factory: manager → failsafe strategy object."""

    # --- Optional: Verification tuning ---
    post_solve_hook: Optional[Callable[[BaseSystemManager], bool]] = None
    """Optional hook called after solve() succeeds.

    Use this for domain-specific post-solve steps (e.g., time-domain
    simulation in power grids). Returns True if the post-solve check
    passes, False otherwise. If None, no post-solve step is performed.
    """

    gating_policy: str = "terminal"
    """Verifier gating policy.

    - ``"terminal"``: the verifier admits an action iff the post-action
      shadow state has *zero* outstanding violations (the original
      single-step recovery semantics). Suitable for domains where a
      single tool call can take the system from any state to a fully
      safe one (e.g., the historical ``grid``/``cluster``/``finance``
      tracks where action granularity matches recovery granularity).

    - ``"progress"``: the verifier additionally admits an action if the
      post-action state is a *recoverability-preserving step* — solver
      converges, no new violation type appears, and the violation count
      does not increase relative to the pre-action baseline. Necessary
      for domains where multiple coordinated single-device actions are
      required to recover from a stressed snapshot (e.g., the ANM track
      under simultaneous multi-generator surges). Aligns with
      permissive/recovery-shielding semantics in the safe-RL literature.

    - ``"progress_mag"``: ``progress`` plus the L3 magnitude guard
      (``docs/method_predicates.md`` §4.2). Adds a quantitative check
      that aggregated violation severity does not inflate beyond a
      relative-OR-absolute threshold (5% relative slack, 1e-3 absolute
      floor). Defends against count-preserving magnitude-drift attacks
      where a prompt-injected LLM proposes setpoints that pass L2 (count
      and types preserved) while quietly worsening violation magnitudes
      across consecutive SP-admitted steps.

    - ``"scalar_progress"``: ablation-only baseline. Admits a non-terminal
      step if the domain's native scalar penalty (``manager.last_penalty``)
      does not increase beyond a small absolute tolerance. This is used to
      test whether structured predicates add value beyond a single numeric
      threshold; it is not the recommended runtime policy.

    - ``"rollback"``: post-hoc rollback baseline for related-work comparison.
      In this shadow-execution implementation it is the permissive,
      support-only version: admit a non-terminal post state iff it introduces
      no violation branch outside the pre-action baseline. It deliberately does
      not apply the ``progress_mag`` per-branch severity envelope.

    The terminal-safety goal (``Verdict.PASS`` = zero violations) is
    preserved in progress-family policies; they only widen the set of
    *admissible* (apply-gated) actions, not the terminal recovery
    criterion. This separation is what lets the verifier double as a
    runtime guard *and* a downstream training signal: gating uses the
    graded verdict, learning targets the terminal verdict.
    """
