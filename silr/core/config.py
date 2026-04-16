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
