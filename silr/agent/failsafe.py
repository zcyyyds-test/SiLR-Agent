"""Base failsafe protocol for domain-specific rule-based fallback actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .types import Observation


class BaseFailsafe(ABC):
    """Domain-specific rule-based failsafe strategy.

    When the LLM's proposals are repeatedly rejected by SiLR verification,
    the failsafe provides a conservative rule-based action as a fallback.
    """

    @abstractmethod
    def suggest(self, obs: Observation) -> Optional[dict]:
        """Suggest a conservative rule-based action.

        Args:
            obs: Current system observation

        Returns:
            Action dict {"tool_name": str, "params": dict}, or None
            if no safe action can be suggested.
        """
        ...

    @abstractmethod
    def suggest_escalated(
        self, obs: Observation, last_rejected: Optional[dict] = None
    ) -> Optional[dict]:
        """Suggest an escalated action after repeated failures.

        Args:
            obs: Current system observation
            last_rejected: The most recently rejected action (for context)

        Returns:
            Action dict or None.
        """
        ...
