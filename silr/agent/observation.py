"""Base observer protocol for domain-specific observation formatting."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import Observation


class BaseObserver(ABC):
    """Domain-specific observer that produces compressed observations for the LLM.

    Each domain implements this to aggregate its tool results into a compact
    JSON representation suitable for LLM consumption.
    """

    @abstractmethod
    def observe(self) -> Observation:
        """Run observation tools and return compressed observation.

        Returns:
            Observation with raw data, compressed JSON, violations, and
            stability flag.
        """
        ...
