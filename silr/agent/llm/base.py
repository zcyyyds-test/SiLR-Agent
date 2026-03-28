"""Base LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    """A tool call extracted from LLM response."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str = ""                            # text content
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Optional[dict[str, int]] = None       # {prompt_tokens, completion_tokens}


class BaseLLMClient(ABC):
    """Abstract base for LLM API clients."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> LLMResponse:
        """Send chat completion request.

        Args:
            messages: OpenAI-format message list
            tools: OpenAI-format tool definitions (optional)
            temperature: Sampling temperature
            seed: Random seed for reproducibility

        Returns:
            Standardized LLMResponse
        """

    def supports_tool_use(self) -> bool:
        """Whether this client supports native tool_calls."""
        return True
