"""Mock LLM client for deterministic testing."""

from __future__ import annotations

from typing import Any

from .base import BaseLLMClient, LLMResponse, ToolCall


class MockClient(BaseLLMClient):
    """Deterministic mock client that returns pre-configured responses.

    Usage:
        client = MockClient([
            LLMResponse(content='{"tool_name": "restore_link", ...}'),
            LLMResponse(tool_calls=[ToolCall(...)]),
        ])
    """

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        default_response: LLMResponse | None = None,
    ):
        self._responses = list(responses or [])
        self._default = default_response or LLMResponse(content="No action needed.")
        self._call_count = 0
        self._call_history: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> LLMResponse:
        self._call_history.append({
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "seed": seed,
        })

        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = self._default
        self._call_count += 1
        return resp

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def call_history(self) -> list[dict[str, Any]]:
        return self._call_history

    def supports_tool_use(self) -> bool:
        return True
