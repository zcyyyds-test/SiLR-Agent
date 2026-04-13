"""OpenAI-compatible LLM client (GPT-4o / DeepSeek / vLLM)."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseLLMClient, LLMResponse, ToolCall

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs.

    Supports GPT-4o (default), DeepSeek, vLLM, and any OpenAI-compatible
    endpoint via base_url override.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install 'silr[agent]'"
            )

        import httpx

        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(60.0, connect=10.0),
            "max_retries": 2,
        }
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        if default_headers is not None:
            kwargs["default_headers"] = default_headers
        self._client = openai.OpenAI(**kwargs)
        self._model = model
        self._is_gemini = "gemini" in model.lower()

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if seed is not None and not self._is_gemini:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        # Parse tool calls
        tool_calls = []
        if msg.tool_calls:
            import json
            for tc in msg.tool_calls:
                logger.debug(
                    "Raw tool call: %s(args=%s)", tc.function.name, tc.function.arguments
                )
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name or "",
                    arguments=args,
                ))

            # Some relay APIs split a single tool call into two entries:
            # one with the name (empty args) and one with args (empty name).
            # Merge them back together.
            if len(tool_calls) >= 2 and not tool_calls[0].arguments:
                for i in range(1, len(tool_calls)):
                    if tool_calls[i].arguments and not tool_calls[i].name:
                        merged = ToolCall(
                            id=tool_calls[0].id,
                            name=tool_calls[0].name,
                            arguments=tool_calls[i].arguments,
                        )
                        tool_calls = [merged]
                        logger.info("Merged split tool call: %s(%s)", merged.name, merged.arguments)
                        break

        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
