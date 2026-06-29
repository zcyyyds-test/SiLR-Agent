"""OpenAI-compatible LLM client (GPT-4o / DeepSeek / vLLM)."""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import BaseLLMClient, LLMResponse, ToolCall

logger = logging.getLogger(__name__)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid integer env %s=%r", name, raw)
        return None
    return value if value > 0 else None


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("Ignoring invalid boolean env %s=%r", name, raw)
    return None


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
        timeout_s: float = 60.0,
        connect_timeout_s: float = 10.0,
        max_retries: int = 2,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install 'silr[agent]'"
            )

        import httpx

        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(timeout_s, connect=connect_timeout_s),
            "max_retries": max_retries,
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
        self._max_tokens = max_tokens if max_tokens is not None else _env_int(
            "SILR_MAX_TOKENS"
        )
        # None = leave server default; False = disable the Qwen3 <think> block so
        # the action parser sees a direct action (the vLLM server path has no
        # LocalModelClient chat-template control). Set via extra_body in chat().
        self._enable_thinking = (
            enable_thinking
            if enable_thinking is not None
            else _env_bool("SILR_QWEN_ENABLE_THINKING")
        )

    def supports_tool_use(self) -> bool:
        # SILR_DISABLE_TOOLS=1 forces bare-text mode (tools=None) so cross-family
        # served models without a matching vLLM tool-call parser still elicit
        # actions via Layer-2 (JSON block) parsing in ActionParser.
        if os.environ.get("SILR_DISABLE_TOOLS", "").strip() in ("1", "true", "True"):
            return False
        return True

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
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._enable_thinking is not None:
            # Qwen3 on vLLM reads chat_template_kwargs.enable_thinking; passing
            # False suppresses the <think> block that otherwise stalls the action
            # parser (retries -> ~6x slower episodes).
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self._enable_thinking}
            }

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

        # vLLM 0.21.0 leaks GPT-2 ByteLevel BPE markers in chat-completion
        # `content` for models like DeepSeek-R1-Distill-Llama-8B (`Ġ`→space,
        # `Ċ`→newline). No-op for tokenizers that decode cleanly (Qwen3, etc.)
        # because those markers never appear in their output.
        content = msg.content or ""
        if "Ġ" in content or "Ċ" in content:
            content = content.replace("Ġ", " ").replace("Ċ", "\n")

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
