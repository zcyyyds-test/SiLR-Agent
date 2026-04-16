"""Kimi Coding endpoint (Anthropic Messages API compatible) LLM client.

The Kimi "coding" endpoint at https://api.kimi.com/coding/v1/messages speaks
the Anthropic Messages API. It requires a claude-cli User-Agent header to
identify as Claude Code.

Design choice: supports_tool_use() returns False on purpose. Passing the
OpenAI-style tools list to the endpoint pushes the model into structured
tool-use mode, where it emits tool_use blocks with empty text content.
That produces empty "Thought" values and defeats the ReAct premise. Keeping
tools=None forces the model into bare-text mode, which yields real Thought
text followed by a JSON action block that ActionParser Layer 2 extracts.

The system prompt must instruct the model to emit
`Thought: <reasoning>\n\n{json_action}` — domains/finance already does.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseLLMClient, LLMResponse, ToolCall

logger = logging.getLogger(__name__)


_DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1/messages"
# Tight enough to look like a real Claude Code session, not loose enough to
# accidentally match an unrelated client UA.
_DEFAULT_UA = "claude-cli/2.0.46 (external, cli)"
_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


class KimiAnthropicClient(BaseLLMClient):
    """Client for Kimi's Anthropic-Messages-compatible coding endpoint.

    Only uses bare-text mode. Tool definitions from the caller are ignored
    intentionally — see module docstring.

    Args:
        model: Kimi model name (e.g. "kimi-k2.5", "kimi-k2-thinking").
            The endpoint may internally remap to a versioned model name.
        api_key: sk-kimi-... token. Falls back to env var KIMI_API_KEY.
        base_url: Full URL to /v1/messages. Default: coding endpoint.
        user_agent: Override the claude-cli UA header.
    """

    def __init__(
        self,
        model: str = "kimi-k2.5",
        api_key: str | None = None,
        base_url: str | None = None,
        user_agent: str | None = None,
        max_tokens: int = 800,
    ):
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError(
                "requests package required for KimiAnthropicClient. "
                "pip install requests"
            )

        import os
        self._api_key = api_key or os.environ.get("KIMI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "KimiAnthropicClient requires api_key or KIMI_API_KEY env var"
            )
        self._base_url = base_url or _DEFAULT_BASE_URL
        self._user_agent = user_agent or _DEFAULT_UA
        self._model = model
        self._max_tokens = max_tokens

    # ── BaseLLMClient protocol ───────────────────────────────────

    def supports_tool_use(self) -> bool:
        """Force bare-text mode so Thought field stays populated."""
        return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> LLMResponse:
        import requests

        # Split system prompt from rest — Anthropic API needs it separate.
        system_parts: list[str] = []
        rest: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(m.get("content", ""))
            else:
                rest.append(self._normalize_message(m))

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "messages": rest,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _DEFAULT_ANTHROPIC_VERSION,
            "User-Agent": self._user_agent,
        }

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                r = requests.post(
                    self._base_url,
                    json=payload,
                    headers=headers,
                    timeout=120,
                )
                if r.status_code != 200:
                    # Surface server errors with short body for diagnosis.
                    snippet = r.text[:300]
                    logger.warning(
                        "Kimi HTTP %s on attempt %d: %s",
                        r.status_code,
                        attempt + 1,
                        snippet,
                    )
                    if r.status_code in (429, 500, 502, 503, 504):
                        last_err = RuntimeError(f"HTTP {r.status_code}: {snippet}")
                        continue
                    r.raise_for_status()
                data = r.json()
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("Kimi request error (attempt %d): %s", attempt + 1, e)
        else:
            raise RuntimeError(f"Kimi request failed after 3 attempts: {last_err}")

        return self._parse_response(data)

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
        """Coerce one message to Anthropic string-content format.

        Anthropic accepts either string content or a list of blocks. For our
        ReAct loop, both user observations and assistant action records are
        plain strings, so the simplest thing is to coerce anything non-string
        to a string and drop role=tool messages (they should never appear when
        supports_tool_use() is False, but guard anyway).
        """
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "tool":
            # Fold tool results into a user message since we never use native
            # tool_use in bare-text mode.
            role = "user"
            content = f"[tool result] {content}"

        if not isinstance(content, str):
            # e.g. list of blocks — flatten to text
            parts: list[str] = []
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict):
                        parts.append(str(blk.get("text") or blk.get("content") or ""))
                    else:
                        parts.append(str(blk))
            content = "\n".join(parts) if parts else str(content)

        return {"role": role, "content": content}

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        # Anthropic response: {content:[{type:"text", text:"..."}], usage:{...}}
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                # Should not happen in bare-text mode, but handle defensively.
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}) or {},
                    )
                )

        usage = data.get("usage") or {}
        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=data.get("stop_reason") or "stop",
            usage={
                "prompt_tokens": int(usage.get("input_tokens", 0)),
                "completion_tokens": int(usage.get("output_tokens", 0)),
            } if usage else None,
        )
