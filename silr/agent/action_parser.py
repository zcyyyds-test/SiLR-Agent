"""Parse LLM output into action dicts — three-layer fallback.

Layer 1: Native tool_calls from API response
Layer 2: JSON code block extraction (```json ... ```)
Layer 3: Regex fallback for loose text
"""

from __future__ import annotations

import json
import logging
import re
from difflib import get_close_matches
from typing import Any

from .llm.base import LLMResponse

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Action parsing failed across all layers."""


class ActionParser:
    """Three-layer action parser with fuzzy matching and type coercion.

    Domain-agnostic: allowed actions, aliases, and coercion rules are
    injected via constructor parameters.
    """

    def __init__(
        self,
        allowed_actions: frozenset[str] | None = None,
        valid_device_ids: dict[str, list] | None = None,
        aliases: dict[str, str] | None = None,
        numeric_fields: set[str] | None = None,
        id_field_map: dict[str, str] | None = None,
    ):
        """
        Args:
            allowed_actions: Set of valid tool names for fuzzy matching.
            valid_device_ids: Mapping of device type to valid IDs,
                e.g. {"gen_id": [1, 2, ...], "bus_id": [...]}
            aliases: Common misspelling/alias → canonical tool name mapping.
            numeric_fields: Parameter names that should be coerced to float.
            id_field_map: tool_name → parameter name that holds a device ID,
                e.g. {"adjust_gen": "gen_id"}.
        """
        self._allowed_actions = allowed_actions or frozenset()
        self._valid_tools = sorted(self._allowed_actions)
        self._valid_ids = valid_device_ids or {}
        self._aliases = aliases or {}
        self._numeric_fields = numeric_fields or set()
        self._id_field_map = id_field_map or {}

    def parse(self, response: LLMResponse) -> tuple[str, dict]:
        """Parse LLM response into (thought, action_dict).

        Returns:
            (thought_text, {"tool_name": str, "params": dict})

        Raises:
            ParseError if all three layers fail.
        """
        # Layer 1: Native tool_calls
        if response.tool_calls:
            tc = response.tool_calls[0]
            tool_name = self._normalize_tool_name(tc.name)
            params = self._coerce_params(tool_name, tc.arguments)
            thought = response.content or ""
            return thought, {"tool_name": tool_name, "params": params}

        text = response.content.strip()
        if not text:
            raise ParseError("Empty LLM response")

        # Layer 2: JSON code block
        action = self._try_json_block(text)
        if action is not None:
            thought = self._extract_thought(text)
            return thought, action

        # Layer 3: Regex fallback
        action = self._try_regex(text)
        if action is not None:
            thought = self._extract_thought(text)
            return thought, action

        raise ParseError(f"Could not parse action from: {text[:200]}")

    def _try_json_block(self, text: str) -> dict | None:
        """Extract action from ```json ... ``` or bare JSON object."""
        patterns = [
            r"```json\s*\n?(.*?)\n?\s*```",
            r"```\s*\n?(.*?)\n?\s*```",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL)
            if m:
                return self._parse_json_action(m.group(1).strip())

        # Try bare JSON object (supports one level of nesting)
        m = re.search(r'\{(?:[^{}]|\{[^{}]*\})*"tool_name"(?:[^{}]|\{[^{}]*\})*\}', text, re.DOTALL)
        if m:
            return self._parse_json_action(m.group(0))

        return None

    def _parse_json_action(self, json_str: str) -> dict | None:
        """Parse a JSON string into an action dict."""
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(obj, dict):
            return None

        tool_name = obj.get("tool_name") or obj.get("action") or obj.get("name")
        if not tool_name:
            return None

        tool_name = self._normalize_tool_name(str(tool_name))

        # params can be nested under "params", "parameters", "arguments", or at top level
        params = (
            obj.get("params")
            or obj.get("parameters")
            or obj.get("arguments")
            or {k: v for k, v in obj.items()
                if k not in ("tool_name", "action", "name", "thought", "reasoning")}
        )

        params = self._coerce_params(tool_name, params)
        return {"tool_name": tool_name, "params": params}

    def _try_regex(self, text: str) -> dict | None:
        """Last resort: match tool_name(param=value, ...) pattern."""
        for tool in self._valid_tools:
            pat = rf"{tool}\s*\(\s*(.*?)\s*\)"
            m = re.search(pat, text, re.DOTALL)
            if m:
                params = self._parse_kwargs_string(m.group(1))
                params = self._coerce_params(tool, params)
                return {"tool_name": tool, "params": params}
        return None

    def _parse_kwargs_string(self, s: str) -> dict:
        """Parse 'key=value, key=value' into dict."""
        params = {}
        for part in re.split(r",\s*", s):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip().strip("'\"")
                v = v.strip().strip("'\"")
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                params[k] = v
        return params

    def _normalize_tool_name(self, name: str) -> str:
        """Fuzzy-match tool name to valid set."""
        name = name.strip().lower().replace("-", "_").replace(" ", "_")
        if name in self._allowed_actions:
            return name

        # Fuzzy match
        if self._valid_tools:
            matches = get_close_matches(name, self._valid_tools, n=1, cutoff=0.8)
            if matches:
                logger.warning(f"Fuzzy-matched tool name '{name}' → '{matches[0]}'")
                return matches[0]

        # Check aliases
        if name in self._aliases:
            return self._aliases[name]

        return name  # return as-is, verifier will reject invalid names

    def _coerce_params(self, tool_name: str, params: dict) -> dict:
        """Type-coerce parameters based on configured rules."""
        if not isinstance(params, dict):
            return {}

        result = dict(params)

        # Numeric coercion for configured fields
        for field_name in self._numeric_fields:
            if field_name in result:
                try:
                    result[field_name] = float(result[field_name])
                except (ValueError, TypeError):
                    pass

        # Device ID validation
        id_field = self._id_field_map.get(tool_name)
        if id_field and id_field in result and id_field in self._valid_ids:
            result[id_field] = self._coerce_device_id(
                result[id_field], self._valid_ids[id_field]
            )

        return result

    def _coerce_device_id(self, raw_id: Any, valid_ids: list) -> Any:
        """Try to match raw_id to a valid device ID."""
        if raw_id in valid_ids:
            return raw_id

        # Try numeric conversion (LLM may return string "1" for int ID 1)
        try:
            numeric = int(raw_id)
            if numeric in valid_ids:
                return numeric
        except (ValueError, TypeError):
            pass

        # Try string match for GENROU_1 style IDs
        raw_str = str(raw_id)
        for vid in valid_ids:
            if str(vid) == raw_str:
                return vid

        return raw_id  # return as-is, tool layer will validate

    @staticmethod
    def _extract_thought(text: str) -> str:
        """Extract reasoning/thought from LLM text before the action."""
        for marker in ["Thought:", "Reasoning:", "Analysis:"]:
            idx = text.find(marker)
            if idx >= 0:
                end = text.find("\n\n", idx)
                if end < 0:
                    end = text.find("```", idx)
                if end < 0:
                    end = len(text)
                return text[idx:end].strip()

        # Return text before JSON block
        json_start = text.find("```")
        if json_start > 0:
            return text[:json_start].strip()

        # Return first paragraph
        first_para = text.split("\n\n")[0]
        if len(first_para) < 500:
            return first_para
        return first_para[:500]
