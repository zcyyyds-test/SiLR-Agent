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
        param_aliases: dict[str, dict[str, str]] | None = None,
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
            param_aliases: tool_name → {wrong_param_name: correct_param_name}.
                Example: {"adjust_position": {"delta_qty": "qty_delta",
                "qty_change": "qty_delta", "qty": "qty_delta"}} redirects
                common typos back to the canonical parameter name so the
                teacher's first proposal doesn't fail on spelling alone.
        """
        self._allowed_actions = allowed_actions or frozenset()
        self._valid_tools = sorted(self._allowed_actions)
        self._valid_ids = valid_device_ids or {}
        self._aliases = aliases or {}
        self._numeric_fields = numeric_fields or set()
        self._id_field_map = id_field_map or {}
        self._param_aliases = param_aliases or {}

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

        # Try bare JSON object (supports one level of nesting). Match any
        # of the tool-name key variants a model might emit: tool_name /
        # tool / action / name. Qwen3 without enable_thinking tends to
        # emit the flat `{"tool": "...", ...}` schema.
        m = re.search(
            r'\{(?:[^{}]|\{[^{}]*\})*"(?:tool_name|tool|action|name)"(?:[^{}]|\{[^{}]*\})*\}',
            text, re.DOTALL)
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

        tool_name = (obj.get("tool_name") or obj.get("tool")
                     or obj.get("action") or obj.get("name"))
        if not tool_name:
            return None

        tool_name = self._normalize_tool_name(str(tool_name))

        # params can be nested under "params", "parameters", "arguments", or at top level
        _TOP_LEVEL_KEYS = {"tool_name", "tool", "action", "name",
                           "thought", "reasoning", "params",
                           "parameters", "arguments"}
        params = (
            obj.get("params")
            or obj.get("parameters")
            or obj.get("arguments")
            or {k: v for k, v in obj.items() if k not in _TOP_LEVEL_KEYS}
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

        # Parameter-name aliasing (e.g. Kimi's "delta_qty" → "qty_delta").
        aliases_for_tool = self._param_aliases.get(tool_name, {})
        for wrong, correct in aliases_for_tool.items():
            if wrong in result and correct not in result:
                result[correct] = result.pop(wrong)
                logger.info(
                    "Param alias: %s.%s → %s.%s", tool_name, wrong, tool_name, correct
                )

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
        """Extract reasoning/thought from LLM text before the action.

        Cuts the thought at whichever of the following comes first:
          - a blank line (``\n\n``)
          - a line starting with ``Action:`` (common in Kimi/Claude output)
          - a triple-backtick fence
          - a bare JSON object opening ``\n{``
        This prevents the action echo from leaking into the Thought record.
        If the text starts with a bare JSON object (no reasoning), return "".
        """
        stripped = text.lstrip()
        if stripped.startswith("{") and '"tool_name"' in stripped[:80]:
            # No reasoning at all — refuse to fabricate one from the JSON.
            return ""

        for marker in ["Thought:", "Reasoning:", "Analysis:"]:
            idx = text.find(marker)
            if idx >= 0:
                body_start = idx + len(marker)
                candidates = [
                    text.find("\n\n", body_start),
                    text.find("\nAction:", body_start),
                    text.find("\n```", body_start),
                    text.find("\n{", body_start),
                ]
                ends = [c for c in candidates if c >= 0]
                end = min(ends) if ends else len(text)
                return text[idx:end].strip()

        # No explicit Thought marker — return everything before the action.
        # Prefer a triple-backtick fence, otherwise stop at a bare JSON block
        # (newline followed by ``{``).
        json_start = text.find("```")
        bare_json = text.find("\n{")
        if json_start > 0 and (bare_json < 0 or json_start < bare_json):
            prose = text[:json_start].strip()
        elif bare_json > 0:
            prose = text[:bare_json].strip()
        else:
            prose = text.strip()

        if not prose or prose.startswith("{"):
            return ""
        # Keep multi-paragraph reasoning but cap to avoid runaway output.
        return prose[:2000]
