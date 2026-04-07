"""JSON Schema definitions for the 4 allowed action tools.

Device ID enums are dynamically populated from the SystemManager.
"""

from __future__ import annotations

from typing import Any

from ..simulator import SystemManager


def build_tool_schemas(manager: SystemManager) -> list[dict[str, Any]]:
    """Build OpenAI-format tool definitions with dynamic device IDs.

    Returns list of tool dicts in OpenAI function-calling format.
    """
    gen_ids = manager.get_all_syn_gen_idx()
    bus_ids = manager.get_bus_idx_list()
    line_ids = manager.get_line_idx_list()

    return [
        _adjust_gen_schema(gen_ids),
        _shed_load_schema(bus_ids),
        _trip_line_schema(line_ids),
        _close_line_schema(line_ids),
    ]


def get_valid_device_ids(manager: SystemManager) -> dict[str, list]:
    """Extract valid device IDs for ActionParser validation."""
    return {
        "gen_id": manager.get_all_syn_gen_idx(),
        "bus_id": manager.get_bus_idx_list(),
        "line_id": manager.get_line_idx_list(),
    }


def _adjust_gen_schema(gen_ids: list) -> dict:
    gen_id_desc = f"Generator ID (string). Valid: {gen_ids}" if len(gen_ids) <= 20 else "Generator ID (string, e.g. 'GENROU_1')"
    return {
        "type": "function",
        "function": {
            "name": "adjust_gen",
            "description": (
                "Adjust a synchronous generator's active power output. "
                "Positive delta_p_mw increases generation, negative decreases. "
                "Clamped to generator limits automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gen_id": {
                        "type": "string",
                        "description": gen_id_desc,
                    },
                    "delta_p_mw": {
                        "type": "number",
                        "description": "Change in active power output (MW). Positive = increase, negative = decrease.",
                    },
                },
                "required": ["gen_id", "delta_p_mw"],
            },
        },
    }


def _shed_load_schema(bus_ids: list) -> dict:
    bus_desc = f"Bus ID. Valid: {bus_ids}" if len(bus_ids) <= 50 else "Bus ID (integer)"
    return {
        "type": "function",
        "function": {
            "name": "shed_load",
            "description": (
                "Shed (reduce) load at a specified bus. "
                "The amount is distributed proportionally across all PQ loads at the bus. "
                "Clamped to total bus load."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bus_id": {
                        "type": "integer",
                        "description": bus_desc,
                    },
                    "amount_mw": {
                        "type": "number",
                        "description": "Amount of load to shed (MW). Must be positive.",
                    },
                },
                "required": ["bus_id", "amount_mw"],
            },
        },
    }


def _trip_line_schema(line_ids: list) -> dict:
    line_desc = f"Line ID (string). Valid: {line_ids}" if len(line_ids) <= 50 else "Line ID (string, e.g. 'Line_1')"
    return {
        "type": "function",
        "function": {
            "name": "trip_line",
            "description": "Trip (disconnect) a transmission line. Use to isolate faulted or overloaded lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line_id": {
                        "type": "string",
                        "description": line_desc,
                    },
                },
                "required": ["line_id"],
            },
        },
    }


def _close_line_schema(line_ids: list) -> dict:
    line_desc = f"Line ID (string). Valid: {line_ids}" if len(line_ids) <= 50 else "Line ID (string, e.g. 'Line_1')"
    return {
        "type": "function",
        "function": {
            "name": "close_line",
            "description": "Close (reconnect) a previously tripped transmission line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line_id": {
                        "type": "string",
                        "description": line_desc,
                    },
                },
                "required": ["line_id"],
            },
        },
    }
