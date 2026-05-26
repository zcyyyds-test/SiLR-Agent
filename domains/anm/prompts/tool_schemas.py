"""OpenAI-format JSON Schema definitions for the gym-anm action tools.

Device IDs (gen / storage) and per-device bounds are populated dynamically from
the manager — the LLM gets the actual feasible range for each device so its
emitted set-points can be validated cheaply by the tool layer (and rejected
with Verdict.ERROR, not Verdict.FAIL, keeping training signal clean).
"""

from __future__ import annotations

from typing import Any

from ..manager import GymANMManager


def build_tool_schemas(manager: GymANMManager) -> list[dict[str, Any]]:
    """Return OpenAI function-calling tool definitions for the ANM action tools."""
    base = manager.base_mva
    gens_meta = []
    for g in manager.get_generator_ids():
        dev = manager._sim.devices[g]
        gens_meta.append(
            {
                "id": g,
                "p_min_mw": float(dev.p_min) * base,
                "p_max_mw": float(dev.p_max) * base,
                "q_min_mvar": float(dev.q_min) * base,
                "q_max_mvar": float(dev.q_max) * base,
            }
        )
    storage_meta = []
    for s in manager.get_storage_ids():
        dev = manager._sim.devices[s]
        storage_meta.append(
            {
                "id": s,
                "p_min_mw": float(dev.p_min) * base,
                "p_max_mw": float(dev.p_max) * base,
                "q_min_mvar": float(dev.q_min) * base,
                "q_max_mvar": float(dev.q_max) * base,
                "soc_min": float(dev.soc_min),
                "soc_max": float(dev.soc_max),
            }
        )
    return [
        _set_generator_setpoint_schema(gens_meta),
        _set_storage_setpoint_schema(storage_meta),
    ]


def get_valid_device_ids(manager: GymANMManager) -> dict[str, list]:
    """Extract valid device IDs for ActionParser validation."""
    return {
        "gen_id": list(manager.get_generator_ids()),
        "storage_id": list(manager.get_storage_ids()),
    }


def _set_generator_setpoint_schema(gens_meta: list[dict]) -> dict:
    gen_id_desc = (
        "Generator ID (integer). "
        + " | ".join(
            f"id={g['id']}: P∈[{g['p_min_mw']:.1f},{g['p_max_mw']:.1f}] MW, "
            f"Q∈[{g['q_min_mvar']:.1f},{g['q_max_mvar']:.1f}] MVAr"
            for g in gens_meta
        )
    )
    return {
        "type": "function",
        "function": {
            "name": "set_generator_setpoint",
            "description": (
                "Set a non-slack (renewable) generator's active power set-point "
                "in MW and optional reactive power in MVAr. p_mw is capped by the "
                "current renewable potential P_pot (visible in get_grid_status) — "
                "you cannot generate above the realized resource. "
                "Use this to curtail renewables that are stressing the network."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gen_id": {
                        "type": "integer",
                        "description": gen_id_desc,
                    },
                    "p_mw": {
                        "type": "number",
                        "description": (
                            "Active power set-point in MW. Must be within "
                            "[p_min_mw, min(p_max_mw, P_pot)] for this generator. "
                            "Out-of-bound values are rejected (Verdict.ERROR)."
                        ),
                    },
                    "q_mvar": {
                        "type": "number",
                        "description": (
                            "Optional reactive power set-point in MVAr. "
                            "Must be within [q_min_mvar, q_max_mvar]."
                        ),
                    },
                },
                "required": ["gen_id", "p_mw"],
            },
        },
    }


def _set_storage_setpoint_schema(storage_meta: list[dict]) -> dict:
    storage_desc = (
        "Storage unit ID (integer). "
        + " | ".join(
            f"id={s['id']}: P∈[{s['p_min_mw']:.1f},{s['p_max_mw']:.1f}] MW, "
            f"Q∈[{s['q_min_mvar']:.1f},{s['q_max_mvar']:.1f}] MVAr, "
            f"SoC∈[{s['soc_min']:.2f},{s['soc_max']:.2f}]"
            for s in storage_meta
        )
    )
    return {
        "type": "function",
        "function": {
            "name": "set_storage_setpoint",
            "description": (
                "Set a storage unit's active power set-point in MW (negative = "
                "charge, positive = discharge) and optional reactive power in MVAr. "
                "Use storage to absorb renewable surplus (charge, p_mw<0) when the "
                "grid is over-supplied, or inject (discharge, p_mw>0) when loads "
                "exceed local generation. SoC bounds [soc_min, soc_max] are "
                "monitored by the storage_soc checker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "storage_id": {
                        "type": "integer",
                        "description": storage_desc,
                    },
                    "p_mw": {
                        "type": "number",
                        "description": (
                            "Active power set-point in MW. Negative=charge, "
                            "positive=discharge. Must be within "
                            "[p_min_mw, p_max_mw]."
                        ),
                    },
                    "q_mvar": {
                        "type": "number",
                        "description": (
                            "Optional reactive power set-point in MVAr. "
                            "Must be within [q_min_mvar, q_max_mvar]."
                        ),
                    },
                },
                "required": ["storage_id", "p_mw"],
            },
        },
    }
