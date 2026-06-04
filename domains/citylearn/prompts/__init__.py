"""CityLearn district-storage domain prompt builders.

Both builders live in this module (mirroring the public API the config wires
up). The system prompt follows the 5-section structure used by the ANM domain:
role, tools, safety constraints, topology, ReAct protocol.
"""

from __future__ import annotations

from typing import Any

from .. import simulator as sim


# ── Tool schemas ─────────────────────────────────────────────────
def build_citylearn_tool_schemas(manager) -> list[dict[str, Any]]:
    """Return OpenAI function-calling tool definitions for the action tool.

    The observation tool (``get_district_status``) is intentionally omitted
    from the gated action schemas (it is not verifier-gated); the system prompt
    documents it separately. The discrete set-point catalog is surfaced as an
    ``enum`` so the LLM emits only admissible values.
    """
    choices = list(sim.ACTIONS_PER_BUILDING)
    indices = list(range(sim.N_BUILDINGS))
    index_desc = (
        "Building index (integer). "
        + " | ".join(
            f"index={b}: battery_{b}, "
            f"SoC∈[{sim.SOC_MIN_KWH[b]:.1f},{sim.SOC_MAX_KWH[b]:.1f}] kWh"
            for b in indices
        )
    )
    return [
        {
            "type": "function",
            "function": {
                "name": "set_building_setpoint",
                "description": (
                    "Set a building's battery power set-point in kW "
                    "(negative = charge the battery / pull from grid, "
                    "positive = discharge the battery / push to grid). "
                    "Use this to relieve SoC bound violations and district "
                    "import/export-limit violations. One building per call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "building_index": {
                            "type": "integer",
                            "description": index_desc,
                            "enum": indices,
                        },
                        "power_kw": {
                            "type": "number",
                            "description": (
                                "Battery set-point in kW. Must be one of the "
                                f"discrete choices {choices}. Negative = charge "
                                "(raises SoC, adds district import), positive = "
                                "discharge (lowers SoC, adds district export). "
                                "Out-of-set values are rejected (Verdict.ERROR)."
                            ),
                            "enum": choices,
                        },
                    },
                    "required": ["building_index", "power_kw"],
                },
            },
        },
    ]


def get_valid_device_ids(manager) -> dict[str, list]:
    """Extract valid device ids for ActionParser validation."""
    return {"building_index": list(range(sim.N_BUILDINGS))}


# ── System prompt ────────────────────────────────────────────────
def build_citylearn_system_prompt(
    manager,
    tool_schemas: list[dict[str, Any]],
) -> str:
    sections = [
        _section_role(),
        _section_tools(tool_schemas),
        _section_constraints(manager),
        _section_topology(manager),
        _section_protocol(),
    ]
    return "\n\n".join(sections)


def _section_role() -> str:
    return (
        "## Role\n\n"
        "You are a district energy manager for a cluster of buildings, each "
        "with its own battery. A single hour is fixed and the district starts "
        "in a constraint-violating state. Your job is to bring the district "
        "back to a feasible state by adjusting each building's battery "
        "set-point (charge or discharge power). Time does NOT advance: you are "
        "solving the steady state of one hour.\n\n"
        "You operate in a ReAct loop: at each step you observe the current "
        "district state, reason about which set-point change relieves a "
        "violation, and propose exactly ONE set_building_setpoint action. Your "
        "proposed action is verified on a shadow copy of the simulator by SiLR "
        "before being applied — if it would violate a constraint, the verifier "
        "rejects it and you receive feedback to revise."
    )


def _section_tools(tool_schemas: list[dict[str, Any]]) -> str:
    lines = ["## Available Action Tools\n"]
    for ts in tool_schemas:
        func = ts["function"]
        lines.append(f"### {func['name']}")
        lines.append(func["description"])
        params = func["parameters"]["properties"]
        req = func["parameters"].get("required", [])
        for pname, pdef in params.items():
            tag = "(required)" if pname in req else "(optional)"
            lines.append(f"- `{pname}` {tag}: {pdef.get('description', '')}")
        lines.append("")
    lines.append(
        "There is also an observation tool `get_district_status` (not gated by "
        "the verifier) that returns per-building SoC and bounds, district "
        "import/export vs. limits, peak demand, price, and the allowed "
        "set-point catalog — use it before each action to pick feasible values."
    )
    return "\n".join(lines)


def _section_constraints(manager) -> str:
    return (
        "## Safety Constraints (verifier-enforced)\n\n"
        f"- **Battery SoC**: each battery's state-of-charge must stay within "
        f"its [soc_min, soc_max] window (per building, in kWh). Discharging "
        f"(positive set-point) lowers SoC; charging (negative) raises it.\n"
        f"- **District import limit**: aggregate feeder import must be "
        f"≤ {sim.DISTRICT_IMPORT_LIMIT_KW:.1f} kW. Charging batteries and high "
        f"building load both raise import.\n"
        f"- **District export limit**: aggregate feeder back-feed (export) must "
        f"be ≤ {sim.DISTRICT_EXPORT_LIMIT_KW:.1f} kW. Discharging batteries and "
        f"high PV both raise export.\n\n"
        "Violating any of these on the verified shadow → Verdict.FAIL and the "
        "action is rejected. Out-of-catalog set-points / unknown building "
        "indices → Verdict.ERROR (parameter problem, not a safety verdict)."
    )


def _section_topology(manager) -> str:
    state = manager.system_state
    lines = [
        "## District Topology\n",
        f"- Hour (fixed): t = {state['t']}",
        f"- Price: {state['price']} $/kWh",
        f"- Buildings: {sim.N_BUILDINGS} (battery_0..battery_{sim.N_BUILDINGS - 1})",
    ]
    for b in state["buildings"]:
        lines.append(
            f"  - {b['id']} (index {b['index']}): "
            f"SoC∈[{b['soc_min_kwh']:.1f}, {b['soc_max_kwh']:.1f}] kWh"
        )
    lines.append(
        f"- Allowed battery set-points (kW): {list(sim.ACTIONS_PER_BUILDING)}"
    )
    lines.append(
        f"- District feeder: import ≤ {sim.DISTRICT_IMPORT_LIMIT_KW:.1f} kW, "
        f"export ≤ {sim.DISTRICT_EXPORT_LIMIT_KW:.1f} kW"
    )
    return "\n".join(lines)


def _section_protocol() -> str:
    return (
        "## Protocol\n\n"
        "1. Call `get_district_status` first to read per-building SoC, the "
        "current district import/export, and the allowed set-point catalog.\n"
        "2. Identify the violated constraint(s): which battery is out of its "
        "SoC window, or whether the district is over-importing / over-exporting.\n"
        "3. Reason about the lever:\n"
        "   - **SoC above max** → discharge that battery (positive set-point) "
        "to draw it down; **SoC below min** → charge it (negative set-point).\n"
        "   - **Over-export** → reduce total discharge: lower discharge "
        "set-points or switch some batteries to charge.\n"
        "   - **Over-import** → reduce total charge: lower charge set-points or "
        "switch some batteries to discharge.\n"
        "   - These couple: charging a low battery to fix SoC adds import; "
        "discharging a full one adds export. Coordinate across buildings.\n"
        "4. Emit exactly ONE tool call with a set-point from the allowed "
        "catalog — out-of-catalog = ERROR (wasted step).\n"
        "5. If the verifier rejects (FAIL), read the violation detail and "
        "revise. When all constraints are satisfied, stop."
    )
