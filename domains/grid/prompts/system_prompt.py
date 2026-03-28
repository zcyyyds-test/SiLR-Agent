"""System prompt builder for the ReAct agent.

4-section structure:
1. Role definition — power system recovery expert
2. Tool Schema — 4 action tools with device ID enumerations
3. Safety constraints — V, f, rotor angle limits
4. Topology summary — IEEE 39-bus system info
"""

from __future__ import annotations

import json
from typing import Any

from ..simulator import SystemManager
from ..config import (
    VOLTAGE_MIN_PU, VOLTAGE_MAX_PU,
    FREQ_DEV_MAX_HZ,
    ROTOR_ANGLE_MAX_DEG,
    LINE_LOADING_NORMAL_PCT,
)


def build_system_prompt(
    manager: SystemManager,
    tool_schemas: list[dict[str, Any]],
) -> str:
    """Build the full system prompt for the ReAct agent.

    Called once at episode start. Includes static topology info.
    """
    sections = [
        _section_role(),
        _section_tools(tool_schemas),
        _section_constraints(),
        _section_topology(manager),
        _section_protocol(),
    ]
    return "\n\n".join(sections)


def _section_role() -> str:
    return """## Role

You are a power system recovery expert agent. Your task is to restore a power system to a safe operating state after a fault or disturbance.

You operate in a ReAct loop: at each step you observe the system state, reason about what action to take, and propose exactly ONE action using the available tools.

Your proposed action will be verified by a Simulation-in-the-Loop Reasoning (SiLR) verifier before being applied. If the verifier rejects your action, you will receive feedback explaining why and can propose a revised action."""


def _section_tools(tool_schemas: list[dict[str, Any]]) -> str:
    lines = ["## Available Tools\n"]
    for ts in tool_schemas:
        func = ts["function"]
        lines.append(f"### {func['name']}")
        lines.append(func["description"])
        params = func["parameters"]["properties"]
        for pname, pdef in params.items():
            req = "(required)" if pname in func["parameters"].get("required", []) else "(optional)"
            lines.append(f"  - `{pname}` ({pdef['type']}) {req}: {pdef['description']}")
        lines.append("")
    return "\n".join(lines)


def _section_constraints() -> str:
    return f"""## Safety Constraints

All proposed actions must satisfy the following constraints (verified by SiLR):

| Constraint | Limit |
|------------|-------|
| Bus voltage | {VOLTAGE_MIN_PU:.2f} – {VOLTAGE_MAX_PU:.2f} p.u. |
| Frequency deviation | |Δf| ≤ {FREQ_DEV_MAX_HZ} Hz |
| Line loading | ≤ {LINE_LOADING_NORMAL_PCT:.0f}% of rate_a |
| Rotor angle separation | < {ROTOR_ANGLE_MAX_DEG:.0f}° (COI-referenced) |

Actions that cause ANY constraint violation will be REJECTED. Start with moderate adjustments. If rejected for insufficient magnitude, increase by 50-100% on next attempt."""


def _section_topology(manager: SystemManager) -> str:
    """Generate topology summary from live system."""
    ss = manager.ss

    n_bus = ss.Bus.n
    n_line = ss.Line.n

    # Count generators
    gen_info = []
    for model_name in ["GENROU", "GENCLS"]:
        mdl = getattr(ss, model_name, None)
        if mdl is None or mdl.n == 0:
            continue
        idx_list = list(mdl.idx.v)
        for i, gid in enumerate(idx_list):
            static_idx = manager.get_static_gen_for_syn(gid, model_name)
            limits = manager.get_gen_limits(static_idx)
            from ..utils.unit_converter import pu_to_mw
            pmax_mw = pu_to_mw(limits["pmax"], manager.base_mva)
            p0_mw = pu_to_mw(limits["p0"], manager.base_mva)
            gen_info.append(
                f"  - {model_name}_{gid}: P0={p0_mw:.0f} MW, Pmax={pmax_mw:.0f} MW"
            )

    gen_lines = "\n".join(gen_info) if gen_info else "  (none)"

    bus_gen_map = """
Generator-Bus connections (IEEE 39-bus):
  GENROU_1 → Bus 30 (nearby: Bus 2, 25)
  GENROU_2 → Bus 31 (nearby: Bus 6, 5, 8)
  GENROU_3 → Bus 32 (nearby: Bus 10, 11, 12)
  GENROU_4 → Bus 33 (nearby: Bus 19, 20)
  GENROU_5 → Bus 34 (nearby: Bus 22, 23)
  GENROU_6 → Bus 35 (nearby: Bus 16, 24)
  GENROU_7 → Bus 36 (nearby: Bus 17, 27)
  GENROU_8 → Bus 37 (nearby: Bus 25, 26)
  GENROU_9 → Bus 38 (nearby: Bus 29)
  GENROU_10 → Bus 39 [SLACK BUS] (nearby: Bus 1, 9) — Note: This is the slack bus. In PFlow mode, adjusting its P output has NO effect (power balance overrides p0).

Voltage support guideline: To raise voltage at Bus X, increase output of the electrically closest generator."""

    return f"""## System Topology

IEEE 39-bus New England test system:
- Buses: {n_bus}
- Lines: {n_line}
- Generators: {len(gen_info)}

Generator capacities:
{gen_lines}
{bus_gen_map}"""


def _section_protocol() -> str:
    return """## Response Protocol

At each step, you MUST respond in this EXACT format:

Thought: <your reasoning here — MANDATORY, must NOT be empty>

<tool call or JSON action>

### Thought Requirements (CRITICAL)
Your Thought MUST include ALL of the following:
1. **Fault diagnosis**: What went wrong? Which buses/lines are affected?
2. **Tool selection rationale**: Why this specific tool and not another?
3. **Parameter justification**: Why this target device and this magnitude?

Example of a GOOD thought:
"Bus 4 voltage is 0.87 p.u., below the 0.90 threshold (gap: 0.03). GENROU_2 at Bus 31 is electrically close (via Line_6). Increasing GENROU_2 output by 80 MW should provide reactive support to raise Bus 4 voltage."

Example of a BAD thought (DO NOT DO THIS):
"" (empty — this is FORBIDDEN)

### Action Format
Call exactly ONE tool, or if the system is stable:
```json
{"tool_name": "none", "params": {}}
```

### Important Guidelines
- Start with moderate adjustments. For voltage violations, 50-100 MW is often needed.
- If your action is rejected due to insufficient magnitude, INCREASE significantly (do not repeat the same or smaller amount).
- For tripped lines: consider close_line FIRST — restoring topology is often more effective than adjust_gen/shed_load.
- If adjust_gen doesn't improve voltage after 2 attempts, switch to shed_load or close_line.
- Prefer adjust_gen over shed_load when no lines are tripped.
- For voltage issues: adjust nearby generators or shed load at affected buses.
- For line overloading: redistribute generation or shed load.
- NEVER skip the Thought section. Every action must be justified."""
