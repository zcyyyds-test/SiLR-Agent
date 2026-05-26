"""System prompt builder for the gym-anm ReAct agent.

5-section structure mirroring grid/cluster prompts:
  1. Role
  2. Tool schema
  3. Safety constraints (voltage / branch / SoC)
  4. Topology summary (devices + base MVA)
  5. ReAct protocol
"""

from __future__ import annotations

import json
from typing import Any

from ..manager import GymANMManager


def build_system_prompt(
    manager: GymANMManager,
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
        "You are an Active Network Management (ANM) operator for a distribution "
        "network. Your job is to keep bus voltages, branch flows, and storage "
        "state-of-charge within their operational limits by setting active and "
        "reactive power set-points for renewable generators and storage units.\n\n"
        "You operate in a ReAct loop: at each step you observe the current grid "
        "state, reason about what action will relieve any stress, and propose "
        "exactly ONE set-point action using the available tools. Your proposed "
        "action is verified on a shadow copy of the simulator by SiLR before "
        "being applied — if a constraint would be violated, the verifier rejects "
        "the action and you receive feedback to propose a revised one."
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
        "There is also an observation tool `get_grid_status` (not gated by the "
        "verifier) that returns bus voltages, branch loadings, storage SoC, and "
        "actionable bounds (P_pot, p/q ranges, SoC range) — use it before each "
        "action to pick feasible set-points."
    )
    return "\n".join(lines)


def _section_constraints(manager: GymANMManager) -> str:
    sim = manager.system_state
    # ANM6-Easy uses per-bus [v_min, v_max] from gym-anm (typically [0.9, 1.1] p.u.).
    vlims = sorted({(float(b.v_min), float(b.v_max)) for b in sim.buses.values()})
    rates = [
        round(float(br.rate), 4)
        for br in sim.branches.values()
        if float(br.rate) > 0
    ]
    return (
        "## Safety Constraints (verifier-enforced)\n\n"
        f"- **Bus voltage**: each bus must satisfy V ∈ [v_min, v_max] p.u. "
        f"(window across buses: {vlims}).\n"
        f"- **Branch loading**: |S| ≤ rate on every branch with rate>0 "
        f"({len(rates)} rated branches; ratings in p.u.: {rates}).\n"
        f"- **Storage SoC**: each storage unit's SoC must stay within "
        f"[soc_min, soc_max] (visible in get_grid_status).\n\n"
        "Violating any of these on the verified shadow → Verdict.FAIL and the "
        "action is rejected. Out-of-bound parameters / unknown devices → "
        "Verdict.ERROR (parameter problem, not a safety verdict)."
    )


def _section_topology(manager: GymANMManager) -> str:
    sim = manager.system_state
    n_bus = len(sim.buses)
    return (
        "## Network Topology\n\n"
        f"- Buses: {n_bus} (slack + distribution buses)\n"
        f"- Loads: ids {list(manager._load_ids)} (passive demand)\n"
        f"- Non-slack generators: ids {list(manager.get_generator_ids())} "
        f"(renewable: wind / PV — output capped by current potential P_pot)\n"
        f"- Storage units: ids {list(manager.get_storage_ids())}\n"
        f"- base MVA: {manager.base_mva}"
    )


def _section_protocol() -> str:
    return (
        "## Protocol\n\n"
        "1. Call `get_grid_status` first to read current voltages, branch "
        "loadings, SoC, and device bounds (especially P_pot for each renewable).\n"
        "2. Identify the stressed buses / branches / storage from the observation.\n"
        "3. Reason about which action relieves the stress: "
        "**curtail** renewables (lower p_mw) when over-supply causes voltage "
        "rise or branch overload; **charge storage** (negative p_mw) to absorb "
        "surplus; **discharge storage** (positive p_mw) when local generation "
        "is insufficient. Note: when overload is *load-driven*, curtailing "
        "renewables can make it worse — storage discharge or accepting slack "
        "import may be the right lever.\n"
        "4. Emit exactly ONE tool call with set-point values within each "
        "device's [p_min, p_max] (in MW) — out-of-bound = ERROR (wasted step).\n"
        "5. If verifier rejects (FAIL), read the violation detail and revise."
    )
