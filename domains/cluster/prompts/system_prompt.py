"""System prompt builder for the GPU cluster scheduling domain."""

from __future__ import annotations

from typing import Any


def build_cluster_system_prompt(manager, tool_schemas: list[dict]) -> str:
    """Build a system prompt describing the GPU cluster and available actions.

    Args:
        manager: ClusterManager instance (for topology introspection).
        tool_schemas: List of OpenAI function-calling schema dicts.

    Returns:
        System prompt string for LLM consumption.
    """
    state = manager.system_state
    nodes = state["nodes"]
    jobs = state["jobs"]

    # Summarise topology
    racks: dict[str, list[str]] = {}
    for nid, n in sorted(nodes.items()):
        racks.setdefault(n["rack"], []).append(nid)

    rack_lines = []
    for rack in sorted(racks.keys()):
        nids = racks[rack]
        types = [nodes[nid]["type"] for nid in nids]
        type_summary = ", ".join(
            f"{t}({nodes[nid]['gpu_total']}G)"
            for nid, t in zip(nids, types)
        )
        rack_lines.append(f"  {rack}: {type_summary}")
    topology_block = "\n".join(rack_lines)

    # Job summary
    total_jobs = len(jobs)
    running = sum(1 for j in jobs.values() if j["status"] == "Running")
    queued = sum(1 for j in jobs.values() if j["status"] == "Queued")

    # Tool names
    tool_names = [s["function"]["name"] for s in tool_schemas]

    prompt = f"""\
You are a GPU cluster scheduler operator. Your job is to manage job scheduling
across a multi-rack GPU cluster, ensuring all constraints are satisfied.

## Cluster Topology ({len(nodes)} nodes, {len(racks)} racks)

{topology_block}

Node types:
  standard : 4 GPU (80GB), 64 CPU,  256 GB RAM
  highmem  : 4 GPU (80GB), 64 CPU,  512 GB RAM
  fat      : 8 GPU (80GB), 128 CPU, 1024 GB RAM

## Current Workload

  Total jobs: {total_jobs}  |  Running: {running}  |  Queued: {queued}

## Constraints

1. **Resource capacity**: No node may exceed its GPU, CPU, or RAM limits.
2. **Rack affinity**: Jobs with rack_affinity must run in the specified rack.
3. **Rack spread**: Urgent job groups (2+ jobs) must span 2+ racks.
4. **Priority**: No urgent job may be queued while preemptible jobs are running.
5. **Queue clearance**: All jobs should be scheduled (no queued jobs remaining).

## Available Actions

{', '.join(tool_names)}

## Response Format

Respond with a JSON object:
{{
  "thought": "<brief reasoning about the current situation>",
  "actions": [
    {{"tool_name": "<action>", "params": {{...}}}}
  ]
}}

Guidelines:
- Propose ONE action at a time for safety verification.
- Prioritise urgent jobs over normal jobs over preemptible jobs.
- Prefer assigning to nodes with spare capacity before preempting.
- Respect rack affinity constraints when placing jobs.
- If the system is stable with no violations, respond with an empty actions list.
"""
    return prompt
