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

Respond with a JSON object containing exactly ONE action:
{{"tool_name": "<action>", "params": {{...}}}}

If the system is stable with no queued jobs and no violations:
{{"tool_name": "none", "params": {{}}}}

## Strategy Guidelines

1. **Read available_nodes first**: The observation lists all nodes with free GPUs,
   including rack, gpu_free, cpu_free, ram_free_gb. Pick target nodes from this list.
2. **Check rack_affinity on each queued job**: If a job has rack_affinity, you MUST
   assign it to a node in that rack. Assigning elsewhere will be rejected.
3. **Preempt to free capacity**: If no available node has enough gpu_free for a
   queued job, use preempt_job on a preemptible running job to free GPUs, then
   assign_job on the next step. This is common when you've placed several jobs
   and remaining nodes are too full — don't keep retrying assign, switch to preempt.
   Example: rack-b is full but job-X needs rack-b → preempt_job a preemptible job
   on rack-b → next step assign_job job-X to the freed node.
4. **Migrate to resolve affinity deadlocks**: If a queued job needs a specific rack
   but all nodes there are full with non-preemptible jobs, use migrate_job to move
   a lower-priority job from that rack to another rack with free capacity, then
   assign the queued job.
   Example: job-X needs rack-a, rack-a-f4 is full → migrate_job a normal job from
   rack-a-f4 to rack-c-h3 (which has free GPUs) → assign_job job-X to rack-a-f4.
5. **Use all 6 tools**: assign_job places queued jobs. preempt_job frees GPUs by
   stopping low-priority jobs. migrate_job moves running jobs between nodes.
   restore_node brings failed nodes back. Don't rely only on assign_job.
6. **Priority conflicts**: Preempt preemptible jobs to make room for urgent ones.
7. **Pick the right node**: Match job GPU/CPU/RAM needs to a node with enough free
   resources. A node with gpu_free=2 cannot host a gpu=4 job.
"""
    return prompt
