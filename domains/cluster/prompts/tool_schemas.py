"""OpenAI function-calling tool schemas for GPU cluster scheduling domain."""

from __future__ import annotations


def build_cluster_tool_schemas(manager) -> list[dict]:
    """Build OpenAI function-calling schemas for all 6 cluster tools.

    Args:
        manager: ClusterManager instance (for valid IDs in enums).

    Returns:
        List of schema dicts in OpenAI function-calling format.
    """
    node_ids = manager.get_node_ids()
    job_ids = manager.get_job_ids()

    return [
        {
            "type": "function",
            "function": {
                "name": "assign_job",
                "description": "Assign a queued job to a specific Ready node.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "ID of a Queued job to assign.",
                            "enum": job_ids,
                        },
                        "node_id": {
                            "type": "string",
                            "description": "ID of a Ready node to place the job on.",
                            "enum": node_ids,
                        },
                    },
                    "required": ["job_id", "node_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "migrate_job",
                "description": "Migrate a running job to a different target node.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "ID of a Running job to migrate.",
                            "enum": job_ids,
                        },
                        "target_node": {
                            "type": "string",
                            "description": "ID of the destination Ready node.",
                            "enum": node_ids,
                        },
                    },
                    "required": ["job_id", "target_node"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "preempt_job",
                "description": "Preempt a running job, returning it to the queue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "ID of a Running job to preempt.",
                            "enum": job_ids,
                        },
                    },
                    "required": ["job_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scale_job",
                "description": "Change the GPU allocation for a running job.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "ID of a Running job to scale.",
                            "enum": job_ids,
                        },
                        "gpu_count": {
                            "type": "integer",
                            "description": "New GPU count (must be positive).",
                            "minimum": 1,
                        },
                    },
                    "required": ["job_id", "gpu_count"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "drain_node",
                "description": (
                    "Drain a node by setting its status to Cordoned "
                    "(no new jobs will be scheduled on it)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "description": "ID of the node to drain.",
                            "enum": node_ids,
                        },
                    },
                    "required": ["node_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "restore_node",
                "description": "Restore a non-Ready node back to Ready status.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "description": "ID of the node to restore.",
                            "enum": node_ids,
                        },
                    },
                    "required": ["node_id"],
                },
            },
        },
    ]
