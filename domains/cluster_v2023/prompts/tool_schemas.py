from __future__ import annotations


def build_tool_schemas(manager) -> list[dict]:
    return [
        {
            "name": "assign_job",
            "description": "Assign a Queued job to a Ready node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "node_id": {"type": "string"},
                },
                "required": ["job_id", "node_id"],
            },
        },
        {
            "name": "migrate_job",
            "description": "Move a Running job to another Ready node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "node_id": {"type": "string"},
                },
                "required": ["job_id", "node_id"],
            },
        },
        {
            "name": "preempt_job",
            "description": "Stop a Running job; it returns to Queued.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    ]
