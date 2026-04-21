"""3 scheduling tools for cluster_v2023: assign / migrate / preempt.

All mutations go through private manager attrs (`_jobs`, `_assignments`,
`_nodes`) — matching the cluster v1 house convention. Callers should
NOT mutate `manager.system_state[...]` directly (see manager.py
docstring).
"""

from __future__ import annotations

from silr.exceptions import (
    DeviceNotFoundError,
    SystemStateError,
    ValidationError,
)
from silr.tools.base import BaseTool


class AssignJobTool(BaseTool):
    name = "assign_job"
    description = "Assign a Queued job to a Ready node."

    def _validate_params(self, job_id: str = "", node_id: str = "", **_) -> None:
        if not job_id:
            raise ValidationError("job_id is required")
        if not node_id:
            raise ValidationError("node_id is required")
        mgr = self.manager
        if job_id not in mgr._jobs:
            raise DeviceNotFoundError(f"Job {job_id} not found")
        if node_id not in mgr._nodes:
            raise DeviceNotFoundError(f"Node {node_id} not found")
        if mgr._jobs[job_id]["status"] != "Queued":
            raise SystemStateError(
                f"Job {job_id} is {mgr._jobs[job_id]['status']}, "
                f"expected Queued")
        if mgr._nodes[node_id]["status"] != "Ready":
            raise SystemStateError(
                f"Node {node_id} is {mgr._nodes[node_id]['status']}, "
                f"expected Ready")

    def _run(self, job_id: str = "", node_id: str = "", **_) -> dict:
        mgr = self.manager
        mgr._jobs[job_id]["status"] = "Running"
        mgr._assignments[job_id] = node_id
        mgr._recompute_node_usage()
        return {"job_id": job_id, "node_id": node_id}


class MigrateJobTool(BaseTool):
    name = "migrate_job"
    description = "Move a Running job to a different Ready node."

    def _validate_params(self, job_id: str = "", node_id: str = "", **_) -> None:
        if not job_id:
            raise ValidationError("job_id is required")
        if not node_id:
            raise ValidationError("node_id is required")
        mgr = self.manager
        if job_id not in mgr._jobs:
            raise DeviceNotFoundError(f"Job {job_id} not found")
        if node_id not in mgr._nodes:
            raise DeviceNotFoundError(f"Node {node_id} not found")
        if mgr._jobs[job_id]["status"] != "Running":
            raise SystemStateError(f"Job {job_id} not Running")
        if mgr._nodes[node_id]["status"] != "Ready":
            raise SystemStateError(f"Node {node_id} not Ready")

    def _run(self, job_id: str = "", node_id: str = "", **_) -> dict:
        mgr = self.manager
        mgr._assignments[job_id] = node_id
        mgr._recompute_node_usage()
        return {"job_id": job_id, "node_id": node_id}


class PreemptJobTool(BaseTool):
    name = "preempt_job"
    description = "Stop a Running job; it returns to Queued."

    def _validate_params(self, job_id: str = "", **_) -> None:
        if not job_id:
            raise ValidationError("job_id is required")
        mgr = self.manager
        if job_id not in mgr._jobs:
            raise DeviceNotFoundError(f"Job {job_id} not found")
        if mgr._jobs[job_id]["status"] != "Running":
            raise SystemStateError(f"Job {job_id} not Running")

    def _run(self, job_id: str = "", **_) -> dict:
        mgr = self.manager
        mgr._jobs[job_id]["status"] = "Queued"
        mgr._assignments.pop(job_id, None)
        mgr._recompute_node_usage()
        return {"job_id": job_id}


def create_toolset(manager) -> dict:
    return {
        "assign_job": AssignJobTool(manager),
        "migrate_job": MigrateJobTool(manager),
        "preempt_job": PreemptJobTool(manager),
    }
