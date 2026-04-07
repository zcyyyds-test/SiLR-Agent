"""GPU cluster scheduling tools for SiLR framework.

6 tools: assign_job, migrate_job, preempt_job, scale_job, drain_node, restore_node.
All inherit from BaseTool for framework compatibility.
"""

from __future__ import annotations

from silr.tools.base import BaseTool
from silr.exceptions import (
    DeviceNotFoundError,
    SystemStateError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Job tools
# ---------------------------------------------------------------------------


class AssignJobTool(BaseTool):
    """Place a Queued job on a Ready node."""

    name = "assign_job"
    description = "Assign a queued job to a specific node"

    def _validate_params(self, job_id: str = "", node_id: str = "",
                         **kwargs) -> None:
        if not job_id:
            raise ValidationError("job_id is required")
        if not node_id:
            raise ValidationError("node_id is required")

        mgr = self.manager
        if job_id not in mgr._jobs:
            raise DeviceNotFoundError(f"Job {job_id} not found")
        if node_id not in mgr._nodes:
            raise DeviceNotFoundError(f"Node {node_id} not found")

        job = mgr._jobs[job_id]
        if job["status"] != "Queued":
            raise SystemStateError(
                f"Job {job_id} is {job['status']}, expected Queued"
            )

        node = mgr._nodes[node_id]
        if node["status"] != "Ready":
            raise SystemStateError(
                f"Node {node_id} is {node['status']}, expected Ready"
            )

    def _run(self, job_id: str = "", node_id: str = "", **kwargs) -> dict:
        mgr = self.manager
        mgr._jobs[job_id]["status"] = "Running"
        mgr._assignments[job_id] = node_id
        mgr._recompute_node_usage()
        return {
            "job_id": job_id,
            "node_id": node_id,
            "message": f"Job {job_id} assigned to {node_id}",
        }


class MigrateJobTool(BaseTool):
    """Move a Running job to a different node."""

    name = "migrate_job"
    description = "Migrate a running job to a different target node"

    def _validate_params(self, job_id: str = "", target_node: str = "",
                         **kwargs) -> None:
        if not job_id:
            raise ValidationError("job_id is required")
        if not target_node:
            raise ValidationError("target_node is required")

        mgr = self.manager
        if job_id not in mgr._jobs:
            raise DeviceNotFoundError(f"Job {job_id} not found")
        if target_node not in mgr._nodes:
            raise DeviceNotFoundError(f"Node {target_node} not found")

        job = mgr._jobs[job_id]
        if job["status"] != "Running":
            raise SystemStateError(
                f"Job {job_id} is {job['status']}, expected Running"
            )

        node = mgr._nodes[target_node]
        if node["status"] != "Ready":
            raise SystemStateError(
                f"Node {target_node} is {node['status']}, expected Ready"
            )

    def _run(self, job_id: str = "", target_node: str = "", **kwargs) -> dict:
        mgr = self.manager
        old_node = mgr._assignments.get(job_id, "unassigned")
        mgr._assignments[job_id] = target_node
        mgr._recompute_node_usage()
        return {
            "job_id": job_id,
            "old_node": old_node,
            "new_node": target_node,
            "message": f"Job {job_id} migrated from {old_node} to {target_node}",
        }


class PreemptJobTool(BaseTool):
    """Suspend a Running job, returning it to the queue."""

    name = "preempt_job"
    description = "Preempt a running job and return it to the queue"

    def _validate_params(self, job_id: str = "", **kwargs) -> None:
        if not job_id:
            raise ValidationError("job_id is required")

        mgr = self.manager
        if job_id not in mgr._jobs:
            raise DeviceNotFoundError(f"Job {job_id} not found")

        job = mgr._jobs[job_id]
        if job["status"] != "Running":
            raise SystemStateError(
                f"Job {job_id} is {job['status']}, expected Running"
            )

    def _run(self, job_id: str = "", **kwargs) -> dict:
        mgr = self.manager
        old_node = mgr._assignments.pop(job_id, "unassigned")
        mgr._jobs[job_id]["status"] = "Queued"
        mgr._recompute_node_usage()
        return {
            "job_id": job_id,
            "old_node": old_node,
            "message": f"Job {job_id} preempted from {old_node}, now Queued",
        }


class ScaleJobTool(BaseTool):
    """Change the GPU allocation for a Running job."""

    name = "scale_job"
    description = "Change the GPU count of a running job"

    def _validate_params(self, job_id: str = "", gpu_count: int = 0,
                         **kwargs) -> None:
        if not job_id:
            raise ValidationError("job_id is required")
        if gpu_count <= 0:
            raise ValidationError("gpu_count must be positive")

        mgr = self.manager
        if job_id not in mgr._jobs:
            raise DeviceNotFoundError(f"Job {job_id} not found")

        job = mgr._jobs[job_id]
        if job["status"] != "Running":
            raise SystemStateError(
                f"Job {job_id} is {job['status']}, expected Running"
            )

    def _run(self, job_id: str = "", gpu_count: int = 0, **kwargs) -> dict:
        mgr = self.manager
        old_gpu = mgr._jobs[job_id]["gpu"]
        mgr._jobs[job_id]["gpu"] = gpu_count
        mgr._recompute_node_usage()
        return {
            "job_id": job_id,
            "old_gpu": old_gpu,
            "new_gpu": gpu_count,
            "message": f"Job {job_id} scaled from {old_gpu} to {gpu_count} GPUs",
        }


# ---------------------------------------------------------------------------
# Node tools
# ---------------------------------------------------------------------------


class DrainNodeTool(BaseTool):
    """Mark a node as Cordoned (no new jobs will be scheduled)."""

    name = "drain_node"
    description = "Drain a node by setting its status to Cordoned"

    def _validate_params(self, node_id: str = "", **kwargs) -> None:
        if not node_id:
            raise ValidationError("node_id is required")

        mgr = self.manager
        if node_id not in mgr._nodes:
            raise DeviceNotFoundError(f"Node {node_id} not found")

    def _run(self, node_id: str = "", **kwargs) -> dict:
        mgr = self.manager
        old_status = mgr._nodes[node_id]["status"]
        mgr._nodes[node_id]["status"] = "Cordoned"
        return {
            "node_id": node_id,
            "old_status": old_status,
            "new_status": "Cordoned",
            "message": f"Node {node_id} drained ({old_status} -> Cordoned)",
        }


class RestoreNodeTool(BaseTool):
    """Bring a non-Ready node back to Ready status."""

    name = "restore_node"
    description = "Restore a node to Ready status"

    def _validate_params(self, node_id: str = "", **kwargs) -> None:
        if not node_id:
            raise ValidationError("node_id is required")

        mgr = self.manager
        if node_id not in mgr._nodes:
            raise DeviceNotFoundError(f"Node {node_id} not found")

        node = mgr._nodes[node_id]
        if node["status"] == "Ready":
            raise SystemStateError(f"Node {node_id} is already Ready")

    def _run(self, node_id: str = "", **kwargs) -> dict:
        mgr = self.manager
        old_status = mgr._nodes[node_id]["status"]
        mgr._nodes[node_id]["status"] = "Ready"
        return {
            "node_id": node_id,
            "old_status": old_status,
            "new_status": "Ready",
            "message": f"Node {node_id} restored ({old_status} -> Ready)",
        }


# ---------------------------------------------------------------------------
# Toolset factory
# ---------------------------------------------------------------------------


def create_cluster_toolset(manager) -> dict:
    """Create all cluster scheduling tools. Returns {name: tool} dict."""
    tools = [
        AssignJobTool(manager),
        MigrateJobTool(manager),
        PreemptJobTool(manager),
        ScaleJobTool(manager),
        DrainNodeTool(manager),
        RestoreNodeTool(manager),
    ]
    return {t.name: t for t in tools}
