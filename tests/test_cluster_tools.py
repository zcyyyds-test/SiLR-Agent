"""Tests for GPU cluster scheduling tools."""

import pytest

from domains.cluster.manager import ClusterManager
from domains.cluster.tools import create_cluster_toolset


@pytest.fixture
def cluster_manager():
    """Fresh ClusterManager with default topology and jobs."""
    return ClusterManager()


@pytest.fixture
def tools(cluster_manager):
    """Full cluster toolset."""
    return create_cluster_toolset(cluster_manager)


@pytest.fixture
def cluster_with_queued_job(cluster_manager):
    """Manager with job-0000 manually set to Queued and unassigned."""
    mgr = cluster_manager
    mgr._jobs["job-0000"]["status"] = "Queued"
    mgr._assignments.pop("job-0000", None)
    mgr._recompute_node_usage()
    return mgr


# -------------------------------------------------------------------
# create_cluster_toolset
# -------------------------------------------------------------------


class TestCreateClusterToolset:
    def test_returns_all_six_tools(self, tools):
        expected = {
            "assign_job",
            "migrate_job",
            "preempt_job",
            "scale_job",
            "drain_node",
            "restore_node",
        }
        assert set(tools.keys()) == expected

    def test_tools_have_descriptions(self, tools):
        for name, tool in tools.items():
            assert tool.description, f"{name} has empty description"


# -------------------------------------------------------------------
# AssignJobTool
# -------------------------------------------------------------------


class TestAssignJobTool:
    def test_assign_success(self, cluster_with_queued_job):
        mgr = cluster_with_queued_job
        tools = create_cluster_toolset(mgr)
        node_id = mgr.get_schedulable_nodes()[0]

        result = tools["assign_job"].execute(job_id="job-0000", node_id=node_id)

        assert result["status"] == "success"
        assert mgr._jobs["job-0000"]["status"] == "Running"
        assert mgr._assignments["job-0000"] == node_id

    def test_assign_missing_job_id(self, tools):
        result = tools["assign_job"].execute(node_id="rack-a-s0")
        assert result["status"] == "error"

    def test_assign_nonexistent_job(self, tools):
        result = tools["assign_job"].execute(job_id="nope", node_id="rack-a-s0")
        assert result["status"] == "error"

    def test_assign_nonexistent_node(self, cluster_with_queued_job):
        tools = create_cluster_toolset(cluster_with_queued_job)
        result = tools["assign_job"].execute(job_id="job-0000", node_id="nope")
        assert result["status"] == "error"

    def test_assign_non_queued_job(self, cluster_manager):
        """Assigning a Running job should fail."""
        tools = create_cluster_toolset(cluster_manager)
        # job-0000 is Running by default
        running_jobs = [
            jid for jid, j in cluster_manager._jobs.items()
            if j["status"] == "Running"
        ]
        assert len(running_jobs) > 0
        jid = running_jobs[0]
        node = cluster_manager.get_schedulable_nodes()[0]
        result = tools["assign_job"].execute(job_id=jid, node_id=node)
        assert result["status"] == "error"

    def test_assign_to_non_ready_node(self, cluster_with_queued_job):
        mgr = cluster_with_queued_job
        tools = create_cluster_toolset(mgr)
        # Drain a node first
        node_id = mgr.get_schedulable_nodes()[0]
        mgr._nodes[node_id]["status"] = "Cordoned"
        result = tools["assign_job"].execute(job_id="job-0000", node_id=node_id)
        assert result["status"] == "error"


# -------------------------------------------------------------------
# MigrateJobTool
# -------------------------------------------------------------------


class TestMigrateJobTool:
    def test_migrate_success(self, cluster_manager):
        tools = create_cluster_toolset(cluster_manager)
        # Pick a running job and a different schedulable node
        running = [
            jid for jid, j in cluster_manager._jobs.items()
            if j["status"] == "Running"
        ]
        jid = running[0]
        old_node = cluster_manager._assignments[jid]
        targets = [
            nid for nid in cluster_manager.get_schedulable_nodes()
            if nid != old_node
        ]
        target = targets[0]

        result = tools["migrate_job"].execute(job_id=jid, target_node=target)
        assert result["status"] == "success"
        assert cluster_manager._assignments[jid] == target

    def test_migrate_missing_params(self, tools):
        result = tools["migrate_job"].execute()
        assert result["status"] == "error"

    def test_migrate_nonexistent_job(self, tools):
        result = tools["migrate_job"].execute(job_id="nope", target_node="rack-a-s0")
        assert result["status"] == "error"

    def test_migrate_non_running_job(self, cluster_with_queued_job):
        tools = create_cluster_toolset(cluster_with_queued_job)
        node = cluster_with_queued_job.get_schedulable_nodes()[0]
        result = tools["migrate_job"].execute(job_id="job-0000", target_node=node)
        assert result["status"] == "error"


# -------------------------------------------------------------------
# PreemptJobTool
# -------------------------------------------------------------------


class TestPreemptJobTool:
    def test_preempt_success(self, cluster_manager):
        tools = create_cluster_toolset(cluster_manager)
        running = [
            jid for jid, j in cluster_manager._jobs.items()
            if j["status"] == "Running"
        ]
        jid = running[0]

        result = tools["preempt_job"].execute(job_id=jid)
        assert result["status"] == "success"
        assert cluster_manager._jobs[jid]["status"] == "Queued"
        assert jid not in cluster_manager._assignments

    def test_preempt_missing_job_id(self, tools):
        result = tools["preempt_job"].execute()
        assert result["status"] == "error"

    def test_preempt_nonexistent_job(self, tools):
        result = tools["preempt_job"].execute(job_id="nope")
        assert result["status"] == "error"

    def test_preempt_non_running_job(self, cluster_with_queued_job):
        tools = create_cluster_toolset(cluster_with_queued_job)
        result = tools["preempt_job"].execute(job_id="job-0000")
        assert result["status"] == "error"


# -------------------------------------------------------------------
# ScaleJobTool
# -------------------------------------------------------------------


class TestScaleJobTool:
    def test_scale_success(self, cluster_manager):
        tools = create_cluster_toolset(cluster_manager)
        running = [
            jid for jid, j in cluster_manager._jobs.items()
            if j["status"] == "Running"
        ]
        jid = running[0]
        old_gpu = cluster_manager._jobs[jid]["gpu"]

        result = tools["scale_job"].execute(job_id=jid, gpu_count=2)
        assert result["status"] == "success"
        assert cluster_manager._jobs[jid]["gpu"] == 2
        assert result["data"]["old_gpu"] == old_gpu

    def test_scale_zero_gpus(self, cluster_manager):
        tools = create_cluster_toolset(cluster_manager)
        running = [
            jid for jid, j in cluster_manager._jobs.items()
            if j["status"] == "Running"
        ]
        result = tools["scale_job"].execute(job_id=running[0], gpu_count=0)
        assert result["status"] == "error"

    def test_scale_negative_gpus(self, cluster_manager):
        tools = create_cluster_toolset(cluster_manager)
        running = [
            jid for jid, j in cluster_manager._jobs.items()
            if j["status"] == "Running"
        ]
        result = tools["scale_job"].execute(job_id=running[0], gpu_count=-1)
        assert result["status"] == "error"

    def test_scale_nonexistent_job(self, tools):
        result = tools["scale_job"].execute(job_id="nope", gpu_count=2)
        assert result["status"] == "error"

    def test_scale_non_running_job(self, cluster_with_queued_job):
        tools = create_cluster_toolset(cluster_with_queued_job)
        result = tools["scale_job"].execute(job_id="job-0000", gpu_count=2)
        assert result["status"] == "error"


# -------------------------------------------------------------------
# DrainNodeTool
# -------------------------------------------------------------------


class TestDrainNodeTool:
    def test_drain_success(self, cluster_manager):
        tools = create_cluster_toolset(cluster_manager)
        node_id = cluster_manager.get_schedulable_nodes()[0]

        result = tools["drain_node"].execute(node_id=node_id)
        assert result["status"] == "success"
        assert cluster_manager._nodes[node_id]["status"] == "Cordoned"

    def test_drain_missing_node_id(self, tools):
        result = tools["drain_node"].execute()
        assert result["status"] == "error"

    def test_drain_nonexistent_node(self, tools):
        result = tools["drain_node"].execute(node_id="nope")
        assert result["status"] == "error"

    def test_drain_already_cordoned(self, cluster_manager):
        """Draining an already-Cordoned node should still succeed."""
        tools = create_cluster_toolset(cluster_manager)
        node_id = cluster_manager.get_schedulable_nodes()[0]
        cluster_manager._nodes[node_id]["status"] = "Cordoned"

        result = tools["drain_node"].execute(node_id=node_id)
        assert result["status"] == "success"
        assert result["data"]["old_status"] == "Cordoned"


# -------------------------------------------------------------------
# RestoreNodeTool
# -------------------------------------------------------------------


class TestRestoreNodeTool:
    def test_restore_success(self, cluster_manager):
        tools = create_cluster_toolset(cluster_manager)
        node_id = cluster_manager.get_schedulable_nodes()[0]
        # Drain it first
        cluster_manager._nodes[node_id]["status"] = "Cordoned"

        result = tools["restore_node"].execute(node_id=node_id)
        assert result["status"] == "success"
        assert cluster_manager._nodes[node_id]["status"] == "Ready"
        assert result["data"]["old_status"] == "Cordoned"

    def test_restore_not_ready_node(self, cluster_manager):
        tools = create_cluster_toolset(cluster_manager)
        node_id = cluster_manager.get_schedulable_nodes()[0]
        cluster_manager._nodes[node_id]["status"] = "NotReady"

        result = tools["restore_node"].execute(node_id=node_id)
        assert result["status"] == "success"
        assert cluster_manager._nodes[node_id]["status"] == "Ready"

    def test_restore_already_ready(self, cluster_manager):
        """Restoring an already-Ready node should fail."""
        tools = create_cluster_toolset(cluster_manager)
        node_id = cluster_manager.get_schedulable_nodes()[0]

        result = tools["restore_node"].execute(node_id=node_id)
        assert result["status"] == "error"

    def test_restore_missing_node_id(self, tools):
        result = tools["restore_node"].execute()
        assert result["status"] == "error"

    def test_restore_nonexistent_node(self, tools):
        result = tools["restore_node"].execute(node_id="nope")
        assert result["status"] == "error"
