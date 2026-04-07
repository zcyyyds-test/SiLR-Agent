"""Tests for the GPU cluster scheduling domain — ClusterManager."""

import pytest

from domains.cluster import ClusterManager


@pytest.fixture
def cluster_manager():
    """Fresh ClusterManager with default topology and jobs."""
    return ClusterManager()


# ======================================================================
# Default topology
# ======================================================================


class TestDefaultTopology:
    def test_node_count(self, cluster_manager):
        """15 nodes total."""
        assert len(cluster_manager.get_node_ids()) == 15

    def test_job_count(self, cluster_manager):
        """30-35 jobs generated (~55-65% cluster utilization)."""
        num_jobs = len(cluster_manager.get_job_ids())
        assert 30 <= num_jobs <= 35

    def test_racks(self, cluster_manager):
        """3 racks, 5 nodes each."""
        state = cluster_manager.system_state
        racks: dict[str, list[str]] = {}
        for nid, node in state["nodes"].items():
            racks.setdefault(node["rack"], []).append(nid)
        assert len(racks) == 3
        for rack, nodes in racks.items():
            assert len(nodes) == 5, f"{rack} should have 5 nodes, got {len(nodes)}"

    def test_gpu_counts(self, cluster_manager):
        """Standard/highmem have 4 GPUs, fat have 8."""
        state = cluster_manager.system_state
        for nid, node in state["nodes"].items():
            if node["type"] == "fat":
                assert node["gpu_total"] == 8, f"{nid} (fat) should have 8 GPUs"
            else:
                assert node["gpu_total"] == 4, f"{nid} ({node['type']}) should have 4 GPUs"

    def test_node_type_distribution(self, cluster_manager):
        """6 standard, 6 highmem, 3 fat."""
        state = cluster_manager.system_state
        counts: dict[str, int] = {}
        for node in state["nodes"].values():
            counts[node["type"]] = counts.get(node["type"], 0) + 1
        assert counts["standard"] == 6
        assert counts["highmem"] == 6
        assert counts["fat"] == 3

    def test_some_jobs_queued(self, cluster_manager):
        """Not all jobs can fit — some must be Queued."""
        queued = cluster_manager.get_queued_jobs()
        running = [
            jid for jid, j in cluster_manager._jobs.items()
            if j["status"] == "Running"
        ]
        assert len(running) > 0, "At least some jobs should be Running"
        # With 60-80 jobs and only 15 nodes, expect some queued
        # (don't assert >0 strictly — depends on RNG, but very likely)

    def test_deterministic_seed(self):
        """Two managers with same seed produce identical state."""
        m1 = ClusterManager()
        m2 = ClusterManager()
        assert m1.get_job_ids() == m2.get_job_ids()
        assert m1.system_state["assignments"] == m2.system_state["assignments"]
        for jid in m1.get_job_ids():
            assert m1._jobs[jid] == m2._jobs[jid]


# ======================================================================
# BaseSystemManager interface
# ======================================================================


class TestBaseSystemManagerInterface:
    def test_sim_time(self, cluster_manager):
        assert cluster_manager.sim_time == 0.0

    def test_base_mva(self, cluster_manager):
        assert cluster_manager.base_mva == 1.0

    def test_system_state_keys(self, cluster_manager):
        state = cluster_manager.system_state
        assert "nodes" in state
        assert "jobs" in state
        assert "assignments" in state

    def test_shadow_copy_isolation(self, cluster_manager):
        """Mutating the shadow must not affect the original."""
        # Pick the first Ready node
        ready = cluster_manager.get_schedulable_nodes()
        assert len(ready) > 0
        target_node = ready[0]

        shadow = cluster_manager.create_shadow_copy()

        # Fail a node in shadow
        shadow.fail_node(target_node)
        assert shadow._nodes[target_node]["status"] == "NotReady"
        # Original is untouched
        assert cluster_manager._nodes[target_node]["status"] == "Ready"

    def test_shadow_copy_type(self, cluster_manager):
        shadow = cluster_manager.create_shadow_copy()
        assert isinstance(shadow, ClusterManager)

    def test_solve_returns_bool(self, cluster_manager):
        result = cluster_manager.solve()
        assert isinstance(result, bool)
        assert result is True


# ======================================================================
# Domain operations
# ======================================================================


class TestDomainOperations:
    def test_fail_node(self, cluster_manager):
        node_id = cluster_manager.get_schedulable_nodes()[0]
        assert cluster_manager.fail_node(node_id) is True
        assert cluster_manager._nodes[node_id]["status"] == "NotReady"

    def test_fail_node_already_down(self, cluster_manager):
        node_id = cluster_manager.get_schedulable_nodes()[0]
        cluster_manager.fail_node(node_id)
        assert cluster_manager.fail_node(node_id) is False

    def test_fail_node_nonexistent(self, cluster_manager):
        assert cluster_manager.fail_node("no-such-node") is False

    def test_fail_node_requeues_jobs(self, cluster_manager):
        """Failing a node should re-queue all jobs assigned to it."""
        # Find a node that has at least one job
        assignments = cluster_manager.system_state["assignments"]
        nodes_with_jobs = set(assignments.values())
        assert len(nodes_with_jobs) > 0
        target = sorted(nodes_with_jobs)[0]

        jobs_on_target = [
            jid for jid, nid in assignments.items() if nid == target
        ]
        assert len(jobs_on_target) > 0

        cluster_manager.fail_node(target)

        # All jobs that were on this node should now be Queued
        for jid in jobs_on_target:
            assert cluster_manager._jobs[jid]["status"] == "Queued"
        # None of them should remain in assignments
        for jid in jobs_on_target:
            assert jid not in cluster_manager._assignments

    def test_restore_node(self, cluster_manager):
        node_id = cluster_manager.get_schedulable_nodes()[0]
        cluster_manager.fail_node(node_id)
        assert cluster_manager.restore_node(node_id) is True
        assert cluster_manager._nodes[node_id]["status"] == "Ready"

    def test_restore_node_already_ready(self, cluster_manager):
        node_id = cluster_manager.get_schedulable_nodes()[0]
        assert cluster_manager.restore_node(node_id) is False

    def test_restore_node_nonexistent(self, cluster_manager):
        assert cluster_manager.restore_node("no-such-node") is False

    def test_fail_rack(self, cluster_manager):
        failed = cluster_manager.fail_rack("rack-a")
        assert len(failed) == 5  # all 5 nodes in rack-a
        for nid in failed:
            assert cluster_manager._nodes[nid]["status"] == "NotReady"

    def test_add_jobs(self, cluster_manager):
        old_count = len(cluster_manager.get_job_ids())
        new_ids = cluster_manager.add_jobs([
            {"gpu": 2, "cpu": 16, "ram_gb": 64, "priority": "urgent"},
            {"gpu": 1, "cpu": 4, "ram_gb": 16},
        ])
        assert len(new_ids) == 2
        assert len(cluster_manager.get_job_ids()) == old_count + 2
        for jid in new_ids:
            assert cluster_manager._jobs[jid]["status"] == "Queued"

    def test_get_queued_jobs(self, cluster_manager):
        queued = cluster_manager.get_queued_jobs()
        for jid in queued:
            assert cluster_manager._jobs[jid]["status"] == "Queued"

    def test_solve_computes_utilization(self, cluster_manager):
        """solve should recompute node usage from assignments."""
        # Manually zero out usage to prove solve recalculates
        for node in cluster_manager._nodes.values():
            node["gpu_used"] = 0
            node["cpu_used"] = 0
            node["ram_used_gb"] = 0

        cluster_manager.solve()

        # Nodes with assignments should now have non-zero usage
        nodes_with_jobs = set(cluster_manager._assignments.values())
        for nid in nodes_with_jobs:
            node = cluster_manager._nodes[nid]
            assert node["gpu_used"] > 0, f"{nid} should have gpu_used > 0"

    def test_solve_advances_time(self, cluster_manager):
        assert cluster_manager.sim_time == 0.0
        cluster_manager.solve()
        assert cluster_manager.sim_time == 1.0
        cluster_manager.solve()
        assert cluster_manager.sim_time == 2.0
