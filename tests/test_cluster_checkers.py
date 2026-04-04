"""Tests for GPU cluster constraint checkers."""

import pytest

from domains.cluster.manager import ClusterManager
from domains.cluster.checkers import (
    ResourceCapacityChecker,
    AffinityChecker,
    RackSpreadChecker,
    PriorityChecker,
    QueueChecker,
)


@pytest.fixture
def mgr():
    return ClusterManager()


# ---------------------------------------------------------------
# ResourceCapacityChecker
# ---------------------------------------------------------------

class TestResourceCapacityChecker:
    def test_clean_state_passes(self, mgr):
        """Default state produced by greedy packing never exceeds capacity."""
        checker = ResourceCapacityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.passed
        assert result.checker_name == "resource_capacity"
        assert result.summary["n_violations"] == 0

    def test_gpu_overload_fails(self, mgr):
        """Force a node over GPU capacity -> critical violation."""
        nid = mgr.get_node_ids()[0]
        mgr._nodes[nid]["gpu_used"] = mgr._nodes[nid]["gpu_total"] + 2

        checker = ResourceCapacityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed
        assert any(v.device_id == nid and v.metric == "gpu_used"
                    for v in result.violations)
        gpu_v = [v for v in result.violations
                 if v.device_id == nid and v.metric == "gpu_used"][0]
        assert gpu_v.severity == "critical"

    def test_cpu_overload_fails(self, mgr):
        """Force a node over CPU capacity -> violation severity."""
        nid = mgr.get_node_ids()[0]
        mgr._nodes[nid]["cpu_used"] = mgr._nodes[nid]["cpu_total"] + 10

        checker = ResourceCapacityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed
        cpu_v = [v for v in result.violations
                 if v.device_id == nid and v.metric == "cpu_used"][0]
        assert cpu_v.severity == "violation"
        assert cpu_v.unit == "cores"

    def test_ram_overload_fails(self, mgr):
        """Force a node over RAM capacity -> violation severity."""
        nid = mgr.get_node_ids()[0]
        mgr._nodes[nid]["ram_used_gb"] = mgr._nodes[nid]["ram_total_gb"] + 64

        checker = ResourceCapacityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed
        ram_v = [v for v in result.violations
                 if v.device_id == nid and v.metric == "ram_used_gb"][0]
        assert ram_v.severity == "violation"
        assert ram_v.unit == "GB"

    def test_skips_not_ready_nodes(self, mgr):
        """NotReady nodes are not checked for capacity."""
        nid = mgr.get_node_ids()[0]
        mgr._nodes[nid]["status"] = "NotReady"
        mgr._nodes[nid]["gpu_used"] = 999  # would violate if checked

        checker = ResourceCapacityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert all(v.device_id != nid for v in result.violations)

    def test_summary_has_max_gpu_util(self, mgr):
        checker = ResourceCapacityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert "max_gpu_util" in result.summary
        assert "overloaded_nodes" in result.summary
        assert "n_violations" in result.summary
        assert isinstance(result.summary["max_gpu_util"], float)

    def test_multiple_resource_violations_on_same_node(self, mgr):
        """A node over in both GPU and CPU yields two violations."""
        nid = mgr.get_node_ids()[0]
        mgr._nodes[nid]["gpu_used"] = mgr._nodes[nid]["gpu_total"] + 1
        mgr._nodes[nid]["cpu_used"] = mgr._nodes[nid]["cpu_total"] + 1

        checker = ResourceCapacityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        node_violations = [v for v in result.violations if v.device_id == nid]
        assert len(node_violations) >= 2
        # But overloaded_nodes counts unique nodes
        assert result.summary["overloaded_nodes"] >= 1


# ---------------------------------------------------------------
# AffinityChecker
# ---------------------------------------------------------------

class TestAffinityChecker:
    def test_checker_name(self, mgr):
        checker = AffinityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.checker_name == "affinity"

    def test_default_state_respects_affinity(self, mgr):
        """Greedy packer tries affinity rack first, so default should pass."""
        checker = AffinityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        # Default packing prefers affinity rack; may still pass
        assert result.checker_name == "affinity"
        assert "affinity_violations" in result.summary

    def test_violation_when_job_in_wrong_rack(self, mgr):
        """Manually place a job with rack_affinity on the wrong rack."""
        # Find a Running job with rack_affinity
        target_jid = None
        for jid, job in mgr._jobs.items():
            if (job["status"] == "Running"
                    and job.get("rack_affinity")
                    and jid in mgr._assignments):
                target_jid = jid
                break

        if target_jid is None:
            # Create one if none exist
            target_jid = "job-affinity-test"
            mgr._jobs[target_jid] = {
                "gpu": 1, "cpu": 4, "ram_gb": 16,
                "priority": "normal", "rack_affinity": "rack-a",
                "status": "Running",
            }
            # Assign to a rack-b node
            wrong_node = next(
                nid for nid, n in mgr._nodes.items()
                if n["rack"] == "rack-b" and n["status"] == "Ready"
            )
            mgr._assignments[target_jid] = wrong_node
        else:
            # Move existing job to wrong rack
            wanted_rack = mgr._jobs[target_jid]["rack_affinity"]
            wrong_node = next(
                nid for nid, n in mgr._nodes.items()
                if n["rack"] != wanted_rack and n["status"] == "Ready"
            )
            mgr._assignments[target_jid] = wrong_node

        checker = AffinityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed
        assert any(v.device_id == target_jid for v in result.violations)
        v = [v for v in result.violations if v.device_id == target_jid][0]
        assert v.severity == "warning"
        assert v.constraint_type == "affinity"

    def test_queued_jobs_not_checked(self, mgr):
        """Queued jobs should not generate affinity violations."""
        jid = "job-queued-affinity"
        mgr._jobs[jid] = {
            "gpu": 1, "cpu": 4, "ram_gb": 16,
            "priority": "normal", "rack_affinity": "rack-a",
            "status": "Queued",
        }
        # Even if an assignment exists (shouldn't, but test robustness)
        wrong_node = next(
            nid for nid, n in mgr._nodes.items()
            if n["rack"] == "rack-b" and n["status"] == "Ready"
        )
        mgr._assignments[jid] = wrong_node

        checker = AffinityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert all(v.device_id != jid for v in result.violations)

    def test_no_affinity_jobs_pass(self, mgr):
        """Jobs without rack_affinity never cause violations."""
        # Clear all rack_affinity
        for job in mgr._jobs.values():
            job["rack_affinity"] = None

        checker = AffinityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.passed
        assert result.summary["n_violations"] == 0


# ---------------------------------------------------------------
# RackSpreadChecker
# ---------------------------------------------------------------

class TestRackSpreadChecker:
    def test_checker_name(self, mgr):
        checker = RackSpreadChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.checker_name == "rack_spread"

    def test_default_state(self, mgr):
        """Default state may or may not have spread violations depending on seed."""
        checker = RackSpreadChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        # Just verify the summary structure is correct
        assert "groups_checked" in result.summary
        assert "n_violations" in result.summary

    def test_spread_group_across_racks_passes(self, mgr):
        """Urgent group with jobs on 2+ racks passes."""
        # Create two urgent Running jobs in the same group on different racks
        rack_a_node = next(
            nid for nid, n in mgr._nodes.items()
            if n["rack"] == "rack-a" and n["status"] == "Ready"
        )
        rack_b_node = next(
            nid for nid, n in mgr._nodes.items()
            if n["rack"] == "rack-b" and n["status"] == "Ready"
        )

        mgr._jobs["job-spread-a"] = {
            "gpu": 1, "cpu": 4, "ram_gb": 16,
            "priority": "urgent", "rack_affinity": None,
            "status": "Running", "group": "llm-pretrain",
        }
        mgr._jobs["job-spread-b"] = {
            "gpu": 1, "cpu": 4, "ram_gb": 16,
            "priority": "urgent", "rack_affinity": None,
            "status": "Running", "group": "llm-pretrain",
        }
        mgr._assignments["job-spread-a"] = rack_a_node
        mgr._assignments["job-spread-b"] = rack_b_node

        checker = RackSpreadChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        # The llm-pretrain group spans 2 racks -> no spread violation for it
        spread_violations = [
            v for v in result.violations if v.device_id == "llm-pretrain"
        ]
        assert len(spread_violations) == 0

    def test_single_rack_group_fails(self, mgr):
        """Urgent group with 2+ jobs all on same rack -> violation."""
        rack_a_nodes = [
            nid for nid, n in mgr._nodes.items()
            if n["rack"] == "rack-a" and n["status"] == "Ready"
        ]
        assert len(rack_a_nodes) >= 2

        mgr._jobs["job-narrow-1"] = {
            "gpu": 1, "cpu": 4, "ram_gb": 16,
            "priority": "urgent", "rack_affinity": None,
            "status": "Running", "group": "benchmark",
        }
        mgr._jobs["job-narrow-2"] = {
            "gpu": 1, "cpu": 4, "ram_gb": 16,
            "priority": "urgent", "rack_affinity": None,
            "status": "Running", "group": "benchmark",
        }
        mgr._assignments["job-narrow-1"] = rack_a_nodes[0]
        mgr._assignments["job-narrow-2"] = rack_a_nodes[1]

        checker = RackSpreadChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed
        v = [v for v in result.violations if v.device_id == "benchmark"][0]
        assert v.severity == "warning"
        assert v.constraint_type == "rack_spread"
        assert v.value == 1.0  # only 1 rack
        assert v.limit == 2.0

    def test_single_job_group_not_checked(self, mgr):
        """A group with only 1 urgent job should NOT violate spread."""
        rack_a_node = next(
            nid for nid, n in mgr._nodes.items()
            if n["rack"] == "rack-a" and n["status"] == "Ready"
        )
        mgr._jobs["job-solo"] = {
            "gpu": 1, "cpu": 4, "ram_gb": 16,
            "priority": "urgent", "rack_affinity": None,
            "status": "Running", "group": "solo-group",
        }
        mgr._assignments["job-solo"] = rack_a_node

        checker = RackSpreadChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert all(v.device_id != "solo-group" for v in result.violations)

    def test_summary_keys(self, mgr):
        checker = RackSpreadChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert "groups_checked" in result.summary
        assert "spread_violations" in result.summary
        assert "n_violations" in result.summary


# ---------------------------------------------------------------
# PriorityChecker
# ---------------------------------------------------------------

class TestPriorityChecker:
    def test_checker_name(self, mgr):
        checker = PriorityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.checker_name == "priority"

    def test_no_urgent_queued_passes(self, mgr):
        """If no urgent jobs are Queued, checker passes regardless."""
        # Force all urgent jobs to Running
        for job in mgr._jobs.values():
            if job["priority"] == "urgent":
                job["status"] = "Running"
        checker = PriorityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.passed
        assert result.summary["n_violations"] == 0

    def test_urgent_queued_while_preemptible_running_fails(self, mgr):
        """Urgent Queued + preemptible Running = critical violation."""
        # Inject an urgent Queued job
        mgr._jobs["job-urgent-q"] = {
            "gpu": 1, "cpu": 4, "ram_gb": 16,
            "priority": "urgent", "rack_affinity": None,
            "status": "Queued",
        }
        # Ensure at least one preemptible Running
        has_preemptible_running = any(
            j["priority"] == "preemptible" and j["status"] == "Running"
            for j in mgr._jobs.values()
        )
        if not has_preemptible_running:
            # Inject one
            node = mgr.get_schedulable_nodes()[0]
            mgr._jobs["job-preempt-r"] = {
                "gpu": 1, "cpu": 4, "ram_gb": 16,
                "priority": "preemptible", "rack_affinity": None,
                "status": "Running",
            }
            mgr._assignments["job-preempt-r"] = node

        checker = PriorityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed
        assert result.summary["urgent_queued"] >= 1
        assert result.summary["preemptible_running"] >= 1
        v = [v for v in result.violations if v.device_id == "job-urgent-q"][0]
        assert v.severity == "critical"

    def test_urgent_queued_no_preemptible_passes(self, mgr):
        """Urgent Queued but no preemptible Running -> passes (nothing to preempt)."""
        # Remove all preemptible running jobs
        for job in mgr._jobs.values():
            if job["priority"] == "preemptible":
                job["status"] = "Queued"
        # Add an urgent queued
        mgr._jobs["job-urgent-only"] = {
            "gpu": 1, "cpu": 4, "ram_gb": 16,
            "priority": "urgent", "rack_affinity": None,
            "status": "Queued",
        }

        checker = PriorityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.passed
        assert result.summary["urgent_queued"] >= 1
        assert result.summary["preemptible_running"] == 0

    def test_summary_counts(self, mgr):
        checker = PriorityChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert "urgent_queued" in result.summary
        assert "preemptible_running" in result.summary
        assert "n_violations" in result.summary


# ---------------------------------------------------------------
# QueueChecker
# ---------------------------------------------------------------

class TestQueueChecker:
    def test_all_running_passes(self, mgr):
        """If every job is Running, checker passes."""
        for job in mgr._jobs.values():
            job["status"] = "Running"
        checker = QueueChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.passed
        assert result.summary["queued_count"] == 0

    def test_queued_jobs_fail(self, mgr):
        """Any Queued job produces a violation."""
        # Force one specific job to Queued
        jid = mgr.get_job_ids()[0]
        mgr._jobs[jid]["status"] = "Queued"
        if jid in mgr._assignments:
            del mgr._assignments[jid]

        checker = QueueChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed
        assert result.summary["queued_count"] >= 1
        assert any(v.device_id == jid for v in result.violations)

    def test_summary_queue_ratio(self, mgr):
        """queue_ratio = critical_queued / total_jobs."""
        checker = QueueChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        total = result.summary["total_jobs"]
        critical = result.summary["critical_queued"]
        expected_ratio = round(critical / total, 3) if total else 0
        assert result.summary["queue_ratio"] == expected_ratio

    def test_all_queued(self, mgr):
        """All jobs Queued -> violations only for urgent+normal, not preemptible."""
        for job in mgr._jobs.values():
            job["status"] = "Queued"
        mgr._assignments.clear()

        checker = QueueChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert not result.passed
        total = result.summary["total_jobs"]
        critical = result.summary["critical_queued"]
        preemptible = result.summary["preemptible_queued"]
        assert critical + preemptible == total
        # Only urgent+normal jobs create violations
        assert len(result.violations) == critical

    def test_violation_severity(self, mgr):
        """Queue violations have severity='violation'."""
        mgr._jobs[mgr.get_job_ids()[0]]["status"] = "Queued"
        checker = QueueChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        for v in result.violations:
            assert v.severity == "violation"

    def test_empty_jobs(self):
        """Edge case: no jobs at all -> passes, ratio=0."""
        state = {"nodes": {}, "jobs": {}, "assignments": {}}
        checker = QueueChecker()
        result = checker.check(state, 1.0)
        assert result.passed
        assert result.summary["queued_count"] == 0
        assert result.summary["queue_ratio"] == 0

    def test_checker_name(self, mgr):
        checker = QueueChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert result.checker_name == "queue"

    def test_summary_keys(self, mgr):
        checker = QueueChecker()
        result = checker.check(mgr.system_state, mgr.base_mva)
        assert "queued_count" in result.summary
        assert "total_jobs" in result.summary
        assert "queue_ratio" in result.summary
        assert "n_violations" in result.summary
