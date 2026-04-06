"""Tests for cluster scheduling scenarios and loader."""

import pytest

from domains.cluster.manager import ClusterManager
from domains.cluster.scenarios import ClusterScenario, ClusterScenarioLoader
from domains.cluster.scenarios.loader import SCENARIOS


class TestClusterScenarioLoader:
    """Basic loader functionality."""

    def test_load_known_scenario(self):
        loader = ClusterScenarioLoader()
        s = loader.load("single_node_failure")
        assert s.id == "single_node_failure"
        assert s.difficulty == "easy"
        assert len(s.node_failures) == 1

    def test_load_unknown_raises_keyerror(self):
        loader = ClusterScenarioLoader()
        with pytest.raises(KeyError, match="Unknown scenario"):
            loader.load("nonexistent_scenario")

    def test_load_all_returns_list(self):
        loader = ClusterScenarioLoader()
        scenarios = loader.load_all()
        assert isinstance(scenarios, list)
        assert all(isinstance(s, ClusterScenario) for s in scenarios)

    def test_load_all_count_at_least_15(self):
        """Parameterized generation should produce 15+ total scenarios."""
        loader = ClusterScenarioLoader()
        scenarios = loader.load_all()
        assert len(scenarios) >= 15

    def test_all_scenarios_have_difficulty(self):
        """Every scenario must declare a difficulty level."""
        loader = ClusterScenarioLoader()
        valid_difficulties = {"easy", "medium", "hard"}
        for s in loader.load_all():
            assert s.difficulty in valid_difficulties, (
                f"Scenario {s.id} has invalid difficulty: {s.difficulty}"
            )

    def test_all_scenario_ids_unique(self):
        loader = ClusterScenarioLoader()
        ids = [s.id for s in loader.load_all()]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"


class TestClusterScenarioSetup:
    """setup_episode applies faults correctly."""

    def test_setup_single_node_failure(self):
        loader = ClusterScenarioLoader()
        mgr = ClusterManager()
        scenario = loader.load("single_node_failure")
        loader.setup_episode(mgr, scenario)

        # rack-a-s0 should be NotReady
        assert mgr._nodes["rack-a-s0"]["status"] == "NotReady"

    def test_setup_rack_failure(self):
        loader = ClusterScenarioLoader()
        mgr = ClusterManager()
        scenario = loader.load("rack_failure_a")
        loader.setup_episode(mgr, scenario)

        # 3 rack-a nodes should be NotReady, 2 should survive
        failed_nodes = {"rack-a-s0", "rack-a-s1", "rack-a-h2"}
        for nid, node in mgr._nodes.items():
            if nid in failed_nodes:
                assert node["status"] == "NotReady", f"{nid} should be NotReady"
        # Surviving rack-a nodes remain Ready for affinity-bound jobs
        assert mgr._nodes["rack-a-h3"]["status"] == "Ready"
        assert mgr._nodes["rack-a-f4"]["status"] == "Ready"

    def test_setup_job_surge_creates_queued(self):
        """Job surge scenario should add new jobs, some of which are Queued."""
        loader = ClusterScenarioLoader()
        mgr = ClusterManager()
        initial_job_count = len(mgr._jobs)

        scenario = loader.load("job_surge")
        loader.setup_episode(mgr, scenario)

        # New jobs were added
        assert len(mgr._jobs) > initial_job_count
        # At least some jobs should be queued (cluster is near capacity)
        queued = mgr.get_queued_jobs()
        assert len(queued) > 0

    def test_setup_compound_failure_surge(self):
        loader = ClusterScenarioLoader()
        mgr = ClusterManager()
        scenario = loader.load("compound_failure_surge")
        loader.setup_episode(mgr, scenario)

        # Fat node should be down
        assert mgr._nodes["rack-b-f4"]["status"] == "NotReady"
        # Queued jobs should exist (from both failure eviction and new arrivals)
        queued = mgr.get_queued_jobs()
        assert len(queued) > 0

    def test_setup_priority_conflict(self):
        loader = ClusterScenarioLoader()
        mgr = ClusterManager()
        scenario = loader.load("priority_conflict")
        loader.setup_episode(mgr, scenario)

        # New jobs should have been added
        # At least some urgent jobs should be queued (cluster already loaded)
        queued = mgr.get_queued_jobs()
        assert len(queued) > 0

    def test_setup_resource_fragmentation(self):
        loader = ClusterScenarioLoader()
        mgr = ClusterManager()
        scenario = loader.load("resource_fragmentation")
        loader.setup_episode(mgr, scenario)

        # Large GPU jobs should create queuing pressure
        queued = mgr.get_queued_jobs()
        assert len(queued) > 0


class TestParameterizedVariants:
    """Ensure parameterized variant generation works correctly."""

    def test_variant_count_at_least_15(self):
        assert len(SCENARIOS) >= 15

    def test_base_scenarios_present(self):
        ids = {s.id for s in SCENARIOS}
        expected_bases = {
            "single_node_failure",
            "rack_failure_a",
            "job_surge",
            "resource_fragmentation",
            "priority_conflict",
            "compound_failure_surge",
        }
        assert expected_bases.issubset(ids)

    def test_node_ids_are_valid(self):
        """All node IDs referenced in scenarios must exist in the manager."""
        mgr = ClusterManager()
        valid_nodes = set(mgr.get_node_ids())

        for s in SCENARIOS:
            for nid in s.node_failures:
                assert nid in valid_nodes, (
                    f"Scenario {s.id} references invalid node: {nid}"
                )

    def test_rack_ids_are_valid(self):
        """All rack IDs referenced in scenarios must be valid."""
        valid_racks = {"rack-a", "rack-b", "rack-c"}
        for s in SCENARIOS:
            if s.rack_failure is not None:
                assert s.rack_failure in valid_racks, (
                    f"Scenario {s.id} references invalid rack: {s.rack_failure}"
                )

    def test_each_difficulty_level_present(self):
        """Scenarios should span all three difficulty levels."""
        difficulties = {s.difficulty for s in SCENARIOS}
        assert difficulties == {"easy", "medium", "hard"}
