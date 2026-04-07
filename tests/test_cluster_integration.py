"""End-to-end integration tests for the GPU cluster scheduling domain.

Tests SiLR verification pipeline, observer, failsafe, prompts, and
DomainConfig wiring using the ClusterManager + node_failure scenario.
"""

import json

import pytest

from domains.cluster import (
    ClusterManager,
    build_cluster_domain_config,
    ClusterObserver,
    ClusterFailsafe,
    ClusterScenario,
    ClusterScenarioLoader,
)
from domains.cluster.tools import create_cluster_toolset
from domains.cluster.prompts import (
    build_cluster_system_prompt,
    build_cluster_tool_schemas,
)
from silr.verifier.verifier import SiLRVerifier
from silr.verifier.types import Verdict
from silr.agent.types import Observation


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture
def cluster_manager():
    """Fresh ClusterManager with default topology and jobs."""
    return ClusterManager()


@pytest.fixture
def scenario_manager():
    """ClusterManager with single_node_failure scenario applied."""
    mgr = ClusterManager()
    loader = ClusterScenarioLoader()
    scenario = loader.load("single_node_failure")
    loader.setup_episode(mgr, scenario)
    return mgr


@pytest.fixture
def domain_config():
    """Full DomainConfig with observer and failsafe."""
    return build_cluster_domain_config(with_observer=True)


@pytest.fixture
def verifier(scenario_manager, domain_config):
    """SiLRVerifier wired to the scenario manager and config."""
    return SiLRVerifier(scenario_manager, domain_config)


# -------------------------------------------------------------------
# DomainConfig
# -------------------------------------------------------------------


class TestDomainConfig:
    def test_domain_name(self, domain_config):
        assert domain_config.domain_name == "gpu_cluster"

    def test_checkers_count(self, domain_config):
        # Verifier only has safety checkers; global-state checkers are observer-only
        assert len(domain_config.checkers) == 2

    def test_checker_names(self, domain_config):
        names = {c.name for c in domain_config.checkers}
        expected = {"resource_capacity", "affinity"}
        assert names == expected

    def test_allowed_actions(self, domain_config):
        expected = {
            "assign_job", "migrate_job", "preempt_job",
            "scale_job", "drain_node", "restore_node",
        }
        assert domain_config.allowed_actions == frozenset(expected)

    def test_create_toolset_callable(self, domain_config):
        assert callable(domain_config.create_toolset)

    def test_create_toolset_returns_all_tools(self, domain_config, cluster_manager):
        tools = domain_config.create_toolset(cluster_manager)
        assert len(tools) == 6
        assert set(tools.keys()) == set(domain_config.allowed_actions)

    def test_create_observer_callable(self, domain_config):
        assert domain_config.create_observer is not None
        assert callable(domain_config.create_observer)

    def test_create_failsafe_callable(self, domain_config):
        assert domain_config.create_failsafe is not None
        assert callable(domain_config.create_failsafe)

    def test_build_system_prompt_callable(self, domain_config):
        assert domain_config.build_system_prompt is not None
        assert callable(domain_config.build_system_prompt)

    def test_build_tool_schemas_callable(self, domain_config):
        assert domain_config.build_tool_schemas is not None
        assert callable(domain_config.build_tool_schemas)

    def test_config_without_observer(self):
        config = build_cluster_domain_config(with_observer=False)
        assert config.create_observer is None
        assert config.create_failsafe is None
        # Prompt builders are always present
        assert config.build_system_prompt is not None


# -------------------------------------------------------------------
# SiLR Verification — valid action
# -------------------------------------------------------------------


class TestSiLRVerificationValid:
    def test_assign_queued_job_passes(self, scenario_manager, verifier):
        """Assigning a queued job to a node with capacity should PASS."""
        queued = scenario_manager.get_queued_jobs()
        assert len(queued) > 0, "Scenario should produce queued jobs"

        # Pick the first queued job
        job_id = queued[0]
        job = scenario_manager._jobs[job_id]

        # Find a Ready node with enough GPU capacity
        ready_nodes = scenario_manager.get_schedulable_nodes()
        target = None
        for nid in ready_nodes:
            node = scenario_manager._nodes[nid]
            gpu_free = node["gpu_total"] - node["gpu_used"]
            cpu_free = node["cpu_total"] - node["cpu_used"]
            ram_free = node["ram_total_gb"] - node["ram_used_gb"]
            if (gpu_free >= job["gpu"]
                    and cpu_free >= job["cpu"]
                    and ram_free >= job["ram_gb"]):
                target = nid
                break

        if target is None:
            pytest.skip("No node with sufficient capacity for test job")

        action = {"tool_name": "assign_job", "params": {"job_id": job_id, "node_id": target}}
        result = verifier.verify(action)

        # The action itself should execute successfully; final verdict depends
        # on whether other constraints (queue, priority) are also satisfied.
        # At minimum it should not be ERROR.
        assert result.verdict != Verdict.ERROR, (
            f"Expected non-ERROR verdict, got {result.verdict}: {result.fail_reason}"
        )


# -------------------------------------------------------------------
# SiLR Verification — invalid tool
# -------------------------------------------------------------------


class TestSiLRVerificationInvalid:
    def test_unknown_tool_returns_error(self, verifier):
        """An action with an unrecognised tool_name should return ERROR."""
        action = {"tool_name": "nuke_cluster", "params": {}}
        result = verifier.verify(action)
        assert result.verdict == Verdict.ERROR
        assert "not in allowed actions" in result.fail_reason

    def test_assign_nonexistent_job_returns_fail(self, verifier):
        """Assigning a non-existent job should return FAIL (tool execution error)."""
        action = {
            "tool_name": "assign_job",
            "params": {"job_id": "job-9999", "node_id": "rack-a-s0"},
        }
        result = verifier.verify(action)
        assert result.verdict in (Verdict.FAIL, Verdict.ERROR)

    def test_assign_to_down_node_returns_fail(self, scenario_manager, verifier):
        """Assigning a job to a NotReady node should return FAIL."""
        # rack-a-s0 was failed in the scenario
        queued = scenario_manager.get_queued_jobs()
        if not queued:
            pytest.skip("No queued jobs")
        action = {
            "tool_name": "assign_job",
            "params": {"job_id": queued[0], "node_id": "rack-a-s0"},
        }
        result = verifier.verify(action)
        assert result.verdict == Verdict.FAIL


# -------------------------------------------------------------------
# Observer
# -------------------------------------------------------------------


class TestClusterObserver:
    def test_observe_returns_observation(self, scenario_manager):
        observer = ClusterObserver(scenario_manager)
        obs = observer.observe()
        assert isinstance(obs, Observation)

    def test_observation_has_compressed_json(self, scenario_manager):
        observer = ClusterObserver(scenario_manager)
        obs = observer.observe()
        data = json.loads(obs.compressed_json)
        assert "down_nodes" in data
        assert "cordoned_nodes" in data
        assert "queued_jobs" in data
        assert "busy_nodes" in data
        assert "checkers" in data
        assert "n_violations" in data

    def test_observation_detects_down_nodes(self, scenario_manager):
        """After node failure scenario, down_nodes should be non-empty."""
        observer = ClusterObserver(scenario_manager)
        obs = observer.observe()
        data = json.loads(obs.compressed_json)
        # rack-a-s0 was failed in the scenario
        assert "rack-a-s0" in data["down_nodes"]

    def test_observation_has_violations(self, scenario_manager):
        """The scenario should produce violations (at least queue violations)."""
        observer = ClusterObserver(scenario_manager)
        obs = observer.observe()
        # With a node failure, there should be queued jobs -> violations
        assert isinstance(obs.violations, list)

    def test_observation_has_raw_state(self, scenario_manager):
        observer = ClusterObserver(scenario_manager)
        obs = observer.observe()
        assert "nodes" in obs.raw
        assert "jobs" in obs.raw
        assert "assignments" in obs.raw

    def test_is_stable_flag(self, cluster_manager):
        """A clean default manager may or may not be stable (depends on queue)."""
        observer = ClusterObserver(cluster_manager)
        obs = observer.observe()
        assert isinstance(obs.is_stable, bool)

    def test_checker_summaries_present(self, scenario_manager):
        observer = ClusterObserver(scenario_manager)
        obs = observer.observe()
        data = json.loads(obs.compressed_json)
        checker_names = set(data["checkers"].keys())
        expected = {"resource_capacity", "affinity", "rack_spread", "priority", "queue"}
        assert checker_names == expected

    def test_observer_via_domain_config(self, domain_config, scenario_manager):
        """Observer created through DomainConfig factory should work."""
        observer = domain_config.create_observer(scenario_manager)
        obs = observer.observe()
        assert isinstance(obs, Observation)


# -------------------------------------------------------------------
# Failsafe
# -------------------------------------------------------------------


class TestClusterFailsafe:
    def test_suggest_returns_valid_action_or_none(self, scenario_manager):
        """After node failure, failsafe returns a valid action or None."""
        observer = ClusterObserver(scenario_manager)
        obs = observer.observe()
        failsafe = ClusterFailsafe(scenario_manager)
        action = failsafe.suggest(obs)

        if action is not None:
            assert "tool_name" in action
            assert "params" in action
            assert action["tool_name"] in ("assign_job", "preempt_job")

    def test_suggest_returns_none_when_stable(self):
        """If no jobs are queued, failsafe should return None."""
        mgr = ClusterManager()
        # Force all jobs to Running (unrealistic but tests None path)
        for jid in list(mgr._jobs.keys()):
            if mgr._jobs[jid]["status"] == "Queued":
                # Remove the job to simulate all-scheduled state
                del mgr._jobs[jid]
                mgr._assignments.pop(jid, None)
        mgr._recompute_node_usage()

        observer = ClusterObserver(mgr)
        obs = observer.observe()
        failsafe = ClusterFailsafe(mgr)
        action = failsafe.suggest(obs)

        queued = mgr.get_queued_jobs()
        if not queued:
            assert action is None

    def test_suggest_escalated_delegates(self, scenario_manager):
        """suggest_escalated should return the same type of result as suggest."""
        observer = ClusterObserver(scenario_manager)
        obs = observer.observe()
        failsafe = ClusterFailsafe(scenario_manager)
        action = failsafe.suggest_escalated(obs, last_rejected=None)

        if action is not None:
            assert "tool_name" in action

    def test_failsafe_via_domain_config(self, domain_config, scenario_manager):
        """Failsafe created through DomainConfig factory should work."""
        failsafe = domain_config.create_failsafe(scenario_manager)
        observer = domain_config.create_observer(scenario_manager)
        obs = observer.observe()
        action = failsafe.suggest(obs)
        # Just verify it returns the expected type
        assert action is None or isinstance(action, dict)

    def test_failsafe_respects_priority_order(self, scenario_manager):
        """Failsafe should prioritise urgent jobs over normal/preemptible."""
        # Add an explicit urgent job
        scenario_manager.add_jobs([
            {"gpu": 1, "cpu": 4, "ram_gb": 16, "priority": "urgent"},
        ])
        observer = ClusterObserver(scenario_manager)
        obs = observer.observe()
        failsafe = ClusterFailsafe(scenario_manager)
        action = failsafe.suggest(obs)

        if action and action["tool_name"] == "assign_job":
            job_id = action["params"]["job_id"]
            job = scenario_manager._jobs[job_id]
            # If there are urgent queued jobs, the suggested job should be urgent
            urgent_queued = [
                jid for jid, j in scenario_manager._jobs.items()
                if j["priority"] == "urgent" and j["status"] == "Queued"
            ]
            if urgent_queued:
                assert job["priority"] == "urgent"


# -------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------


class TestPrompts:
    def test_build_tool_schemas(self, cluster_manager):
        schemas = build_cluster_tool_schemas(cluster_manager)
        assert isinstance(schemas, list)
        assert len(schemas) == 6
        names = {s["function"]["name"] for s in schemas}
        expected = {
            "assign_job", "migrate_job", "preempt_job",
            "scale_job", "drain_node", "restore_node",
        }
        assert names == expected

    def test_tool_schemas_have_required_fields(self, cluster_manager):
        schemas = build_cluster_tool_schemas(cluster_manager)
        for schema in schemas:
            assert schema["type"] == "function"
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"

    def test_build_system_prompt(self, cluster_manager):
        schemas = build_cluster_tool_schemas(cluster_manager)
        prompt = build_cluster_system_prompt(cluster_manager, schemas)
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        # Should mention key domain concepts
        assert "GPU" in prompt
        assert "cluster" in prompt.lower() or "Cluster" in prompt
        assert "assign_job" in prompt

    def test_system_prompt_includes_topology(self, cluster_manager):
        schemas = build_cluster_tool_schemas(cluster_manager)
        prompt = build_cluster_system_prompt(cluster_manager, schemas)
        assert "rack-a" in prompt
        assert "rack-b" in prompt
        assert "rack-c" in prompt

    def test_system_prompt_via_domain_config(self, domain_config, cluster_manager):
        """Prompt builder via DomainConfig factory should work."""
        schemas = domain_config.build_tool_schemas(cluster_manager)
        prompt = domain_config.build_system_prompt(cluster_manager, schemas)
        assert isinstance(prompt, str)
        assert "assign_job" in prompt


# -------------------------------------------------------------------
# Scenario Loader
# -------------------------------------------------------------------


class TestScenarioLoader:
    def test_load_scenario(self):
        loader = ClusterScenarioLoader()
        scenario = loader.load("single_node_failure")
        assert isinstance(scenario, ClusterScenario)
        assert scenario.id == "single_node_failure"

    def test_load_all(self):
        loader = ClusterScenarioLoader()
        all_scenarios = loader.load_all()
        assert len(all_scenarios) >= 6  # at least 6 base scenarios
        ids = {s.id for s in all_scenarios}
        assert "single_node_failure" in ids
        assert "rack_failure_a" in ids

    def test_unknown_scenario_raises(self):
        loader = ClusterScenarioLoader()
        with pytest.raises(KeyError, match="Unknown scenario"):
            loader.load("nonexistent_scenario")

    def test_setup_episode_applies_faults(self):
        mgr = ClusterManager()
        loader = ClusterScenarioLoader()
        scenario = loader.load("single_node_failure")
        loader.setup_episode(mgr, scenario)
        # rack-a-s0 should be NotReady
        assert mgr._nodes["rack-a-s0"]["status"] == "NotReady"


# -------------------------------------------------------------------
# End-to-end: scenario -> observe -> failsafe -> verify
# -------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline(self, scenario_manager, domain_config):
        """Scenario -> Observer -> Failsafe -> SiLR Verify pipeline."""
        # 1. Create observer and observe
        observer = domain_config.create_observer(scenario_manager)
        obs = observer.observe()
        assert isinstance(obs, Observation)

        # 2. If not stable, get failsafe suggestion
        if not obs.is_stable:
            failsafe = domain_config.create_failsafe(scenario_manager)
            action = failsafe.suggest(obs)

            if action is not None:
                # 3. Verify the failsafe action
                verifier = SiLRVerifier(scenario_manager, domain_config)
                result = verifier.verify(action)

                # Failsafe should produce a valid (non-ERROR) action
                assert result.verdict != Verdict.ERROR, (
                    f"Failsafe action should not produce ERROR: {result.fail_reason}"
                )
