"""Test cascading fault scenarios and network specialists."""

import pytest

from domains.network import (
    NetworkManager,
    NetworkObserver,
    NetworkScenarioLoader,
    build_network_domain_config,
    build_connectivity_specialist_config,
    build_utilization_specialist_config,
)
from domains.network.scenarios import SCENARIOS
from silr.verifier import SiLRVerifier, Verdict


class TestNetworkObserver:
    def test_observe_healthy(self):
        mgr = NetworkManager()
        mgr.solve()
        obs = NetworkObserver(mgr).observe()
        assert obs.is_stable is True
        assert obs.violations == []

    def test_observe_with_failure(self):
        """Isolating node 3 creates connectivity violations."""
        mgr = NetworkManager()
        mgr.fail_link(2, 3)
        mgr.fail_link(3, 5)
        mgr.solve()
        obs = NetworkObserver(mgr).observe()
        assert obs.is_stable is False
        assert len(obs.violations) > 0
        assert any(v["type"] == "connectivity" for v in obs.violations)

    def test_compressed_json_parseable(self):
        import json
        mgr = NetworkManager()
        mgr.fail_link(1, 2)
        mgr.solve()
        obs = NetworkObserver(mgr).observe()
        data = json.loads(obs.compressed_json)
        assert "down_links" in data
        assert "n_violations" in data


class TestNetworkScenarios:
    def test_load_all(self):
        loader = NetworkScenarioLoader()
        scenarios = loader.load_all()
        assert len(scenarios) >= 3

    def test_load_by_id(self):
        loader = NetworkScenarioLoader()
        s = loader.load("cascade_easy")
        assert s.id == "cascade_easy"
        assert len(s.faults) > 0

    def test_load_unknown_raises(self):
        loader = NetworkScenarioLoader()
        with pytest.raises(KeyError):
            loader.load("nonexistent")

    def test_setup_cascade_easy(self):
        loader = NetworkScenarioLoader()
        mgr = NetworkManager()
        scenario = loader.load("cascade_easy")
        loader.setup_episode(mgr, scenario)

        # Link 1-2 should be down
        assert mgr._links[(1, 2)]["up"] is False
        # Link 2-5 should be overloaded (>90% of capacity 60)
        assert mgr._links[(2, 5)]["traffic"] >= 55

    def test_setup_cascade_medium(self):
        loader = NetworkScenarioLoader()
        mgr = NetworkManager()
        scenario = loader.load("cascade_medium")
        loader.setup_episode(mgr, scenario)

        assert mgr._links[(2, 3)]["up"] is False
        assert mgr._links[(3, 5)]["up"] is False

    def test_setup_cascade_hard(self):
        loader = NetworkScenarioLoader()
        mgr = NetworkManager()
        scenario = loader.load("cascade_hard")
        loader.setup_episode(mgr, scenario)

        assert mgr._links[(1, 2)]["up"] is False
        assert mgr._links[(2, 5)]["traffic"] >= 58

    def test_all_scenarios_create_violations(self):
        """Every cascading scenario should produce at least one violation."""
        loader = NetworkScenarioLoader()
        observer_cls = NetworkObserver
        for scenario in loader.load_all():
            mgr = NetworkManager()
            loader.setup_episode(mgr, scenario)
            obs = observer_cls(mgr).observe()
            assert not obs.is_stable, f"Scenario {scenario.id} should have violations"


class TestSpecialistConfigs:
    def test_connectivity_specialist(self):
        config = build_connectivity_specialist_config()
        assert "restore_link" in config.allowed_actions
        assert "reroute_traffic" not in config.allowed_actions
        assert len(config.checkers) == 1
        assert config.checkers[0].name == "connectivity"

    def test_utilization_specialist(self):
        config = build_utilization_specialist_config()
        assert "reroute_traffic" in config.allowed_actions
        assert "restore_link" not in config.allowed_actions
        assert len(config.checkers) == 1
        assert config.checkers[0].name == "link_utilization"

    def test_specialist_toolset_restricted(self):
        mgr = NetworkManager()
        conn_tools = build_connectivity_specialist_config().create_toolset(mgr)
        assert "restore_link" in conn_tools
        assert "reroute_traffic" not in conn_tools

    def test_full_verifier_checks_all_constraints(self):
        """Specialist uses restricted config, but verifier uses full config."""
        mgr = NetworkManager()
        mgr.fail_link(1, 2)
        mgr.solve()

        full_config = build_network_domain_config()
        verifier = SiLRVerifier(mgr, domain_config=full_config)

        # Restore link via full verifier (checks both connectivity AND utilization)
        result = verifier.verify({"tool_name": "restore_link", "params": {"src": 1, "dst": 2}})
        assert result.verdict == Verdict.PASS
        # Both checkers were evaluated
        assert len(result.check_results) == 2
