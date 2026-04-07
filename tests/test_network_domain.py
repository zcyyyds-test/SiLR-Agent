"""Full coverage tests for the toy network domain."""

import pytest

from domains.network import NetworkManager, build_network_domain_config
from domains.network.checkers import LinkUtilizationChecker, ConnectivityChecker
from domains.network.tools import create_network_toolset


class TestNetworkManager:
    def test_initial_state(self, network_manager):
        assert network_manager.sim_time == 0.0
        assert network_manager.base_mva == 1.0
        assert len(network_manager.get_node_ids()) == 5
        assert len(network_manager.get_link_ids()) == 6

    def test_solve(self, network_manager):
        ok = network_manager.solve()
        assert ok is True
        assert network_manager.sim_time == 1.0

    def test_fail_link(self, network_manager):
        assert network_manager.fail_link(1, 2) is True
        # Already down
        assert network_manager.fail_link(1, 2) is False
        # Non-existent
        assert network_manager.fail_link(1, 3) is False

    def test_restore_link(self, network_manager):
        network_manager.fail_link(1, 2)
        assert network_manager.restore_link(1, 2) is True
        # Already up
        assert network_manager.restore_link(1, 2) is False

    def test_reroute_traffic(self, network_manager):
        ok = network_manager.reroute_traffic(1, 2, 10)
        assert ok is True

    def test_create_shadow_copy(self, network_manager):
        network_manager.fail_link(1, 2)
        shadow = network_manager.create_shadow_copy()
        # Shadow should be independent
        assert shadow._links[(1, 2)]["up"] is False
        shadow.restore_link(1, 2)
        assert shadow._links[(1, 2)]["up"] is True
        assert network_manager._links[(1, 2)]["up"] is False

    def test_system_state(self, network_manager):
        state = network_manager.system_state
        assert "links" in state
        assert "demands" in state
        assert "nodes" in state


class TestNetworkCheckers:
    def test_link_utilization_pass(self, network_manager):
        checker = LinkUtilizationChecker()
        result = checker.check(network_manager.system_state, 1.0)
        assert result.passed is True

    def test_link_utilization_fail(self, network_manager):
        # Overload a link
        network_manager._links[(2, 5)]["traffic"] = 60  # capacity=60 → 100%
        checker = LinkUtilizationChecker()
        result = checker.check(network_manager.system_state, 1.0)
        assert result.passed is False
        assert len(result.violations) > 0

    def test_connectivity_pass(self, network_manager):
        checker = ConnectivityChecker()
        result = checker.check(network_manager.system_state, 1.0)
        assert result.passed is True

    def test_connectivity_fail(self, network_manager):
        # Isolate node 3 by failing both its links
        network_manager.fail_link(2, 3)
        network_manager.fail_link(3, 5)
        checker = ConnectivityChecker()
        result = checker.check(network_manager.system_state, 1.0)
        assert result.passed is False
        # Demands (1,3) and (4,3) should be unreachable
        assert len(result.violations) >= 1


class TestNetworkTools:
    def test_create_toolset(self, network_manager):
        tools = create_network_toolset(network_manager)
        assert "get_link_status" in tools
        assert "check_network_health" in tools
        assert "restore_link" in tools
        assert "reroute_traffic" in tools

    def test_get_link_status(self, network_manager):
        tools = create_network_toolset(network_manager)
        result = tools["get_link_status"].execute()
        assert result["status"] == "success"
        assert result["data"]["total"] == 6

    def test_restore_link_tool(self, network_manager):
        network_manager.fail_link(1, 2)
        tools = create_network_toolset(network_manager)
        result = tools["restore_link"].execute(src=1, dst=2)
        assert result["status"] == "success"
        assert result["data"]["restored"] is True


class TestNetworkDomainConfig:
    def test_build_config(self, network_config):
        assert network_config.domain_name == "toy_network"
        assert len(network_config.checkers) == 2
        assert "restore_link" in network_config.allowed_actions
        assert "reroute_traffic" in network_config.allowed_actions
