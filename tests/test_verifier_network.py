"""Test SiLR verifier using the network domain as backend."""

import pytest

from domains.network import NetworkManager, build_network_domain_config
from silr.verifier import SiLRVerifier, Verdict


@pytest.fixture
def verifier():
    manager = NetworkManager()
    manager.fail_link(1, 2)
    manager.solve()
    config = build_network_domain_config()
    return SiLRVerifier(manager, domain_config=config)


class TestSiLRVerifier:
    def test_verify_pass(self, verifier):
        result = verifier.verify(
            {"tool_name": "restore_link", "params": {"src": 1, "dst": 2}},
        )
        assert result.verdict == Verdict.PASS
        assert result.solver_converged is True

    def test_verify_unknown_action(self, verifier):
        result = verifier.verify(
            {"tool_name": "delete_node", "params": {"node_id": 1}},
        )
        assert result.verdict == Verdict.ERROR

    def test_verify_generates_report(self, verifier):
        result = verifier.verify(
            {"tool_name": "restore_link", "params": {"src": 1, "dst": 2}},
        )
        assert result.report_text is not None
        assert len(result.report_text) > 0

    def test_verify_preserves_original(self, verifier):
        """Verification should not modify the original manager."""
        manager = verifier._manager
        orig_links = {k: dict(v) for k, v in manager._links.items()}

        verifier.verify(
            {"tool_name": "restore_link", "params": {"src": 1, "dst": 2}},
        )

        # Original should still have link 1-2 down
        assert manager._links[(1, 2)]["up"] is False


class TestSiLRVerifierReroute:
    def test_reroute_pass(self):
        manager = NetworkManager()
        manager.solve()
        config = build_network_domain_config()
        verifier = SiLRVerifier(manager, domain_config=config)

        result = verifier.verify(
            {"tool_name": "reroute_traffic",
             "params": {"src": 1, "dst": 2, "amount_mbps": 5}},
        )
        assert result.verdict == Verdict.PASS
