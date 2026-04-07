"""Specialist agent configurations for the network domain.

Each specialist gets a restricted DomainConfig with a subset of tools
and checkers. The SiLRVerifier still uses the full checker set for
global safety verification.
"""

from __future__ import annotations

from silr.core.config import DomainConfig
from .checkers import LinkUtilizationChecker, ConnectivityChecker
from .tools import create_network_toolset


def _create_restore_only_toolset(manager):
    """Toolset restricted to restore_link only."""
    full = create_network_toolset(manager)
    return {k: v for k, v in full.items() if k in ("restore_link", "get_link_status")}


def _create_reroute_only_toolset(manager):
    """Toolset restricted to reroute_traffic only."""
    full = create_network_toolset(manager)
    return {k: v for k, v in full.items() if k in ("reroute_traffic", "get_link_status")}


def build_connectivity_specialist_config() -> DomainConfig:
    """DomainConfig for the connectivity specialist (restore_link only)."""
    return DomainConfig(
        domain_name="network_connectivity",
        checkers=[ConnectivityChecker()],
        allowed_actions=frozenset(["restore_link"]),
        create_toolset=_create_restore_only_toolset,
    )


def build_utilization_specialist_config() -> DomainConfig:
    """DomainConfig for the utilization specialist (reroute_traffic only)."""
    return DomainConfig(
        domain_name="network_utilization",
        checkers=[LinkUtilizationChecker()],
        allowed_actions=frozenset(["reroute_traffic"]),
        create_toolset=_create_reroute_only_toolset,
    )
