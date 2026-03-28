"""Network DomainConfig factory."""

from silr.core.config import DomainConfig
from .checkers import LinkUtilizationChecker, ConnectivityChecker
from .tools import create_network_toolset


def build_network_domain_config() -> DomainConfig:
    """Build a DomainConfig for the toy 5-node network domain.

    Returns:
        DomainConfig with network-specific tools and checkers.
        No post-solve hook (discrete event domain, not continuous simulation).
    """
    return DomainConfig(
        domain_name="toy_network",
        checkers=[LinkUtilizationChecker(), ConnectivityChecker()],
        allowed_actions=frozenset(["restore_link", "reroute_traffic"]),
        create_toolset=create_network_toolset,
    )
