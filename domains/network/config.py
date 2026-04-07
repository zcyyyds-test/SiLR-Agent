"""Network DomainConfig factory."""

from silr.core.config import DomainConfig
from .checkers import LinkUtilizationChecker, ConnectivityChecker
from .tools import create_network_toolset
from .observation import NetworkObserver


def build_network_domain_config(with_observer: bool = False) -> DomainConfig:
    """Build a DomainConfig for the toy 5-node network domain.

    Args:
        with_observer: If True, include NetworkObserver. Required for
            CoordinatorAgent. Default False for backward compatibility
            with single-agent tests that rely on _MinimalObserver.
    """
    return DomainConfig(
        domain_name="toy_network",
        checkers=[LinkUtilizationChecker(), ConnectivityChecker()],
        allowed_actions=frozenset(["restore_link", "reroute_traffic"]),
        create_toolset=create_network_toolset,
        create_observer=(lambda mgr: NetworkObserver(mgr)) if with_observer else None,
    )
