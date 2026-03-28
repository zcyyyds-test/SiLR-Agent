"""Toy network domain: 5-node network routing for SiLR framework demo."""

from .manager import NetworkManager
from .config import build_network_domain_config

__all__ = ["NetworkManager", "build_network_domain_config"]
