"""Shared test fixtures."""

import pytest

from domains.network import NetworkManager, build_network_domain_config


@pytest.fixture
def network_manager():
    """Fresh NetworkManager with default topology."""
    return NetworkManager()


@pytest.fixture
def network_config():
    """Network DomainConfig."""
    return build_network_domain_config()
