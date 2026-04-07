"""Power grid domain for the SiLR framework (IEEE 39-bus demo).

Requires: pip install silr[grid]
"""

from .config import build_grid_domain_config
from .simulator import SystemManager

__all__ = ["build_grid_domain_config", "SystemManager"]
