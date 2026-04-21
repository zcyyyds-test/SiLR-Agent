"""Alibaba cluster-trace-GPU-v2023 domain for SiLR.

Based on the OpenB production trace from FGD (ATC'23).
Plugs into SiLR via BaseSystemManager + BaseConstraintChecker.
"""

__version__ = "0.1.0"

from .config import build_cluster_v2023_domain_config  # noqa: E402,F401
from .manager import ClusterV2023Manager              # noqa: E402,F401

__all__ = ["build_cluster_v2023_domain_config", "ClusterV2023Manager"]
