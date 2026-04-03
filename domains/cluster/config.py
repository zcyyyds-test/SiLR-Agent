"""GPU cluster DomainConfig factory."""

from silr.core.config import DomainConfig
from .checkers import (
    ResourceCapacityChecker,
    AffinityChecker,
    RackSpreadChecker,
    PriorityChecker,
    QueueChecker,
)
from .tools import create_cluster_toolset
from .observation import ClusterObserver
from .failsafe import ClusterFailsafe
from .prompts import build_cluster_system_prompt, build_cluster_tool_schemas


def build_cluster_domain_config(with_observer: bool = True) -> DomainConfig:
    """Build a DomainConfig for the GPU cluster scheduling domain.

    Args:
        with_observer: If True, include ClusterObserver and ClusterFailsafe.
            Default True. Set False for lightweight verifier-only usage.
    """
    return DomainConfig(
        domain_name="gpu_cluster",
        # Verifier checkers: only per-action SAFETY constraints.
        # Global state checks (queue, priority, rack_spread) are handled by
        # the observer for episode termination — they require multiple actions
        # to resolve and would reject every intermediate step otherwise.
        checkers=[
            ResourceCapacityChecker(),
            AffinityChecker(),
        ],
        allowed_actions=frozenset([
            "assign_job",
            "migrate_job",
            "preempt_job",
            "scale_job",
            "drain_node",
            "restore_node",
        ]),
        create_toolset=create_cluster_toolset,
        create_observer=(lambda mgr: ClusterObserver(mgr)) if with_observer else None,
        create_failsafe=(lambda mgr: ClusterFailsafe(mgr)) if with_observer else None,
        build_system_prompt=build_cluster_system_prompt,
        build_tool_schemas=build_cluster_tool_schemas,
    )
