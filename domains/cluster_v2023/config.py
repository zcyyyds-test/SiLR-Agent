"""DomainConfig factory for cluster_v2023.

Per-action gate (SiLRVerifier.checkers): Capacity + Affinity only.

Priority / Queue / Fragmentation are episode-level — evaluated by the
observer and consumed by the reward function. Putting them on the
per-action gate would reject 100% of intermediate actions (see spec
§5.1 "Verifier 策略" and cluster v1 QueueChecker pitfall).
"""

from __future__ import annotations

from silr.core.config import DomainConfig

from .checkers import AffinityChecker, ResourceCapacityChecker
from .observation import ClusterV2023Observer
from .prompts import build_system_prompt, build_tool_schemas
from .tools import create_toolset


def build_cluster_v2023_domain_config(
    *, f_threshold: float = 10.0,
    with_observer: bool = True,
) -> DomainConfig:
    return DomainConfig(
        domain_name="cluster_v2023",
        checkers=[ResourceCapacityChecker(), AffinityChecker()],
        allowed_actions=frozenset(["assign_job", "migrate_job", "preempt_job"]),
        create_toolset=create_toolset,
        create_observer=(
            (lambda mgr: ClusterV2023Observer(mgr, f_threshold=f_threshold))
            if with_observer else None
        ),
        build_system_prompt=build_system_prompt,
        build_tool_schemas=build_tool_schemas,
    )
