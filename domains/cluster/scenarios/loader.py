"""Cluster scheduling scenarios for multi-agent coordinator testing.

Each scenario defines a combination of node/rack failures, job surges,
and scheduling conflicts that require coordinated specialist dispatch.

Node IDs follow the rack-based naming convention from ClusterManager:
    rack-{a,b,c}-{s,h,f}{0..4}
where s=standard, h=highmem, f=fat.  Full topology: 15 nodes, 3 racks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..manager import ClusterManager

# ---------------------------------------------------------------------------
# Concrete node / rack identifiers (must match manager._build_default_nodes)
# ---------------------------------------------------------------------------
_RACKS = ["rack-a", "rack-b", "rack-c"]

# All 15 node IDs, deterministic order
_ALL_NODE_IDS = [
    # rack-a
    "rack-a-s0", "rack-a-s1", "rack-a-h2", "rack-a-h3", "rack-a-f4",
    # rack-b
    "rack-b-s0", "rack-b-s1", "rack-b-h2", "rack-b-h3", "rack-b-f4",
    # rack-c
    "rack-c-s0", "rack-c-s1", "rack-c-h2", "rack-c-h3", "rack-c-f4",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClusterScenario:
    """Cluster fault / scheduling scenario definition."""

    id: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    node_failures: list[str] = field(default_factory=list)
    rack_failure: Optional[str] = None
    new_jobs: list[dict] = field(default_factory=list)
    force_queued: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Job templates (reusable across scenarios)
# ---------------------------------------------------------------------------

def _urgent_jobs(count: int, gpu: int = 2, cpu: int = 16,
                 ram_gb: int = 64) -> list[dict]:
    """Generate a batch of urgent job definitions."""
    return [
        {"gpu": gpu, "cpu": cpu, "ram_gb": ram_gb, "priority": "urgent"}
        for _ in range(count)
    ]


def _large_gpu_jobs(count: int, gpu: int = 4, cpu: int = 32,
                    ram_gb: int = 128) -> list[dict]:
    """Generate large GPU jobs that are hard to place (fragmentation)."""
    return [
        {"gpu": gpu, "cpu": cpu, "ram_gb": ram_gb, "priority": "normal"}
        for _ in range(count)
    ]


def _preemptible_jobs(count: int, gpu: int = 1, cpu: int = 8,
                      ram_gb: int = 32) -> list[dict]:
    """Generate preemptible jobs for priority-conflict scenarios."""
    return [
        {"gpu": gpu, "cpu": cpu, "ram_gb": ram_gb, "priority": "preemptible"}
        for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# Base scenarios (6 types)
# ---------------------------------------------------------------------------

_BASE_SCENARIOS: list[ClusterScenario] = [
    # 1. Single node failure (easy)
    ClusterScenario(
        id="single_node_failure",
        description="Standard node rack-a-s0 goes down, its jobs need rescheduling",
        difficulty="easy",
        node_failures=["rack-a-s0"],
    ),
    # 2. Partial rack failure (hard) — 3 of 5 nodes in rack-a go down,
    # but 2 survive so affinity-bound jobs can still be placed there.
    ClusterScenario(
        id="rack_failure_a",
        description="3 nodes in rack-a fail; affinity jobs must use surviving rack-a nodes",
        difficulty="hard",
        node_failures=["rack-a-s0", "rack-a-s1", "rack-a-h2"],
    ),
    # 3. Job surge (medium)
    ClusterScenario(
        id="job_surge",
        description="Batch of 10 urgent jobs submitted, cluster already near capacity",
        difficulty="medium",
        new_jobs=_urgent_jobs(10, gpu=2, cpu=16, ram_gb=64),
    ),
    # 4. Resource fragmentation (medium)
    ClusterScenario(
        id="resource_fragmentation",
        description="Large 4-GPU jobs need consolidation across fragmented nodes",
        difficulty="medium",
        new_jobs=_large_gpu_jobs(6, gpu=4, cpu=32, ram_gb=128),
    ),
    # 5. Priority conflict (easy)
    ClusterScenario(
        id="priority_conflict",
        description="Urgent jobs queued while preemptible jobs occupy resources",
        difficulty="easy",
        new_jobs=(
            _urgent_jobs(4, gpu=2, cpu=16, ram_gb=64)
            + _preemptible_jobs(8, gpu=1, cpu=8, ram_gb=32)
        ),
    ),
    # 6. Compound: node failure + surge (hard)
    ClusterScenario(
        id="compound_failure_surge",
        description="Fat node rack-b-f4 fails while 4 urgent jobs arrive simultaneously",
        difficulty="hard",
        node_failures=["rack-b-f4"],
        new_jobs=_urgent_jobs(4, gpu=2, cpu=16, ram_gb=64),
    ),
]


# ---------------------------------------------------------------------------
# Parameterized variants — different nodes, racks, job sizes
# ---------------------------------------------------------------------------

def _generate_variants() -> list[ClusterScenario]:
    """Generate parameterized variants to reach 15+ total scenarios."""
    variants: list[ClusterScenario] = []

    # --- Single node failure variants (easy) ---
    for node_id in ["rack-b-h2", "rack-c-f4"]:
        rack_label = node_id.split("-")[1]  # "b" or "c"
        variants.append(ClusterScenario(
            id=f"single_node_failure_{rack_label}_{node_id.split('-')[-1]}",
            description=f"Node {node_id} goes down, its jobs need rescheduling",
            difficulty="easy",
            node_failures=[node_id],
        ))

    # --- Rack failure variants (hard) ---
    for rack in ["rack-b", "rack-c"]:
        variants.append(ClusterScenario(
            id=f"rack_failure_{rack.split('-')[1]}",
            description=f"Entire {rack} goes down; 5 nodes lost",
            difficulty="hard",
            rack_failure=rack,
        ))

    # --- Job surge variants (medium) ---
    variants.append(ClusterScenario(
        id="job_surge_small",
        description="5 urgent GPU-heavy jobs submitted",
        difficulty="medium",
        new_jobs=_urgent_jobs(5, gpu=4, cpu=32, ram_gb=128),
    ))
    variants.append(ClusterScenario(
        id="job_surge_large",
        description="8 urgent small jobs flood the queue",
        difficulty="medium",
        new_jobs=_urgent_jobs(8, gpu=1, cpu=8, ram_gb=32),
    ))

    # --- Resource fragmentation variant (medium) ---
    # Use 4-GPU normal jobs instead of 8-GPU fat-only jobs so any node
    # type with 4 free GPUs can host them (rack-c has ~20 GPU free).
    variants.append(ClusterScenario(
        id="resource_fragmentation_fat",
        description="Large 4-GPU jobs compete for remaining capacity across nodes",
        difficulty="medium",
        new_jobs=_large_gpu_jobs(3, gpu=4, cpu=32, ram_gb=128),
    ))

    # --- Priority conflict variant (easy) ---
    variants.append(ClusterScenario(
        id="priority_conflict_rack_affinity",
        description="Urgent jobs with rack-c affinity queued behind preemptible jobs",
        difficulty="easy",
        new_jobs=[
            {"gpu": 2, "cpu": 16, "ram_gb": 64,
             "priority": "urgent", "rack_affinity": "rack-c"}
            for _ in range(4)
        ],
    ))

    # --- Compound variants (hard) ---
    # Partial rack-c failure (3 nodes down, 2 survive ~8 GPU) + 5 urgent
    # jobs (10 GPU).  Evicted jobs + new arrivals fit in remaining capacity.
    variants.append(ClusterScenario(
        id="compound_rack_surge",
        description="3 nodes in rack-c fail while 5 urgent jobs arrive",
        difficulty="hard",
        node_failures=["rack-c-s0", "rack-c-s1", "rack-c-h2"],
        new_jobs=_urgent_jobs(5, gpu=2, cpu=16, ram_gb=64),
    ))
    variants.append(ClusterScenario(
        id="compound_multi_node_failure",
        description="Fat node fails with 2 urgent jobs arriving",
        difficulty="hard",
        node_failures=["rack-a-f4"],
        new_jobs=_urgent_jobs(2, gpu=2, cpu=16, ram_gb=64),
    ))
    variants.append(ClusterScenario(
        id="compound_fragmentation_failure",
        description="Highmem node fails while large jobs need placement",
        difficulty="hard",
        node_failures=["rack-b-h3"],
        new_jobs=_large_gpu_jobs(5, gpu=4, cpu=32, ram_gb=128),
    ))

    return variants


# ---------------------------------------------------------------------------
# Combined scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: list[ClusterScenario] = _BASE_SCENARIOS + _generate_variants()

_SCENARIO_MAP: dict[str, ClusterScenario] = {s.id: s for s in SCENARIOS}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ClusterScenarioLoader:
    """Load and apply cluster scheduling scenarios to a ClusterManager."""

    def load(self, scenario_id: str) -> ClusterScenario:
        """Load a scenario by ID. Raises KeyError if not found."""
        if scenario_id not in _SCENARIO_MAP:
            raise KeyError(f"Unknown scenario: {scenario_id}")
        return _SCENARIO_MAP[scenario_id]

    def load_all(self) -> list[ClusterScenario]:
        """Return all registered scenarios."""
        return list(SCENARIOS)

    def setup_episode(
        self,
        manager: ClusterManager,
        scenario: ClusterScenario,
    ) -> None:
        """Apply scenario faults and jobs to a fresh ClusterManager.

        Order of operations:
        1. Fail individual nodes (re-queues their jobs)
        2. Fail entire rack if specified (re-queues all rack jobs)
        3. Add new jobs (all start Queued)
        4. Force specific existing jobs to Queued status
        5. Recompute scheduling state via solve()
        """
        # 1. Node failures
        for node_id in scenario.node_failures:
            manager.fail_node(node_id)

        # 2. Rack failure
        if scenario.rack_failure is not None:
            manager.fail_rack(scenario.rack_failure)

        # 3. New jobs
        if scenario.new_jobs:
            manager.add_jobs(scenario.new_jobs)

        # 4. Force specific jobs to Queued
        for job_id in scenario.force_queued:
            if job_id in manager._jobs:
                # Remove from assignments if running
                if job_id in manager._assignments:
                    del manager._assignments[job_id]
                manager._jobs[job_id]["status"] = "Queued"

        # 5. Recompute
        manager.solve()
