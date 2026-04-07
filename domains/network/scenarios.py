"""Cascading fault scenarios for multi-agent coordinator testing.

Each scenario defines multiple simultaneous faults that create
cross-constraint conflicts, requiring coordinated specialist dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .manager import NetworkManager


@dataclass
class CascadingScenario:
    """Multi-fault scenario definition."""

    id: str
    description: str
    faults: list[dict] = field(default_factory=list)
    overloads: list[dict] = field(default_factory=list)
    difficulty: str = "easy"


# Scenario definitions for the 5-node network:
#
#   1 ---100--- 2 ---100--- 3
#   |           |           |
#  80          60          80
#   |           |           |
#   4 ----100---- 5 --------+

SCENARIOS = [
    CascadingScenario(
        id="cascade_easy",
        description="Link failure + overload: restore then reroute",
        faults=[{"type": "fail_link", "src": 1, "dst": 2}],
        overloads=[{"src": 2, "dst": 5, "traffic": 55}],  # 91.7% > 90% threshold
        difficulty="easy",
    ),
    CascadingScenario(
        id="cascade_medium",
        description="Two link failures: node isolation causes connectivity violation",
        faults=[
            {"type": "fail_link", "src": 2, "dst": 3},
            {"type": "fail_link", "src": 3, "dst": 5},
        ],
        difficulty="medium",
    ),
    CascadingScenario(
        id="cascade_hard",
        description="Link failure + critical overload: restoring link worsens overload",
        faults=[{"type": "fail_link", "src": 1, "dst": 2}],
        overloads=[{"src": 2, "dst": 5, "traffic": 58}],  # 96.7% > 90% threshold
        difficulty="hard",
    ),
]

_SCENARIO_MAP = {s.id: s for s in SCENARIOS}


class NetworkScenarioLoader:
    """Load and apply cascading fault scenarios to a NetworkManager."""

    def load(self, scenario_id: str) -> CascadingScenario:
        if scenario_id not in _SCENARIO_MAP:
            raise KeyError(f"Unknown scenario: {scenario_id}")
        return _SCENARIO_MAP[scenario_id]

    def load_all(self) -> list[CascadingScenario]:
        return list(SCENARIOS)

    def setup_episode(self, manager: NetworkManager, scenario: CascadingScenario) -> None:
        """Apply all faults and overloads to the manager.

        Overloads are applied AFTER solve() because the solver
        resets all traffic to 0 and recalculates from demands.
        Manual overloads simulate pre-existing stress that the solver
        alone doesn't produce.
        """
        for fault in scenario.faults:
            if fault["type"] == "fail_link":
                manager.fail_link(fault["src"], fault["dst"])

        manager.solve()

        # Apply overloads after solver to avoid being wiped out
        for overload in scenario.overloads:
            key = (overload["src"], overload["dst"])
            if key in manager._links:
                manager._links[key]["traffic"] = overload["traffic"]
