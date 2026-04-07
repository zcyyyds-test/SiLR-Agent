"""Abstract base classes for the SiLR framework.

These ABCs define the minimal interface any domain must implement
to participate in the SiLR verification pipeline.

Design principle: zero silr imports — this module is a leaf node
in the dependency graph, preventing circular imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from silr.verifier.types import CheckResult


class BaseSystemManager(ABC):
    """Domain-agnostic system manager interface.

    Any domain's simulator wrapper must implement these methods to work
    with the SiLR verification pipeline and ReAct agent loop.

    The framework calls these methods during:
    - Verification: create_shadow_copy() → tool execution → solve() → checkers
    - Observation: system_state for reading current state
    - Timing: sim_time for ToolResult timestamps
    """

    @property
    @abstractmethod
    def sim_time(self) -> float:
        """Current simulation time."""
        ...

    @property
    @abstractmethod
    def base_mva(self) -> float:
        """Base power for per-unit conversions.

        Domains without per-unit systems should return 1.0.
        """
        ...

    @property
    @abstractmethod
    def system_state(self) -> Any:
        """Domain-specific system state object.

        Passed to constraint checkers. For power grids this is andes.System,
        for network domains it could be a graph object, etc.
        """
        ...

    @abstractmethod
    def create_shadow_copy(self) -> BaseSystemManager:
        """Create an independent copy for SiLR verification.

        The shadow copy must be fully isolated — modifications to it
        must not affect the original system.
        """
        ...

    @abstractmethod
    def solve(self) -> bool:
        """Re-compute steady-state from current configuration. Returns True if the solver converged.

        Each domain implements this differently:
        - Power grid: runs the AC power flow solver
        - GPU cluster: recomputes per-node resource utilisation
        - Network: redistributes traffic along shortest paths
        - Robotics: kinematics feasibility check
        """
        ...


class BaseConstraintChecker(ABC):
    """Domain-agnostic constraint checker interface.

    Each checker inspects the system state after an action and returns
    a CheckResult indicating whether domain-specific safety constraints
    are satisfied.

    Examples:
    - Power grid: VoltageChecker, FrequencyChecker, LineLoadingChecker
    - Network: LinkUtilizationChecker, LatencyChecker
    - Traffic: CongestionChecker, SignalTimingChecker
    """

    name: str  # Unique checker name (e.g. "voltage", "link_utilization")

    @abstractmethod
    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        """Check constraints against the given system state.

        Args:
            system_state: Domain-specific state object from
                          BaseSystemManager.system_state
            base_mva: Base value for unit conversion (1.0 if not applicable)

        Returns:
            CheckResult with pass/fail verdict and violations list.
        """
        ...
