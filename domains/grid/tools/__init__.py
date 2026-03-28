"""Tool registry and toolset factory for the power grid domain."""

from ..simulator import SystemManager
from .simulation import InjectFaultTool, RunTdsTool, RunPowerFlowTool
from .observation import (
    GetBusVoltageTool, GetFrequencyTool, GetRotorAngleTool, GetLineFlowTool,
)
from .action import AdjustGenTool, ShedLoadTool, TripLineTool, CloseLineTool
from .stability import CheckStabilityTool

TOOL_CLASSES = [
    InjectFaultTool,
    RunTdsTool,
    RunPowerFlowTool,
    GetBusVoltageTool,
    GetFrequencyTool,
    GetRotorAngleTool,
    GetLineFlowTool,
    AdjustGenTool,
    ShedLoadTool,
    TripLineTool,
    CloseLineTool,
    CheckStabilityTool,
]

TOOL_REGISTRY = {cls.name: cls for cls in TOOL_CLASSES}


def create_toolset(manager: SystemManager) -> dict:
    """Create a complete toolset bound to a SystemManager instance."""
    return {name: cls(manager) for name, cls in TOOL_REGISTRY.items()}
