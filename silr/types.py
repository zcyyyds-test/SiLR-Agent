"""Type definitions for SiLR tool results."""

from typing import TypedDict, Optional, Literal, Any


class ToolResult(TypedDict):
    status: Literal["success", "error"]
    tool_name: str
    timestamp: float
    data: dict[str, Any]
    error: Optional[str]


class ViolationFlag(TypedDict):
    violated: bool
    limit: float
    value: float
    unit: str
