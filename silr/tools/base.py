"""Base class for all SiLR tools.

Provides uniform execute() → ToolResult wrapping with error handling.
Subclasses implement _validate_params() and _run().
"""

import logging
from abc import ABC, abstractmethod

from silr.types import ToolResult
from silr.core.interfaces import BaseSystemManager
from silr.exceptions import SiLRError

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base for all tools. Never crashes — errors become ToolResult."""

    name: str = "base_tool"
    description: str = ""

    def __init__(self, manager: BaseSystemManager):
        self.manager = manager

    def execute(self, **kwargs) -> ToolResult:
        """Execute tool with uniform error handling."""
        try:
            self._validate_params(**kwargs)
            data = self._run(**kwargs)
            return ToolResult(
                status="success",
                tool_name=self.name,
                timestamp=self.manager.sim_time,
                data=data,
                error=None,
            )
        except SiLRError as e:
            logger.error(f"{self.name} failed: {e}")
            return ToolResult(
                status="error",
                tool_name=self.name,
                timestamp=self.manager.sim_time,
                data={},
                error=str(e),
            )
        except Exception as e:
            logger.exception(f"{self.name} unexpected error")
            return ToolResult(
                status="error",
                tool_name=self.name,
                timestamp=self.manager.sim_time,
                data={},
                error=f"Internal error: {type(e).__name__}: {e}",
            )

    @abstractmethod
    def _validate_params(self, **kwargs) -> None:
        pass

    @abstractmethod
    def _run(self, **kwargs) -> dict:
        pass
