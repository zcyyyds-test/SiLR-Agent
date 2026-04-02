"""SiLR evaluation: batch runner and metrics."""

from .runner import EvalRunner
from .metrics import compute_metrics, compute_multi_agent_metrics
from .multi_runner import MultiAgentEvalRunner

__all__ = [
    "EvalRunner",
    "compute_metrics",
    "MultiAgentEvalRunner",
    "compute_multi_agent_metrics",
]
