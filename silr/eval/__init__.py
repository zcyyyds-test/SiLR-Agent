"""SiLR evaluation: batch runner and metrics."""

from .runner import EvalRunner
from .metrics import compute_metrics

__all__ = ["EvalRunner", "compute_metrics"]
