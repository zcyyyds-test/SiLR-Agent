"""Data types for SiLR Verifier results."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Verdict(Enum):
    """Verifier verdicts.

    Graded safety semantics aligned with recoverability-set shielding:

    - ``PASS``: post-action state has *zero* outstanding violations
      (= recovered). The terminal-safety goal of the episode.
    - ``SAFE_PROGRESS``: post-action state is *admissible* under the
      selected progress-family criterion. The structured policies use
      violation types/counts and optional severity magnitude; the
      ``scalar_progress`` ablation uses the domain-native scalar penalty.
      Only emitted when the domain's gating policy explicitly enables
      non-terminal recovery admission.
    - ``FAIL``: action is unsafe — solver diverged, new violation type
      appeared, violation count grew, or (in ``terminal`` policy) the
      state is not yet recovered.
    - ``ERROR``: action could not even be evaluated (tool-layer error,
      invalid params, exception). Kept distinct so DPO/GRPO training
      does not conflate parser typos with physical unsafety.
    """

    PASS = "PASS"
    SAFE_PROGRESS = "SAFE_PROGRESS"
    FAIL = "FAIL"
    ERROR = "ERROR"


@dataclass
class Violation:
    constraint_type: str          # e.g. "voltage", "link_utilization"
    device_type: str              # e.g. "bus", "generator", "link"
    device_id: Any
    metric: str                   # e.g. "v_pu", "utilization_pct"
    value: float
    limit: float
    unit: str
    severity: str                 # "warning" | "violation" | "critical"
    detail: str


@dataclass
class CheckResult:
    checker_name: str             # e.g. "voltage", "link_utilization"
    passed: bool
    summary: dict
    violations: list[Violation] = field(default_factory=list)


@dataclass
class VerificationResult:
    verdict: Verdict
    action: dict                  # {"tool_name": str, "params": dict}
    check_results: list[CheckResult] = field(default_factory=list)
    action_result: Optional[dict] = None
    solver_converged: Optional[bool] = None
    post_solve_passed: Optional[bool] = None
    fail_reason: Optional[str] = None
    report_text: str = ""
    elapsed_seconds: float = 0.0
