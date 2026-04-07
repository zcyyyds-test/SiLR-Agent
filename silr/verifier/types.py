"""Data types for SiLR Verifier results."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Verdict(Enum):
    PASS = "PASS"
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
