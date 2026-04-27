"""SWE domain constraint checkers.

Two checkers:
- RegressionChecker: instance's PASS_TO_PASS tests must all stay green.
- TargetTestChecker: instance's FAIL_TO_PASS tests must all become green.

Both invoke pytest as a subprocess in the manager's work_dir. The checker
reads `system_state["instance_id"]` / `["work_dir"]` and uses the instance
metadata stashed on the manager (via module-level registry) to resolve test
node-id lists. For simplicity the manager is accessed through a thread-local
registry populated by the verifier just before checks — same pattern used
for the cluster domain's observer.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation


# Module-level registry populated by RepoManager when system_state is read.
# This lets the checker look up the instance metadata (fail_to_pass /
# pass_to_pass lists) without putting them in the state dict (which would
# bloat observer output and churn serialization).
_INSTANCE_REGISTRY: dict[str, Any] = {}


def register_instance(instance_id: str, instance: Any) -> None:
    _INSTANCE_REGISTRY[instance_id] = instance


def _run_pytest(work_dir: str, node_ids: list[str], timeout: int = 180) -> tuple[int, int, list[str]]:
    """Return (n_total, n_passed, failed_node_ids).

    - No `-x`: pytest runs ALL supplied tests so n_passed = n_total - n_failed
      is actually accurate. With `-x` (stop-on-first-failure) we'd miscount
      un-run tests as passed, inflating the GRPO reward signal.
    - returncode check: pytest exits 0 (all pass) or 1 (some fail); any other
      code (2=interrupted, 3=internal error, 4=usage, 5=nothing collected)
      means we can't trust the output. Treat as "everything failed" so
      reward conservatively reflects an unusable run.
    - Parse both FAILED and ERROR lines: setup/teardown/collection errors
      produce ERROR prefixes which would otherwise be miscounted as passes.
    """
    if not node_ids:
        return (0, 0, [])
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "-q", "--no-header", *node_ids],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (len(node_ids), 0, node_ids)
    if result.returncode not in (0, 1):
        # 2=interrupted, 3=internal, 4=usage, 5=no tests collected. Any of
        # these means the run is meaningless — conservative: all "failed".
        return (len(node_ids), 0, node_ids)
    failed: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            parts = line.split()
            if len(parts) >= 2:
                failed.append(parts[1])
    n_total = len(node_ids)
    n_passed = n_total - len(failed)
    return (n_total, n_passed, failed)


class RegressionChecker(BaseConstraintChecker):
    name = "swe_regression"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        instance_id = system_state["instance_id"]
        work_dir = system_state["work_dir"]
        inst = _INSTANCE_REGISTRY.get(instance_id)
        if inst is None:
            return CheckResult(
                checker_name=self.name,
                passed=False,
                summary={"error": "instance not registered"},
                violations=[],
            )
        n_total, n_passed, failed = _run_pytest(work_dir, inst.pass_to_pass)
        violations = [
            Violation(
                constraint_type="regression",
                device_type="test",
                device_id=node_id,
                metric="pass",
                value=0.0,
                limit=1.0,
                unit="bool",
                severity="violation",
                detail=f"PASS_TO_PASS test failed: {node_id}",
            )
            for node_id in failed
        ]
        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={"n_total": n_total, "n_passed": n_passed, "n_failed": len(failed)},
            violations=violations,
        )


class TargetTestChecker(BaseConstraintChecker):
    name = "swe_target"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        instance_id = system_state["instance_id"]
        work_dir = system_state["work_dir"]
        inst = _INSTANCE_REGISTRY.get(instance_id)
        if inst is None:
            return CheckResult(
                checker_name=self.name,
                passed=False,
                summary={"error": "instance not registered"},
                violations=[],
            )
        n_target, n_green, failed = _run_pytest(work_dir, inst.fail_to_pass)
        violations = [
            Violation(
                constraint_type="target",
                device_type="test",
                device_id=node_id,
                metric="green",
                value=0.0,
                limit=1.0,
                unit="bool",
                severity="critical",
                detail=f"FAIL_TO_PASS test still red: {node_id}",
            )
            for node_id in failed
        ]
        return CheckResult(
            checker_name=self.name,
            passed=n_target > 0 and len(violations) == 0,
            summary={"n_target": n_target, "n_green": n_green, "n_red": len(failed)},
            violations=violations,
        )
