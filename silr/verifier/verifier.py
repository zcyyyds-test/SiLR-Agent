"""SiLR Verifier: Simulation-in-the-Loop Reasoning pipeline.

Domain-agnostic verification pipeline:
1. Validate action name against allowed_actions
2. Create shadow copy of the system
3. Execute action on shadow via Tool layer
4. Run steady-state solver on shadow → check convergence
5. (Optional) Run post-solve hook (e.g., TDS for power grids)
6. Run all constraint checkers
7. Generate report
"""

import time
import logging

from .types import Verdict, VerificationResult
from .report import ReportGenerator
from silr.core.interfaces import BaseSystemManager
from silr.core.config import DomainConfig

logger = logging.getLogger(__name__)


class SiLRVerifier:
    """Simulation-in-the-Loop Reasoning verifier.

    Validates proposed actions by executing them on a shadow copy,
    running the steady-state solver + optional post-solve hook,
    and checking all domain constraints.
    """

    def __init__(
        self,
        manager: BaseSystemManager,
        domain_config: DomainConfig,
        shadow_setup_hook: "Callable[[BaseSystemManager], None] | None" = None,
    ):
        self._manager = manager
        self._domain_config = domain_config
        self._shadow_setup_hook = shadow_setup_hook

        self._checkers = list(domain_config.checkers)
        self._allowed_actions = domain_config.allowed_actions
        self._create_toolset = domain_config.create_toolset
        self._post_solve_hook = domain_config.post_solve_hook
        self._reporter = ReportGenerator()

    def verify(self, action: dict) -> VerificationResult:
        """Verify a proposed action.

        Args:
            action: {"tool_name": str, "params": dict}

        Returns:
            VerificationResult with verdict, check results, and report.
        """
        t0 = time.perf_counter()

        tool_name = action.get("tool_name", "")
        params = action.get("params", {})

        # 1. Validate action name
        if tool_name not in self._allowed_actions:
            result = VerificationResult(
                verdict=Verdict.ERROR,
                action=action,
                fail_reason=f"Action '{tool_name}' not in allowed actions: {sorted(self._allowed_actions)}",
                elapsed_seconds=time.perf_counter() - t0,
            )
            result.report_text = self._reporter.generate(result)
            return result

        try:
            # 2. Create shadow copy
            shadow = self._manager.create_shadow_copy()
            try:
                if self._shadow_setup_hook is not None:
                    self._shadow_setup_hook(shadow)
            except Exception:
                del shadow
                raise
            shadow_tools = self._create_toolset(shadow)

            # 3. Execute action on shadow
            action_tool = shadow_tools.get(tool_name)
            if action_tool is None:
                result = VerificationResult(
                    verdict=Verdict.ERROR,
                    action=action,
                    fail_reason=f"Tool '{tool_name}' not found in toolset",
                    elapsed_seconds=time.perf_counter() - t0,
                )
                result.report_text = self._reporter.generate(result)
                return result

            action_result = action_tool.execute(**params)
            if action_result["status"] == "error":
                result = VerificationResult(
                    verdict=Verdict.FAIL,
                    action=action,
                    action_result=action_result,
                    fail_reason=f"Action execution failed: {action_result.get('error', 'unknown')}",
                    elapsed_seconds=time.perf_counter() - t0,
                )
                result.report_text = self._reporter.generate(result)
                return result

            # 4. Run steady-state solver on shadow
            solver_converged = shadow.solve()

            if not solver_converged:
                result = VerificationResult(
                    verdict=Verdict.FAIL,
                    action=action,
                    action_result=action_result,
                    solver_converged=False,
                    fail_reason="Steady-state solver did not converge after action",
                    elapsed_seconds=time.perf_counter() - t0,
                )
                result.report_text = self._reporter.generate(result)
                return result

            # 5. Run post-solve hook (e.g., TDS for power grids)
            post_solve_passed = None
            if self._post_solve_hook is not None:
                post_solve_passed = self._post_solve_hook(shadow)
                if not post_solve_passed:
                    result = VerificationResult(
                        verdict=Verdict.FAIL,
                        action=action,
                        action_result=action_result,
                        solver_converged=True,
                        post_solve_passed=False,
                        fail_reason="Post-solve check failed",
                        elapsed_seconds=time.perf_counter() - t0,
                    )
                    result.report_text = self._reporter.generate(result)
                    return result

            # 6. Run all constraint checkers
            check_results = []
            for checker in self._checkers:
                cr = checker.check(shadow.system_state, shadow.base_mva)
                check_results.append(cr)

            # 7. Determine verdict
            all_passed = all(cr.passed for cr in check_results)
            failed_names = [cr.checker_name for cr in check_results if not cr.passed]

            if all_passed:
                verdict = Verdict.PASS
                fail_reason = None
            else:
                verdict = Verdict.FAIL
                fail_reason = f"Constraint violations: {', '.join(failed_names)}"

            result = VerificationResult(
                verdict=verdict,
                action=action,
                check_results=check_results,
                action_result=action_result,
                solver_converged=True,
                post_solve_passed=post_solve_passed,
                fail_reason=fail_reason,
                elapsed_seconds=time.perf_counter() - t0,
            )
            result.report_text = self._reporter.generate(result)
            return result

        except Exception as e:
            logger.exception("SiLR verification failed with exception")
            result = VerificationResult(
                verdict=Verdict.ERROR,
                action=action,
                fail_reason=f"{type(e).__name__}: {e}",
                elapsed_seconds=time.perf_counter() - t0,
            )
            result.report_text = self._reporter.generate(result)
            return result
