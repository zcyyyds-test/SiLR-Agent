"""SiLR Verifier: Simulation-in-the-Loop Reasoning pipeline.

Domain-agnostic verification pipeline:

1. Validate action name against allowed_actions
2. Create shadow copy of the system
3. (progress-family policies) Snapshot pre-action violation baseline on the shadow
4. Execute action on shadow via Tool layer
5. Run steady-state solver on shadow → check convergence
6. (Optional) Run post-solve hook (e.g., TDS for power grids)
7. Run all constraint checkers
8. Map (baseline, post) into a graded verdict according to the domain
   ``gating_policy``:

   - PASS         post state has zero violations (= recovered)
   - SAFE_PROGRESS  progress-family only: post state is an admissible
                  non-terminal recovery step under the selected predicate
   - FAIL         solver diverged, post-solve hook failed, post state
                  worsened, or (``terminal`` policy) state not recovered
   - ERROR        action could not be evaluated (tool error / exception)

The split between PASS (terminal recovery) and SAFE_PROGRESS
(admissible step) lets a single verifier double as runtime guard and
GRPO reward signal: apply-gating reads the graded verdict, downstream
training targets the terminal verdict.
"""

import math
import os
import time
import logging
from typing import Optional

from .types import Verdict, VerificationResult
from .report import ReportGenerator
from silr.core.interfaces import BaseSystemManager
from silr.core.config import DomainConfig

logger = logging.getLogger(__name__)


# L3 magnitude-guard hyperparameters (see docs/method_predicates.md §4.2).
# A magnitude-aware step passes iff post_score is within a relative slack
# of α-1 over baseline (default 5%) OR an absolute floor (default 1e-3) —
# whichever is larger. The relative threshold dominates for non-trivial
# baselines; the absolute floor is used when baseline ≈ 0 to keep the
# guard robust to deterministic-solver round-off.
_MAGNITUDE_RELATIVE_SLACK = 0.05
_MAGNITUDE_ABS_FLOOR = 1e-3

# Scalar-progress is a deliberately simple ablation baseline: it admits a
# non-terminal step only when the domain's native scalar penalty does not
# increase beyond deterministic-solver jitter. This tests whether the paper's
# structured product predicates do real work beyond threshold tuning.
_SCALAR_PROGRESS_ABS_FLOOR = 1e-3
_SCALAR_PROGRESS_RELATIVE_SLACK = 0.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid float env %s=%r", name, raw)
        return default


def _scalar_progress_threshold(baseline: float) -> float:
    """Configurable scalar baseline threshold for ablation sweeps.

    Defaults preserve the original strict non-increase baseline. Environment
    variables are opt-in so existing remote runs and reports keep the same
    semantics unless a batch script explicitly requests a sensitivity sweep.
    """
    abs_floor = _env_float(
        "SILR_SCALAR_PROGRESS_ABS_FLOOR",
        _SCALAR_PROGRESS_ABS_FLOOR,
    )
    rel_slack = _env_float(
        "SILR_SCALAR_PROGRESS_RELATIVE_SLACK",
        _SCALAR_PROGRESS_RELATIVE_SLACK,
    )
    return max(baseline + abs_floor, baseline * (1.0 + rel_slack))


def _severity_score(check_results) -> float:
    """Domain-agnostic magnitude proxy: sum of |value - limit| over violations.

    Non-finite values map to a large constant so that a divergent solver
    output (NaN propagation) trips the L3 guard rather than silently
    comparing as zero.
    """
    score = 0.0
    for cr in check_results:
        for v in cr.violations:
            try:
                val = float(v.value)
                lim = float(v.limit)
            except (TypeError, ValueError):
                score += 1.0
                continue
            if not (math.isfinite(val) and math.isfinite(lim)):
                score += 1e6
                continue
            score += abs(val - lim)
    return score


def _native_scalar_penalty(manager: BaseSystemManager) -> Optional[float]:
    """Best-effort domain-native scalar penalty for scalar baseline ablations."""
    try:
        value = getattr(manager, "last_penalty")
    except Exception:
        return None
    try:
        penalty = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(penalty):
        return None
    return penalty


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
        self._gating_policy = domain_config.gating_policy
        if self._gating_policy not in (
            "terminal",
            "progress",
            "progress_mag",
            "scalar_progress",
        ):
            raise ValueError(
                f"Unknown gating_policy {self._gating_policy!r}; "
                f"expected 'terminal', 'progress', 'progress_mag', or "
                f"'scalar_progress'"
            )
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

            # 2b. Snapshot pre-action violation baseline on the shadow.
            # Progress-family verdicts compare post-action state to this
            # baseline; terminal policy ignores it. Snapshotting on the shadow
            # (not the live manager) guarantees a self-consistent before/after
            # comparison even if the shadow_setup_hook mutates state.
            if self._gating_policy in ("progress", "progress_mag", "scalar_progress"):
                baseline_checks = [
                    checker.check(shadow.system_state, shadow.base_mva)
                    for checker in self._checkers
                ]
                baseline_violation_count = sum(
                    len(cr.violations) for cr in baseline_checks if not cr.passed
                )
                baseline_violation_types = {
                    cr.checker_name for cr in baseline_checks if not cr.passed
                }
                if self._gating_policy == "progress_mag":
                    baseline_severity_score = _severity_score(baseline_checks)
                else:
                    baseline_severity_score = None
                if self._gating_policy == "scalar_progress":
                    baseline_scalar_penalty = _native_scalar_penalty(shadow)
                else:
                    baseline_scalar_penalty = None
            else:
                baseline_violation_count = None
                baseline_violation_types = None
                baseline_severity_score = None
                baseline_scalar_penalty = None

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
                # Tool-layer error (ValidationError, DeviceNotFoundError, etc.) means
                # the action could not even be evaluated — not a safety verdict.
                # Verdict.ERROR keeps it out of FAIL-based training signals
                # (e.g., DPO rejected pairs in trajectory.export_dpo_pairs), which
                # would otherwise conflate "LLM made a typo / out-of-bounds param"
                # with "the action is physically unsafe".
                result = VerificationResult(
                    verdict=Verdict.ERROR,
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

            # 7. Determine graded verdict (PASS / SAFE_PROGRESS / FAIL).
            # PASS is the terminal-recovery verdict (zero outstanding
            # violations). SAFE_PROGRESS is emitted only by progress-family
            # policies when the post-action state satisfies that policy's
            # non-terminal recovery predicate. FAIL covers both "unsafe"
            # (worsened under the selected predicate) and (terminal-policy
            # only) "not yet recovered".
            all_passed = all(cr.passed for cr in check_results)
            failed_names = [cr.checker_name for cr in check_results if not cr.passed]

            if all_passed:
                verdict = Verdict.PASS
                fail_reason = None
            elif self._gating_policy == "scalar_progress":
                baseline = baseline_scalar_penalty
                post = _native_scalar_penalty(shadow)
                if baseline is None or post is None:
                    verdict = Verdict.FAIL
                    fail_reason = (
                        "Scalar progress baseline requires finite "
                        "manager.last_penalty before and after solve"
                    )
                else:
                    threshold = _scalar_progress_threshold(baseline)
                    if post > threshold:
                        verdict = Verdict.FAIL
                        fail_reason = (
                            f"Scalar penalty worsened: {baseline:.4f} -> "
                            f"{post:.4f} > threshold "
                            f"{threshold:.4f}"
                        )
                    else:
                        verdict = Verdict.SAFE_PROGRESS
                        fail_reason = (
                            f"Admissible scalar recovery step "
                            f"(penalty {baseline:.4f} -> {post:.4f})"
                        )
            elif self._gating_policy in ("progress", "progress_mag"):
                post_violation_count = sum(
                    len(cr.violations) for cr in check_results if not cr.passed
                )
                post_violation_types = set(failed_names)
                new_violation_types = post_violation_types - baseline_violation_types
                if new_violation_types:
                    verdict = Verdict.FAIL
                    fail_reason = (
                        f"Introduced new violation type(s): "
                        f"{sorted(new_violation_types)}"
                    )
                elif post_violation_count > baseline_violation_count:
                    verdict = Verdict.FAIL
                    fail_reason = (
                        f"Violation count worsened: "
                        f"{baseline_violation_count} -> {post_violation_count}"
                    )
                elif self._gating_policy == "progress_mag":
                    # L3 magnitude guard: violation count and types are
                    # admissible, but check that aggregated severity score
                    # does not inflate beyond the relative-OR-absolute
                    # threshold. Defends against count-preserving magnitude
                    # drift attacks (see decisions.md "Magnitude-aware
                    # SAFE_PROGRESS" entry).
                    post_severity_score = _severity_score(check_results)
                    threshold = max(
                        baseline_severity_score * (1.0 + _MAGNITUDE_RELATIVE_SLACK),
                        baseline_severity_score + _MAGNITUDE_ABS_FLOOR,
                    )
                    if post_severity_score > threshold:
                        verdict = Verdict.FAIL
                        fail_reason = (
                            f"Magnitude drift: severity {baseline_severity_score:.4f} "
                            f"-> {post_severity_score:.4f} > threshold {threshold:.4f}"
                        )
                    else:
                        verdict = Verdict.SAFE_PROGRESS
                        fail_reason = (
                            f"Admissible recovery step "
                            f"(viol {baseline_violation_count} -> "
                            f"{post_violation_count}, severity "
                            f"{baseline_severity_score:.4f} -> "
                            f"{post_severity_score:.4f}, no new types)"
                        )
                else:
                    verdict = Verdict.SAFE_PROGRESS
                    fail_reason = (
                        f"Admissible recovery step "
                        f"(viol {baseline_violation_count} -> "
                        f"{post_violation_count}, no new types)"
                    )
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
