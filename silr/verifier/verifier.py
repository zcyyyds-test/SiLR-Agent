"""SiLR Verifier: Simulation-in-the-Loop Reasoning pipeline.

Domain-agnostic verification pipeline:

1. Validate action name against allowed_actions
2. Create shadow copy of the system
3. (progress-family / rollback policies) Snapshot pre-action violation baseline on the shadow
4. Execute action on shadow via Tool layer
5. Run steady-state solver on shadow → check convergence
6. (Optional) Run post-solve hook (e.g., TDS for power grids)
7. Run all constraint checkers
8. Map (baseline, post) into a graded verdict according to the domain
   ``gating_policy``:

   - PASS         post state has zero violations (= recovered)
   - SAFE_PROGRESS  progress-family/rollback only: post state is an admissible
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
# The guard is per-branch (ψ₃), not on the aggregate Σσ: a magnitude-aware
# step passes iff *every* surviving violation branch i has post-severity σ_i
# within a relative slack α-1 over its own baseline σ_i (default 5%) OR an
# absolute floor (default 1e-3) — whichever is larger. The relative threshold
# dominates for non-trivial baselines; the absolute floor handles σ_i ≈ 0
# robustly against deterministic-solver round-off.
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


def _violation_severity(v) -> float:
    """Per-branch magnitude σ_i = |value - limit| with robust fallbacks.

    Non-castable fields contribute a unit penalty; non-finite values map to a
    large constant so a divergent solver (NaN propagation) trips the L3 guard
    rather than silently comparing as zero.
    """
    try:
        val = float(v.value)
        lim = float(v.limit)
    except (TypeError, ValueError):
        return 1.0
    if not (math.isfinite(val) and math.isfinite(lim)):
        return 1e6
    return abs(val - lim)


def _severity_score(check_results) -> float:
    """Aggregate magnitude proxy: Σ_i σ_i over all violations.

    This is the *scalar projection* of the structured state — retained for
    reporting / scalar-baseline diagnostics. The structured progress_mag guard
    uses :func:`_violation_branches` (per-branch σ_i), not this sum.
    """
    return sum(_violation_severity(v) for cr in check_results for v in cr.violations)


def _violation_branches(check_results) -> dict:
    """Structured safety state Φ = (S, σ) as a per-branch severity map.

    Keys are the violation support set S — each a unique branch identifier
    ``(constraint_type, device_type, device_id, metric)`` — and values are the
    per-branch severities σ_i = |value - limit|. This is the per-branch state
    over which the structured admission predicates are evaluated: ψ₂ (support
    inclusion S(ŝ) ⊆ S(s)) and ψ₃ (per-branch severity envelope), as opposed
    to the scalar projection Σ_i σ_i used by the scalar baseline.

    Only violations from *failed* checkers are included; passed checkers carry
    no violations by the ``passed == (len(violations) == 0)`` invariant upheld
    by every checker.

    INVARIANT (relied upon by ψ₂/ψ₃): a branch — a unique physical quantity —
    is identified by exactly one ``(constraint_type, device_type, device_id,
    metric)`` key, and no two distinct checkers emit the same key. ``device_id``
    is coerced with ``str`` so that a domain reporting it as ``1`` in one call
    and ``"1"`` in another (the type is ``Any``) cannot be mistaken for two
    distinct branches and trip a spurious support expansion. On the rare event
    of a duplicate key (same checker, repeated entry) the worst (max) severity
    is kept — a conservative upper bound that keeps the guard sound.
    """
    branches: dict = {}
    for cr in check_results:
        if cr.passed:
            continue
        for v in cr.violations:
            key = (v.constraint_type, v.device_type, str(v.device_id), v.metric)
            sev = _violation_severity(v)
            prev = branches.get(key)
            if prev is None or sev > prev:
                branches[key] = sev
    return branches


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
            "rollback",
            "scalar_maxsev",
            "scalar_lex",
        ):
            raise ValueError(
                f"Unknown gating_policy {self._gating_policy!r}; "
                f"expected 'terminal', 'progress', 'progress_mag', "
                f"'scalar_progress', 'rollback', 'scalar_maxsev', "
                f"or 'scalar_lex'"
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
            if self._gating_policy in (
                "progress",
                "progress_mag",
                "scalar_progress",
                "rollback",
                "scalar_maxsev",
                "scalar_lex",
            ):
                baseline_checks = [
                    checker.check(shadow.system_state, shadow.base_mva)
                    for checker in self._checkers
                ]
                if self._gating_policy in (
                    "progress",
                    "progress_mag",
                    "rollback",
                    # Strong-scalar baselines (sup-norm, lexicographic) project the
                    # same per-branch state Φ=(S,σ) onto a total order; they read the
                    # branch map but, unlike ψ₂/ψ₃, do not enforce support inclusion.
                    "scalar_maxsev",
                    "scalar_lex",
                ):
                    # Φ(s) = (S, σ): the per-branch baseline against which the
                    # structured predicates ψ₂/ψ₃ are evaluated.
                    baseline_branches = _violation_branches(baseline_checks)
                else:
                    baseline_branches = None
                if self._gating_policy == "scalar_progress":
                    baseline_scalar_penalty = _native_scalar_penalty(shadow)
                else:
                    baseline_scalar_penalty = None
            else:
                baseline_branches = None
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
            elif self._gating_policy == "scalar_maxsev":
                # Strongest scalar candidate #1 (sup-norm gate): admit iff the
                # max per-branch severity ‖σ‖_∞ does not increase beyond the
                # scalar-progress slack. This is a TOTAL order on σ — it keeps
                # more geometry than Σσ but, projecting onto max alone, is blind
                # to a count-preserving redistribution that lifts a non-max
                # branch and to support swaps. Prop 1 predicts it is unsound.
                post_branches = _violation_branches(check_results)
                base_max = max(baseline_branches.values(), default=0.0)
                post_max = max(post_branches.values(), default=0.0)
                threshold = _scalar_progress_threshold(base_max)
                if post_max > threshold:
                    verdict = Verdict.FAIL
                    fail_reason = (
                        f"Sup-norm severity worsened: {base_max:.4f} -> "
                        f"{post_max:.4f} > threshold {threshold:.4f}"
                    )
                else:
                    verdict = Verdict.SAFE_PROGRESS
                    fail_reason = (
                        f"Admissible sup-norm scalar step "
                        f"(‖σ‖∞ {base_max:.4f} -> {post_max:.4f})"
                    )
            elif self._gating_policy == "scalar_lex":
                # Strongest scalar candidate #2 (lexicographic gate): admit iff
                # (|S|, Σσ) does not increase in lexicographic order — first the
                # violation count, then the aggregate severity within the scalar
                # slack. Still a TOTAL order: it ranks every Φ-antichain pair and
                # so, per Prop 1, accepts a non-improving step on some such pair.
                post_branches = _violation_branches(check_results)
                base_count = len(baseline_branches)
                post_count = len(post_branches)
                base_sum = sum(baseline_branches.values())
                post_sum = sum(post_branches.values())
                if post_count > base_count:
                    verdict = Verdict.FAIL
                    fail_reason = (
                        f"Lexicographic worsened on count: |S| "
                        f"{base_count} -> {post_count}"
                    )
                elif post_count < base_count:
                    verdict = Verdict.SAFE_PROGRESS
                    fail_reason = (
                        f"Admissible lexicographic step (|S| {base_count} -> "
                        f"{post_count})"
                    )
                else:
                    sum_threshold = _scalar_progress_threshold(base_sum)
                    if post_sum > sum_threshold:
                        verdict = Verdict.FAIL
                        fail_reason = (
                            f"Lexicographic worsened on Σσ at equal count "
                            f"{post_count}: {base_sum:.4f} -> {post_sum:.4f} "
                            f"> threshold {sum_threshold:.4f}"
                        )
                    else:
                        verdict = Verdict.SAFE_PROGRESS
                        fail_reason = (
                            f"Admissible lexicographic step (|S|={post_count}, "
                            f"Σσ {base_sum:.4f} -> {post_sum:.4f})"
                        )
            elif self._gating_policy in ("progress", "progress_mag", "rollback"):
                # ψ₂ — support inclusion: the post-action violation support set
                # S(ŝ) must be contained in the baseline support set S(s),
                # evaluated *per branch*. This fails count-preserving support
                # swaps (e.g. {A} -> {B}, same count but a different branch)
                # that an aggregate count/type check would wrongly admit, and
                # it subsumes both "no new violation type" and "violation count
                # does not grow" (S(ŝ) ⊆ S(s) ⟹ |S(ŝ)| ≤ |S(s)|).
                post_branches = _violation_branches(check_results)
                new_branches = [
                    key for key in post_branches if key not in baseline_branches
                ]
                if new_branches:
                    verdict = Verdict.FAIL
                    fail_reason = (
                        f"Support set expanded — {len(new_branches)} new "
                        f"violation branch(es) outside baseline: "
                        f"{[str(k) for k in new_branches[:3]]}"
                    )
                elif self._gating_policy == "progress_mag":
                    # ψ₃ — per-branch severity envelope: for every surviving
                    # branch i ∈ S(ŝ) ⊆ S(s), σ_i(ŝ) ≤ max(α·σ_i(s), σ_i(s)+ε).
                    # Enforced per branch (not on the scalar sum Σσ), so a
                    # severity reallocation that lowers the aggregate while
                    # inflating a single branch is rejected. Defends against
                    # count-preserving magnitude drift (see decisions.md
                    # "Magnitude-aware SAFE_PROGRESS" entry).
                    rel = _env_float(
                        "SILR_MAGNITUDE_RELATIVE_SLACK",
                        _MAGNITUDE_RELATIVE_SLACK,
                    )
                    abs_floor = _env_float(
                        "SILR_MAGNITUDE_ABS_FLOOR",
                        _MAGNITUDE_ABS_FLOOR,
                    )
                    drifted = []
                    for key, post_sev in post_branches.items():
                        # ψ₂ above (empty ``new_branches``) guarantees every
                        # post key is present in ``baseline_branches``, so this
                        # lookup is total — a resolved baseline branch simply
                        # drops out of ``post_branches`` and is not checked.
                        base_sev = baseline_branches[key]
                        branch_threshold = max(
                            base_sev * (1.0 + rel),
                            base_sev + abs_floor,
                        )
                        if post_sev > branch_threshold:
                            drifted.append(
                                (key, base_sev, post_sev, branch_threshold)
                            )
                    if drifted:
                        worst = max(drifted, key=lambda d: d[2] - d[3])
                        key, base_sev, post_sev, branch_threshold = worst
                        verdict = Verdict.FAIL
                        fail_reason = (
                            f"Per-branch magnitude drift on {key}: severity "
                            f"{base_sev:.4f} -> {post_sev:.4f} > threshold "
                            f"{branch_threshold:.4f} "
                            f"({len(drifted)} branch(es) drifted)"
                        )
                    else:
                        verdict = Verdict.SAFE_PROGRESS
                        fail_reason = (
                            f"Admissible recovery step (support "
                            f"{len(baseline_branches)} -> {len(post_branches)} "
                            f"branches ⊆ baseline, per-branch severity within "
                            f"envelope)"
                        )
                elif self._gating_policy == "rollback":
                    verdict = Verdict.SAFE_PROGRESS
                    fail_reason = (
                        f"Rollback baseline admits post-hoc state (support "
                        f"{len(baseline_branches)} -> {len(post_branches)} "
                        f"branches; no new violation branch outside baseline)"
                    )
                else:
                    verdict = Verdict.SAFE_PROGRESS
                    fail_reason = (
                        f"Admissible recovery step (support "
                        f"{len(baseline_branches)} -> {len(post_branches)} "
                        f"branches ⊆ baseline)"
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
                # Persist Φ = (S, σ) for downstream GRPO process-reward scoring.
                # baseline_branches is in scope from the pre-action snapshot
                # (None for non-progress policies); post_branches is recomputed
                # from check_results here so it is total across every verdict
                # path (PASS → {}, FAIL/SAFE_PROGRESS → post support set).
                baseline_branches=baseline_branches,
                post_branches=_violation_branches(check_results),
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
