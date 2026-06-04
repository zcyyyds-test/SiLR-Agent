"""Test GRPO reward computation from verification results."""

import pytest

from silr.verifier.types import (
    CheckResult, Verdict, VerificationResult, Violation,
)
from silr.training.reward import (
    RewardConfig, compute_grpo_reward, _fail_penalty, _margin_for_check,
)


def _make_result(verdict, check_results=None, **kw):
    return VerificationResult(
        verdict=verdict,
        action={"tool_name": "test", "params": {}},
        check_results=check_results or [],
        **kw,
    )


def _make_check(name, passed, summary=None, violations=None):
    return CheckResult(
        checker_name=name,
        passed=passed,
        summary=summary or {},
        violations=violations or [],
    )


def _make_violation(severity="violation"):
    return Violation(
        constraint_type="test", device_type="test", device_id=1,
        metric="x", value=1.0, limit=0.5, unit="pu",
        severity=severity, detail="test violation",
    )


class TestComputeGrpoReward:
    def test_error_returns_negative_one(self):
        r = _make_result(Verdict.ERROR)
        assert compute_grpo_reward(r) == -1.0

    def test_pass_no_config_returns_one(self):
        checks = [_make_check("voltage", True)]
        r = _make_result(Verdict.PASS, checks)
        assert compute_grpo_reward(r) == 1.0

    def test_pass_with_margin_bonus(self):
        checks = [_make_check("voltage", True, summary={
            "v_min_pu": 0.98, "v_max_pu": 1.02,
        })]
        config = RewardConfig(thresholds={
            "voltage": {"min_pu": 0.90, "max_pu": 1.10},
        })
        r = _make_result(Verdict.PASS, checks)
        reward = compute_grpo_reward(r, config)
        assert reward > 1.0  # has bonus
        assert reward <= 1.5  # capped at 0.5 bonus

    def test_fail_warning_severity(self):
        checks = [_make_check("test", False, violations=[
            _make_violation("warning"),
        ])]
        r = _make_result(Verdict.FAIL, checks)
        reward = compute_grpo_reward(r)
        assert reward == -0.3

    def test_fail_critical_severity(self):
        checks = [_make_check("test", False, violations=[
            _make_violation("critical"),
        ])]
        r = _make_result(Verdict.FAIL, checks)
        reward = compute_grpo_reward(r)
        assert reward == -1.0

    def test_fail_worst_severity_wins(self):
        checks = [_make_check("test", False, violations=[
            _make_violation("warning"),
            _make_violation("critical"),
        ])]
        r = _make_result(Verdict.FAIL, checks)
        assert compute_grpo_reward(r) == -1.0

    def test_fail_no_violations_default(self):
        """FAIL with no violations still gives minimum penalty."""
        checks = [_make_check("test", False)]
        r = _make_result(Verdict.FAIL, checks)
        assert compute_grpo_reward(r) == -0.3


class TestMarginForCheck:
    def test_voltage_margin(self):
        cr = _make_check("voltage", True, summary={
            "v_min_pu": 1.0, "v_max_pu": 1.0,
        })
        thresholds = {"voltage": {"min_pu": 0.90, "max_pu": 1.10}}
        m = _margin_for_check(cr, thresholds)
        assert m is not None
        assert 0.0 <= m <= 1.0

    def test_frequency_margin(self):
        cr = _make_check("frequency", True, summary={
            "max_abs_delta_f_hz": 0.1,
        })
        thresholds = {"frequency": {"max_hz": 0.5}}
        m = _margin_for_check(cr, thresholds)
        assert m is not None
        assert m == pytest.approx(0.8)

    def test_line_loading_margin(self):
        cr = _make_check("line_loading", True, summary={
            "max_loading_pct": 50.0,
        })
        thresholds = {"line_loading": {"max_pct": 100.0}}
        m = _margin_for_check(cr, thresholds)
        assert m == pytest.approx(0.5)

    def test_transient_margin(self):
        cr = _make_check("transient", True, summary={
            "max_separation_deg": 45.0,
        })
        thresholds = {"transient": {"max_deg": 180.0}}
        m = _margin_for_check(cr, thresholds)
        assert m == pytest.approx(0.75)

    def test_unknown_checker_with_violations(self):
        cr = _make_check("custom", True, summary={"n_violations": 0})
        m = _margin_for_check(cr, {})
        assert m == 1.0

    def test_unknown_checker_no_summary(self):
        cr = _make_check("custom", True, summary={})
        m = _margin_for_check(cr, {})
        assert m is None

    def test_missing_threshold_returns_none(self):
        cr = _make_check("voltage", True, summary={
            "v_min_pu": 1.0, "v_max_pu": 1.0,
        })
        m = _margin_for_check(cr, {})
        assert m is None


# ---------------------------------------------------------------------------
# SAFE_PROGRESS branch + three-arm reward (panel 2026-06-03: arm C/D/E).
# Φ = (S, σ) is supplied via baseline_branches / post_branches on the result.
# Branch key schema = (constraint_type, device_type, device_id, metric).
# ---------------------------------------------------------------------------
from silr.training.reward import _safe_progress_reward, compute_scalar_reward, compute_binary_reward


def _sp(pre, post):
    """SAFE_PROGRESS result carrying pre/post per-branch geometry."""
    return _make_result(Verdict.SAFE_PROGRESS, baseline_branches=pre, post_branches=post)


class TestSafeProgressBugFix:
    def test_safe_progress_no_longer_penalised(self):
        # Regression: pre-fix, SAFE_PROGRESS fell through to _fail_penalty (<0).
        r = _sp({("bl", "line", "0-1", "load"): 1.0}, {})  # fully resolved
        assert compute_grpo_reward(r) > 0.0

    def test_safe_progress_below_pass(self):
        r = _sp({("bl", "line", "0-1", "load"): 1.0}, {})
        assert compute_grpo_reward(r) < compute_grpo_reward(_make_result(Verdict.PASS))

    def test_no_geometry_falls_back_positive(self):
        r = _make_result(Verdict.SAFE_PROGRESS)  # no baseline_branches
        assert compute_grpo_reward(r) > 0.0


class TestArmD_ProductOrderGeometry:
    def test_high_severity_elimination_beats_low(self):
        # Arm D is severity-weighted: eliminating a high-σ branch is worth more.
        pre = {("bl", "line", "0-1", "load"): 8.0, ("bl", "line", "1-2", "load"): 1.0}
        elim_high = _sp(pre, {("bl", "line", "1-2", "load"): 1.0})  # killed the 8.0
        elim_low = _sp(pre, {("bl", "line", "0-1", "load"): 8.0})   # killed the 1.0
        assert _safe_progress_reward(elim_high) > _safe_progress_reward(elim_low)

    def test_no_progress_earns_near_zero(self):
        # Admissible but non-progressing (support + severity unchanged) ~ 0.
        pre = {("bl", "line", "0-1", "load"): 2.0}
        r = _sp(pre, dict(pre))
        assert abs(_safe_progress_reward(r)) < 1e-6

    def test_support_dominates_severity(self):
        # W2 > W3: eliminating a branch beats merely shrinking severity by the
        # same normalised amount on a surviving branch.
        pre = {("bl", "line", "0-1", "load"): 1.0, ("bl", "line", "1-2", "load"): 1.0}
        eliminate = _sp(pre, {("bl", "line", "1-2", "load"): 1.0})        # |S| 2->1
        shrink = _sp(pre, {("bl", "line", "0-1", "load"): 0.0,
                           ("bl", "line", "1-2", "load"): 1.0})           # |S| stays 2, σ down
        assert _safe_progress_reward(eliminate) > _safe_progress_reward(shrink)


class TestArmD_vs_ArmE_Separation:
    """The core thesis test: arm D (geometry) must diverge from arm E (count)."""

    def test_magnitude_drift_D_penalises_E_blind(self):
        # Count-preserving magnitude reallocation: same branches, one inflated.
        # (This is the projection-trap shape; SAFE_PROGRESS here is hypothetical,
        #  exercising the reward function's discrimination, not the gate.)
        pre = {("bl", "line", "0-1", "load"): 1.0, ("bl", "line", "1-2", "load"): 1.0}
        post = {("bl", "line", "0-1", "load"): 0.5, ("bl", "line", "1-2", "load"): 1.8}
        d = _safe_progress_reward(_sp(pre, post))
        e = compute_scalar_reward(_sp(pre, post))
        # Arm E is count-blind: |S| unchanged -> 0. Arm D sees the drift penalty.
        assert e == 0.0
        assert d < e  # D punishes the single-branch inflation, E does not

    def test_high_vs_low_severity_elim_E_identical_D_differs(self):
        pre = {("bl", "line", "0-1", "load"): 8.0, ("bl", "line", "1-2", "load"): 1.0}
        elim_high = _sp(pre, {("bl", "line", "1-2", "load"): 1.0})
        elim_low = _sp(pre, {("bl", "line", "0-1", "load"): 8.0})
        # Arm E (count fraction) cannot tell them apart; arm D can.
        assert compute_scalar_reward(elim_high) == compute_scalar_reward(elim_low)
        assert _safe_progress_reward(elim_high) != _safe_progress_reward(elim_low)


class TestArmC_Binary:
    def test_admitted_positive_rejected_negative(self):
        assert compute_binary_reward(_make_result(Verdict.PASS)) == 0.5
        assert compute_binary_reward(_make_result(Verdict.SAFE_PROGRESS)) == 0.5
        assert compute_binary_reward(_make_result(Verdict.FAIL)) == -0.5
        assert compute_binary_reward(_make_result(Verdict.ERROR)) == -0.5
