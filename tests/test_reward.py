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
