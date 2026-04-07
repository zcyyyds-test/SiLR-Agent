"""GRPO reward computation from SiLR verification results.

Converts VerificationResult into a scalar reward for reinforcement learning.
Pure function — no side effects, easy to test.

Threshold constants are passed via RewardConfig to keep the framework
domain-agnostic. Each domain provides its own thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..verifier.types import CheckResult, Verdict, VerificationResult


@dataclass
class RewardConfig:
    """Domain-specific thresholds for margin computation.

    Keys are checker names (e.g. "voltage", "frequency").
    Values are dicts with domain-specific limit fields.

    Example for power grid:
        thresholds = {
            "voltage": {"min_pu": 0.90, "max_pu": 1.10},
            "frequency": {"max_hz": 0.5},
            "line_loading": {"max_pct": 100.0},
            "transient": {"max_deg": 180.0},
        }
    """
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)


def compute_grpo_reward(
    result: VerificationResult,
    config: Optional[RewardConfig] = None,
) -> float:
    """Compute scalar reward from a SiLR verification result.

    Reward design:
        PASS : +1.0 + margin_bonus (0 ~ 0.5)
        FAIL : -0.3 ~ -1.0 (scaled by worst severity)
        ERROR: -1.0

    The margin bonus rewards actions that leave safety headroom.
    The FAIL penalty scales with severity so the model learns to
    distinguish near-miss from catastrophic violations.
    """
    if result.verdict == Verdict.ERROR:
        return -1.0

    if result.verdict == Verdict.PASS:
        return _pass_reward(result.check_results, config)

    # FAIL
    return _fail_penalty(result.check_results)


def _pass_reward(
    checks: list[CheckResult],
    config: Optional[RewardConfig] = None,
) -> float:
    """PASS reward: 1.0 + average margin bonus across checkers."""
    if not checks or config is None or not config.thresholds:
        return 1.0

    margins = []
    for cr in checks:
        m = _margin_for_check(cr, config.thresholds)
        if m is not None:
            margins.append(m)

    bonus = sum(margins) / len(margins) * 0.5 if margins else 0.0
    return 1.0 + bonus


def _fail_penalty(checks: list[CheckResult]) -> float:
    """FAIL penalty: -0.3 (minor) to -1.0 (critical).

    Severity mapping:
        warning  → -0.3
        violation → -0.6
        critical → -1.0
    """
    severity_scores = {"warning": 0.3, "violation": 0.6, "critical": 1.0}
    worst = 0.3  # minimum penalty

    for cr in checks:
        for v in cr.violations:
            score = severity_scores.get(v.severity, 0.6)
            worst = max(worst, score)

    return -worst


def _get_first(d: dict, *keys) -> float | None:
    """Get the first non-None value from dict for the given keys."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _margin_for_check(
    cr: CheckResult,
    thresholds: dict[str, dict[str, float]],
) -> float | None:
    """Compute normalized margin (0~1) for a single checker.

    Each margin measures how far the worst-case value is from its limit,
    normalized so 0 = at the limit and 1 = maximum headroom.
    """
    summary = cr.summary
    name = cr.checker_name
    limits = thresholds.get(name, {})

    if name == "voltage":
        v_min_limit = limits.get("min_pu")
        v_max_limit = limits.get("max_pu")
        if v_min_limit is None or v_max_limit is None:
            return None
        v_min = _get_first(summary, "v_min_pu", "min_pu")
        v_max = _get_first(summary, "v_max_pu", "max_pu")
        if v_min is None or v_max is None:
            return None
        low_margin = (v_min - v_min_limit) / (1.0 - v_min_limit + 1e-10)
        high_margin = (v_max_limit - v_max) / (v_max_limit - 1.0 + 1e-10)
        return max(0.0, min(1.0, min(low_margin, high_margin)))

    if name == "frequency":
        max_hz = limits.get("max_hz")
        if max_hz is None:
            return None
        max_dev = _get_first(summary, "max_abs_delta_f_hz", "max_deviation_hz")
        if max_dev is None:
            return None
        margin = 1.0 - (max_dev / max_hz)
        return max(0.0, min(1.0, margin))

    if name == "line_loading":
        max_pct = limits.get("max_pct")
        if max_pct is None:
            return None
        max_load = _get_first(summary, "max_loading_pct", "max_pct")
        if max_load is None:
            return None
        margin = 1.0 - (max_load / max_pct)
        return max(0.0, min(1.0, margin))

    if name == "transient":
        max_deg = limits.get("max_deg")
        if max_deg is None:
            return None
        max_sep = _get_first(summary, "max_separation_deg", "max_angle_deg")
        if max_sep is None:
            return None
        margin = 1.0 - (max_sep / max_deg)
        return max(0.0, min(1.0, margin))

    # --- Cluster domain checkers ---

    if name == "resource_capacity":
        max_util = _get_first(summary, "max_gpu_util")
        if max_util is not None:
            return max(0.0, min(1.0, 1.0 - max_util))

    if name == "queue":
        ratio = _get_first(summary, "queue_ratio")
        if ratio is not None:
            return max(0.0, min(1.0, 1.0 - ratio))

    # Unknown checker — generic margin from n_violations
    n_viol = summary.get("n_violations")
    if n_viol is not None:
        return 1.0 if n_viol == 0 else 0.0

    return None
