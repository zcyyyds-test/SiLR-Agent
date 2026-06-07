"""GRPO reward computation from SiLR verification results.

Converts VerificationResult into a scalar reward for reinforcement learning.
Pure function — no side effects, easy to test.

Threshold constants are passed via RewardConfig to keep the framework
domain-agnostic. Each domain provides its own thresholds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from ..verifier.types import CheckResult, Verdict, VerificationResult


def _env_float(name: str, default: float) -> float:
    """Read a float from env, falling back to ``default`` on missing/garbage."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# --- Arm D (structured) SAFE_PROGRESS weights ---------------------------------
# The per-step SAFE_PROGRESS reward is W2·support_elim + W3·severity_reduction
# − WD·branch_drift, every term derived from the PER-BRANCH map Φ = (S, σ)
# (never from the scalar sum Σσ). W2 > W3 encodes the product-order priority
# "support inclusion ψ₂ dominates per-branch severity ψ₃" (decisions §2.3).
# Bounded by construction (each term ∈ [0,1]) so the value stays well inside
# GRPO's advantage-normalisation + clamp[-3,3] range — no exploding lexicographic
# tiers (panel 2026-06-03). All three sweep-able via env for the reward-scale study.
_SP_W_SUPPORT = _env_float("SILR_SP_W_SUPPORT", 0.6)   # ψ₂ — primary
_SP_W_SEVERITY = _env_float("SILR_SP_W_SEVERITY", 0.3)  # ψ₃ — secondary
_SP_W_DRIFT = _env_float("SILR_SP_W_DRIFT", 0.3)        # per-branch worsening penalty
# Flat-constant ablation (NOT the claim arm): set >0 to override the graded form
# with a constant, to isolate "does graded geometry matter vs just non-terminal
# positive feedback". 0 = use the graded Φ-descent form.
_SP_FLAT = _env_float("SILR_SP_FLAT", 0.0)


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
    """Arm D — structured (product-order) GRPO process reward.

    This is the *claim* arm: the per-step scalar reward is an
    **order-preserving** encoding of the product-order descent
    Φ(s) → Φ(ŝ), so the geometric distinction the runtime gate makes
    (support inclusion ψ₂ dominating per-branch severity ψ₃) survives
    into the training signal rather than collapsing into the scalar
    projection it criticises (that collapse is arm E,
    :func:`compute_scalar_reward`).

    Reward design:
        PASS          : +1.0 + margin_bonus (0 ~ 0.5)   — terminal recovery
        SAFE_PROGRESS : bounded Φ-descent (see _safe_progress_reward),
                        always < PASS; ≈0 when admissible-but-non-progressing
                        (anti reward-hacking)
        FAIL          : -0.3 ~ -1.0 (scaled by worst severity)
        ERROR         : -1.0 (kept distinct so parser typos are not
                        conflated with physical unsafety)

    The terminal recovery bonus is added by the training loop at the
    trajectory level; this function scores a single step.
    """
    if result.verdict == Verdict.ERROR:
        return -1.0

    if result.verdict == Verdict.PASS:
        return _pass_reward(result.check_results, config)

    if result.verdict == Verdict.SAFE_PROGRESS:
        return _safe_progress_reward(result)

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


def _safe_progress_reward(result: VerificationResult) -> float:
    """Arm D SAFE_PROGRESS reward — bounded, order-preserving Φ-descent.

    Scores the product-order descent Φ(s) = (S, σ) → Φ(ŝ) = (Ŝ, σ̂) using the
    per-branch geometry persisted on the VerificationResult:

        support_elim   = Σ_{k ∈ S\\Ŝ} σ_k / Σ_{k ∈ S} σ_k   (ψ₂, severity-weighted)
        severity_red   = Σ_{k ∈ S∩Ŝ} max(0, σ_k - σ̂_k) / Σ σ   (ψ₃, surviving branches)
        drift          = max_{k ∈ S∩Ŝ} max(0, σ̂_k - σ_k)/σ_k    (per-branch worsening)
        r = W2·support_elim + W3·severity_red − WD·min(drift, 1)

    Key properties (vs the scalar arm E):
    * **severity-weighted, not count-based** — eliminating a high-σ branch is
      worth more than a low-σ one; a count-preserving magnitude reallocation
      (the projection trap) earns ~0 here but can score positive under arm E.
    * **W2 > W3** — support elimination dominates severity polishing
      (product-order priority, not a free weighted sum over Σσ).
    * **anti reward-hacking** — an admissible-but-non-progressing step
      (support_elim ≈ 0, severity_red ≈ 0) earns ≈ 0, not a positive constant,
      so the policy cannot farm SAFE_PROGRESS by stalling.
    * **bounded < PASS** — max ≈ W2 + W3 < 1.0 (PASS).

    Falls back to a flat constant when the geometry is unavailable (non-progress
    gating policy, missing baseline) or when SILR_SP_FLAT > 0 (ablation arm).
    """
    if _SP_FLAT > 0.0:
        return _SP_FLAT

    pre = result.baseline_branches
    post = result.post_branches or {}
    if not pre:
        # No pre-action geometry (non-progress policy / no prior violations):
        # cannot score a descent — fall back to a small positive constant so the
        # verdict is still distinguished from FAIL.
        return _SP_W_SEVERITY  # modest, < PASS

    pre_keys = set(pre)
    post_keys = set(post)
    eliminated = pre_keys - post_keys
    surviving = pre_keys & post_keys

    # Per-family normalization (k[0] = constraint_type). The product order over
    # Phi=(S, sigma) is over PHYSICALLY INCOMPARABLE families; a single
    # sum(sigma) normalizer collapses it into a dimensionally-mixed scalar, so a
    # large-magnitude family (or branch) dominates and the policy is rewarded
    # ~0 for clearing a small-sigma branch -- which silently turned arm D into a
    # "fix-the-largest-branch" scalar under high sigma-heterogeneity (the very
    # multi-type regime; CityLearn sigma-het ~286). We instead score each family
    # as a fraction of ITS OWN total severity and average the families with equal
    # weight: within-family severity weighting (the single-type geometric
    # advantage) is preserved, while no family can hijack the signal by raw
    # magnitude. For a single family (e.g. ANM) this reduces EXACTLY to the
    # original sum(sigma) normalization, so the published single-type result is
    # unchanged. (7-way panel 2026-06-05; root cause of the multi-type null.)
    by_family: dict = {}
    for k in pre_keys:
        by_family.setdefault(k[0], []).append(k)

    se_terms, sr_terms, drift_vals = [], [], []
    for keys in by_family.values():
        fam_total = sum(pre[k] for k in keys) + 1e-8
        se_terms.append(sum(pre[k] for k in keys if k in eliminated) / fam_total)
        sr_terms.append(
            sum(max(0.0, pre[k] - post[k]) for k in keys if k in surviving) / fam_total)
        for k in keys:
            if k in surviving:
                drift_vals.append(max(0.0, post[k] - pre[k]) / (pre[k] + 1e-8))

    support_elim = sum(se_terms) / len(se_terms)
    severity_red = sum(sr_terms) / len(sr_terms)
    drift = max(drift_vals, default=0.0)

    return (
        _SP_W_SUPPORT * support_elim
        + _SP_W_SEVERITY * severity_red
        - _SP_W_DRIFT * min(drift, 1.0)
    )


def compute_scalar_reward(
    result: VerificationResult,
    config: Optional[RewardConfig] = None,
) -> float:
    """Arm E — scalar-projection GRPO process reward (the criticised baseline).

    Identical verdict scaffold to arm D, but the SAFE_PROGRESS step is scored by
    the **count-based projection** — the violation-count reduction fraction,
    blind to *which* branch or *how severe*. This is the training-signal analogue
    of the scalar gate: it collapses Φ = (S, σ) to |S|, so a high-severity and a
    low-severity branch elimination are indistinguishable, and a count-preserving
    magnitude drift scores 0 (no penalty). Arm D must beat this to show the
    geometry matters as a learning signal, not just at inference.
    """
    if result.verdict == Verdict.ERROR:
        return -1.0
    if result.verdict == Verdict.PASS:
        return _pass_reward(result.check_results, config)
    if result.verdict == Verdict.SAFE_PROGRESS:
        pre = result.baseline_branches
        post = result.post_branches or {}
        if not pre:
            return _SP_W_SEVERITY
        # Pure count-delta fraction — the severity-blind scalar projection.
        return (len(pre) - len(post)) / (len(pre) + 1e-8)
    # FAIL
    return _fail_penalty(result.check_results)


def compute_severity_scalar_reward(
    result: VerificationResult,
    config: Optional[RewardConfig] = None,
) -> float:
    """Arm E2 — severity-scalar GRPO process reward (the strongest scalar baseline).

    Identical verdict scaffold to arms D/E, but the SAFE_PROGRESS step is scored by
    the **total-severity reduction fraction** — (Σσ_pre − Σσ_post)/Σσ_pre — a single
    scalar that keeps severity information (unlike count E) but COLLAPSES the product
    order over physically-incomparable families into one dimension. Under sigma-
    heterogeneity it is dominated by the large-σ family, so it rewards clearing the
    big family and under-weights a small-σ bottleneck family — the projection that
    arm D's per-family normalization is designed to beat. (In a single-family domain
    this reduces to arm D's within-family weighting, so D > E2 only shows in the
    multi-type / sigma-het regime.)
    """
    if result.verdict == Verdict.ERROR:
        return -1.0
    if result.verdict == Verdict.PASS:
        return _pass_reward(result.check_results, config)
    if result.verdict == Verdict.SAFE_PROGRESS:
        pre = result.baseline_branches
        post = result.post_branches or {}
        if not pre:
            return _SP_W_SEVERITY
        sum_pre = sum(pre.values())
        sum_post = sum(post.get(k, 0.0) for k in pre)  # surviving severity of pre-branches
        # add any drift on surviving branches into the residual so a magnitude
        # reallocation is not free (matches the severity-scalar used in landscaping)
        sum_post = sum(post.values()) if post else 0.0
        return max(-1.0, (sum_pre - sum_post) / (sum_pre + 1e-8))
    # FAIL
    return _fail_penalty(result.check_results)


def compute_binary_reward(
    result: VerificationResult,
    config: Optional[RewardConfig] = None,
) -> float:
    """Arm C — binary baseline reward (no graded verifier signal).

    Admitted (PASS or SAFE_PROGRESS) → +0.5; rejected (FAIL/ERROR) → -0.5.
    The terminal recovery bonus is added by the training loop. This mirrors the
    hard-coded reward the existing cluster/finance trainers use, exposed here as
    a function so all three arms share one tested code path.
    """
    if result.verdict in (Verdict.PASS, Verdict.SAFE_PROGRESS):
        return 0.5
    return -0.5


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
