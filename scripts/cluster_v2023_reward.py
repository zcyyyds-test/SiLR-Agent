"""Dense reward for cluster_v2023 GRPO (mirrors finance domain).

Formula (spec §5.2):
    r_step = +0.10  if total_violation_count_decreased
           + 0.30  if fragmentation_decreased_by_ε   (ε = 0.05 × F_baseline)
           + 1.00  if all per-action checkers (Capacity + Affinity) pass
           - 0.50  if action rejected (verdict=FAIL) by verifier

The four components are additive. Plumbed into train_grpo_cluster_v2023.py
at the reward computation site.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.cluster_v2023.checkers import (
    AffinityChecker,
    DEFAULT_JOB_SIZE_DIST,
    FragmentationChecker,
    PriorityChecker,
    QueueChecker,
    ResourceCapacityChecker,
)

# Per-action gate (reward bonus when all of these pass)
_GATE_CHECKERS = [ResourceCapacityChecker(), AffinityChecker()]
# All 4 non-frag checkers contribute to the `violation_count_decreased` signal
_ALL_CHECKERS = _GATE_CHECKERS + [PriorityChecker(), QueueChecker()]


def _total_violations(state: Any) -> int:
    return sum(len(c.check(state, 1.0).violations) for c in _ALL_CHECKERS)


def _gate_passes(state: Any) -> bool:
    return all(c.check(state, 1.0).passed for c in _GATE_CHECKERS)


def _frag(state: Any) -> float:
    return FragmentationChecker(
        f_threshold=1e9,
        job_size_dist=DEFAULT_JOB_SIZE_DIST,
    ).check(state, 1.0).summary["F"]


def dense_reward(*, pre_state: Any, post_state: Any,
                 verdict: str, f_baseline: float,
                 eps_frag: float | None = None) -> float:
    """Step reward; additive across four components."""
    if eps_frag is None:
        eps_frag = 0.05 * max(f_baseline, 1e-6)

    if verdict == "FAIL":
        return -0.5

    pre_v = _total_violations(pre_state)
    post_v = _total_violations(post_state)
    pre_f = _frag(pre_state)
    post_f = _frag(post_state)

    r = 0.0
    if post_v < pre_v:
        r += 0.10
    if (pre_f - post_f) >= eps_frag:
        r += 0.30
    if _gate_passes(post_state):
        r += 1.00
    return r
