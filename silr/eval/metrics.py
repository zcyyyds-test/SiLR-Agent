"""Evaluation metrics computation — domain-agnostic."""

from __future__ import annotations

from typing import Any

from ..agent.types import EpisodeResult


def compute_metrics(
    episodes: list[EpisodeResult],
    difficulty_fn: Any = None,
) -> dict[str, Any]:
    """Compute aggregate metrics over a list of episode results.

    Args:
        episodes: List of completed episode results.
        difficulty_fn: Optional callable(scenario_id) -> str that maps
            scenario IDs to difficulty labels for per-difficulty breakdown.
            If None, no per-difficulty breakdown is computed.

    Returns dict with:
    - recovery_rate: fraction of episodes where system was recovered
    - avg_steps: average steps per episode
    - avg_proposals: average proposals per episode
    - rejection_rate: total rejections / total proposals
    - failsafe_rate: fraction of episodes where failsafe triggered
    - per_difficulty: metrics broken down by difficulty level (if difficulty_fn provided)
    """
    if not episodes:
        return {"error": "No episodes to evaluate"}

    n = len(episodes)
    recovered = sum(1 for ep in episodes if ep.recovered)
    total_steps = sum(ep.total_steps for ep in episodes)
    total_proposals = sum(ep.total_proposals for ep in episodes)
    total_rejections = sum(ep.total_rejections for ep in episodes)
    failsafe_count = sum(1 for ep in episodes if ep.failsafe_triggered)

    metrics: dict[str, Any] = {
        "n_episodes": n,
        "recovery_rate": recovered / n,
        "avg_steps": total_steps / n,
        "avg_proposals": total_proposals / n,
        "rejection_rate": total_rejections / total_proposals if total_proposals > 0 else 0.0,
        "failsafe_rate": failsafe_count / n,
    }

    # Per-difficulty breakdown
    if difficulty_fn is not None:
        difficulty_map: dict[str, list[EpisodeResult]] = {}
        for ep in episodes:
            diff = difficulty_fn(ep.scenario_id)
            difficulty_map.setdefault(diff, []).append(ep)

        per_diff = {}
        for diff, eps in difficulty_map.items():
            nd = len(eps)
            per_diff[diff] = {
                "n_episodes": nd,
                "recovery_rate": sum(1 for e in eps if e.recovered) / nd,
                "avg_steps": sum(e.total_steps for e in eps) / nd,
            }
        metrics["per_difficulty"] = per_diff

    return metrics


def compute_unsafe_action_rate(
    post_hoc_results: list[list[Any]],
) -> float:
    """Compute unsafe action rate for NoVerify ablation.

    Args:
        post_hoc_results: Per-episode list of VerificationResults
            from running each committed action through the real verifier.

    Returns:
        Fraction of committed actions that would have been rejected.
    """
    total_actions = 0
    unsafe_actions = 0

    for ep_results in post_hoc_results:
        for vr in ep_results:
            total_actions += 1
            if hasattr(vr, 'verdict') and vr.verdict.value != "PASS":
                unsafe_actions += 1

    if total_actions == 0:
        return 0.0
    return unsafe_actions / total_actions


def compute_multi_agent_metrics(
    episodes: list[Any],
    difficulty_fn: Any = None,
) -> dict[str, Any]:
    """Compute aggregate metrics for multi-agent coordinator episodes.

    Args:
        episodes: List of MultiAgentEpisodeResult.
        difficulty_fn: Optional callable(scenario_id) -> difficulty label.

    Returns dict with single-agent compatible metrics plus:
    - avg_rounds: average coordinator rounds per episode
    - avg_specialist_activations: average specialist dispatches
    - conflict_rate: fraction of activations that worsened a constraint
    - per_specialist: per-specialist activation and success breakdown
    """
    if not episodes:
        return {"error": "No episodes to evaluate"}

    # Flatten to single-agent view for base metrics
    flat = [ep.to_single_agent_view() for ep in episodes]
    base = compute_metrics(flat, difficulty_fn)

    n = len(episodes)
    total_rounds = sum(ep.total_rounds for ep in episodes)
    total_activations = sum(len(ep.activations) for ep in episodes)
    total_conflicts = sum(ep.conflict_count for ep in episodes)

    base["avg_rounds"] = total_rounds / n
    base["avg_specialist_activations"] = total_activations / n
    base["conflict_rate"] = total_conflicts / total_activations if total_activations > 0 else 0.0

    # Per-specialist breakdown
    specialist_stats: dict[str, dict[str, int]] = {}
    for ep in episodes:
        for a in ep.activations:
            stats = specialist_stats.setdefault(a.specialist_name, {
                "activations": 0, "steps": 0, "improved": 0, "worsened": 0,
            })
            stats["activations"] += 1
            stats["steps"] += a.episode_result.total_steps
            if a.constraints_improved:
                stats["improved"] += 1
            if a.constraints_worsened:
                stats["worsened"] += 1

    base["per_specialist"] = specialist_stats

    return base
