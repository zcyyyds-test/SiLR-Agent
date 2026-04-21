"""5 constraint checkers for cluster_v2023 domain.

Per-action gate (SiLRVerifier.checkers): ResourceCapacity + Affinity only.
Observer-only (episode-level signals): Priority / Queue / Fragmentation.

Rationale (per spec §5.1): Queue/Priority evaluated per-action would
reject every intermediate step (LS still queued while scheduling is in
progress → 100% fail). This mirrors cluster v1 and finance designs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation

logger = logging.getLogger(__name__)


class ResourceCapacityChecker(BaseConstraintChecker):
    """No Ready node may exceed cpu_milli / ram_mib / gpu_total."""

    name = "resource_capacity"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        nodes = system_state["nodes"]
        violations: list[Violation] = []
        max_gpu_util = 0.0

        for nid, node in nodes.items():
            if node["status"] != "Ready":
                continue
            g_used, g_total = node["gpu_used"], node["gpu_total"]
            if g_total > 0:
                max_gpu_util = max(max_gpu_util, g_used / g_total)
            if g_used > g_total:
                violations.append(Violation(
                    constraint_type="resource_capacity", device_type="node",
                    device_id=nid, metric="gpu_used",
                    value=float(g_used), limit=float(g_total), unit="GPUs",
                    severity="critical",
                    detail=f"{nid}: {g_used}/{g_total} GPUs",
                ))
            if node["cpu_used"] > node["cpu_total"]:
                violations.append(Violation(
                    constraint_type="resource_capacity", device_type="node",
                    device_id=nid, metric="cpu_milli",
                    value=float(node["cpu_used"]), limit=float(node["cpu_total"]),
                    unit="cpu_milli", severity="violation",
                    detail=f"{nid}: cpu {node['cpu_used']}/{node['cpu_total']}",
                ))
            if node["ram_used_mib"] > node["ram_total_mib"]:
                violations.append(Violation(
                    constraint_type="resource_capacity", device_type="node",
                    device_id=nid, metric="ram_mib",
                    value=float(node["ram_used_mib"]),
                    limit=float(node["ram_total_mib"]),
                    unit="MiB", severity="violation",
                    detail=f"{nid}: ram {node['ram_used_mib']}/{node['ram_total_mib']}",
                ))

        return CheckResult(
            checker_name=self.name,
            passed=not violations,
            summary={
                "max_gpu_util": round(max_gpu_util, 3),
                "n_violations": len(violations),
            },
            violations=violations,
        )


class AffinityChecker(BaseConstraintChecker):
    """Running jobs with gpu_spec_required must be placed on matching node.model."""

    name = "affinity"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        jobs = system_state["jobs"]
        nodes = system_state["nodes"]
        assignments = system_state["assignments"]
        violations: list[Violation] = []

        for jid, job in jobs.items():
            if job["status"] != "Running":
                continue
            req = job.get("gpu_spec_required")
            if not req:
                continue
            nid = assignments.get(jid)
            if nid is None:
                continue
            node_model = nodes[nid]["model"]
            if node_model != req:
                violations.append(Violation(
                    constraint_type="affinity", device_type="job",
                    device_id=jid, metric="gpu_spec_match",
                    value=0.0, limit=1.0, unit="bool", severity="warning",
                    detail=f"{jid} needs {req} but on {nid} ({node_model})",
                ))

        return CheckResult(
            checker_name=self.name, passed=not violations,
            summary={"n_violations": len(violations)},
            violations=violations,
        )


class PriorityChecker(BaseConstraintChecker):
    """No LS job queued while BE job running (higher-priority preemption).

    Per spec §5.1: OBSERVER-ONLY (not in SiLRVerifier.checkers). Used
    by observer for reward shaping and recovery-target signalling.
    """

    name = "priority"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        jobs = system_state["jobs"]
        ls_queued = [j for j, v in jobs.items()
                     if v["qos"] == "LS" and v["status"] == "Queued"]
        be_running = [j for j, v in jobs.items()
                      if v["qos"] == "BE" and v["status"] == "Running"]
        violations: list[Violation] = []
        if ls_queued and be_running:
            for jid in ls_queued:
                violations.append(Violation(
                    constraint_type="priority", device_type="job",
                    device_id=jid, metric="ls_queued_while_be_running",
                    value=1.0, limit=0.0, unit="bool", severity="critical",
                    detail=(f"LS {jid} queued while {len(be_running)} "
                            f"BE running"),
                ))
        return CheckResult(
            checker_name=self.name, passed=not violations,
            summary={"ls_queued": len(ls_queued),
                     "be_running": len(be_running),
                     "n_violations": len(violations)},
            violations=violations,
        )


class QueueChecker(BaseConstraintChecker):
    """LS and Burstable jobs must not be in Queued status. BE may be.

    Per spec §5.1: OBSERVER-ONLY. Per-action gating would make every
    intermediate step violate (queue drains gradually) — see cluster v1
    坑点记录 "QueueChecker 在 per-action 级别必定失败 (1.2% recovery)".
    """

    name = "queue"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        jobs = system_state["jobs"]
        critical_queued = [j for j, v in jobs.items()
                           if v["status"] == "Queued"
                           and v["qos"] in ("LS", "Burstable")]
        violations: list[Violation] = []
        for jid in critical_queued:
            job = jobs[jid]
            sev = "critical" if job["qos"] == "LS" else "violation"
            violations.append(Violation(
                constraint_type="queue", device_type="job",
                device_id=jid, metric="queued",
                value=1.0, limit=0.0, unit="bool", severity=sev,
                detail=f"{jid} ({job['qos']}) queued, needs {job['gpu']} gpu",
            ))
        return CheckResult(
            checker_name=self.name, passed=not violations,
            summary={"critical_queued": len(critical_queued),
                     "n_violations": len(violations)},
            violations=violations,
        )


# --- FragmentationChecker (FGD ATC'23) ---

_HARDCODED_FALLBACK_DIST = {1: 0.55, 2: 0.25, 4: 0.15, 8: 0.05}


def _load_default_dist() -> dict[int, float]:
    """Load p(g) from trace-derived JSON; fall back to hardcoded if missing.

    The trace-derived distribution lives at
    `domains/cluster_v2023/data_pipeline/job_size_dist.json` and is
    produced by `scripts` + `compute_job_size_dist.py`. Using the
    hardcoded fallback makes F values NON-COMPARABLE to the FGD ATC'23
    paper — only acceptable during smoke tests, not in published
    benchmark numbers (see spec §5.1a).
    """
    p = Path(__file__).parent / "data_pipeline" / "job_size_dist.json"
    if p.is_file():
        try:
            raw = json.loads(p.read_text())
            loaded = {int(k): float(v) for k, v in raw.items()}
            if loaded:
                return loaded
            logger.warning("job_size_dist.json is empty; using hardcoded fallback")
        except Exception as e:
            logger.warning("failed to load job_size_dist.json (%s); "
                           "using hardcoded fallback", e)
    else:
        logger.info("job_size_dist.json not found at %s; using hardcoded "
                    "fallback (NOT FGD-comparable)", p)
    return dict(_HARDCODED_FALLBACK_DIST)


# Exposed as module-level for back-compat with tests/imports
DEFAULT_JOB_SIZE_DIST = _load_default_dist()


class FragmentationChecker(BaseConstraintChecker):
    """FGD ATC'23: F = Σ_node Σ_g p(g) · 𝟙[0 < rem(n) < g] · rem(n).

    Per spec §5.1: OBSERVER-ONLY. Fragmentation is an episode-level
    signal; it enters the reward function as a shaping term but does
    not gate individual actions.
    """

    name = "fragmentation"

    def __init__(self, *, f_threshold: float,
                 job_size_dist: dict[int, float] | None = None):
        self.f_threshold = float(f_threshold)
        # Kimi #5 guard: empty dict would collapse F→0 silently. Fall back.
        if job_size_dist is None or not job_size_dist:
            self.job_size_dist = dict(DEFAULT_JOB_SIZE_DIST)
        else:
            self.job_size_dist = dict(job_size_dist)
        # Final invariant: never operate on empty dist.
        assert self.job_size_dist, "FragmentationChecker requires non-empty p(g)"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        f_total = 0.0
        for node in system_state["nodes"].values():
            if node["status"] != "Ready":
                continue
            rem = node["gpu_total"] - node["gpu_used"]
            if rem <= 0:
                continue
            for g, p in self.job_size_dist.items():
                if 0 < rem < g:
                    f_total += p * rem
        passed = f_total <= self.f_threshold
        violations: list[Violation] = []
        if not passed:
            violations.append(Violation(
                constraint_type="fragmentation", device_type="cluster",
                device_id="global", metric="F",
                value=float(f_total), limit=float(self.f_threshold),
                unit="weighted_gpus", severity="warning",
                detail=f"F={f_total:.3f} > threshold={self.f_threshold:.3f}",
            ))
        return CheckResult(
            checker_name=self.name, passed=passed,
            summary={"F": round(f_total, 3),
                     "f_threshold": self.f_threshold,
                     "n_violations": len(violations)},
            violations=violations,
        )
