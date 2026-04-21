from domains.cluster_v2023.checkers import (
    AffinityChecker,
    FragmentationChecker,
    PriorityChecker,
    QueueChecker,
    ResourceCapacityChecker,
)


def _state():
    return {
        "nodes": {
            "n0": {"model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
                   "gpu_total": 8, "status": "Ready",
                   "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
        },
        "jobs": {},
        "assignments": {},
        "sim_time": 0.0,
    }


def test_capacity_pass_when_under_limits():
    s = _state()
    s["nodes"]["n0"]["gpu_used"] = 4
    r = ResourceCapacityChecker().check(s, base_mva=1.0)
    assert r.passed


def test_capacity_fail_when_gpu_overcommitted():
    s = _state()
    s["nodes"]["n0"]["gpu_used"] = 9
    r = ResourceCapacityChecker().check(s, base_mva=1.0)
    assert not r.passed
    assert any(v.metric == "gpu_used" for v in r.violations)


def test_capacity_fail_on_cpu_overcommit():
    s = _state()
    s["nodes"]["n0"]["cpu_used"] = 100000
    r = ResourceCapacityChecker().check(s, base_mva=1.0)
    assert not r.passed
    assert any(v.metric == "cpu_milli" for v in r.violations)


def test_capacity_fail_on_ram_overcommit():
    s = _state()
    s["nodes"]["n0"]["ram_used_mib"] = 2_000_000
    r = ResourceCapacityChecker().check(s, base_mva=1.0)
    assert not r.passed
    assert any(v.metric == "ram_mib" for v in r.violations)


def test_capacity_skips_down_nodes():
    s = _state()
    s["nodes"]["n0"]["status"] = "Down"
    s["nodes"]["n0"]["gpu_used"] = 999  # normally a violation
    r = ResourceCapacityChecker().check(s, base_mva=1.0)
    assert r.passed  # Down nodes don't count


# --- AffinityChecker ---


def test_affinity_pass_when_no_requirement():
    s = _state()
    s["jobs"] = {"j0": {"cpu": 0, "ram_mib": 0, "gpu": 1,
                        "gpu_spec_required": None, "qos": "LS",
                        "status": "Running"}}
    s["assignments"] = {"j0": "n0"}
    r = AffinityChecker().check(s, base_mva=1.0)
    assert r.passed


def test_affinity_fail_on_mismatch():
    s = _state()
    s["jobs"] = {"j0": {"cpu": 0, "ram_mib": 0, "gpu": 1,
                        "gpu_spec_required": "T4", "qos": "LS",
                        "status": "Running"}}
    s["assignments"] = {"j0": "n0"}  # n0 is V100M32
    r = AffinityChecker().check(s, base_mva=1.0)
    assert not r.passed


def test_affinity_ignores_queued_jobs():
    """A Queued job with affinity requirement is not yet violating —
    the verifier only checks placement correctness for Running jobs."""
    s = _state()
    s["jobs"] = {"j0": {"cpu": 0, "ram_mib": 0, "gpu": 1,
                        "gpu_spec_required": "T4", "qos": "LS",
                        "status": "Queued"}}
    r = AffinityChecker().check(s, base_mva=1.0)
    assert r.passed


# --- PriorityChecker ---


def test_priority_pass_when_no_ls_queued():
    s = _state()
    s["jobs"] = {"j0": {"qos": "BE", "status": "Running",
                        "cpu": 0, "ram_mib": 0, "gpu": 1,
                        "gpu_spec_required": None}}
    r = PriorityChecker().check(s, base_mva=1.0)
    assert r.passed


def test_priority_fail_when_ls_queued_with_be_running():
    s = _state()
    s["jobs"] = {
        "j0": {"qos": "LS", "status": "Queued",
               "cpu": 0, "ram_mib": 0, "gpu": 1, "gpu_spec_required": None},
        "j1": {"qos": "BE", "status": "Running",
               "cpu": 0, "ram_mib": 0, "gpu": 1, "gpu_spec_required": None},
    }
    r = PriorityChecker().check(s, base_mva=1.0)
    assert not r.passed


def test_priority_pass_when_ls_queued_but_no_be_running():
    """LS queued + only Burstable running → acceptable (can't preempt equals)."""
    s = _state()
    s["jobs"] = {
        "j0": {"qos": "LS", "status": "Queued",
               "cpu": 0, "ram_mib": 0, "gpu": 1, "gpu_spec_required": None},
        "j1": {"qos": "Burstable", "status": "Running",
               "cpu": 0, "ram_mib": 0, "gpu": 1, "gpu_spec_required": None},
    }
    r = PriorityChecker().check(s, base_mva=1.0)
    assert r.passed


# --- QueueChecker ---


def test_queue_pass_when_only_be_queued():
    s = _state()
    s["jobs"] = {"j0": {"qos": "BE", "status": "Queued",
                        "cpu": 0, "ram_mib": 0, "gpu": 1,
                        "gpu_spec_required": None}}
    r = QueueChecker().check(s, base_mva=1.0)
    assert r.passed


def test_queue_fail_when_ls_queued():
    s = _state()
    s["jobs"] = {"j0": {"qos": "LS", "status": "Queued",
                        "cpu": 0, "ram_mib": 0, "gpu": 1,
                        "gpu_spec_required": None}}
    r = QueueChecker().check(s, base_mva=1.0)
    assert not r.passed


def test_queue_fail_when_burstable_queued():
    s = _state()
    s["jobs"] = {"j0": {"qos": "Burstable", "status": "Queued",
                        "cpu": 0, "ram_mib": 0, "gpu": 1,
                        "gpu_spec_required": None}}
    r = QueueChecker().check(s, base_mva=1.0)
    assert not r.passed


# --- FragmentationChecker (FGD ATC'23 formula) ---


def test_fragmentation_score_zero_when_node_full_or_empty():
    s = _state()
    # gpu_used=0 → remaining=8, not fragmentary for any g ∈ {1,2,4,8}
    r = FragmentationChecker(f_threshold=1000.0).check(s, base_mva=1.0)
    assert r.summary["F"] == 0.0


def test_fragmentation_score_positive_when_partially_filled():
    s = _state()
    s["nodes"]["n0"]["gpu_used"] = 6   # remaining=2 → fragmentary for g ∈ {4, 8}
    r = FragmentationChecker(
        f_threshold=100.0,
        job_size_dist={1: 0.0, 2: 0.0, 4: 0.5, 8: 0.5},
    ).check(s, base_mva=1.0)
    # F = (0.5 * 2)  (g=4 matches 0 < 2 < 4)  +  (0.5 * 2)  (g=8 matches 0 < 2 < 8)
    assert r.summary["F"] == 2.0


def test_fragmentation_fail_above_threshold():
    s = _state()
    s["nodes"]["n0"]["gpu_used"] = 6
    r = FragmentationChecker(
        f_threshold=1.0,
        job_size_dist={4: 1.0},  # single mass at 4
    ).check(s, base_mva=1.0)
    # F = 1.0 * 2 = 2.0 > 1.0 → fail
    assert not r.passed


def test_fragmentation_falls_back_when_empty_dist_passed():
    """Regression (Codex Q4 / Kimi #5): empty job_size_dist must not
    collapse F to always-zero — falls back to DEFAULT_JOB_SIZE_DIST."""
    s = _state()
    s["nodes"]["n0"]["gpu_used"] = 6
    chk = FragmentationChecker(f_threshold=1e9, job_size_dist={})
    assert chk.job_size_dist, "fallback to DEFAULT must populate dict"
    r = chk.check(s, base_mva=1.0)
    assert r.summary["F"] > 0


def test_fragmentation_skips_down_nodes():
    s = _state()
    s["nodes"]["n0"]["status"] = "Down"
    s["nodes"]["n0"]["gpu_used"] = 6  # would be fragmentary if Ready
    r = FragmentationChecker(f_threshold=1000.0,
                             job_size_dist={4: 1.0}).check(s, base_mva=1.0)
    assert r.summary["F"] == 0.0


# --- Kimi review follow-ups (non-blocking coverage additions) ---


def test_priority_pass_on_unknown_qos():
    """Kimi Suggest: unknown qos value (not LS/Burstable/BE) shouldn't trigger
    Priority. Current logic ignores non-LS/non-BE → passes. Documents the
    implicit contract — if qos schema expands, this test will force a
    decision rather than silently accept the default."""
    s = _state()
    s["jobs"] = {
        "j0": {"qos": "Premium", "status": "Queued",
               "cpu": 0, "ram_mib": 0, "gpu": 1, "gpu_spec_required": None},
        "j1": {"qos": "BE", "status": "Running",
               "cpu": 0, "ram_mib": 0, "gpu": 1, "gpu_spec_required": None},
    }
    assert PriorityChecker().check(s, 1.0).passed


def test_queue_treats_unknown_qos_as_non_critical():
    """Kimi Suggest: qos not in {LS, Burstable} is allowed to queue."""
    s = _state()
    s["jobs"] = {"j0": {"qos": "Premium", "status": "Queued",
                        "cpu": 0, "ram_mib": 0, "gpu": 1,
                        "gpu_spec_required": None}}
    assert QueueChecker().check(s, 1.0).passed


def test_fragmentation_multi_node_additive():
    """Kimi Suggest: F must sum across multiple Ready nodes."""
    s = _state()
    # n0 rem=8 (dist={4:1.0} → not fragmentary)
    # n1 rem=2 (dist={4:1.0} → 0 < 2 < 4 → F += 1.0 * 2)
    s["nodes"]["n1"] = {
        "model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
        "gpu_total": 8, "status": "Ready",
        "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 6,
    }
    r = FragmentationChecker(
        f_threshold=1e9, job_size_dist={4: 1.0}).check(s, 1.0)
    assert r.summary["F"] == 2.0


def test_all_checkers_pass_on_empty_state():
    """Kimi Suggest: empty cluster (0 nodes, 0 jobs) is trivially clean."""
    empty = {"nodes": {}, "jobs": {}, "assignments": {}, "sim_time": 0.0}
    assert ResourceCapacityChecker().check(empty, 1.0).passed
    assert AffinityChecker().check(empty, 1.0).passed
    assert PriorityChecker().check(empty, 1.0).passed
    assert QueueChecker().check(empty, 1.0).passed
    assert FragmentationChecker(f_threshold=0.0).check(empty, 1.0).passed


def test_fragmentation_zero_gpu_total_node():
    """Kimi Suggest: Ready node with gpu_total=0 contributes nothing."""
    s = _state()
    s["nodes"]["n0"]["gpu_total"] = 0
    s["nodes"]["n0"]["gpu_used"] = 0
    r = FragmentationChecker(
        f_threshold=1e9, job_size_dist={4: 1.0}).check(s, 1.0)
    assert r.summary["F"] == 0.0


def test_fragmentation_rejects_corrupted_overcommit_state():
    """Kimi Suggest: locking behavior when gpu_used > gpu_total (transient
    mid-transition). rem becomes negative, skipped by `rem <= 0`."""
    s = _state()
    s["nodes"]["n0"]["gpu_total"] = 8
    s["nodes"]["n0"]["gpu_used"] = 10  # corrupt state
    r = FragmentationChecker(
        f_threshold=1e9, job_size_dist={4: 1.0}).check(s, 1.0)
    assert r.summary["F"] == 0.0
