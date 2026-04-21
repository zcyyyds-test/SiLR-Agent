from domains.cluster_v2023.checkers import ResourceCapacityChecker


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

from domains.cluster_v2023.checkers import AffinityChecker  # noqa: E402


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

from domains.cluster_v2023.checkers import PriorityChecker  # noqa: E402


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

from domains.cluster_v2023.checkers import QueueChecker  # noqa: E402


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
