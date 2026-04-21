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
