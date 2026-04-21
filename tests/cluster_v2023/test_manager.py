from domains.cluster_v2023.manager import ClusterV2023Manager


def _mk_nodes():
    return {
        "n0": {"model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
        "n1": {"model": "G1", "cpu_total": 64000, "ram_total_mib": 524288,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
    }


def _mk_jobs():
    return {
        "j0": {"cpu": 8000, "ram_mib": 16384, "gpu": 1,
               "gpu_spec_required": None, "qos": "LS", "status": "Queued"},
    }


def test_manager_properties_present():
    m = ClusterV2023Manager(nodes=_mk_nodes(), jobs=_mk_jobs())
    assert m.sim_time == 0.0
    assert m.base_mva == 1.0
    assert "nodes" in m.system_state
    assert "jobs" in m.system_state
    assert "assignments" in m.system_state


def test_initial_assignments_empty():
    m = ClusterV2023Manager(nodes=_mk_nodes(), jobs=_mk_jobs())
    assert m.system_state["assignments"] == {}


def test_solve_recomputes_usage():
    nodes = _mk_nodes()
    jobs = _mk_jobs()
    jobs["j0"]["status"] = "Running"
    m = ClusterV2023Manager(nodes=nodes, jobs=jobs, assignments={"j0": "n0"})
    assert m.solve() is True
    assert m.system_state["nodes"]["n0"]["gpu_used"] == 1
    assert m.system_state["nodes"]["n0"]["cpu_used"] == 8000
    assert m.system_state["nodes"]["n1"]["gpu_used"] == 0


def test_solve_ignores_down_nodes_but_keeps_their_assignments():
    nodes = _mk_nodes()
    nodes["n0"]["status"] = "Down"
    jobs = _mk_jobs()
    jobs["j0"]["status"] = "Preempted"
    m = ClusterV2023Manager(nodes=nodes, jobs=jobs, assignments={})
    assert m.solve() is True
    assert m.system_state["nodes"]["n0"]["gpu_used"] == 0


def test_shadow_is_isolated():
    m = ClusterV2023Manager(nodes=_mk_nodes(), jobs=_mk_jobs())
    shadow = m.create_shadow_copy()
    shadow._jobs["j0"]["status"] = "Running"
    shadow._assignments["j0"] = "n0"
    shadow.solve()
    assert shadow.system_state["nodes"]["n0"]["gpu_used"] == 1
    assert m.system_state["nodes"]["n0"]["gpu_used"] == 0


def test_shadow_isolates_node_mutation():
    """Regression (Kimi suggest): original was only covered for _jobs mutation."""
    m = ClusterV2023Manager(nodes=_mk_nodes(), jobs=_mk_jobs())
    shadow = m.create_shadow_copy()
    shadow._nodes["n0"]["status"] = "Down"
    shadow._nodes["n0"]["gpu_total"] = 0
    assert m.system_state["nodes"]["n0"]["status"] == "Ready"
    assert m.system_state["nodes"]["n0"]["gpu_total"] == 8
