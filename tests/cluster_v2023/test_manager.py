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
