from domains.cluster_v2023.manager import ClusterV2023Manager
from domains.cluster_v2023.tools import (
    AssignJobTool, MigrateJobTool, PreemptJobTool, create_toolset,
)


def _nodes():
    return {
        "n0": {"model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
        "n1": {"model": "G1", "cpu_total": 64000, "ram_total_mib": 524288,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
    }


def _jobs():
    return {
        "j0": {"cpu": 8000, "ram_mib": 16384, "gpu": 1,
               "gpu_spec_required": None, "qos": "LS", "status": "Queued"},
    }


def test_assign_happy_path():
    m = ClusterV2023Manager(nodes=_nodes(), jobs=_jobs())
    r = AssignJobTool(m).execute(job_id="j0", node_id="n0")
    assert r.status == "success"
    assert m.system_state["jobs"]["j0"]["status"] == "Running"
    assert m.system_state["assignments"]["j0"] == "n0"
    m.solve()
    assert m.system_state["nodes"]["n0"]["gpu_used"] == 1


def test_assign_fails_on_down_node():
    n = _nodes(); n["n0"]["status"] = "Down"
    m = ClusterV2023Manager(nodes=n, jobs=_jobs())
    r = AssignJobTool(m).execute(job_id="j0", node_id="n0")
    assert r.status == "error"


def test_assign_fails_on_missing_job():
    m = ClusterV2023Manager(nodes=_nodes(), jobs=_jobs())
    r = AssignJobTool(m).execute(job_id="nonexistent", node_id="n0")
    assert r.status == "error"


def test_assign_fails_on_already_running_job():
    j = _jobs(); j["j0"]["status"] = "Running"
    m = ClusterV2023Manager(nodes=_nodes(), jobs=j, assignments={"j0": "n0"})
    r = AssignJobTool(m).execute(job_id="j0", node_id="n1")
    assert r.status == "error"


def test_migrate_moves_running_job():
    j = _jobs(); j["j0"]["status"] = "Running"
    m = ClusterV2023Manager(nodes=_nodes(), jobs=j, assignments={"j0": "n0"})
    r = MigrateJobTool(m).execute(job_id="j0", node_id="n1")
    assert r.status == "success"
    assert m.system_state["assignments"]["j0"] == "n1"


def test_migrate_fails_on_queued_job():
    m = ClusterV2023Manager(nodes=_nodes(), jobs=_jobs())
    r = MigrateJobTool(m).execute(job_id="j0", node_id="n1")
    assert r.status == "error"


def test_preempt_queues_job_and_frees_node():
    j = _jobs(); j["j0"]["status"] = "Running"
    m = ClusterV2023Manager(nodes=_nodes(), jobs=j, assignments={"j0": "n0"})
    r = PreemptJobTool(m).execute(job_id="j0")
    assert r.status == "success"
    assert m.system_state["jobs"]["j0"]["status"] == "Queued"
    assert "j0" not in m.system_state["assignments"]


def test_preempt_fails_on_queued_job():
    m = ClusterV2023Manager(nodes=_nodes(), jobs=_jobs())
    r = PreemptJobTool(m).execute(job_id="j0")
    assert r.status == "error"


def test_toolset_has_three_tools():
    m = ClusterV2023Manager(nodes=_nodes(), jobs=_jobs())
    ts = create_toolset(m)
    assert {"assign_job", "migrate_job", "preempt_job"} <= set(ts)
    # Sanity: every tool wraps the same manager
    for tool in ts.values():
        assert tool.manager is m
