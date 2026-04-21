from domains.cluster_v2023.data_pipeline.inject_faults import (
    inject_fragmentation_surge,
    inject_gpu_spec_mismatch,
    inject_node_failure,
    inject_qos_pressure,
)
from domains.cluster_v2023.manager import ClusterV2023Manager


def _cluster():
    nodes = {
        f"n{i}": {"model": ["V100M32", "G1", "T4"][i % 3],
                  "cpu_total": 96000, "ram_total_mib": 786432,
                  "gpu_total": 8, "status": "Ready",
                  "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0}
        for i in range(6)
    }
    jobs = {f"j{i}": {"cpu": 8000, "ram_mib": 16384, "gpu": 1,
                      "gpu_spec_required": None, "qos": "BE",
                      "status": "Running"}
            for i in range(6)}
    assignments = {f"j{i}": f"n{i}" for i in range(6)}
    return ClusterV2023Manager(nodes=nodes, jobs=jobs,
                               assignments=assignments)


def test_inject_node_failure_marks_node_down():
    m = _cluster()
    meta = inject_node_failure(m, n_nodes=2, seed=0)
    downs = [nid for nid, n in m.system_state["nodes"].items()
             if n["status"] == "Down"]
    assert len(downs) == 2
    assert set(downs) == set(meta["fault_nodes"])


def test_inject_node_failure_requeues_affected_jobs():
    """Affected jobs go back to Queued (not Preempted) so the Best-fit
    expert's qos-ordered assign loop picks them up for recovery."""
    m = _cluster()
    meta = inject_node_failure(m, n_nodes=2, seed=0)
    requeued = [j for j, v in m.system_state["jobs"].items()
                if v["status"] == "Queued"]
    assert set(meta["affected_jobs"]).issubset(set(requeued))
    for jid in meta["affected_jobs"]:
        assert jid not in m.system_state["assignments"]


def test_inject_gpu_spec_mismatch_adds_requirement():
    m = _cluster()
    meta = inject_gpu_spec_mismatch(m, n_jobs=3, seed=0)
    required = [j for j, v in m.system_state["jobs"].items()
                if v.get("gpu_spec_required")]
    assert len(required) == 3
    assert set(required) == set(meta["affected_jobs"])


def test_inject_gpu_spec_mismatch_creates_real_mismatch():
    """The injected requirement must differ from the current node's model."""
    m = _cluster()
    inject_gpu_spec_mismatch(m, n_jobs=3, seed=0)
    for jid, job in m.system_state["jobs"].items():
        req = job.get("gpu_spec_required")
        if req is None:
            continue
        nid = m.system_state["assignments"].get(jid)
        if nid is None:
            continue
        assert m.system_state["nodes"][nid]["model"] != req


def test_inject_qos_pressure_queues_ls_while_be_runs():
    m = _cluster()
    meta = inject_qos_pressure(m, n_ls_queued=2, seed=0)
    ls_queued = [j for j, v in m.system_state["jobs"].items()
                 if v["qos"] == "LS" and v["status"] == "Queued"]
    assert len(ls_queued) == 2
    assert set(ls_queued) == set(meta["ls_queued"])


def test_inject_fragmentation_surge_produces_many_small_jobs():
    m = _cluster()
    meta = inject_fragmentation_surge(m, seed=0)
    # affected list should contain many 1-GPU BE jobs + 1 large LS queued
    assert len(meta["affected_jobs"]) >= 2
    # there should be at least one LS queued job requesting 4+ GPUs
    big_queued = [j for j, v in m.system_state["jobs"].items()
                  if v["status"] == "Queued" and v["qos"] == "LS"
                  and v["gpu"] >= 4]
    assert len(big_queued) >= 1
