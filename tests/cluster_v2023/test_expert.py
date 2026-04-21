from domains.cluster_v2023.expert import BestFitExpert
from domains.cluster_v2023.manager import ClusterV2023Manager


def _simple_cluster():
    nodes = {
        "n0": {"model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
        "n1": {"model": "G1", "cpu_total": 64000, "ram_total_mib": 524288,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
    }
    jobs = {
        "j0": {"cpu": 8000, "ram_mib": 16384, "gpu": 1,
               "gpu_spec_required": None, "qos": "LS", "status": "Queued"},
    }
    return ClusterV2023Manager(nodes=nodes, jobs=jobs)


def test_expert_assigns_queued_job():
    m = _simple_cluster()
    actions = BestFitExpert().plan(m, max_steps=5)
    assert len(actions) >= 1
    # apply all actions to a fresh manager, verify goal state
    for a in actions:
        BestFitExpert.apply(m, a)
    assert m.system_state["jobs"]["j0"]["status"] == "Running"


def test_expert_migrates_job_off_down_node():
    nodes = {
        "n0": {"model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
               "gpu_total": 8, "status": "Down",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
        "n1": {"model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
    }
    jobs = {"j0": {"cpu": 0, "ram_mib": 0, "gpu": 1,
                   "gpu_spec_required": None, "qos": "LS", "status": "Running"}}
    m = ClusterV2023Manager(nodes=nodes, jobs=jobs,
                            assignments={"j0": "n0"})
    actions = BestFitExpert().plan(m, max_steps=5)
    # Should at least propose to migrate j0 off n0
    assert any(a["tool_name"] == "migrate_job" for a in actions)


def test_expert_preempts_be_for_ls():
    """LS queued + BE running occupying all capacity → expert should preempt BE."""
    nodes = {
        "n0": {"model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
               "gpu_total": 1, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 1},
    }
    jobs = {
        "j_be": {"cpu": 0, "ram_mib": 0, "gpu": 1,
                 "gpu_spec_required": None, "qos": "BE", "status": "Running"},
        "j_ls": {"cpu": 0, "ram_mib": 0, "gpu": 1,
                 "gpu_spec_required": None, "qos": "LS", "status": "Queued"},
    }
    m = ClusterV2023Manager(nodes=nodes, jobs=jobs,
                            assignments={"j_be": "n0"})
    actions = BestFitExpert().plan(m, max_steps=5)
    # First action should be preempt of j_be
    assert actions[0] == {"tool_name": "preempt_job",
                          "params": {"job_id": "j_be"}}


def test_seed_changes_tiebreak():
    """Two identical Ready nodes → different seeds pick different targets."""
    nodes = {
        "n0": {"model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
        "n1": {"model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
    }
    jobs = {"j0": {"cpu": 0, "ram_mib": 0, "gpu": 1,
                   "gpu_spec_required": None, "qos": "LS",
                   "status": "Queued"}}
    picks = set()
    for s in range(8):
        m = ClusterV2023Manager(nodes=nodes, jobs=jobs)
        actions = BestFitExpert(seed=s).plan(m, max_steps=2)
        picks.add(actions[0]["params"]["node_id"])
    assert picks == {"n0", "n1"}


def test_expert_respects_affinity():
    nodes = {
        "n0": {"model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
        "n1": {"model": "G1", "cpu_total": 64000, "ram_total_mib": 524288,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
    }
    jobs = {"j0": {"cpu": 0, "ram_mib": 0, "gpu": 1,
                   "gpu_spec_required": "G1", "qos": "LS",
                   "status": "Queued"}}
    m = ClusterV2023Manager(nodes=nodes, jobs=jobs)
    actions = BestFitExpert().plan(m, max_steps=5)
    assert actions[0]["params"]["node_id"] == "n1"
