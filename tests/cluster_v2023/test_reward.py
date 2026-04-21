import copy

from domains.cluster_v2023.manager import ClusterV2023Manager
from scripts.cluster_v2023_reward import dense_reward


def _mgr_with_one_queued():
    return ClusterV2023Manager(
        nodes={"n0": {"model": "V100M32", "cpu_total": 96000,
                      "ram_total_mib": 786432, "gpu_total": 8,
                      "status": "Ready", "cpu_used": 0,
                      "ram_used_mib": 0, "gpu_used": 0}},
        jobs={"j0": {"cpu": 0, "ram_mib": 0, "gpu": 1,
                     "gpu_spec_required": None, "qos": "LS",
                     "status": "Queued"}},
    )


def test_reward_positive_when_violation_drops():
    pre = _mgr_with_one_queued()
    pre_state = copy.deepcopy(pre.system_state)
    # apply an assign
    post = _mgr_with_one_queued()
    post._jobs["j0"]["status"] = "Running"
    post._assignments["j0"] = "n0"
    post.solve()
    r = dense_reward(pre_state=pre_state,
                     post_state=post.system_state,
                     verdict="PASS", f_baseline=1.0)
    # violation count dropped (+0.10) + all checkers pass (+1.00)
    assert r >= 1.10 - 1e-9


def test_reward_penalty_when_rejected():
    mgr = _mgr_with_one_queued()
    r = dense_reward(pre_state=mgr.system_state,
                     post_state=mgr.system_state,
                     verdict="FAIL", f_baseline=1.0)
    assert r == -0.5


def test_reward_neutral_when_no_change():
    """Same state pre and post → no violation_count drop and no F drop →
    reward is 0 (unless all checkers pass, in which case +1.0 bonus)."""
    mgr = _mgr_with_one_queued()
    snapshot = copy.deepcopy(mgr.system_state)
    r = dense_reward(pre_state=snapshot,
                     post_state=snapshot,
                     verdict="PASS", f_baseline=1.0)
    # LS queued → violations present → not all pass → reward = 0
    assert r == 0.0


def test_reward_fragmentation_bonus():
    """Dropping fragmentation (F) earns +0.30 bonus."""
    high_frag = {
        "nodes": {f"n{i}": {"model": "V100M32", "cpu_total": 96000,
                            "ram_total_mib": 786432, "gpu_total": 8,
                            "status": "Ready", "cpu_used": 0,
                            "ram_used_mib": 0, "gpu_used": 6}
                  for i in range(5)},
        "jobs": {}, "assignments": {}, "sim_time": 0.0,
    }
    low_frag = copy.deepcopy(high_frag)
    for n in low_frag["nodes"].values():
        n["gpu_used"] = 0  # all remaining = 8, not fragmentary
    r = dense_reward(pre_state=high_frag, post_state=low_frag,
                     verdict="PASS", f_baseline=1.0)
    # violation count same (0 → 0), fragmentation dropped (+0.30),
    # all checkers pass (+1.00)
    assert r >= 1.30 - 1e-9
