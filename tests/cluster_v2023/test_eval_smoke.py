from domains.cluster_v2023.scenarios.loader import ScenarioLoader
from scripts import eval_cluster_v2023 as ev


def _fake_scenario_payload(sid: str) -> dict:
    return {
        "scenario_id": sid,
        "fault_type": "node_failure", "fault_meta": {},
        "nodes": {}, "jobs": {}, "assignments": {},
        "expert_solution": [], "f_bestfit_baseline": 1.0,
        "f_threshold": 1.2,
    }


def test_scenarios_in_returns_sorted_list(tmp_path):
    for i in range(2):
        ScenarioLoader.save(_fake_scenario_payload(f"v2023_fake_{i:02d}"),
                            tmp_path / f"s{i}.json")
    paths = ev._scenarios_in(tmp_path)
    assert len(paths) == 2


def test_aggregate_stats_structure():
    """`_aggregate` must produce the schema downstream tooling expects."""
    episodes = [
        {"recovered": True, "total_rejections": 0, "total_proposals": 3,
         "fragmentation_F_normalized": 1.0},
        {"recovered": False, "total_rejections": 2, "total_proposals": 5,
         "fragmentation_F_normalized": 1.5},
    ]
    agg = ev._aggregate(episodes)
    assert agg["n_episodes"] == 2
    assert agg["recovery_rate"] == 0.5
    assert abs(agg["mean_F_normalized"] - 1.25) < 1e-9
    assert abs(agg["reject_rate"] - (2 / 8)) < 1e-9
