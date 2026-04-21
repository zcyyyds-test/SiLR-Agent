from domains.cluster_v2023.manager import ClusterV2023Manager
from domains.cluster_v2023.scenarios.loader import ScenarioLoader


def test_roundtrip_save_and_load(tmp_path):
    nodes = {"n0": {"model": "V100M32", "cpu_total": 96000,
                    "ram_total_mib": 786432, "gpu_total": 8,
                    "status": "Ready", "cpu_used": 0, "ram_used_mib": 0,
                    "gpu_used": 0}}
    jobs = {"j0": {"cpu": 8000, "ram_mib": 16384, "gpu": 1,
                   "gpu_spec_required": None, "qos": "LS",
                   "status": "Queued"}}
    scenario = {
        "scenario_id": "v2023_test_01",
        "fault_type": "node_failure",
        "fault_meta": {"fault_nodes": []},
        "nodes": nodes, "jobs": jobs, "assignments": {},
        "expert_solution": [],
        "f_bestfit_baseline": 3.0,
    }
    p = tmp_path / "s.json"
    ScenarioLoader.save(scenario, p)
    loaded = ScenarioLoader.load(p)
    assert loaded["scenario_id"] == "v2023_test_01"


def test_build_manager_from_scenario(tmp_path):
    nodes = {"n0": {"model": "V100M32", "cpu_total": 96000,
                    "ram_total_mib": 786432, "gpu_total": 8,
                    "status": "Ready", "cpu_used": 0, "ram_used_mib": 0,
                    "gpu_used": 0}}
    jobs = {"j0": {"cpu": 0, "ram_mib": 0, "gpu": 1,
                   "gpu_spec_required": None, "qos": "LS",
                   "status": "Queued"}}
    scenario = {"scenario_id": "v2023_test",
                "fault_type": "node_failure", "fault_meta": {},
                "nodes": nodes, "jobs": jobs, "assignments": {},
                "expert_solution": [], "f_bestfit_baseline": 1.0}
    m = ScenarioLoader.build_manager(scenario)
    assert isinstance(m, ClusterV2023Manager)
    assert "n0" in m.system_state["nodes"]
    assert m.system_state["jobs"]["j0"]["status"] == "Queued"


def test_list_scenarios(tmp_path):
    for i in range(3):
        ScenarioLoader.save(
            {"scenario_id": f"v2023_{i:02d}", "fault_type": "node_failure",
             "fault_meta": {}, "nodes": {}, "jobs": {}, "assignments": {},
             "expert_solution": [], "f_bestfit_baseline": 1.0},
            tmp_path / f"s{i}.json",
        )
    paths = ScenarioLoader.list_scenarios(tmp_path)
    assert len(paths) == 3
    # Paths returned in sorted order (deterministic iteration)
    assert [p.name for p in paths] == ["s0.json", "s1.json", "s2.json"]
