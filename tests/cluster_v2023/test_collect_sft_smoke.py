import json
from pathlib import Path

from domains.cluster_v2023.scenarios.loader import ScenarioLoader
from scripts import collect_cluster_v2023_sft as collect


def _make_minimal_scenario(path: Path) -> None:
    ScenarioLoader.save(
        {
            "scenario_id": "v2023_test_00",
            "fault_type": "node_failure",
            "fault_meta": {},
            "nodes": {"n0": {"model": "V100M32", "cpu_total": 96000,
                              "ram_total_mib": 786432, "gpu_total": 8,
                              "status": "Ready", "cpu_used": 0,
                              "ram_used_mib": 0, "gpu_used": 0}},
            "jobs": {"j0": {"cpu": 0, "ram_mib": 0, "gpu": 1,
                             "gpu_spec_required": None, "qos": "LS",
                             "status": "Queued"}},
            "assignments": {},
            "expert_solution": [{"tool_name": "assign_job",
                                 "params": {"job_id": "j0", "node_id": "n0"}}],
            "f_bestfit_baseline": 1.0,
            "f_threshold": 1.2,
        },
        path,
    )


def test_collect_emits_trajectories(tmp_path):
    scen_dir = tmp_path / "scen"
    scen_dir.mkdir()
    _make_minimal_scenario(scen_dir / "s.json")
    out = tmp_path / "sft_base.jsonl"
    n = collect.collect(scenario_dir=scen_dir, out_path=out, seeds=[0])
    assert n == 1
    assert out.exists() and out.stat().st_size > 0

    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["scenario_id"] == "v2023_test_00"
    assert rec["seed"] == 0
    assert len(rec["messages"]) >= 3  # system + user + assistant


def test_collect_multi_seed_produces_multiple_records(tmp_path):
    scen_dir = tmp_path / "scen"
    scen_dir.mkdir()
    _make_minimal_scenario(scen_dir / "s.json")
    out = tmp_path / "sft_base.jsonl"
    n = collect.collect(scenario_dir=scen_dir, out_path=out,
                        seeds=[0, 1, 2, 3])
    assert n == 4
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    seeds = {r["seed"] for r in lines}
    assert seeds == {0, 1, 2, 3}
