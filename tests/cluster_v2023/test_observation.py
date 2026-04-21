from silr.agent.types import Observation

from domains.cluster_v2023.manager import ClusterV2023Manager
from domains.cluster_v2023.observation import ClusterV2023Observer


def _mgr():
    nodes = {
        "n0": {"model": "V100M32", "cpu_total": 96000, "ram_total_mib": 786432,
               "gpu_total": 8, "status": "Ready",
               "cpu_used": 4000, "ram_used_mib": 8192, "gpu_used": 2},
    }
    jobs = {"j0": {"cpu": 8000, "ram_mib": 16384, "gpu": 1,
                   "gpu_spec_required": None, "qos": "LS", "status": "Queued"}}
    return ClusterV2023Manager(nodes=nodes, jobs=jobs)


def test_observation_is_dataclass_with_required_fields():
    obs = ClusterV2023Observer(_mgr()).observe()
    assert isinstance(obs, Observation)
    # ReActAgent reads these three fields
    assert isinstance(obs.raw, dict)
    assert isinstance(obs.compressed_json, str) and len(obs.compressed_json) > 0
    assert isinstance(obs.is_stable, bool)


def test_observation_raw_contains_key_fields():
    obs = ClusterV2023Observer(_mgr()).observe()
    assert "nodes" in obs.raw
    assert "queued_jobs" in obs.raw
    assert "fragmentation_F" in obs.raw
    assert obs.raw["queued_jobs"][0]["id"] == "j0"


def test_observation_is_stable_false_when_ls_queued():
    """LS queued with a stable cluster should flag Queue violation → is_stable=False."""
    obs = ClusterV2023Observer(_mgr()).observe()
    assert obs.is_stable is False
    # Queue violation should appear in violations list
    assert any(v["constraint_type"] == "queue" for v in obs.violations)


def test_observation_is_stable_true_when_all_satisfied():
    mgr = ClusterV2023Manager(
        nodes={"n0": {"model": "V100M32", "cpu_total": 96000,
                      "ram_total_mib": 786432, "gpu_total": 8, "status": "Ready",
                      "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0}},
        jobs={},  # no jobs — nothing to violate
    )
    obs = ClusterV2023Observer(mgr).observe()
    assert obs.is_stable is True


def test_observation_fragmentation_is_observer_only():
    """Fragmentation violations do NOT affect is_stable (per spec §5.1)."""
    mgr = ClusterV2023Manager(
        nodes={"n0": {"model": "V100M32", "cpu_total": 96000,
                      "ram_total_mib": 786432, "gpu_total": 8, "status": "Ready",
                      "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 6}},
        jobs={},  # no jobs: Queue/Priority/Affinity/Capacity all pass
    )
    # f_threshold=0 means any fragmentation is flagged, but it's observer-only
    obs = ClusterV2023Observer(mgr, f_threshold=0.0).observe()
    assert obs.is_stable is True
    assert obs.raw["fragmentation_F"] > 0
