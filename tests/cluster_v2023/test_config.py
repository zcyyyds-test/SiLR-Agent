from domains.cluster_v2023 import build_cluster_v2023_domain_config, ClusterV2023Manager
from domains.cluster_v2023.observation import ClusterV2023Observer


def test_domain_config_wiring():
    cfg = build_cluster_v2023_domain_config(f_threshold=5.0)
    assert cfg.domain_name == "cluster_v2023"
    # per-action gate is ONLY Capacity + Affinity (spec §5.1)
    names = {c.name for c in cfg.checkers}
    assert names == {"resource_capacity", "affinity"}
    assert cfg.allowed_actions == frozenset(
        ["assign_job", "migrate_job", "preempt_job"])
    assert cfg.create_observer is not None
    assert cfg.build_system_prompt is not None
    assert cfg.build_tool_schemas is not None


def test_observer_factory_returns_cluster_v2023_observer():
    cfg = build_cluster_v2023_domain_config()
    mgr = ClusterV2023Manager(nodes={}, jobs={})
    obs = cfg.create_observer(mgr)
    assert isinstance(obs, ClusterV2023Observer)


def test_toolset_factory_returns_three_tools():
    cfg = build_cluster_v2023_domain_config()
    mgr = ClusterV2023Manager(nodes={}, jobs={})
    ts = cfg.create_toolset(mgr)
    assert set(ts.keys()) == {"assign_job", "migrate_job", "preempt_job"}


def test_with_observer_false_disables_observer():
    cfg = build_cluster_v2023_domain_config(with_observer=False)
    assert cfg.create_observer is None
