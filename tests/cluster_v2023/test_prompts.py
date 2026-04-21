from domains.cluster_v2023.manager import ClusterV2023Manager
from domains.cluster_v2023.prompts import (
    build_system_prompt, build_tool_schemas,
)


def _mgr():
    return ClusterV2023Manager(
        nodes={"n0": {"model": "V100M32", "cpu_total": 96000,
                      "ram_total_mib": 786432, "gpu_total": 8,
                      "status": "Ready", "cpu_used": 0,
                      "ram_used_mib": 0, "gpu_used": 0}},
        jobs={},
    )


def test_system_prompt_contains_qos_hint():
    p = build_system_prompt(_mgr(), build_tool_schemas(_mgr()))
    assert "LS" in p and "Burstable" in p and "BE" in p
    assert "assign_job" in p


def test_system_prompt_mentions_all_three_tools():
    p = build_system_prompt(_mgr(), build_tool_schemas(_mgr()))
    for name in ("assign_job", "migrate_job", "preempt_job"):
        assert name in p


def test_tool_schemas_are_three():
    schemas = build_tool_schemas(_mgr())
    names = {s["name"] for s in schemas}
    assert names == {"assign_job", "migrate_job", "preempt_job"}


def test_tool_schemas_have_required_params():
    schemas = {s["name"]: s for s in build_tool_schemas(_mgr())}
    assert schemas["assign_job"]["parameters"]["required"] == ["job_id", "node_id"]
    assert schemas["migrate_job"]["parameters"]["required"] == ["job_id", "node_id"]
    assert schemas["preempt_job"]["parameters"]["required"] == ["job_id"]
