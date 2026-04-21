"""End-to-end smoke: scenario → manager + verifier + observer →
verify one action → check Observation contract + verdict.

This is the contract test that would have caught Task 14's original
dict-vs-Observation bug at pytest time instead of at deployment time.
"""

from silr.agent.types import Observation
from silr.verifier import SiLRVerifier
from silr.verifier.types import Verdict

from domains.cluster_v2023.config import build_cluster_v2023_domain_config
from domains.cluster_v2023.scenarios.loader import ScenarioLoader


def _minimal_scenario(tmp_path):
    """One-node, one-queued-job scenario — guaranteed solvable in 1 step."""
    s = {
        "scenario_id": "v2023_integration_smoke",
        "fault_type": "node_failure",
        "fault_meta": {},
        "nodes": {
            "n0": {"model": "V100M32", "cpu_total": 96000,
                   "ram_total_mib": 786432, "gpu_total": 8, "status": "Ready",
                   "cpu_used": 0, "ram_used_mib": 0, "gpu_used": 0},
        },
        "jobs": {
            "j0": {"cpu": 0, "ram_mib": 0, "gpu": 1, "gpu_spec_required": None,
                   "qos": "LS", "status": "Queued"},
        },
        "assignments": {},
        "expert_solution": [{"tool_name": "assign_job",
                             "params": {"job_id": "j0", "node_id": "n0"}}],
        "f_bestfit_baseline": 1.0,
        "f_threshold": 1.2,
    }
    ScenarioLoader.save(s, tmp_path / "s.json")
    return ScenarioLoader.load(tmp_path / "s.json")


def test_scenario_roundtrip_verify_loop(tmp_path):
    s = _minimal_scenario(tmp_path)
    mgr = ScenarioLoader.build_manager(s)
    mgr.solve()

    cfg = build_cluster_v2023_domain_config(f_threshold=s["f_threshold"])

    # Observation contract (would catch Task 14 dict-vs-Observation bug)
    observer = cfg.create_observer(mgr)
    obs = observer.observe()
    assert isinstance(obs, Observation)
    assert obs.is_stable is False  # LS job queued → Queue observer violation
    assert isinstance(obs.compressed_json, str) and len(obs.compressed_json) > 0

    # Verifier pipeline (catches DomainConfig wiring mistakes)
    verifier = SiLRVerifier(mgr, domain_config=cfg)
    result = verifier.verify(
        {"tool_name": "assign_job",
         "params": {"job_id": "j0", "node_id": "n0"}},
    )
    assert result.verdict == Verdict.PASS
    assert result.solver_converged is True

    # Shadow isolation: main mgr unchanged after verify()
    assert mgr.system_state["jobs"]["j0"]["status"] == "Queued"


def test_verifier_rejects_action_on_down_node(tmp_path):
    """Tool validation failure surfaces through verifier as non-PASS."""
    s = _minimal_scenario(tmp_path)
    s["nodes"]["n0"]["status"] = "Down"
    mgr = ScenarioLoader.build_manager(s)
    mgr.solve()

    cfg = build_cluster_v2023_domain_config(f_threshold=s["f_threshold"])
    verifier = SiLRVerifier(mgr, domain_config=cfg)
    result = verifier.verify(
        {"tool_name": "assign_job",
         "params": {"job_id": "j0", "node_id": "n0"}},
    )
    # Tool validation raises SystemStateError before checkers run
    assert result.verdict != Verdict.PASS


def test_verifier_rejects_unknown_tool(tmp_path):
    """Tools not in allowed_actions → ERROR verdict."""
    s = _minimal_scenario(tmp_path)
    mgr = ScenarioLoader.build_manager(s)
    mgr.solve()

    cfg = build_cluster_v2023_domain_config(f_threshold=s["f_threshold"])
    verifier = SiLRVerifier(mgr, domain_config=cfg)
    result = verifier.verify(
        {"tool_name": "not_a_real_tool", "params": {}},
    )
    assert result.verdict == Verdict.ERROR
