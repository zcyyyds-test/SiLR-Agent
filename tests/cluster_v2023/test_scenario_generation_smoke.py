from pathlib import Path

from domains.cluster_v2023.scenarios.loader import ScenarioLoader
from scripts import build_cluster_v2023_scenarios as build

FIX_DIR = Path(__file__).parent / "fixtures"


def test_build_smoke(tmp_path):
    """End-to-end smoke: build 2 scenarios from tiny fixtures, verify
    they load and have expert solutions."""
    n_built = build.build_scenarios(
        out_dir=tmp_path,
        nodes_csv=FIX_DIR / "tiny_nodes.csv",
        jobs_csv=FIX_DIR / "tiny_jobs.csv",
        target_nodes=4,
        window_start=0, window_end=10000,
        max_jobs=6,
        n_scenarios=2,
        seed=0,
    )
    assert n_built == 2
    files = ScenarioLoader.list_scenarios(tmp_path)
    assert len(files) == 2
    for p in files:
        s = ScenarioLoader.load(p)
        assert "scenario_id" in s
        assert "fault_type" in s
        assert s["expert_solution"], "must be solvable"
        assert s["f_bestfit_baseline"] >= 0
        assert s["f_threshold"] >= 0


def test_build_returns_partial_on_unsolvable_scenarios(tmp_path, monkeypatch):
    """If scenarios can't be generated solvably, builder should return
    the number it did manage to build (may be < n_scenarios)."""
    # Force expert to return no actions → unsolvable every time
    monkeypatch.setattr(
        "scripts.build_cluster_v2023_scenarios.BestFitExpert",
        _UnsolvableExpert,
    )
    n_built = build.build_scenarios(
        out_dir=tmp_path,
        nodes_csv=FIX_DIR / "tiny_nodes.csv",
        jobs_csv=FIX_DIR / "tiny_jobs.csv",
        target_nodes=4,
        window_start=0, window_end=10000,
        max_jobs=6,
        n_scenarios=3,
        seed=0,
    )
    assert n_built == 0


class _UnsolvableExpert:
    def __init__(self, *_, **__):
        pass

    def plan(self, *_args, **_kwargs):
        return []

    @staticmethod
    def apply(*_args, **_kwargs):
        pass
