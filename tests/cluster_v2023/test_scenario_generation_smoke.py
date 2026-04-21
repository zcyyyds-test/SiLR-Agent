from pathlib import Path

from domains.cluster_v2023.scenarios.loader import ScenarioLoader
from scripts import build_cluster_v2023_scenarios as build

FIX_DIR = Path(__file__).parent / "fixtures"


def test_build_smoke(tmp_path):
    """End-to-end smoke: build scenarios from tiny fixtures.

    Tiny fixtures (4 nodes) can't support aggressive faults like
    node_failure(2) — expected to sometimes hit the retry cap. Smoke
    assertion is "at least one scenario builds successfully" — full
    production (40 nodes) has much more headroom.
    """
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
    assert n_built >= 1, f"no scenario was solvable at smoke scale (n_built={n_built})"
    files = ScenarioLoader.list_scenarios(tmp_path)
    assert len(files) == n_built
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
