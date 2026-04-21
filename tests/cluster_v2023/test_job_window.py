from pathlib import Path
from domains.cluster_v2023.data_pipeline.job_window import select_window

FIX = Path(__file__).parent / "fixtures" / "tiny_jobs.csv"


def test_window_respects_time_bounds():
    rows = select_window(FIX, start=1000, end=2000, max_jobs=10)
    names = {r["name"] for r in rows}
    assert names == {"j0", "j1", "j2", "j3"}  # j4, j5 outside window


def test_window_caps_max_jobs():
    rows = select_window(FIX, start=0, end=10000, max_jobs=2)
    assert len(rows) == 2


def test_window_respects_node_capacity_filter():
    rows = select_window(
        FIX, start=0, end=10000, max_jobs=99,
        max_gpus_per_job=1,
    )
    assert all(int(r["num_gpu"]) <= 1 for r in rows)


def test_window_max_gpus_per_job_zero_filters_all_gpu_jobs():
    """Regression (Kimi review Q2): passing max_gpus_per_job=0 must exclude
    all jobs with num_gpu >= 1, not silently bypass the filter due to
    `if max_gpus_per_job and ...` truthiness."""
    rows = select_window(FIX, start=0, end=10000, max_jobs=99, max_gpus_per_job=0)
    # All fixture jobs have num_gpu >= 1, so the filter should yield empty.
    assert rows == []
