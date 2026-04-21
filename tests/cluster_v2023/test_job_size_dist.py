from pathlib import Path
from domains.cluster_v2023.data_pipeline.compute_job_size_dist import (
    compute_dist, save_dist, load_dist,
)

FIX = Path(__file__).parent / "fixtures" / "tiny_jobs.csv"


def test_compute_dist_sums_to_one():
    d = compute_dist(FIX)
    assert abs(sum(d.values()) - 1.0) < 1e-6


def test_dist_keys_are_ints():
    d = compute_dist(FIX)
    assert all(isinstance(k, int) for k in d)


def test_save_load_roundtrip(tmp_path):
    d = {1: 0.5, 2: 0.3, 4: 0.2}
    p = tmp_path / "d.json"
    save_dist(d, p)
    assert load_dist(p) == d
