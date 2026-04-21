from pathlib import Path
from domains.cluster_v2023.data_pipeline import download


def test_expected_files_declared():
    assert "openb_node_list_gpu_node.csv" in download.EXPECTED_FILES
    assert "openb_pod_list_default.csv" in download.EXPECTED_FILES


def test_verify_returns_false_when_missing(tmp_path):
    assert download.verify(tmp_path) is False
