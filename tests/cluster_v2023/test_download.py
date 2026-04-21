from pathlib import Path
from domains.cluster_v2023.data_pipeline import download


def test_expected_files_declared():
    assert "openb_node_list_gpu_node.csv" in download.EXPECTED_FILES
    assert "openb_pod_list_default.csv" in download.EXPECTED_FILES


def test_verify_returns_false_when_missing(tmp_path):
    assert download.verify(tmp_path) is False


def test_verify_returns_false_on_zero_byte(tmp_path):
    """Regression (Kimi review Q6): verify() must reject 0-byte files,
    not just absent ones (covers the `st_size == 0` branch in download.py)."""
    for fname in download.EXPECTED_FILES:
        (tmp_path / fname).write_text("")
    assert download.verify(tmp_path) is False


def test_verify_returns_false_on_partial_presence(tmp_path):
    """Regression (Kimi review Q6): only one of two expected files present."""
    names = list(download.EXPECTED_FILES)
    (tmp_path / names[0]).write_text("content")
    # names[1] deliberately missing
    assert download.verify(tmp_path) is False


def test_verify_returns_true_when_all_present_non_empty(tmp_path):
    """Regression (Kimi review Q6): happy path — all files exist + non-empty."""
    for fname in download.EXPECTED_FILES:
        (tmp_path / fname).write_text("non-empty")
    assert download.verify(tmp_path) is True
