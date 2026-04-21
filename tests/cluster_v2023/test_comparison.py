import json
from scripts import build_cluster_v2023_comparison as bcv


def _write_agg(path, recovery, f, reject):
    path.write_text(json.dumps({
        "aggregate": {
            "recovery_rate": recovery,
            "mean_F_normalized": f,
            "reject_rate": reject,
        },
        "episodes": [],
    }))


def test_build_table_handles_missing_files(tmp_path):
    """Missing JSONs render as em-dash placeholders (not crashes)."""
    table = bcv.build_table(tmp_path)
    assert "Qwen3-14B zero-shot |" in table
    # All missing → em-dashes in cells
    assert "—" in table


def test_build_table_renders_all_four(tmp_path):
    _write_agg(tmp_path / "zero_shot_14b.json", 0.05, 1.40, 0.20)
    _write_agg(tmp_path / "zero_shot_32b.json", 0.35, 1.20, 0.10)
    _write_agg(tmp_path / "eval_sft.json", 0.82, 1.10, 0.05)
    _write_agg(tmp_path / "eval_grpo.json", 0.93, 1.05, 0.02)
    table = bcv.build_table(tmp_path)
    assert "5.0%" in table     # zs14 recovery
    assert "35.0%" in table    # zs32 recovery
    assert "82.0%" in table    # sft recovery
    assert "**93.0%**" in table   # grpo recovery bold
    assert "1.400" in table    # zs14 f
    assert "**1.050**" in table   # grpo f bold
