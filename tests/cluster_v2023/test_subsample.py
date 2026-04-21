from pathlib import Path
from domains.cluster_v2023.data_pipeline.subsample import stratified_nodes

FIX = Path(__file__).parent / "fixtures" / "tiny_nodes.csv"


def test_stratified_preserves_proportions():
    out = stratified_nodes(FIX, target=4, seed=42)
    models = [n["model"] for n in out]
    # tiny_nodes has 3:2:1 V100M32:G1:T4; target=4 should keep at least 1 of each
    assert models.count("V100M32") == 2
    assert models.count("G1") == 1
    assert models.count("T4") == 1


def test_stratified_deterministic():
    a = stratified_nodes(FIX, target=4, seed=42)
    b = stratified_nodes(FIX, target=4, seed=42)
    assert [n["sn"] for n in a] == [n["sn"] for n in b]


def test_stratified_does_not_oversample_small_last_group(tmp_path):
    """Regression (Codex review Q2): target=8 with group sizes [3,3,3,1]
    formerly raised ValueError because last-group take=remaining=2 > len=1."""
    csv_path = tmp_path / "skewed_nodes.csv"
    csv_path.write_text(
        "sn,cpu_milli,memory_mib,gpu,model\n"
        + "\n".join(f"a{i},96000,786432,8,A" for i in range(3)) + "\n"
        + "\n".join(f"b{i},96000,786432,8,B" for i in range(3)) + "\n"
        + "\n".join(f"c{i},96000,786432,8,C" for i in range(3)) + "\n"
        + "d0,96000,786432,8,D\n"
    )
    out = stratified_nodes(csv_path, target=8, seed=0)
    assert len(out) <= 8
    models_out = [n["model"] for n in out]
    assert models_out.count("D") <= 1  # never more than exists
