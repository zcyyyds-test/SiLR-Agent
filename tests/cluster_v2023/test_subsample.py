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
