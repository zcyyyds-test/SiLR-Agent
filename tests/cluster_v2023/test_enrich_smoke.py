import os
import json
import pytest


@pytest.mark.skipif(not os.environ.get("LEMON_API_KEY"),
                    reason="LEMON_API_KEY env not set — live API not exercised")
def test_enrich_one_line_live(tmp_path):
    """Live smoke — only runs when LEMON_API_KEY is set. Writes one
    record through the enrichment pipeline and checks the assistant
    message got a CoT prefix."""
    from scripts import enrich_cluster_v2023_sft as enrich
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(json.dumps({
        "scenario_id": "v2023_enrich_test",
        "seed": 0,
        "messages": [
            {"role": "system", "content": "Schedule GPUs."},
            {"role": "user",
             "content": '{"nodes":[{"id":"n0","gpu":"0/8"}],"queued_jobs":[{"id":"j0","gpu":1,"qos":"LS"}]}'},
            {"role": "assistant",
             "content": '{"tool_name":"assign_job","params":{"job_id":"j0","node_id":"n0"}}'},
        ],
    }) + "\n")
    n = enrich.enrich(inp, out, max_lines=1)
    assert n == 1
    record = json.loads(out.read_text().splitlines()[0])
    assistant_msg = record["messages"][-1]["content"]
    # CoT should be prepended, original JSON preserved at end
    assert '"tool_name":"assign_job"' in assistant_msg
    # Reasoning text is non-empty (at least one line before the JSON)
    assert "\n\n" in assistant_msg


def test_jsonl_to_json_array_converter(tmp_path):
    """Plan Task 23 Step 7: JSONL → JSON array converter (inline snippet
    in the plan must actually work). Locked in a test."""
    from scripts import enrich_cluster_v2023_sft as enrich

    jsonl = tmp_path / "input.jsonl"
    out_json = tmp_path / "output.json"
    records = [
        {"scenario_id": "s1", "messages": [{"role": "user", "content": "x"}]},
        {"scenario_id": "s2", "messages": [{"role": "user", "content": "y"}]},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    n = enrich.jsonl_to_json_array(jsonl, out_json)
    assert n == 2
    loaded = json.loads(out_json.read_text())
    assert len(loaded) == 2
    assert loaded[0]["scenario_id"] == "s1"
    assert loaded[1]["scenario_id"] == "s2"
