"""ID anonymization post-processor for SFT per-step samples.

Both Codex and DeepSeek diagnosed v12's gpu_spec 0/11 as job/node id
overfitting: 3 unique gpu_spec scenarios x 8 seeds taught the policy
"job 2698 -> node 0024" memorization rather than the general rule
"match aff_run.required_model with free node's model".

Fix: per-sample, build a {old_id -> random_new_id} mapping that
preserves model-name correspondences, then rewrite all id references
across user content + assistant action. The required_model strings
(G3, V100M16, ...) are NOT remapped — that's the only signal the
policy is supposed to learn.

Eval-side observation will still emit real ids; the policy learned to
look up "free node whose model == aff_run job's required_model" so it
will still work on real ids.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

NODE_PREFIX = "openb-node-"
POD_PREFIX = "openb-pod-"


def _short(full: str, prefix: str) -> str:
    return full[len(prefix):] if full.startswith(prefix) else full


def _restore(short: str, full_was: str, prefix: str) -> str:
    return prefix + short if full_was.startswith(prefix) else short


def _gen_unique(rng: random.Random, used: set) -> str:
    for _ in range(100000):
        candidate = f"{rng.randint(0, 9999):04d}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise RuntimeError("ran out of 4-digit ids")


def _collect_ids(user_obj: dict, assistant_obj: dict) -> tuple[set, set]:
    """Walk every field and gather all node-ids and job-ids referenced."""
    nodes, jobs = set(), set()
    for n in user_obj.get("free", []) or []:
        nodes.add(n[0])
    for n in user_obj.get("down", []) or []:
        nodes.add(n)
    for j in user_obj.get("q", []) or []:
        jobs.add(j[0])
    for s in user_obj.get("strand", []) or []:
        jobs.add(s[0])
        nodes.add(s[1])
    for b in user_obj.get("be_run", []) or []:
        jobs.add(b[0])
        nodes.add(b[1])
    for a in user_obj.get("aff_run", []) or []:
        jobs.add(a[0])
        nodes.add(a[1])
    params = assistant_obj.get("params", {}) or {}
    if "job_id" in params:
        jobs.add(_short(params["job_id"], POD_PREFIX))
    if "node_id" in params:
        nodes.add(_short(params["node_id"], NODE_PREFIX))
    return nodes, jobs


def anonymize_sample(sample: dict, seed: int) -> dict:
    msgs = sample.get("messages", [])
    if len(msgs) < 3:
        return sample
    user_msg, assistant_msg = msgs[1], msgs[2]
    try:
        user_obj = json.loads(user_msg["content"])
        assistant_obj = json.loads(assistant_msg["content"])
    except (json.JSONDecodeError, KeyError):
        return sample

    rng = random.Random(seed)
    old_nodes, old_jobs = _collect_ids(user_obj, assistant_obj)

    used_n: set = set()
    used_j: set = set()
    node_map = {old: _gen_unique(rng, used_n) for old in old_nodes}
    job_map = {old: _gen_unique(rng, used_j) for old in old_jobs}

    # Rewrite user content
    for n in user_obj.get("free", []) or []:
        n[0] = node_map.get(n[0], n[0])
    user_obj["down"] = [node_map.get(n, n) for n in user_obj.get("down", []) or []]
    for j in user_obj.get("q", []) or []:
        j[0] = job_map.get(j[0], j[0])
    for s in user_obj.get("strand", []) or []:
        s[0] = job_map.get(s[0], s[0])
        s[1] = node_map.get(s[1], s[1])
    for b in user_obj.get("be_run", []) or []:
        b[0] = job_map.get(b[0], b[0])
        b[1] = node_map.get(b[1], b[1])
    for a in user_obj.get("aff_run", []) or []:
        a[0] = job_map.get(a[0], a[0])
        a[1] = node_map.get(a[1], a[1])

    # Rewrite assistant action
    params = assistant_obj.get("params", {}) or {}
    if "job_id" in params:
        old_full = params["job_id"]
        short = _short(old_full, POD_PREFIX)
        params["job_id"] = _restore(job_map.get(short, short), old_full, POD_PREFIX)
    if "node_id" in params:
        old_full = params["node_id"]
        short = _short(old_full, NODE_PREFIX)
        params["node_id"] = _restore(node_map.get(short, short), old_full, NODE_PREFIX)

    msgs[1]["content"] = json.dumps(user_obj, separators=(",", ":"))
    msgs[2]["content"] = json.dumps(assistant_obj, separators=(",", ":"))
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Per-step JSON array.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = json.loads(Path(args.input).read_text())
    rng = random.Random(args.seed)
    out = []
    for s in samples:
        sample_seed = rng.randint(0, 2**31 - 1)
        out.append(anonymize_sample(s, sample_seed))
    Path(args.output).write_text(json.dumps(out, indent=None))
    print(f"Anonymized {len(out)} samples -> {args.output}")


if __name__ == "__main__":
    main()
