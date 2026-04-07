"""Clean, deduplicate, and enrich SFT data from multiple collection runs.

Fixes:
1. Thought+JSON duplication in assistant responses → clean JSON only
2. Old observation format → replay scenario to regenerate with current observer
3. Missing system message → inject from system_prompt builder
4. Missing chain-of-thought → teacher model generates reasoning per step
5. Cross-version deduplication → keep best version of each trajectory
6. Remove trivial none-only samples

Usage:
    python scripts/clean_sft_data.py \
        --input-dirs outputs/sft_collection_v1 outputs/sft_collection_v2 ... \
        --output outputs/sft_cleaned \
        --enrich-model <teacher-model-name> \
        --enrich-base-url <openai-compatible-endpoint> \
        --enrich-api-key <key>
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.cluster import ClusterManager, ClusterScenarioLoader
from domains.cluster.observation import ClusterObserver
from domains.cluster.prompts.system_prompt import build_cluster_system_prompt
from domains.cluster.prompts.tool_schemas import build_cluster_tool_schemas

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Clean assistant format (Thought+JSON duplication)
# ---------------------------------------------------------------------------

def clean_assistant_content(content: str) -> str:
    """Remove Thought: prefix duplication, keep the final JSON action."""
    # Pattern: "Thought: {JSON}\n\n{JSON}" → keep last JSON
    # Also handles "Thought: some text\n\n{JSON}"
    content = content.strip()

    # Try to find the last valid JSON action block
    json_pattern = r'\{["\s]*"?tool_name"?\s*:'
    matches = list(re.finditer(json_pattern, content))

    if len(matches) >= 2:
        # Multiple JSON blocks — keep the last one
        last_start = matches[-1].start()
        content = content[last_start:].strip()
    elif len(matches) == 1 and content.startswith("Thought:"):
        # "Thought: ..." then JSON — keep just JSON
        last_start = matches[0].start()
        content = content[last_start:].strip()

    return content


# ---------------------------------------------------------------------------
# Step 2: Replay scenario to rebuild observations with current observer
# ---------------------------------------------------------------------------

def extract_actions(messages: list[dict]) -> list[dict]:
    """Extract (tool_name, params) from assistant messages."""
    actions = []
    for m in messages:
        if m["role"] != "assistant":
            continue
        cleaned = clean_assistant_content(m["content"])
        try:
            action = json.loads(cleaned)
            actions.append(action)
        except json.JSONDecodeError:
            actions.append({"tool_name": "none", "params": {}})
    return actions


def replay_scenario(scenario_id: str, actions: list[dict]) -> list[str]:
    """Replay scenario with current observer to get updated observations.

    Returns list of observation JSON strings (one per step).
    """
    loader = ClusterScenarioLoader()
    scenario = loader.load(scenario_id)
    mgr = ClusterManager()
    loader.setup_episode(mgr, scenario)
    mgr.solve()

    observer = ClusterObserver(mgr)
    observations = []

    for action in actions:
        obs = observer.observe()
        observations.append(obs.compressed_json)

        tool_name = action.get("tool_name", "none")
        params = action.get("params", {})

        if tool_name == "none":
            continue

        # Apply action to manager
        try:
            if tool_name == "assign_job":
                jid = params.get("job_id", "")
                nid = params.get("node_id", params.get("target_node", ""))
                if jid in mgr._jobs and nid in mgr._nodes:
                    mgr._assignments[jid] = nid
                    mgr._jobs[jid]["status"] = "Running"
            elif tool_name == "preempt_job":
                jid = params.get("job_id", "")
                if jid in mgr._assignments:
                    del mgr._assignments[jid]
                    mgr._jobs[jid]["status"] = "Queued"
            elif tool_name == "migrate_job":
                jid = params.get("job_id", "")
                nid = params.get("target_node", params.get("target_node_id", ""))
                if jid in mgr._assignments and nid in mgr._nodes:
                    mgr._assignments[jid] = nid
            elif tool_name == "restore_node":
                nid = params.get("node_id", "")
                if nid in mgr._nodes:
                    mgr._nodes[nid]["status"] = "Ready"
            elif tool_name == "drain_node":
                nid = params.get("node_id", "")
                if nid in mgr._nodes:
                    mgr._nodes[nid]["status"] = "Cordoned"

            mgr.solve()
        except Exception as e:
            logger.warning(f"Replay action failed: {tool_name} {params}: {e}")

    return observations


# ---------------------------------------------------------------------------
# Step 3: Build system message
# ---------------------------------------------------------------------------

def build_system_message() -> str:
    """Build system message from current prompt builder."""
    mgr = ClusterManager()
    schemas = build_cluster_tool_schemas(mgr)
    return build_cluster_system_prompt(mgr, schemas)


# ---------------------------------------------------------------------------
# Step 4: GPT enrichment — add chain-of-thought reasoning
# ---------------------------------------------------------------------------

def enrich_with_reasoning(
    observation: str,
    action_json: str,
    system_msg: str,
    client,
) -> str:
    """Use GPT to generate chain-of-thought reasoning for an action.

    Returns: "Reasoning: <explanation>\n\n<action_json>"
    """
    prompt = f"""You are analyzing a GPU cluster scheduling decision.

Given this cluster observation:
{observation}

The correct action taken was:
{action_json}

Write a brief (2-3 sentences) reasoning explaining WHY this action was chosen.
Focus on: which job needs placement, why this specific node was picked (rack affinity,
free GPUs), and whether preemption was needed.

Reply ONLY with the reasoning text, no JSON, no prefixes."""

    try:
        resp = client.chat.completions.create(
            model=client._model_name,
            messages=[
                {"role": "system", "content": "You explain GPU scheduling decisions concisely."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        reasoning = resp.choices[0].message.content.strip()
        return f"{reasoning}\n\n{action_json}"
    except Exception as e:
        logger.warning(f"Enrichment failed: {e}")
        return action_json


# ---------------------------------------------------------------------------
# Step 5: Main pipeline
# ---------------------------------------------------------------------------

def get_action_fingerprint(sample: dict) -> tuple:
    """Create a fingerprint for deduplication."""
    actions = extract_actions(sample["messages"])
    act_strs = []
    for a in actions:
        tn = a.get("tool_name", "none")
        params = a.get("params", {})
        act_strs.append(f"{tn}:{json.dumps(params, sort_keys=True)}")
    return (sample["scenario_id"], tuple(act_strs))


def main():
    parser = argparse.ArgumentParser(description="Clean and enrich SFT data")
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/sft_cleaned")
    parser.add_argument("--enrich-model", default=None,
                        help="Model for chain-of-thought enrichment (skip if not set)")
    parser.add_argument("--enrich-base-url", default=None)
    parser.add_argument("--enrich-api-key", default=None)
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip SFT deduplication (keep all non-trivial)")
    parser.add_argument("--dpo-only-dedup", action="store_true",
                        help="Only deduplicate DPO pairs, skip enrichment")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output, "clean.log"),
                                encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Load all SFT data
    all_sft = []
    all_dpo = []
    for d in args.input_dirs:
        sft_path = os.path.join(d, "sft_data.json")
        dpo_path = os.path.join(d, "dpo_pairs.json")
        source = os.path.basename(d)
        if os.path.exists(sft_path):
            data = json.load(open(sft_path, encoding="utf-8"))
            for item in data:
                item["_source"] = source
            all_sft.extend(data)
            logger.info(f"Loaded {len(data)} SFT from {source}")
        if os.path.exists(dpo_path):
            data = json.load(open(dpo_path, encoding="utf-8"))
            for item in data:
                item["_source"] = source
            all_dpo.extend(data)
            logger.info(f"Loaded {len(data)} DPO from {source}")

    logger.info(f"Total raw: {len(all_sft)} SFT, {len(all_dpo)} DPO")

    # --- Phase 1: Remove trivial samples ---
    non_trivial = []
    trivial_count = 0
    for s in all_sft:
        actions = extract_actions(s["messages"])
        if all(a.get("tool_name") == "none" for a in actions):
            trivial_count += 1
            continue
        non_trivial.append(s)
    logger.info(f"Phase 1: Removed {trivial_count} trivial none-only samples")

    # --- Phase 2: Deduplicate (optional) ---
    if args.no_dedup:
        unique_sft = non_trivial
        logger.info(f"Phase 2: Skipped dedup (--no-dedup), keeping all {len(unique_sft)}")
    else:
        seen_fps = {}
        unique_sft = []
        dup_count = 0
        for s in non_trivial:
            fp = get_action_fingerprint(s)
            if fp in seen_fps:
                dup_count += 1
                existing_idx = seen_fps[fp]
                existing = unique_sft[existing_idx]
                new_priority = 1 if s["_source"] in ("sft_collection_v5",
                                                       "sft_collection_v6",
                                                       "sft_collection_v7b") else 0
                old_priority = 1 if existing["_source"] in ("sft_collection_v5",
                                                             "sft_collection_v6",
                                                             "sft_collection_v7b") else 0
                if new_priority > old_priority:
                    unique_sft[existing_idx] = s
            else:
                seen_fps[fp] = len(unique_sft)
                unique_sft.append(s)
        logger.info(f"Phase 2: Removed {dup_count} duplicates, {len(unique_sft)} unique remain")

    # --- Phase 3: Clean format + rebuild observations ---
    system_msg = build_system_message()
    cleaned_sft = []
    replay_ok = 0
    replay_fail = 0

    for s in unique_sft:
        actions = extract_actions(s["messages"])
        scenario_id = s["scenario_id"]

        # Try to replay for fresh observations
        try:
            new_obs = replay_scenario(scenario_id, actions)
            replay_ok += 1
        except Exception as e:
            logger.warning(f"Replay failed for {scenario_id}: {e}")
            new_obs = None
            replay_fail += 1

        # Build clean messages
        messages = [{"role": "system", "content": system_msg}]
        orig_user_msgs = [m for m in s["messages"] if m["role"] == "user"]
        orig_asst_msgs = [m for m in s["messages"] if m["role"] == "assistant"]

        for i, action in enumerate(actions):
            # User message: use replayed observation if available, else original
            if new_obs and i < len(new_obs):
                user_content = new_obs[i]
            elif i < len(orig_user_msgs):
                user_content = orig_user_msgs[i]["content"]
            else:
                break

            messages.append({"role": "user", "content": user_content})

            # Assistant message: clean format
            action_json = json.dumps(action, ensure_ascii=False)
            messages.append({"role": "assistant", "content": action_json})

        cleaned_sft.append({
            "scenario_id": scenario_id,
            "messages": messages,
            "total_steps": len(actions),
            "_source": s["_source"],
        })

    logger.info(f"Phase 3: Rebuilt observations: {replay_ok} ok, {replay_fail} failed")

    # --- Phase 4: GPT enrichment (optional) ---
    if args.enrich_model and args.enrich_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=args.enrich_api_key,
                base_url=args.enrich_base_url,
            )
            client._model_name = args.enrich_model

            enriched = 0
            total_msgs = sum(
                len([m for m in s["messages"] if m["role"] == "assistant"])
                for s in cleaned_sft
            )
            logger.info(f"Phase 4: Enriching {total_msgs} assistant messages with {args.enrich_model}")

            for s in cleaned_sft:
                for j, m in enumerate(s["messages"]):
                    if m["role"] != "assistant":
                        continue
                    action_json = m["content"]
                    if action_json.strip().startswith('{"tool_name": "none"'):
                        continue

                    # Get preceding observation
                    obs = s["messages"][j - 1]["content"] if j > 0 else ""

                    enriched_content = enrich_with_reasoning(
                        obs, action_json, system_msg, client
                    )
                    m["content"] = enriched_content
                    enriched += 1

                    if enriched % 20 == 0:
                        logger.info(f"  Enriched {enriched}/{total_msgs} messages")

            logger.info(f"Phase 4: Enriched {enriched} assistant messages")
        except Exception as e:
            logger.error(f"Phase 4 failed: {e}")
    else:
        logger.info("Phase 4: Skipped (no enrich model specified)")

    # --- Phase 5: DPO dedup ---
    seen_dpo = set()
    unique_dpo = []
    for d in all_dpo:
        key = (
            d.get("scenario_id", ""),
            d.get("step", 0),
            str(d.get("chosen", ""))[:200],
            str(d.get("rejected", ""))[:200],
        )
        if key not in seen_dpo:
            seen_dpo.add(key)
            # Clean DPO format too
            if "chosen" in d and isinstance(d["chosen"], str):
                d["chosen"] = clean_assistant_content(d["chosen"])
            if "rejected" in d and isinstance(d["rejected"], str):
                d["rejected"] = clean_assistant_content(d["rejected"])
            unique_dpo.append(d)

    logger.info(f"Phase 5: DPO {len(all_dpo)} → {len(unique_dpo)} unique pairs")

    # --- Save ---
    # Remove internal fields
    for s in cleaned_sft:
        s.pop("_source", None)
    for d in unique_dpo:
        d.pop("_source", None)

    sft_path = os.path.join(args.output, "sft_data.json")
    dpo_path = os.path.join(args.output, "dpo_pairs.json")
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_sft, f, indent=2, ensure_ascii=False)
    with open(dpo_path, "w", encoding="utf-8") as f:
        json.dump(unique_dpo, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Final: {len(cleaned_sft)} SFT, {len(unique_dpo)} DPO")
    logger.info(f"Saved to {args.output}")

    # Stats
    scen_dist = Counter(s["scenario_id"] for s in cleaned_sft)
    logger.info(f"\nSFT per scenario:")
    for sid, cnt in sorted(scen_dist.items(), key=lambda x: -x[1]):
        logger.info(f"  {sid}: {cnt}")

    step_dist = Counter(s["total_steps"] for s in cleaned_sft)
    logger.info(f"\nSFT step distribution: {dict(sorted(step_dist.items()))}")


if __name__ == "__main__":
    main()
