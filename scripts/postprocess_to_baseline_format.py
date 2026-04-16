"""Convert a Kimi-style SFT dataset (wrapped user + verbose Thought) into
baseline format (raw user JSON + Thought=JSON echo).

Baseline format reference:
  user: <raw compressed_json observation>
  assistant: Thought: {json_action}\\n\\n{json_action}

This preserves Kimi's action choices but strips verbose reasoning text,
letting us isolate "teacher quality" from "data format" as variables.
"""
import argparse
import json
import re
from pathlib import Path


def unwrap_user_content(content: str) -> str:
    """Strip the '## Step N — System Observation\\n{json}\\n...' wrapper back
    to bare compressed_json. If already raw JSON, return as-is."""
    if not content.startswith("## Step"):
        return content
    # Expected shape: '## Step N — System Observation\n\n{json}\n...trailing...'
    # Keep only the JSON body
    lines = content.split("\n")
    # Find the JSON object start
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("{") and '"positions"' in stripped:
            return stripped
    return content


def to_echo_assistant(content: str) -> str:
    """Convert assistant content to 'Thought: {json}\\n\\n{json}' echo form."""
    # Extract the JSON action — search for last bare JSON block
    # Content shape: 'Thought: <text>\n\n{json_action}' possibly with extra lines
    # Grab the last JSON object with "tool_name"
    matches = list(re.finditer(
        r'\{(?:[^{}]|\{[^{}]*\})*"tool_name"(?:[^{}]|\{[^{}]*\})*\}',
        content,
        re.DOTALL,
    ))
    if not matches:
        return content
    json_str = matches[-1].group(0)
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return content
    # Re-emit canonical JSON (match baseline spacing: "tool_name" with space after colon)
    canonical = json.dumps(obj, ensure_ascii=False)
    return f"Thought: {canonical}\n\n{canonical}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Source SFT json (Kimi wrapped/verbose)")
    ap.add_argument("--output", required=True, help="Output SFT json (baseline format)")
    args = ap.parse_args()

    src = json.load(open(args.input))
    out = []
    for conv in src:
        new_messages = []
        for m in conv["messages"]:
            if m["role"] == "user":
                new_messages.append({
                    "role": "user",
                    "content": unwrap_user_content(m["content"]),
                })
            elif m["role"] == "assistant":
                new_messages.append({
                    "role": "assistant",
                    "content": to_echo_assistant(m["content"]),
                })
            else:
                new_messages.append(m)
        out.append({
            "scenario_id": conv["scenario_id"],
            "messages": new_messages,
            "recovered": conv.get("recovered", True),
            "total_steps": conv.get("total_steps", len(new_messages)//2),
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Verify
    total_turns = sum(1 for c in out for m in c["messages"] if m["role"]=="assistant")
    sample = out[0]["messages"][1]["content"]
    print(f"Wrote {len(out)} conversations, {total_turns} assistant turns.")
    print(f"Sample assistant message (first conv, first turn):")
    print(f"  {sample[:200]}")


if __name__ == "__main__":
    main()
