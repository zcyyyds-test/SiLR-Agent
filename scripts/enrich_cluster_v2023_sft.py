"""Prepend a concise CoT paragraph to each assistant JSON action.

Uses the OpenAI-compatible LemonAPI relay already configured for the
project. See memory/reference_api_config.md for endpoint + key names.

Enrichment is ASYNCHRONOUS with a bounded concurrency semaphore
(default 10) — sequential processing took ~75s/record × 200 = 4 hours.
Async 10-way brings this down to ~25 minutes at the same API cost.

Also provides `jsonl_to_json_array(src, dst)` helper — `train_sft.py`
uses `json.load(f)` which expects a JSON array at top level, NOT
per-line JSONL. Run after enrichment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REASONING_PROMPT = """You are writing a short chain-of-thought for a GPU-cluster
scheduler agent. Given the observation JSON and the chosen tool call,
output 1–2 sentences explaining WHY the tool call is the right next
action. Do not repeat the tool call JSON. Be specific about node IDs,
job IDs, qos class, and the constraint(s) being addressed."""


# Model id for LemonAPI Gemini relay — the [L] prefix is REQUIRED
# (without it returns 503 model_not_found). Can be overridden via env
# LEMON_API_MODEL if the relay catalog changes.
_DEFAULT_MODEL = "[L]gemini-3-flash-preview"


async def _enrich_one_async(
    obs: str, assistant: str, client, model: str, sem: asyncio.Semaphore,
) -> str:
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REASONING_PROMPT},
                {"role": "user",
                 "content": f"OBSERVATION:\n{obs}\n\nACTION:\n{assistant}"},
            ],
            temperature=0.3,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()


async def _enrich_record_async(
    rec: dict, client, model: str, sem: asyncio.Semaphore,
) -> dict:
    """Enrich all (user, assistant) pairs inside one record concurrently."""
    tasks = []
    indices = []
    msgs = rec["messages"]
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] != "user":
            continue
        j = i + 1
        if j >= len(msgs) or msgs[j]["role"] != "assistant":
            continue
        tasks.append(_enrich_one_async(
            msgs[i]["content"], msgs[j]["content"], client, model, sem))
        indices.append(j)

    if not tasks:
        return rec

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for j, r in zip(indices, results):
        if isinstance(r, Exception):
            logger.warning("enrich failed scenario=%s turn=%d: %s",
                           rec.get("scenario_id"), j, r)
            continue
        msgs[j]["content"] = f"{r}\n\n{msgs[j]['content']}"
    return rec


async def _enrich_async(
    inp: Path, out: Path, *,
    max_lines: int | None = None,
    concurrency: int = 10,
) -> int:
    # Import OpenAI only here — tests without API key never touch this path.
    from openai import AsyncOpenAI
    api_key = os.environ.get("LEMON_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LEMON_API_KEY env var is required to run enrichment.")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.environ.get("LEMON_API_BASE",
                                "https://new.lemonapi.site/v1"),
    )
    model = os.environ.get("LEMON_API_MODEL", _DEFAULT_MODEL)
    sem = asyncio.Semaphore(concurrency)

    # Load all records first (input is ~200 lines × ~20KB, fits in RAM)
    records = []
    with open(inp) as fin:
        for line in fin:
            if max_lines and len(records) >= max_lines:
                break
            records.append(json.loads(line))

    # Process all records concurrently (semaphore caps in-flight API calls)
    logger.info("enriching %d records with concurrency=%d ...",
                len(records), concurrency)
    enriched = await asyncio.gather(
        *(_enrich_record_async(r, client, model, sem) for r in records)
    )

    # Write in original order
    with open(out, "w") as fout:
        for rec in enriched:
            fout.write(json.dumps(rec) + "\n")
    return len(enriched)


def enrich(inp: Path, out: Path, *,
           max_lines: int | None = None,
           concurrency: int = 10) -> int:
    """Synchronous entry point — wraps the async impl for CLI / tests."""
    return asyncio.run(_enrich_async(inp, out,
                                     max_lines=max_lines,
                                     concurrency=concurrency))


def jsonl_to_json_array(src: Path, dst: Path) -> int:
    """Convert a JSONL file to a top-level JSON array.

    Required before feeding to `scripts/train_sft.py:49-50` which
    uses `json.load(f)` on the file.
    """
    src = Path(src)
    dst = Path(dst)
    records = [json.loads(line) for line in src.read_text().splitlines()
               if line.strip()]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(records, indent=2))
    return len(records)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True,
                   help="JSONL output path (enriched)")
    p.add_argument("--final-json", default=None,
                   help="Optional: also write JSON array format at this path "
                        "(required by train_sft.py)")
    p.add_argument("--max-lines", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=10,
                   help="Max concurrent in-flight API calls")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    n = enrich(Path(args.inp), Path(args.out),
               max_lines=args.max_lines,
               concurrency=args.concurrency)
    logger.info("enriched %d lines → %s", n, args.out)

    if args.final_json:
        total = jsonl_to_json_array(Path(args.out), Path(args.final_json))
        logger.info("converted JSONL → JSON array: %d records → %s",
                    total, args.final_json)


if __name__ == "__main__":
    sys.exit(main() or 0)
