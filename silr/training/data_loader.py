"""Load trajectory data into HuggingFace Dataset for SFT / DPO training.

Input contract: files produced by TrajectoryRecorder.save()
  - sft_data.jsonl (or .json): {scenario_id, messages: [{role, content}], recovered, total_steps}
  - dpo_pairs.jsonl (or .json): {scenario_id, step, prompt, chosen, rejected}

This loader requires an explicit system_prompt parameter — there is no
frozen prompt, since different domains have different system prompts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrainingDataLoader:
    """Load trajectory JSON files and convert to HuggingFace Dataset format."""

    def __init__(self, data_dir: str | Path):
        self._dir = Path(data_dir)
        self._sft_raw: list[dict] | None = None
        self._dpo_raw: list[dict] | None = None

    @staticmethod
    def _load_json_or_jsonl(path: Path) -> list[dict]:
        """Load data from either .json or .jsonl file."""
        jsonl_path = path.with_suffix(".jsonl")
        json_path = path.with_suffix(".json")

        if jsonl_path.exists():
            data = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        elif json_path.exists():
            with open(json_path) as f:
                return json.load(f)
        else:
            raise FileNotFoundError(
                f"Neither {jsonl_path} nor {json_path} found"
            )

    def _load_sft_raw(self) -> list[dict]:
        if self._sft_raw is None:
            self._sft_raw = self._load_json_or_jsonl(self._dir / "sft_data")
        return self._sft_raw

    def _load_dpo_raw(self) -> list[dict]:
        if self._dpo_raw is None:
            self._dpo_raw = self._load_json_or_jsonl(self._dir / "dpo_pairs")
        return self._dpo_raw

    def load_sft_dataset(
        self,
        system_prompt: str,
        upsample_min: int = 0,
    ):
        """Load SFT data as HuggingFace Dataset.

        Each row: {"messages": [{"role":"system",...}, {"role":"user",...}, ...]}
        Prepends system prompt to each conversation.

        Args:
            system_prompt: Domain-specific system prompt (required).
            upsample_min: If > 0, repeat under-represented scenarios.
        """
        from datasets import Dataset

        raw = self._load_sft_raw()

        rows = []
        for conv in raw:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conv["messages"])
            rows.append({
                "messages": messages,
                "_scenario_id": conv.get("scenario_id", "unknown"),
            })

        if upsample_min > 0:
            by_scenario: dict[str, list[dict]] = {}
            for row in rows:
                by_scenario.setdefault(row["_scenario_id"], []).append(row)

            upsampled: list[dict] = []
            for sid, srows in sorted(by_scenario.items()):
                n = len(srows)
                if n < upsample_min:
                    repeats = (upsample_min + n - 1) // n
                    srows = (srows * repeats)[:upsample_min]
                    logger.info(f"Upsampled {sid}: {n} → {len(srows)}")
                upsampled.extend(srows)
            rows = upsampled

        for row in rows:
            row.pop("_scenario_id", None)

        ds = Dataset.from_list(rows)
        logger.info(f"Loaded SFT dataset: {len(ds)} conversations")
        return ds

    def load_dpo_dataset(self, system_prompt: str):
        """Load DPO pairs as HuggingFace Dataset in TRL format.

        Each row: {
            "prompt": [{"role":"system",...}, {"role":"user",...}],
            "chosen": [{"role":"assistant", "content": ...}],
            "rejected": [{"role":"assistant", "content": ...}],
        }

        Args:
            system_prompt: Domain-specific system prompt (required).
        """
        from datasets import Dataset

        raw = self._load_dpo_raw()

        rows = []
        for pair in raw:
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pair["prompt"]},
            ]
            chosen_messages = [
                {"role": "assistant", "content": pair["chosen"]},
            ]
            rejected_messages = [
                {"role": "assistant", "content": pair["rejected"]},
            ]
            rows.append({
                "prompt": prompt_messages,
                "chosen": chosen_messages,
                "rejected": rejected_messages,
            })

        ds = Dataset.from_list(rows)
        logger.info(f"Loaded DPO dataset: {len(ds)} pairs")
        return ds

    def stats(self) -> dict[str, Any]:
        """Return dataset statistics."""
        sft = self._load_sft_raw()
        dpo = self._load_dpo_raw()

        sft_total_turns = sum(len(c["messages"]) for c in sft)
        scenarios = set()
        for c in sft:
            scenarios.add(c.get("scenario_id", "unknown"))
        for p in dpo:
            scenarios.add(p.get("scenario_id", "unknown"))

        return {
            "sft_conversations": len(sft),
            "sft_total_turns": sft_total_turns,
            "dpo_pairs": len(dpo),
            "unique_scenarios": len(scenarios),
            "scenarios": sorted(scenarios),
        }
