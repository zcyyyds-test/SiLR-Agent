"""Scenario JSON persistence + manager construction."""

from __future__ import annotations

import json
from pathlib import Path

from ..manager import ClusterV2023Manager


class ScenarioLoader:
    @staticmethod
    def save(scenario: dict, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(scenario, f, indent=2)

    @staticmethod
    def load(path: Path | str) -> dict:
        with open(Path(path)) as f:
            return json.load(f)

    @staticmethod
    def build_manager(scenario: dict) -> ClusterV2023Manager:
        return ClusterV2023Manager(
            nodes=scenario["nodes"],
            jobs=scenario["jobs"],
            assignments=scenario.get("assignments", {}),
        )

    @staticmethod
    def list_scenarios(dir_path: Path | str) -> list[Path]:
        return sorted(Path(dir_path).glob("*.json"))
