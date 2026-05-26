"""Audit paper-facing ANM evidence artifacts.

The paper has several tables assembled from separate sweeps. This script keeps
the submission numbers tied to concrete JSON artifacts and fails loudly when a
required artifact is missing, malformed, or no longer matches the paper-facing
claim.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STALE_MINED_MSG = (
    "eval_mined_v1.json has no scenario_manifest after the mined initial_soc "
    "replay fix; rerun mined benign eval before treating mined claims as "
    "paper-facing evidence."
)


@dataclass(frozen=True)
class Claim:
    label: str
    artifact: str
    path: tuple[Any, ...]
    expected: float
    tol: float = 1e-6


def load_json(name: str) -> Any:
    path = ROOT / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick(data: Any, path: tuple[Any, ...]) -> Any:
    cur = data
    for part in path:
        cur = cur[part]
    return cur


def assert_close(claim: Claim, data: dict[str, Any]) -> tuple[bool, str]:
    actual = pick(data[claim.artifact], claim.path)
    ok = math.isclose(float(actual), claim.expected, rel_tol=0.0, abs_tol=claim.tol)
    status = "OK" if ok else "FAIL"
    return ok, (
        f"{status}  {claim.label}: actual={float(actual):.6g} "
        f"expected={claim.expected:.6g}  source={claim.artifact}"
    )


CLAIMS = [
    Claim(
        "Table 1 hard terminal recovery deadlock",
        "eval_sweep_v1.json",
        ("aggregates", "hard_renewable_surge__terminal", "recovery_rate"),
        0.0,
    ),
    Claim(
        "Table 1 hard terminal final penalty",
        "eval_sweep_v1.json",
        ("aggregates", "hard_renewable_surge__terminal", "final_penalty", "mean"),
        25.848,
        1e-3,
    ),
    Claim(
        "Table 1 hard progress recovery",
        "eval_sweep_v1.json",
        ("aggregates", "hard_renewable_surge__progress", "recovery_rate"),
        1.0,
    ),
    Claim(
        "Table 1 hard progress_mag recovery",
        "eval_progmag_baseline_v1.json",
        ("aggregates", "hard_renewable_surge__progress_mag", "recovery_rate"),
        1.0,
    ),
    Claim(
        "Table 1 hard progress_mag proposals",
        "eval_progmag_baseline_v1.json",
        ("aggregates", "hard_renewable_surge__progress_mag", "proposals_per_episode", "mean"),
        4.0,
    ),
    Claim(
        "A1 medium succeeds under progress",
        "adversarial_v1.json",
        ("aggregates", "medium_seed42_default__prompt_injection", "attack_success_rate"),
        1.0,
    ),
    Claim(
        "A1 medium blocked by progress_mag",
        "adversarial_progmag_v1.json",
        ("aggregates", "medium_seed42_default__prompt_injection", "attack_success_rate"),
        0.0,
    ),
    Claim(
        "A1 medium blocked by progress_mag at N=5",
        "adversarial_n5_sanity_v1.json",
        ("aggregates", "medium_seed42_default__prompt_injection", "attack_success_rate"),
        0.0,
    ),
    Claim(
        "A3 medium detected by full L2+L3+L4",
        "adversarial_full_v1.json",
        ("aggregates", "medium_seed42_default__stall", "stall_termination_rate"),
        1.0,
    ),
    Claim(
        "A3 hard detected by full L2+L3+L4",
        "adversarial_full_v1.json",
        ("aggregates", "hard_renewable_surge__stall", "stall_termination_rate"),
        1.0,
    ),
    Claim(
        "Mined MPC-unsolved OFF final penalty",
        "eval_mined_v1.json",
        ("aggregates", "mined_mpc_unsolved_2_l2p0g1p0_s20__OFF", "final_penalty", "mean"),
        58.009,
        1e-3,
    ),
    Claim(
        "Mined MPC-unsolved progress recovery",
        "eval_mined_v1.json",
        ("aggregates", "mined_mpc_unsolved_2_l2p0g1p0_s20__progress", "recovery_rate"),
        1.0,
    ),
    Claim(
        "8B hard progress_mag final penalty",
        "eval_8b_v1.json",
        ("aggregates", "hard_renewable_surge__progress_mag", "final_penalty", "mean"),
        2.867,
        1e-3,
    ),
    Claim(
        "Offline graded reward rescues binary-degenerate groups",
        "trajectory_offline_v1.json",
        ("rescued",),
        10,
    ),
]


def latency_summary(data: dict[str, Any]) -> str:
    rows = data["verifier_latency_v1.json"]["rows"]
    mean_min = min(row["mean_ms"] for row in rows)
    mean_max = max(row["mean_ms"] for row in rows)
    p95_max = max(row["p95_ms"] for row in rows)
    return (
        "OK  Verifier latency: "
        f"mean range={mean_min:.3f}-{mean_max:.3f} ms, p95 max={p95_max:.3f} ms "
        "source=verifier_latency_v1.json"
    )


def stale_guard(data: dict[str, Any]) -> list[str]:
    mined = data.get("eval_mined_v1.json")
    if isinstance(mined, dict) and "scenario_manifest" not in mined:
        return [STALE_MINED_MSG]
    return []


def main() -> None:
    artifacts = sorted({claim.artifact for claim in CLAIMS} | {"verifier_latency_v1.json"})
    data: dict[str, Any] = {}
    for artifact in artifacts:
        data[artifact] = load_json(artifact)

    failures = 0
    print("=== ANM paper evidence audit ===")
    for msg in stale_guard(data):
        print(f"FAIL  {msg}")
        failures += 1
    for claim in CLAIMS:
        ok, line = assert_close(claim, data)
        print(line)
        failures += int(not ok)
    print(latency_summary(data))

    if failures:
        raise SystemExit(f"{failures} evidence checks failed")


if __name__ == "__main__":
    main()
