"""Build a CPU-only experiment coverage matrix for ANM/WISE26.

This script reads existing JSON artifacts and the frozen scenario library. It
does not call an LLM or touch a GPU. The output answers two practical questions:

1. Which scenario/policy/attack cells already have evidence?
2. Which GPU jobs should be queued next once compute is allowed?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from domains.anm import ANMScenarioLoader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments" / "anm_experiment_matrix.md"


def load_json(name: str) -> dict[str, Any] | None:
    path = ROOT / name
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def artifact_status() -> list[tuple[str, str, int, int]]:
    rows = []
    for name in sorted(ROOT.glob("*_v1.json")):
        if name.name.startswith("._"):
            continue
        try:
            data = json.loads(name.read_text(encoding="utf-8"))
        except Exception:
            rows.append((name.name, "BROKEN", 0, 0))
            continue
        episodes = len(data.get("episodes", [])) if isinstance(data, dict) else 0
        aggregates = len(data.get("aggregates", {})) if isinstance(data, dict) else 0
        rows.append((name.name, "OK", episodes, aggregates))
    return rows


def _float_dict(data: dict[int, float] | None) -> dict[str, float] | None:
    if data is None:
        return None
    return {str(k): float(v) for k, v in sorted(data.items())}


def _scenario_record(s: Any) -> dict[str, Any]:
    return {
        "id": s.id,
        "difficulty": s.difficulty,
        "P_load": _float_dict(s.P_load),
        "P_pot": _float_dict(s.P_pot),
        "initial_P_set": _float_dict(s.initial_P_set),
        "initial_Q_set": _float_dict(s.initial_Q_set),
        "initial_soc": _float_dict(s.initial_soc),
        "source_seed": s.source_seed,
        "source_step": s.source_step,
    }


def _record_hash(record: dict[str, Any]) -> str:
    blob = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def current_scenario_manifest() -> dict[str, dict[str, Any]]:
    out = {}
    for s in ANMScenarioLoader().load_all():
        record = _scenario_record(s)
        out[s.id] = {"sha256": _record_hash(record), "record": record}
    return out


def scenario_rows() -> list[dict[str, str]]:
    rows = []
    for s in ANMScenarioLoader().load_all():
        rows.append(
            {
                "id": s.id,
                "difficulty": s.difficulty,
                "source": "mined" if s.id.startswith("mined_") else "hand",
                "initial_soc": "yes" if s.initial_soc else "no",
            }
        )
    return rows


def _cell_is_current(
    source_name: str,
    data: dict[str, Any],
    scenario: str,
    current_manifest: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    manifest = data.get("scenario_manifest")
    if isinstance(manifest, dict) and scenario in manifest:
        got = manifest[scenario]
        got_hash = got.get("sha256") if isinstance(got, dict) else got
        expected = current_manifest.get(scenario, {}).get("sha256")
        if got_hash == expected:
            return True, ""
        return False, f"{source_name}: scenario manifest mismatch"
    if scenario.startswith("mined_"):
        return False, (
            f"{source_name}: no scenario manifest after mined initial_soc replay fix"
        )
    return True, ""


def benign_matrix() -> tuple[
    list[str],
    list[str],
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
]:
    scenarios = [r["id"] for r in scenario_rows()]
    policies = ["OFF", "terminal", "progress", "progress_mag"]
    matrix: dict[tuple[str, str], str] = {}
    stale: dict[tuple[str, str], str] = {}
    current_manifest = current_scenario_manifest()

    sources = [
        ("eval_sweep_v1.json", load_json("eval_sweep_v1.json")),
        ("eval_progmag_baseline_v1.json", load_json("eval_progmag_baseline_v1.json")),
        ("eval_mined_v1.json", load_json("eval_mined_v1.json")),
        ("eval_8b_v1.json", load_json("eval_8b_v1.json")),
    ]
    for source_name, data in sources:
        if not data:
            continue
        for key, agg in data.get("aggregates", {}).items():
            if "__" not in key:
                continue
            scenario, policy = key.rsplit("__", 1)
            if scenario not in scenarios:
                continue
            if policy not in policies:
                continue
            ok, reason = _cell_is_current(source_name, data, scenario, current_manifest)
            if not ok:
                stale[(scenario, policy)] = f"STALE: {reason}"
                continue
            if (scenario, policy) in matrix:
                # Keep the primary 14B / mined-library evidence in the main
                # matrix; 8B is cross-scale support, not the headline source.
                continue
            rec = agg.get("recovery_rate", "?")
            pen = agg.get("final_penalty", {}).get("mean", "?")
            matrix[(scenario, policy)] = f"{source_name}: rec={rec}, pen={pen}"
    return scenarios, policies, matrix, stale


def attack_matrix() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    sources = [
        ("progress", "adversarial_v1.json"),
        ("progress_mag", "adversarial_progmag_v1.json"),
        ("progress+L4", "adversarial_stallmit_v1.json"),
        ("progress_mag+L4", "adversarial_full_v1.json"),
        ("progress_mag N=5", "adversarial_n5_sanity_v1.json"),
    ]
    for policy, name in sources:
        data = load_json(name)
        if not data:
            rows.append((policy, name, "MISSING", ""))
            continue
        for key, agg in data.get("aggregates", {}).items():
            if "__" not in key:
                continue
            scenario, attack = key.rsplit("__", 1)
            metric = (
                f"attack={agg.get('attack_success_rate')}, "
                f"rec={agg.get('recovery_rate')}, "
                f"stall={agg.get('stall_termination_rate', '-')}"
            )
            rows.append((policy, scenario, attack, metric))
    return rows


def next_jobs(
    scenarios: list[str],
    policies: list[str],
    matrix: dict[tuple[str, str], str],
) -> list[dict[str, str]]:
    mined = [s for s in scenarios if s.startswith("mined_")]
    missing_mined_policies = [
        p for p in policies if any((s, p) not in matrix for s in mined)
    ]
    jobs = []
    if missing_mined_policies:
        smoke_scenario = "mined_single_action_1_l1p0g1p0_s18"
        if smoke_scenario not in mined and mined:
            smoke_scenario = mined[0]
        jobs.append(
            {
                "id": "anm_mined_refresh_smoke_n1",
                "question": (
                    "Before the full mined refresh, verify the vLLM endpoint, "
                    "tool parsing, verifier path, and new scenario manifest on "
                    "one corrected mined scenario."
                ),
                "command": (
                    "PYTHONPATH=. python scripts/anm_eval_sweep.py "
                    "--base-url http://localhost:8001/v1 --model qwen3-14b "
                    f"--scenarios {smoke_scenario} "
                    "--policies progress_mag --reps 1 --max-steps 6 "
                    "--output eval_mined_smoke_v1.json"
                ),
            }
        )
        jobs.append(
            {
                "id": "anm_mined_benign_refresh_n3",
                "question": (
                    "Refresh mined-library benign evidence after the initial_soc "
                    "replay fix; stale mined cells must not be mixed with new baselines."
                ),
                "command": (
                    "PYTHONPATH=. python scripts/anm_eval_sweep.py "
                    "--base-url http://localhost:8001/v1 --model qwen3-14b "
                    "--scenarios "
                    + " ".join(mined)
                    + " --policies "
                    + " ".join(missing_mined_policies)
                    + " --reps 3 --max-steps 6 "
                    "--output eval_mined_refresh_v1.json"
                ),
            }
        )
    jobs.append(
        {
            "id": "anm_adversarial_full_n5",
            "question": "Do the A1/A3 mitigation claims hold with N=5 rather than N=3?",
            "command": (
                "PYTHONPATH=. python scripts/anm_adversarial_sweep.py "
                "--base-url http://localhost:8002/v1 --model qwen3-14b-adv "
                "--gating-policy progress_mag --stall-budget 2 "
                "--attacks prompt_injection stall stall_rag "
                "--reps 5 --output adversarial_full_n5_v1.json"
            ),
        }
    )
    jobs.append(
        {
            "id": "anm_grpo_sanity",
            "question": "Can the graded verdict improve policy behavior in a tiny online GRPO sanity run?",
            "command": "Design only for now; do not launch until benign/attack matrices are complete.",
        }
    )
    return jobs


def render() -> str:
    scenarios, policies, matrix, stale = benign_matrix()
    missing = [
        (s, p)
        for s in scenarios
        for p in policies
        if (s, p) not in matrix
    ]

    lines: list[str] = []
    lines.append("# ANM Experiment Matrix")
    lines.append("")
    lines.append("CPU-only generated report. No LLM or GPU call was made.")
    lines.append("STALE benign cells are treated as missing for the next-job queue.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("| Artifact | Status | Episodes | Aggregate Cells |")
    lines.append("|---|---:|---:|---:|")
    for name, status, episodes, aggregates in artifact_status():
        lines.append(f"| `{name}` | {status} | {episodes} | {aggregates} |")
    lines.append("")
    lines.append("## Scenario Library")
    lines.append("")
    lines.append("| Scenario | Difficulty | Source | Initial SoC Override |")
    lines.append("|---|---|---|---:|")
    for r in scenario_rows():
        lines.append(
            f"| `{r['id']}` | {r['difficulty']} | {r['source']} | "
            f"{r['initial_soc']} |"
        )
    lines.append("")
    lines.append("## Benign Coverage")
    lines.append("")
    lines.append("| Scenario | OFF | terminal | progress | progress_mag |")
    lines.append("|---|---|---|---|---|")
    for s in scenarios:
        cells = [matrix.get((s, p), stale.get((s, p), "MISSING")) for p in policies]
        lines.append("| `" + s + "` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Missing benign cells: {len(missing)}")
    for s, p in missing:
        detail = stale.get((s, p))
        suffix = f" — {detail}" if detail else ""
        lines.append(f"- `{s}` x `{p}`{suffix}")
    lines.append("")
    lines.append("## Attack Coverage")
    lines.append("")
    lines.append("| Policy | Scenario | Attack | Metrics |")
    lines.append("|---|---|---|---|")
    for policy, scenario, attack, metric in attack_matrix():
        lines.append(f"| {policy} | `{scenario}` | `{attack}` | {metric} |")
    lines.append("")
    lines.append("## Next GPU Jobs To Queue Later")
    lines.append("")
    for job in next_jobs(scenarios, policies, matrix):
        lines.append(f"### {job['id']}")
        lines.append("")
        lines.append(f"Question: {job['question']}")
        lines.append("")
        lines.append("```bash")
        lines.append(job["command"])
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    report = render()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
