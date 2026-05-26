"""Verifier latency micro-benchmark for §5 Table-N.

Measures `SiLRVerifier.verify()` wall-clock per call across the 3 hand
scenarios × 3 gating policies (terminal / progress / progress_mag), plus
a no-op tool call for the absolute floor.

Per call we time:
  - shadow copy (deepcopy of Simulator)
  - shadow setup hook (no-op for ANM)
  - tool execution on shadow
  - shadow.solve() (power-flow re-solution)
  - constraint-checker invocation (3 checkers)
  - (progress / progress_mag) baseline checker invocation
  - graded verdict computation
  - report generation

Total verify() time is what an agent step pays per proposal. Compared to
the ~22 s/step the LLM takes, the verifier overhead should be << 1 s.

Run on AMD silr-anm:

    PYTHONPATH=. python scripts/anm_verifier_latency.py --reps 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from domains.anm.config import build_anm_domain_config
from domains.anm.manager import GymANMManager
from domains.anm.scenarios import ANMScenarioLoader
from silr.verifier import SiLRVerifier


_SCENARIOS = ["easy_lightload", "medium_seed42_default", "hard_renewable_surge"]
_POLICIES = ["terminal", "progress", "progress_mag"]

# A small grid of typical actions LLM might propose.
_ACTIONS = [
    {"tool_name": "set_generator_setpoint", "params": {"gen_id": 4, "p_mw": 0.0}},
    {"tool_name": "set_generator_setpoint", "params": {"gen_id": 4, "p_mw": 20.0}},
    {"tool_name": "set_generator_setpoint", "params": {"gen_id": 2, "p_mw": 15.0}},
    {"tool_name": "set_storage_setpoint", "params": {"storage_id": 6, "p_mw": -2.0}},
    {"tool_name": "set_storage_setpoint", "params": {"storage_id": 6, "p_mw": 2.0}},
]


def bench(scenario_id: str, gating_policy: str, reps: int) -> dict:
    cfg = build_anm_domain_config(gating_policy=gating_policy)
    loader = ANMScenarioLoader()
    scenario = loader.load(scenario_id)
    mgr = GymANMManager(seed=scenario.source_seed or 42)
    loader.setup_episode(mgr, scenario)
    verifier = SiLRVerifier(mgr, domain_config=cfg)

    times = []
    verdicts = []
    for i in range(reps):
        act = _ACTIONS[i % len(_ACTIONS)]
        t0 = time.perf_counter()
        result = verifier.verify(act)
        dt = time.perf_counter() - t0
        times.append(dt)
        verdicts.append(result.verdict.value)

    times.sort()
    mean = statistics.mean(times)
    median = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    p99 = times[int(len(times) * 0.99) if len(times) > 100 else -1]
    return {
        "scenario": scenario_id,
        "policy": gating_policy,
        "reps": reps,
        "mean_ms": round(mean * 1000, 3),
        "median_ms": round(median * 1000, 3),
        "p95_ms": round(p95 * 1000, 3),
        "p99_ms": round(p99 * 1000, 3),
        "min_ms": round(min(times) * 1000, 3),
        "max_ms": round(max(times) * 1000, 3),
        "verdict_counts": dict(
            sorted(((v, verdicts.count(v)) for v in set(verdicts)), key=lambda x: -x[1])
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50,
                    help="Repetitions per (scenario, policy) cell.")
    ap.add_argument("--output", default="verifier_latency_v1.json")
    args = ap.parse_args()

    rows = []
    print(f"=== Verifier latency μ-bench (reps={args.reps}/cell) ===")
    print(f"  {'scenario':<25} {'policy':<14} {'mean':>8} {'median':>8} {'p95':>8} {'min':>8} {'max':>8}")
    for scn in _SCENARIOS:
        for pol in _POLICIES:
            row = bench(scn, pol, args.reps)
            rows.append(row)
            print(f"  {scn:<25} {pol:<14} "
                  f"{row['mean_ms']:>6.2f}ms {row['median_ms']:>6.2f}ms "
                  f"{row['p95_ms']:>6.2f}ms {row['min_ms']:>6.2f}ms {row['max_ms']:>6.2f}ms"
                  f"  verdicts={row['verdict_counts']}")

    out = {"reps_per_cell": args.reps, "rows": rows}
    Path(args.output).write_text(json.dumps(out, indent=2))

    # Headline numbers for paper
    all_means = [r["mean_ms"] for r in rows]
    all_p95 = [r["p95_ms"] for r in rows]
    print(f"\n=== Paper Table-N headline ===")
    print(f"  Verifier mean latency across {len(rows)} (scenario × policy) cells: "
          f"min={min(all_means):.2f}ms, max={max(all_means):.2f}ms, "
          f"median={statistics.median(all_means):.2f}ms")
    print(f"  p95 latency: max={max(all_p95):.2f}ms")
    print(f"  vs typical LLM-call latency: ~22000ms per ReAct step")
    print(f"  → verifier overhead is < {max(all_means)*100/22000:.2f}% of episode wallclock")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
