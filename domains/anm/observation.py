"""gym-anm domain observer: compresses grid state for LLM consumption."""

from __future__ import annotations

import json
import math

from silr.agent.observation import BaseObserver
from silr.agent.types import Observation
from .checkers import (
    ANMVoltageChecker,
    ANMBranchLoadingChecker,
    ANMStorageSoCChecker,
)


class ANMObserver(BaseObserver):
    """Observer for the gym-anm distribution-network domain.

    Runs the voltage / branch-loading / SoC checkers and emits a compact JSON
    summary highlighting out-of-range buses, overloaded branches, and storage
    near SoC bounds. Also surfaces actionable device bounds (generator P_pot,
    p_min/p_max, q_min/q_max, soc_min/soc_max) so the LLM can infer the
    feasible action range without trial-and-error.
    """

    def __init__(self, manager, with_admission_criteria: bool = False,
                 magnitude_alpha: float = 1.05, magnitude_floor: float = 1e-3,
                 stall_budget: int | None = None):
        self._manager = manager
        self._checkers = [
            ANMVoltageChecker(),
            ANMBranchLoadingChecker(),
            ANMStorageSoCChecker(),
        ]
        self._with_admission = with_admission_criteria
        self._mag_alpha = magnitude_alpha
        self._mag_floor = magnitude_floor
        self._stall_budget = stall_budget

    @staticmethod
    def _severity_score(check_results) -> float:
        """Same Σ|v - limit| metric the L3 verifier uses for admission."""
        score = 0.0
        for cr in check_results:
            for v in cr.violations:
                try:
                    val = float(v.value)
                    lim = float(v.limit)
                except (TypeError, ValueError):
                    score += 1.0
                    continue
                if not (math.isfinite(val) and math.isfinite(lim)):
                    score += 1e6
                    continue
                score += abs(val - lim)
        return score

    def observe(self) -> Observation:
        sim = self._manager.system_state

        violations = []
        checker_summaries = {}
        for checker in self._checkers:
            cr = checker.check(sim, self._manager.base_mva)
            checker_summaries[checker.name] = cr.summary
            for v in cr.violations:
                violations.append(
                    {
                        "type": v.constraint_type,
                        "device": v.device_id,
                        "detail": v.detail,
                        "severity": v.severity,
                    }
                )

        # stressed margin = 20% of the bus's own [v_min, v_max] window,
        # so it scales with the checker's actual thresholds instead of a
        # fixed 0.02 pu that disconnects on tighter envelopes.
        stressed_buses = []
        for bid, bus in sim.buses.items():
            v = abs(bus.v)
            if not math.isfinite(v):
                continue
            v = float(v)
            vmin, vmax = float(bus.v_min), float(bus.v_max)
            margin = 0.2 * (vmax - vmin)
            if v < vmin + margin or v > vmax - margin:
                stressed_buses.append({"bus": bid, "v_pu": round(v, 4)})

        stressed_branches = []
        for br in sim.branches.values():
            rate = float(br.rate)
            if rate <= 0:
                continue
            s = float(br.s_apparent_max)
            if not math.isfinite(s):
                stressed_branches.append(
                    {"branch": f"{br.f_bus}-{br.t_bus}", "loading_pct": None}
                )
                continue
            loading = abs(s) / rate * 100.0
            if loading > 80.0:
                stressed_branches.append(
                    {"branch": f"{br.f_bus}-{br.t_bus}", "loading_pct": round(loading, 1)}
                )

        # Actionable device bounds for the LLM, in MW/MVAr (action's unit).
        base = self._manager.base_mva
        gen_bounds = [
            {
                "gen_id": g,
                "P_pot_mw": round(float(self._manager._P_pot.get(g, 0.0)), 4),
                "p_range_mw": [
                    float(sim.devices[g].p_min) * base,
                    float(sim.devices[g].p_max) * base,
                ],
                "q_range_mvar": [
                    float(sim.devices[g].q_min) * base,
                    float(sim.devices[g].q_max) * base,
                ],
            }
            for g in self._manager._gen_ids
        ]
        storage_bounds = [
            {
                "storage_id": s,
                "soc": round(float(sim.devices[s].soc), 4),
                "soc_range": [
                    float(sim.devices[s].soc_min),
                    float(sim.devices[s].soc_max),
                ],
                "p_range_mw": [
                    float(sim.devices[s].p_min) * base,
                    float(sim.devices[s].p_max) * base,
                ],
                "q_range_mvar": [
                    float(sim.devices[s].q_min) * base,
                    float(sim.devices[s].q_max) * base,
                ],
            }
            for s in self._manager._des_ids
        ]

        compressed = {
            "stressed_buses": stressed_buses,
            "stressed_branches": stressed_branches,
            "generators": gen_bounds,
            "storage": storage_bounds,
            "checkers": checker_summaries,
            "n_violations": len(violations),
        }

        # Progress certificate: inject L2 + L3 (+ L4) admission criteria
        # into the structured observation so the LLM can self-filter
        # before proposing. This tests whether forward-communicating the
        # apply-gate predicates reduces verifier rejection rate.
        if self._with_admission:
            check_results = [
                ck.check(sim, self._manager.base_mva) for ck in self._checkers
            ]
            baseline_n = sum(len(cr.violations) for cr in check_results)
            baseline_types = sorted({cr.checker_name for cr in check_results if not cr.passed})
            baseline_severity = self._severity_score(check_results)
            magnitude_ceiling = max(
                self._mag_alpha * baseline_severity,
                baseline_severity + self._mag_floor,
            )
            compressed["admission_criteria"] = {
                "L2_count_ceiling": baseline_n,
                "L2_forbidden_new_violation_types_outside": baseline_types,
                "L3_severity_baseline": round(baseline_severity, 3),
                "L3_severity_ceiling": round(magnitude_ceiling, 3),
                "L4_stall_budget": self._stall_budget,
                "note": (
                    "Your next tool-call is admitted iff post-action state "
                    f"has n_violations <= {baseline_n}, no new violation "
                    f"type outside {baseline_types}, and severity_score "
                    f"<= {magnitude_ceiling:.3f}. Otherwise rejected. "
                    "Choose action that monotonically descends both."
                ),
            }

        # raw is annotated dict[str, Any] in silr.agent.types.Observation;
        # wrap the live Simulator so downstream dict-style access does not crash.
        return Observation(
            raw={"simulator": sim},
            compressed_json=json.dumps(compressed, separators=(",", ":")),
            violations=violations,
            is_stable=len(violations) == 0,
        )
