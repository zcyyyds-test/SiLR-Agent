"""Rule-based failsafe strategies.

Triggered when the LLM fails consecutive verification checks.
Produces conservative, safe actions based on violation type.
"""

from __future__ import annotations

import logging
from typing import Optional

from silr.agent.types import Observation

logger = logging.getLogger(__name__)

# Voltage violation thresholds for adaptive failsafe sizing
_V_NOMINAL = 1.0
_V_LOW_THRESHOLD = 0.90
_V_HIGH_THRESHOLD = 1.10


class DefaultFailsafe:
    """Adaptive rule-based action selection.

    Strategy per violation type:
    - Frequency low → increase largest generator
    - Frequency high → decrease largest generator
    - Voltage low → shed load at bus WITH PQ load, amount scaled by gap
      (20/50/100/up to 300 MW depending on gap severity)
    - Line overload → shed load at bus WITH PQ load
    """

    def __init__(
        self,
        gen_ids: list,
        bus_ids: list,
        pq_bus_ids: list[int] | None = None,
    ):
        self._gen_ids = gen_ids
        self._bus_ids = bus_ids
        self._pq_bus_ids: set[int] = set(pq_bus_ids) if pq_bus_ids else set(bus_ids)

    def suggest_escalated(
        self, obs: Observation, last_rejected: Optional[dict] = None
    ) -> Optional[dict]:
        """Suggest an escalated action based on last rejected proposal.

        If the last rejected action was shed_load on a valid PQ bus,
        double its magnitude (min 100 MW). Otherwise fall back to suggest().
        """
        if last_rejected:
            tool = last_rejected.get("tool_name", "")
            params = last_rejected.get("params", {})
            if tool == "shed_load":
                bus_id = params.get("bus_id")
                prev_mw = params.get("amount_mw", 0)
                if bus_id is not None and bus_id in self._pq_bus_ids:
                    escalated_mw = max(prev_mw * 2, 100.0)
                    return {
                        "tool_name": "shed_load",
                        "params": {"bus_id": bus_id, "amount_mw": escalated_mw},
                    }
            elif tool == "adjust_gen":
                gen_id = params.get("gen_id")
                prev_delta = params.get("delta_p_mw", 0)
                if gen_id is not None and prev_delta != 0:
                    escalated_delta = prev_delta * 2
                    return {
                        "tool_name": "adjust_gen",
                        "params": {"gen_id": gen_id, "delta_p_mw": escalated_delta},
                    }
        return self.suggest(obs)

    def suggest(self, obs: Observation) -> Optional[dict]:
        """Suggest a conservative action based on current violations.

        Returns action dict or None if no suggestion.
        """
        if not obs.violations:
            return None

        # Prioritize by violation type
        freq_violations = [v for v in obs.violations if v["type"] == "frequency"]
        volt_violations = [v for v in obs.violations if v["type"] == "voltage"]
        line_violations = [v for v in obs.violations if v["type"] == "line_loading"]

        if freq_violations:
            return self._handle_frequency(freq_violations)
        if volt_violations:
            return self._handle_voltage(volt_violations)
        if line_violations:
            return self._handle_line_loading(line_violations)

        return None

    def _handle_frequency(self, violations: list[dict]) -> Optional[dict]:
        """Handle frequency violation with conservative gen adjustment."""
        if not self._gen_ids:
            return None
        avg_delta = sum(v["value"] for v in violations) / len(violations)
        if avg_delta < 0:
            return {
                "tool_name": "adjust_gen",
                "params": {"gen_id": self._gen_ids[0], "delta_p_mw": 10.0},
            }
        else:
            return {
                "tool_name": "adjust_gen",
                "params": {"gen_id": self._gen_ids[0], "delta_p_mw": -10.0},
            }

    def _handle_voltage(self, violations: list[dict]) -> Optional[dict]:
        """Handle voltage violation with adaptive load shedding.

        Amount scales with the voltage gap via _adaptive_amount():
          gap < 0.005  →  20 MW
          gap < 0.015  →  50 MW
          gap < 0.03   → 100 MW
          gap >= 0.03  → up to 300 MW (linear)

        Only targets buses that have PQ loads.
        Falls back to adjust_gen if no PQ bus is available.
        """
        for v in violations:
            device = v.get("device", "")
            if not device.startswith("Bus_"):
                continue
            try:
                bus_id = int(device.split("_")[1])
            except (ValueError, IndexError):
                continue

            # Compute adaptive amount based on voltage gap
            voltage = v.get("value", 0.0)
            gap = abs(_V_LOW_THRESHOLD - voltage) if voltage < _V_LOW_THRESHOLD else 0.0
            amount_mw = self._adaptive_amount(gap)

            # Prefer the violation bus if it has PQ load
            if bus_id in self._pq_bus_ids:
                return {
                    "tool_name": "shed_load",
                    "params": {"bus_id": bus_id, "amount_mw": amount_mw},
                }

            # Otherwise find nearest PQ bus
            if self._pq_bus_ids:
                fallback_bus = min(self._pq_bus_ids, key=lambda b: abs(b - bus_id))
                logger.info(
                    f"Failsafe: Bus {bus_id} has no PQ load, "
                    f"redirecting to Bus {fallback_bus}"
                )
                return {
                    "tool_name": "shed_load",
                    "params": {"bus_id": fallback_bus, "amount_mw": amount_mw},
                }

        # Fallback: gen adjustment
        if self._gen_ids:
            return {
                "tool_name": "adjust_gen",
                "params": {"gen_id": self._gen_ids[0], "delta_p_mw": 20.0},
            }
        return None

    def _handle_line_loading(self, violations: list[dict]) -> Optional[dict]:
        """Handle line overload with load shedding at a PQ bus."""
        if self._pq_bus_ids:
            target = sorted(self._pq_bus_ids)[0]
            return {
                "tool_name": "shed_load",
                "params": {"bus_id": target, "amount_mw": 20.0},
            }
        return None

    @staticmethod
    def _adaptive_amount(gap: float) -> float:
        """Scale failsafe MW amount by voltage gap.

        Shallow violations get lighter touch; deep ones get aggressive.
        """
        if gap < 0.005:
            return 20.0
        if gap < 0.015:
            return 50.0
        if gap < 0.03:
            return 100.0
        # Linear scale for extreme gaps, capped at 300 MW
        return min(100.0 + gap * 2000, 300.0)
