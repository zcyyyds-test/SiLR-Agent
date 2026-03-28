"""ObservationFormatter: collect system state from tools → compressed JSON for LLM."""

from __future__ import annotations

import json
import logging
from typing import Any

from silr.agent.types import Observation
from .tools import create_toolset
from .simulator import SystemManager

logger = logging.getLogger(__name__)


class ObservationFormatter:
    """Collects observations from 4 tools and compresses for LLM consumption.

    Normal devices → statistical summary only.
    Violated devices → detailed info.
    Includes action hints based on violation types.
    """

    def __init__(self, manager: SystemManager):
        self._manager = manager
        self._tools = create_toolset(manager)

    def observe(self) -> Observation:
        """Collect full observation from the current system state."""
        raw = {}
        violations = []

        # 1. Bus voltages
        v_result = self._tools["get_bus_voltage"].execute()
        raw["bus_voltage"] = v_result
        if v_result["status"] == "success":
            for bus in v_result["data"].get("buses", []):
                if bus.get("violation"):
                    violations.append({
                        "type": "voltage",
                        "device": f"Bus_{bus['bus_id']}",
                        "value": bus["v_pu"],
                        "detail": f"V={bus['v_pu']:.4f} p.u.",
                    })

        # 2. Frequency
        f_result = self._tools["get_frequency"].execute()
        raw["frequency"] = f_result
        if f_result["status"] == "success":
            for gen in f_result["data"].get("generators", []):
                if gen.get("violation"):
                    violations.append({
                        "type": "frequency",
                        "device": f"{gen['model']}_{gen['gen_id']}",
                        "value": gen["delta_f_hz"],
                        "detail": f"Δf={gen['delta_f_hz']:.4f} Hz",
                    })

        # 3. Line flow
        l_result = self._tools["get_line_flow"].execute()
        raw["line_flow"] = l_result
        if l_result["status"] == "success":
            for line in l_result["data"].get("lines", []):
                if line.get("violation"):
                    violations.append({
                        "type": "line_loading",
                        "device": f"Line_{line['line_id']}",
                        "value": line["loading_pct"],
                        "detail": f"Loading={line['loading_pct']:.1f}%",
                    })

        # 4. Stability check
        s_result = self._tools["check_stability"].execute()
        raw["stability"] = s_result
        is_stable = (
            s_result["status"] == "success"
            and s_result["data"].get("stable", False)
        )

        # Compress
        compressed = self._compress(raw, violations, is_stable)

        return Observation(
            raw=raw,
            compressed_json=json.dumps(compressed, ensure_ascii=False),
            violations=violations,
            is_stable=is_stable,
        )

    def _compress(
        self,
        raw: dict[str, Any],
        violations: list[dict],
        is_stable: bool,
    ) -> dict:
        """Create compact JSON representation for LLM prompt."""
        obs = {"stable": is_stable, "violations": [], "summary": {}}

        # Voltage summary
        v_data = raw.get("bus_voltage", {})
        if v_data.get("status") == "success":
            buses = (v_data.get("data") or {}).get("buses", [])
            v_vals = [b["v_pu"] for b in buses]
            if v_vals:
                obs["summary"]["voltage"] = {
                    "min_pu": round(min(v_vals), 4),
                    "max_pu": round(max(v_vals), 4),
                    "n_buses": len(v_vals),
                }

        # Frequency summary
        f_data = raw.get("frequency", {})
        if f_data.get("status") == "success":
            gens = (f_data.get("data") or {}).get("generators", [])
            if gens:
                delta_fs = [abs(g["delta_f_hz"]) for g in gens]
                obs["summary"]["frequency"] = {
                    "max_abs_delta_f_hz": round(max(delta_fs), 4),
                    "n_generators": len(gens),
                }

        # Line loading summary
        l_data = raw.get("line_flow", {})
        if l_data.get("status") == "success":
            lines = (l_data.get("data") or {}).get("lines", [])
            active_lines = [ln for ln in lines if ln["status"] == 1]
            loadings = [
                ln["loading_pct"] for ln in active_lines
                if ln["loading_pct"] is not None
            ]
            if loadings:
                obs["summary"]["line_loading"] = {
                    "max_loading_pct": round(max(loadings), 1),
                    "n_active_lines": len(active_lines),
                }

            # Tripped lines
            tripped = [ln for ln in lines if ln["status"] == 0]
            if tripped:
                obs["summary"]["tripped_lines"] = [
                    {"line_id": ln["line_id"],
                     "from_bus": ln["from_bus"],
                     "to_bus": ln["to_bus"]}
                    for ln in tripped
                ]

        # Detailed violations
        for v in violations:
            obs["violations"].append(v)

        # Action hints based on violation types and system state
        hints = self._generate_hints(violations, obs)
        if hints:
            obs["action_hints"] = hints

        return obs

    @staticmethod
    def _generate_hints(violations: list[dict], obs: dict) -> list[str]:
        """Generate action hints based on active violations and system state."""
        v_types = {v["type"] for v in violations}
        hints = []

        # Tripped lines hint (highest priority)
        tripped = obs.get("summary", {}).get("tripped_lines", [])
        if tripped:
            line_ids = ", ".join(t["line_id"] for t in tripped)
            hints.append(
                f"Tripped lines detected: {line_ids}. "
                "Consider: close_line to restore topology before adjust_gen or shed_load."
            )

        if "voltage" in v_types:
            low_v = [v for v in violations if v["type"] == "voltage" and v["value"] < 0.90]
            high_v = [v for v in violations if v["type"] == "voltage" and v["value"] > 1.10]
            if low_v:
                hints.append(
                    "Low voltage detected. Consider: adjust_gen (increase nearby gen output) "
                    "or shed_load (reduce load at affected bus)."
                )
            if high_v:
                hints.append(
                    "High voltage detected. Consider: adjust_gen (decrease nearby gen output)."
                )

        if "frequency" in v_types:
            low_f = [v for v in violations if v["type"] == "frequency" and v["value"] < 0]
            high_f = [v for v in violations if v["type"] == "frequency" and v["value"] > 0]
            if low_f:
                hints.append(
                    "Low frequency (generation deficit). Consider: adjust_gen (increase output) "
                    "or shed_load (reduce demand)."
                )
            if high_f:
                hints.append(
                    "High frequency (generation excess). Consider: adjust_gen (decrease output)."
                )

        if "line_loading" in v_types:
            hints.append(
                "Line overloaded. Consider: adjust_gen (redistribute generation) "
                "or shed_load (reduce load near overloaded line)."
            )

        return hints
