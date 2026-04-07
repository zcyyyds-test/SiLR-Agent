"""YAML scenario loader with fault injection and prestress support."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..simulator import SystemManager
from ..utils.unit_converter import mw_to_pu
from ..tools import create_toolset
from silr.exceptions import SiLRError

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass
class Scenario:
    """A loaded scenario definition."""
    id: str
    name: str
    difficulty: str
    fault_sequence: list[dict[str, Any]]
    prestress: dict[str, Any] = field(default_factory=dict)
    ground_truth: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ScenarioLoader:
    """Load scenario YAML files and apply faults to a SystemManager."""

    def __init__(self, definitions_dir: str | Path | None = None):
        if yaml is None:
            raise ImportError(
                "pyyaml required for scenarios. Install with: pip install 'silr[grid]'"
            )
        if definitions_dir is not None:
            self._dir = Path(definitions_dir)
        else:
            self._dir = Path(__file__).parent / "definitions"

    def list_scenarios(self) -> list[str]:
        """List available scenario YAML files."""
        return sorted(p.stem for p in self._dir.glob("*.yaml"))

    def load(self, scenario_id: str) -> Scenario:
        """Load a scenario definition from YAML."""
        path = self._dir / f"{scenario_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return Scenario(
            id=data["id"],
            name=data.get("name", scenario_id),
            difficulty=data.get("difficulty", "unknown"),
            fault_sequence=data.get("fault_sequence", []),
            prestress=data.get("prestress", {}),
            ground_truth=data.get("ground_truth"),
            metadata=data.get("metadata", {}),
        )

    def load_all(self) -> list[Scenario]:
        """Load all scenarios from definitions directory."""
        return [self.load(sid) for sid in self.list_scenarios()]

    @staticmethod
    def apply_prestress(manager: SystemManager, prestress: dict[str, Any]) -> None:
        """Apply prestress conditions to push system toward operational limits.

        Supports:
        - load_scale: multiply all PQ loads by this factor
        - gen_adjustments: list of {gen_id, delta_p_mw}
        """
        ss = manager.ss
        base = manager.base_mva

        # Scale loads
        load_scale = prestress.get("load_scale", 1.0)
        if load_scale != 1.0 and ss.PQ.n > 0:
            for i, idx in enumerate(ss.PQ.idx.v):
                old_p = float(ss.PQ.p0.v[i])
                old_q = float(ss.PQ.q0.v[i])
                ss.PQ.alter("p0", idx, old_p * load_scale)
                ss.PQ.alter("q0", idx, old_q * load_scale)
            logger.info(f"Applied load_scale={load_scale}")

        # Generator adjustments
        gen_adjs = prestress.get("gen_adjustments", [])
        tools = create_toolset(manager)
        for adj in gen_adjs:
            gen_id = adj["gen_id"]
            delta = adj["delta_p_mw"]
            result = tools["adjust_gen"].execute(gen_id=gen_id, delta_p_mw=delta)
            if result["status"] == "error":
                logger.warning(f"Prestress gen adjustment failed: {result['error']}")

    @staticmethod
    def apply_faults(manager: SystemManager, fault_sequence: list[dict]) -> None:
        """Apply fault sequence to the system.

        Supported fault types:
        - line_trip: trip a line (alter u=0)
        - gen_reduce: reduce generator output by percentage
        - load_increase: increase load at bus by percentage
        - bus_fault: inject three-phase fault (requires TDS)
        """
        ss = manager.ss
        base = manager.base_mva
        tools = create_toolset(manager)

        for fault in fault_sequence:
            ftype = fault["type"]
            target = fault.get("target_id")

            if ftype == "line_trip":
                result = tools["trip_line"].execute(line_id=target)
                if result["status"] == "error":
                    logger.warning(f"Fault line_trip failed: {result['error']}")
                else:
                    logger.info(f"Tripped line {target}")

            elif ftype == "gen_reduce":
                pct = fault.get("reduce_pct", 50)
                gen_id = target
                limits = manager.get_gen_limits(
                    manager.get_static_gen_for_syn(
                        gen_id, manager.get_gen_model_name(gen_id)
                    )
                )
                current_mw = limits["p0"] * base
                delta = -(current_mw * pct / 100)
                result = tools["adjust_gen"].execute(gen_id=gen_id, delta_p_mw=delta)
                if result["status"] == "error":
                    logger.warning(f"Fault gen_reduce failed: {result['error']}")
                else:
                    logger.info(f"Reduced gen {gen_id} by {pct}%")

            elif ftype == "load_increase":
                bus_id = target
                pct = fault.get("increase_pct", 50)
                pq_loads = manager.get_pq_on_bus(bus_id)
                for ld in pq_loads:
                    old_p = ld["p0"]
                    old_q = ld["q0"]
                    ss.PQ.alter("p0", ld["idx"], old_p * (1 + pct / 100))
                    ss.PQ.alter("q0", ld["idx"], old_q * (1 + pct / 100))
                logger.info(f"Increased load at bus {bus_id} by {pct}%")

            elif ftype == "bus_fault":
                tf = fault.get("tf", 0.1)
                tc = fault.get("tc", 0.2)
                tools["inject_fault"].execute(
                    fault_type="bus_fault",
                    target_id=target,
                    tf=tf,
                    tc=tc,
                )
                logger.info(f"Injected bus fault at {target}, tf={tf}, tc={tc}")

            else:
                logger.warning(f"Unknown fault type: {ftype}")

    @staticmethod
    def setup_episode(
        manager: SystemManager,
        scenario: Scenario,
    ) -> None:
        """Full episode setup: prestress → PFlow → faults → PFlow.

        After this call, the system is ready for the agent to begin recovery.
        """
        # 1. Apply prestress (before PFlow)
        if scenario.prestress:
            ScenarioLoader.apply_prestress(manager, scenario.prestress)

        # 2. Run initial PFlow with prestress
        converged = manager.solve()
        if not converged:
            raise SiLRError(
                f"PFlow did not converge after prestress for scenario '{scenario.id}'"
            )

        # 3. Apply faults
        ScenarioLoader.apply_faults(manager, scenario.fault_sequence)

        # 4. Run PFlow again to get post-fault steady state
        converged = manager.solve()
        if not converged:
            logger.warning(
                f"PFlow did not converge after faults for scenario '{scenario.id}'. "
                f"System may be severely disturbed."
            )
