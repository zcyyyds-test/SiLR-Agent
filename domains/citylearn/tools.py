"""District-storage tools for SiLR verification.

2 tools: 1 observe + 1 action.
Per-building battery set-points are restricted to the discrete catalog in
``simulator.ACTIONS_PER_BUILDING`` to force coordinated multi-building
recovery (a single set-point change rarely clears a coupled district
import/export violation). All inherit from BaseTool for framework
compatibility.
"""

from __future__ import annotations

from silr.tools.base import BaseTool
from silr.exceptions import DeviceNotFoundError, ValidationError

from . import simulator as sim


class GetDistrictStatusTool(BaseTool):
    """Observe district state: per-building SoC/bounds/set-point, district
    import/export against feeder limits, peak demand, price, and cost."""

    name = "get_district_status"
    description = (
        "Get current district state: per-building battery SoC, SoC bounds, "
        "and pending set-point; district import/export vs. feeder limits; "
        "peak demand; price; and cost"
    )

    def _validate_params(self, **kwargs) -> None:
        pass

    def _run(self, **kwargs) -> dict:
        state = self.manager.system_state
        buildings = []
        for b in state["buildings"]:
            buildings.append({
                "id": b["id"],
                "index": b["index"],
                "soc_kwh": round(b["soc_kwh"], 4),
                "soc_min_kwh": b["soc_min_kwh"],
                "soc_max_kwh": b["soc_max_kwh"],
                "action_kw": b["action_kw"],
            })
        return {
            "t": state["t"],
            "price": state["price"],
            "buildings": buildings,
            "district_import_kw": round(state["district_import_kw"], 4),
            "district_import_limit_kw": state["district_import_limit_kw"],
            "district_export_kw": round(state["district_export_kw"], 4),
            "district_export_limit_kw": state["district_export_limit_kw"],
            "peak_import_kw": round(state["peak_import_kw"], 4),
            "cost": round(state["cost"], 4),
            "action_choices_kw": list(sim.ACTIONS_PER_BUILDING),
        }


class SetBuildingSetpointTool(BaseTool):
    """Set a building's battery set-point (discrete charge<0 / discharge>0 kW).

    Rejects (ValidationError -> Verdict.ERROR): missing params, non-numeric
    power, or a power_kw not in the discrete action catalog. Unknown building
    index -> DeviceNotFoundError (also Verdict.ERROR), keeping parameter typos
    out of safety-FAIL training signals.
    """

    name = "set_building_setpoint"
    description = (
        "Set a building's battery set-point in kW "
        "(negative = charge, positive = discharge). "
        f"Allowed values: {list(sim.ACTIONS_PER_BUILDING)}."
    )

    def _validate_params(self, building_index=None, power_kw=None, **kwargs) -> None:
        if building_index is None or power_kw is None:
            raise ValidationError("building_index and power_kw are required")
        try:
            idx = int(building_index)
        except (TypeError, ValueError):
            raise ValidationError(f"building_index must be an integer, got {building_index!r}")
        if idx < 0 or idx >= sim.N_BUILDINGS:
            raise DeviceNotFoundError(
                f"Building {building_index} not found. "
                f"Available indices: {list(range(sim.N_BUILDINGS))}"
            )
        try:
            p = float(power_kw)
        except (TypeError, ValueError):
            raise ValidationError(f"power_kw must be numeric, got {power_kw!r}")
        if p not in sim.ACTIONS_PER_BUILDING:
            raise ValidationError(
                f"power_kw={p} not in allowed set-points "
                f"{list(sim.ACTIONS_PER_BUILDING)}"
            )

    def _run(self, building_index=None, power_kw=None, **kwargs) -> dict:
        idx = int(building_index)
        p = float(power_kw)
        success = self.manager.set_building_action(idx, p)
        action = "charge" if p < 0 else ("discharge" if p > 0 else "idle")
        return {
            "building_index": idx,
            "power_kw": p,
            "action": action,
            "success": success,
            "message": (
                f"battery_{idx} set-point -> {p:+.1f} kW ({action})"
                if success
                else f"Failed to set battery_{idx} set-point to {p:+.1f} kW"
            ),
        }


def create_citylearn_toolset(manager) -> dict:
    """Create the district-storage toolset bound to a manager instance."""
    tools = [
        GetDistrictStatusTool(manager),
        SetBuildingSetpointTool(manager),
    ]
    return {t.name: t for t in tools}
