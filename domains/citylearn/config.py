"""CityLearn district-storage DomainConfig factory."""

from __future__ import annotations

from silr.core.config import DomainConfig

from .checkers import (
    SoCChecker,
    DistrictImportChecker,
    DistrictExportChecker,
)
from .tools import create_citylearn_toolset
from .prompts import (
    build_citylearn_system_prompt,
    build_citylearn_tool_schemas,
    get_valid_device_ids,
)


def build_citylearn_domain_config(
    with_observer: bool = True,
    gating_policy: str = "terminal",
) -> DomainConfig:
    """Build a DomainConfig for the CityLearn district-storage domain.

    Unlike the finance track (whose compliance metrics are observer-only),
    CityLearn ships verifier checkers (SoC + district import/export), like the
    ANM track: a battery set-point change must be shadow-verified against all
    three constraint families before being applied.

    ``gating_policy`` defaults to ``"terminal"`` per the domain spec. The
    progress-family policies (``"progress"`` / ``"progress_mag"`` /
    ``"scalar_progress"``) are available for the ablation sweep — the
    coupled multi-building violations (e.g. a district export that needs two
    batteries to back off) often cannot be cleared by any single set-point
    change, so the sweep exercises those policies to study admission breadth.
    """
    create_observer = None
    if with_observer:
        from .observation import CityLearnObserver

        def _make_observer(mgr):
            return CityLearnObserver(mgr)

        create_observer = _make_observer

    return DomainConfig(
        domain_name="citylearn",
        checkers=[
            SoCChecker(),
            DistrictImportChecker(),
            DistrictExportChecker(),
        ],
        allowed_actions=frozenset({"set_building_setpoint"}),
        create_toolset=create_citylearn_toolset,
        build_system_prompt=build_citylearn_system_prompt,
        build_tool_schemas=build_citylearn_tool_schemas,
        get_valid_device_ids=get_valid_device_ids,
        create_observer=create_observer,
        gating_policy=gating_policy,
    )
