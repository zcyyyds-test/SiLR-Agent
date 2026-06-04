"""CityLearn district battery-storage domain for SiLR.

An ANM-isomorphic district-storage recovery testbed derived from the CityLearn
Challenge 2022 (Phase-1) 3-building summer-day profile. A single hour is fixed
and the district starts in a constraint-violating state (battery SoC out of
bounds, or district feeder import/export over limit). The agent adjusts
per-building battery set-points until SoC bounds and the district import/export
limits are all satisfied. Time does not advance — used to validate SiLR
shadow-execution verification on a second public-benchmark physics domain.
"""

from .manager import CityLearnManager
from .checkers import (
    SoCChecker,
    DistrictImportChecker,
    DistrictExportChecker,
)
from .tools import create_citylearn_toolset
from .config import build_citylearn_domain_config
from .scenarios import CityLearnScenario, CityLearnScenarioLoader, SCENARIOS

__all__ = [
    "CityLearnManager",
    "SoCChecker",
    "DistrictImportChecker",
    "DistrictExportChecker",
    "create_citylearn_toolset",
    "build_citylearn_domain_config",
    "CityLearnScenario",
    "CityLearnScenarioLoader",
    "SCENARIOS",
]
