"""DomainConfig factory for the SWE (code repair) domain."""
from __future__ import annotations

from silr.core.config import DomainConfig

from .checkers import RegressionChecker, TargetTestChecker
from .observation import SWEObserver
from .prompts import build_swe_system_prompt, build_swe_tool_schemas
from .tools import create_swe_toolset


def build_swe_domain_config(with_observer: bool = True) -> DomainConfig:
    return DomainConfig(
        domain_name="swe_code_repair",
        checkers=[RegressionChecker(), TargetTestChecker()],
        allowed_actions=frozenset(["localize", "patch"]),
        create_toolset=create_swe_toolset,
        create_observer=(lambda mgr: SWEObserver(mgr)) if with_observer else None,
        build_system_prompt=build_swe_system_prompt,
        build_tool_schemas=build_swe_tool_schemas,
    )
