"""gym-anm DomainConfig factory."""

from __future__ import annotations

from silr.core.config import DomainConfig

from .checkers import (
    ANMVoltageChecker,
    ANMBranchLoadingChecker,
    ANMStorageSoCChecker,
)
from .tools import create_anm_toolset
from .prompts import build_system_prompt, build_tool_schemas, get_valid_device_ids


def build_anm_domain_config(
    with_observer: bool = False,
    gating_policy: str = "progress",
    with_admission_criteria: bool = False,
    stall_budget: int | None = None,
) -> DomainConfig:
    """Build a DomainConfig for the gym-anm distribution-network domain.

    ANM defaults to ``gating_policy="progress"`` (recoverability-preserving
    admission) rather than the framework-wide ``"terminal"`` default: the
    ANM6-Easy action layer exposes per-device set-points, so stressed
    snapshots that need multi-device coordination (e.g. simultaneous
    PV / wind surge under light load) cannot be cleared by any single
    set-point change. Under ``"terminal"`` gating, the verifier rejects
    every single-device proposal and the recovery loop deadlocks even
    though each proposal is monotonically improving (empirically
    validated by ``scripts/anm_trajectory_probe.py``). ``"progress"``
    admits these steps while keeping the terminal-PASS = recovered
    semantics for training-reward and termination.

    Other domains (single-action-coverable recovery, e.g. the historical
    grid / cluster / finance tracks) can override back to ``"terminal"``.
    """
    create_observer = None
    if with_observer:
        from .observation import ANMObserver

        _wac = with_admission_criteria
        _sb = stall_budget

        def _make_observer(mgr):
            return ANMObserver(mgr, with_admission_criteria=_wac, stall_budget=_sb)
        create_observer = _make_observer

    return DomainConfig(
        domain_name="gym_anm",
        checkers=[
            ANMVoltageChecker(),
            ANMBranchLoadingChecker(),
            ANMStorageSoCChecker(),
        ],
        allowed_actions=frozenset(["set_generator_setpoint", "set_storage_setpoint"]),
        create_toolset=create_anm_toolset,
        build_system_prompt=build_system_prompt,
        build_tool_schemas=build_tool_schemas,
        get_valid_device_ids=get_valid_device_ids,
        create_observer=create_observer,
        gating_policy=gating_policy,
    )
