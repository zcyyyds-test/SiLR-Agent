"""Constants and domain configuration factory for the power grid domain.

Merges grid-specific constants (thresholds, TDS defaults) with the
DomainConfig factory that wires all grid domain components together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silr.core.config import DomainConfig

# ── System base ──────────────────────────────────────────────────────
SYSTEM_BASE_MVA = 100.0
SYSTEM_FREQ_HZ = 60.0

# ── Stability thresholds ─────────────────────────────────────────────
VOLTAGE_MIN_PU = 0.90
VOLTAGE_MAX_PU = 1.10
FREQ_DEV_MAX_HZ = 0.5
LINE_LOADING_NORMAL_PCT = 100.0
LINE_LOADING_EMERGENCY_PCT = 120.0
ROTOR_ANGLE_MAX_DEG = 180.0

# ── TDS defaults ─────────────────────────────────────────────────────
TDS_DEFAULT_STEP = 1 / 30  # ~0.033s
TDS_DEFAULT_TF = 20.0

# ── Snapshot ─────────────────────────────────────────────────────────
MAX_SNAPSHOTS = 10

# ── SiLR Verifier defaults ──────────────────────────────────────────
SILR_TDS_DURATION = 10.0
SILR_ALLOWED_ACTIONS = frozenset(["adjust_gen", "shed_load", "trip_line", "close_line"])


def build_grid_domain_config(pflow_only: bool = False) -> "DomainConfig":
    """Build a DomainConfig for the IEEE 39-bus power grid domain.

    Args:
        pflow_only: If True, skip transient stability checker and TDS.

    Returns:
        DomainConfig with all grid-specific components.
    """
    # Late imports to avoid circular dependency at module level
    from silr.core.config import DomainConfig
    from .constraints import (
        VoltageChecker,
        FrequencyChecker,
        LineLoadingChecker,
        TransientStabilityChecker,
    )
    from .tools import create_toolset
    from .observation import ObservationFormatter
    from .failsafe import DefaultFailsafe
    from .prompts.system_prompt import build_system_prompt
    from .prompts.tool_schemas import build_tool_schemas, get_valid_device_ids

    checkers = [VoltageChecker(), FrequencyChecker(), LineLoadingChecker()]
    if not pflow_only:
        checkers.append(TransientStabilityChecker())

    def _create_failsafe(manager):
        return DefaultFailsafe(
            gen_ids=manager.get_all_syn_gen_idx(),
            bus_ids=manager.get_bus_idx_list(),
            pq_bus_ids=manager.get_pq_bus_ids(),
        )

    return DomainConfig(
        domain_name="power_grid",
        checkers=checkers,
        allowed_actions=SILR_ALLOWED_ACTIONS,
        create_toolset=create_toolset,
        build_system_prompt=build_system_prompt,
        build_tool_schemas=build_tool_schemas,
        get_valid_device_ids=get_valid_device_ids,
        create_observer=lambda mgr: ObservationFormatter(mgr),
        create_failsafe=_create_failsafe,
        post_solve_hook=None if pflow_only else _tds_hook,
    )


def _tds_hook(manager) -> bool:
    """Run time-domain simulation as post-solve verification step.

    Called by SiLRVerifier after solve() succeeds on the shadow copy.
    Returns True if TDS completes without divergence.
    """
    try:
        manager.run_tds(SILR_TDS_DURATION)
        return not getattr(manager.ss.TDS, 'busted', False)
    except Exception:
        return False
