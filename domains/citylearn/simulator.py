"""CityLearn district-storage physics — self-contained, no external deps.

Derived from the CityLearn Challenge 2022 (Phase-1) 3-building summer-day
profile (load + PV bell + ERCOT-style TOU price). The transition is a pure
function of (state, joint-action): each building owns a battery, the controller
picks a discrete charge(<0)/discharge(>0) power per building, and the per-
building grid draws aggregate through a district feeder that couples them via a
shared import/export limit and a peak-demand charge.

This module is the domain's "simulator". The SiLR adapter (manager.py) wraps it.
Physics ported from the sibling SpeculativeControl env; the SiLR recovery task
fixes a single hour `t` and asks the agent to adjust per-building set-points
until SoC and district import/export constraints are satisfied (ANM-isomorphic).

Design note: district PV-curtailment (which would silently cap export at the
limit) is intentionally DISABLED here, so an over-export is surfaced as a
recoverable `export_limit` violation for the agent to resolve, rather than
being clipped away. This mirrors ANM, where the gate — not a built-in clip —
is what enforces feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── State ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CityLearnState:
    """Immutable district state. Trivially copyable for shadow execution."""
    t: int
    soc_kwh: tuple[float, ...]
    peak_import_kw: float


# ── Per-building catalog (CityLearn 2022 Phase-1, parameterized) ─
# N_BUILDINGS is a difficulty knob: the joint discrete action space is
# 5^N_BUILDINGS, so raising it past the original 3 enlarges the search the
# policy must solve UNGATED (no reject feedback) -- the headroom the 3-building
# slice lacked (base 8B saturated, decisions-aamas.md A-route). Buildings beyond
# the 3 real CityLearn-2022 buildings reuse base battery column b%3 with a scaled
# demand profile (_LOAD_SCALE), so N_BUILDINGS=3 reproduces the original domain
# exactly. Feeder limits are NOT scaled with N, so more buildings tighten the
# import/export constraints relative to demand (harder + activates the feeder
# families more often).
N_BUILDINGS = 4

_BASE_SOC_MIN = (0.5, 0.5, 0.5)
_BASE_SOC_MAX = (6.4, 5.6, 4.0)
_BASE_INITIAL_SOC = (3.2, 2.8, 2.0)
_BASE_MAX_CHARGE = (3.0, 2.5, 2.0)
_BASE_MAX_DISCHARGE = (3.0, 2.5, 2.0)
_BASE_ETA = (0.95, 0.95, 0.95)
# Demand scale per building (len >= any supported N); 1.0 for the 3 real ones.
_LOAD_SCALE = (1.0, 1.0, 1.0, 1.15, 0.85, 1.2)


def _derive(base: tuple) -> tuple:
    return tuple(base[b % 3] for b in range(N_BUILDINGS))


SOC_MIN_KWH = _derive(_BASE_SOC_MIN)
SOC_MAX_KWH = _derive(_BASE_SOC_MAX)
INITIAL_SOC_KWH = _derive(_BASE_INITIAL_SOC)
MAX_CHARGE_KW = _derive(_BASE_MAX_CHARGE)
MAX_DISCHARGE_KW = _derive(_BASE_MAX_DISCHARGE)
ETA_CHARGE = _derive(_BASE_ETA)
ETA_DISCHARGE = _derive(_BASE_ETA)

# ── District feeder limits + cost coefficients ───────────────────
DISTRICT_IMPORT_LIMIT_KW = 16.0
DISTRICT_EXPORT_LIMIT_KW = 8.0
FEED_IN_PRICE = 0.03
DEMAND_CHARGE_RATE = 1.25
CYCLING_COST = 0.012
DT_H = 1.0

# Per-building discrete action set (charge negative, discharge positive).
ACTIONS_PER_BUILDING = (-3.0, -1.5, 0.0, 1.5, 3.0)

# ── 24h base profiles (hour 0..23, one tuple per hour for buildings B1,B2,B3) ─
_BASE_LOAD = (
    (0.48, 0.41, 0.32), (0.45, 0.39, 0.29), (0.42, 0.37, 0.27), (0.41, 0.35, 0.26),
    (0.43, 0.36, 0.27), (0.55, 0.45, 0.32), (0.72, 0.58, 0.48), (1.18, 0.92, 0.74),
    (1.45, 1.21, 0.93), (1.32, 1.08, 0.82), (1.15, 0.94, 0.71), (1.04, 0.87, 0.68),
    (1.01, 0.85, 0.66), (0.98, 0.82, 0.63), (0.99, 0.83, 0.65), (1.08, 0.91, 0.72),
    (1.31, 1.15, 0.89), (1.78, 1.52, 1.18), (2.05, 1.74, 1.36), (1.92, 1.61, 1.23),
    (1.61, 1.34, 1.04), (1.24, 1.03, 0.81), (0.86, 0.71, 0.56), (0.58, 0.49, 0.39),
)
_BASE_PV = (
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0), (0.02, 0.01, 0.01), (0.18, 0.12, 0.08), (0.62, 0.41, 0.28),
    (1.21, 0.82, 0.55), (1.78, 1.21, 0.81), (2.18, 1.49, 1.01), (2.41, 1.65, 1.12),
    (2.48, 1.71, 1.16), (2.32, 1.59, 1.08), (1.92, 1.31, 0.89), (1.32, 0.91, 0.61),
    (0.61, 0.42, 0.28), (0.18, 0.12, 0.08), (0.02, 0.01, 0.01), (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
)


def _derive_profile(base_24x3: tuple) -> tuple:
    return tuple(
        tuple(round(hour[b % 3] * _LOAD_SCALE[b], 4) for b in range(N_BUILDINGS))
        for hour in base_24x3
    )


LOAD_KW = _derive_profile(_BASE_LOAD)
PV_KW = _derive_profile(_BASE_PV)
PRICE = (
    0.10, 0.10, 0.10, 0.10, 0.10, 0.12, 0.16, 0.21, 0.21, 0.18, 0.16, 0.16,
    0.18, 0.21, 0.24, 0.28, 0.32, 0.36, 0.36, 0.32, 0.24, 0.18, 0.14, 0.12,
)

EPISODE_LENGTH = 24


def evaluate(state: CityLearnState, action: tuple[float, ...]) -> dict:
    """Pure (state, joint-action) -> derived district outcome at hour state.t.

    Does NOT advance time: this is a steady-state evaluation of applying
    `action` at the current hour, for SiLR shadow verification. Returns a dict
    consumed by the manager's system_state and the constraint checkers.
    """
    t = state.t
    load = LOAD_KW[t % EPISODE_LENGTH]
    pv = PV_KW[t % EPISODE_LENGTH]
    price = PRICE[t % EPISODE_LENGTH]

    soc_next: list[float] = []
    per_building_grid: list[float] = []
    cycling_total = 0.0
    for b in range(N_BUILDINGS):
        a = action[b]
        discharge = max(a, 0.0)
        charge = max(-a, 0.0)
        s = (
            state.soc_kwh[b]
            + ETA_CHARGE[b] * charge * DT_H
            - discharge * DT_H / ETA_DISCHARGE[b]
        )
        soc_next.append(round(s, 6))
        per_building_grid.append(load[b] - pv[b] + charge - discharge)
        cycling_total += CYCLING_COST * (charge + discharge) * DT_H

    district_grid = sum(per_building_grid)
    district_import = max(district_grid, 0.0)
    district_export = max(-district_grid, 0.0)
    peak_next = max(state.peak_import_kw, district_import)
    peak_increment = peak_next - state.peak_import_kw
    cost = (
        price * district_import * DT_H
        - FEED_IN_PRICE * district_export * DT_H
        + cycling_total
        + DEMAND_CHARGE_RATE * peak_increment
    )

    return {
        "t": t,
        "price": price,
        "soc_next_kwh": tuple(soc_next),
        "per_building_grid_kw": tuple(per_building_grid),
        "district_import_kw": round(district_import, 6),
        "district_export_kw": round(district_export, 6),
        "peak_import_kw": round(peak_next, 6),
        "cost": round(cost, 6),
        "load_kw": load,
        "pv_kw": pv,
    }
