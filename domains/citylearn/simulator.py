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


# ── Per-building catalog (CityLearn 2022 Phase-1, 3-building slice) ─
N_BUILDINGS = 3
SOC_MIN_KWH = (0.5, 0.5, 0.5)
SOC_MAX_KWH = (6.4, 5.6, 4.0)
INITIAL_SOC_KWH = (3.2, 2.8, 2.0)
MAX_CHARGE_KW = (3.0, 2.5, 2.0)
MAX_DISCHARGE_KW = (3.0, 2.5, 2.0)
ETA_CHARGE = (0.95, 0.95, 0.95)
ETA_DISCHARGE = (0.95, 0.95, 0.95)

# ── District feeder limits + cost coefficients ───────────────────
DISTRICT_IMPORT_LIMIT_KW = 16.0
DISTRICT_EXPORT_LIMIT_KW = 8.0
FEED_IN_PRICE = 0.03
DEMAND_CHARGE_RATE = 1.25
CYCLING_COST = 0.012
DT_H = 1.0

# Per-building discrete action set (charge negative, discharge positive).
ACTIONS_PER_BUILDING = (-3.0, -1.5, 0.0, 1.5, 3.0)

# ── 24h profiles (hour 0..23, one tuple per hour for buildings B1,B2,B3) ─
LOAD_KW = (
    (0.48, 0.41, 0.32), (0.45, 0.39, 0.29), (0.42, 0.37, 0.27), (0.41, 0.35, 0.26),
    (0.43, 0.36, 0.27), (0.55, 0.45, 0.32), (0.72, 0.58, 0.48), (1.18, 0.92, 0.74),
    (1.45, 1.21, 0.93), (1.32, 1.08, 0.82), (1.15, 0.94, 0.71), (1.04, 0.87, 0.68),
    (1.01, 0.85, 0.66), (0.98, 0.82, 0.63), (0.99, 0.83, 0.65), (1.08, 0.91, 0.72),
    (1.31, 1.15, 0.89), (1.78, 1.52, 1.18), (2.05, 1.74, 1.36), (1.92, 1.61, 1.23),
    (1.61, 1.34, 1.04), (1.24, 1.03, 0.81), (0.86, 0.71, 0.56), (0.58, 0.49, 0.39),
)
PV_KW = (
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0), (0.02, 0.01, 0.01), (0.18, 0.12, 0.08), (0.62, 0.41, 0.28),
    (1.21, 0.82, 0.55), (1.78, 1.21, 0.81), (2.18, 1.49, 1.01), (2.41, 1.65, 1.12),
    (2.48, 1.71, 1.16), (2.32, 1.59, 1.08), (1.92, 1.31, 0.89), (1.32, 0.91, 0.61),
    (0.61, 0.42, 0.28), (0.18, 0.12, 0.08), (0.02, 0.01, 0.01), (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
)
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
