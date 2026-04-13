"""Portfolio stress scenarios grounded in real US equity market events.

Each scenario applies historical return magnitudes (sourced from Yahoo
Finance 2019-2024, see data/close_prices.csv) to the 2024-01-02 baseline
prices.  The percentage moves are taken from actual market events; only
the absolute price levels are rebased to the common baseline date.

30 scenarios across 3 difficulty levels (8 easy / 10 medium / 12 hard).
At least 15 are BIDIR (require both sell + buy actions).

Compliance thresholds:
  - Position ceiling: single stock ≤ 20%
  - Position floor: every held stock ≥ 4%
  - Sector ceiling: single sector ≤ 40%
  - Sector floor: every sector ≥ 15%
  - Cash reserve: ≥ 5% of portfolio
  - Drawdown from peak: ≤ 8% (monitoring only)

Trade limit: $15K per action.  max_steps=6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .manager import FinanceManager


@dataclass
class FinanceScenario:
    """Portfolio stress scenario definition."""

    id: str
    description: str
    source_event: str
    price_changes: dict[str, float] = field(default_factory=dict)
    position_overrides: dict[str, int] = field(default_factory=dict)
    cash_override: float | None = None
    difficulty: str = "easy"


# ── Baseline prices (2024-01-02): ────────────────────────────────
# AAPL 183.73  MSFT 364.59  NVDA  48.14
# JNJ  149.66  PFE   25.70  UNH  514.26
# XOM   94.85  CVX  135.63
#
# Each stock ~11.4% weight, tech ~34%, health ~34%, energy ~23%
# Cash ~9.1% of $1.1M total
#
# BIDIR = requires both sell (ceiling fix) and buy (floor fix)

SCENARIOS = [
    # ═══════════════════════ EASY (8) ════════════════════════════════
    # Solvable in 3-5 steps with $15K limit, max_steps=6

    # --- BIDIR easy ---
    FinanceScenario(
        id="nvda_surge_energy_lag",
        description="NVDA AI rally + energy rotation out; tech ceiling + energy floor",
        source_event="NVDA H1 2024 (+100%) + energy sector outflow",
        price_changes={
            "NVDA": 96.28,      # +100%
            "XOM":  56.91,      # -40%
            "CVX":  81.38,      # -40%
        },
        difficulty="easy",
    ),
    FinanceScenario(
        id="energy_crash_tech_flight",
        description="COVID-era energy collapse + flight to tech; sector rotation",
        source_event="COVID 2020: XOM -48%, CVX -51%, tech +10%",
        price_changes={
            "XOM":  49.32,      # -48%
            "CVX":  66.46,      # -51%
            "AAPL": 202.10,     # +10%
            "MSFT": 401.05,     # +10%
            "NVDA":  55.36,     # +15%
        },
        difficulty="easy",
    ),
    FinanceScenario(
        id="tech_rally_energy_slump",
        description="Tech momentum + energy slump; tech ceiling + energy floor",
        source_event="2023 AI narrative: tech +20~30%, energy -45%",
        price_changes={
            "AAPL": 220.48,     # +20%
            "MSFT": 437.51,     # +20%
            "NVDA":  62.58,     # +30%
            "XOM":  52.17,      # -45%
            "CVX":  74.60,      # -45%
        },
        difficulty="easy",
    ),

    # --- Non-BIDIR easy ---
    FinanceScenario(
        id="rate_hike_cash_drain",
        description="Broad decline on rate hike + margin call depletes cash",
        source_event="Fed 75bp hike Jun 2022 + margin call",
        price_changes={
            "AAPL": 160.59,     # -12.6%
            "MSFT": 328.49,     #  -9.9%
            "NVDA":  40.24,     # -16.4%
            "JNJ":  141.73,     #  -5.3%
            "PFE":   23.00,     # -10.5%
            "UNH":  473.65,     #  -7.9%
            "XOM":   90.30,     #  -4.8%
            "CVX":  120.71,     # -11.0%
        },
        cash_override=35_000.0,
        difficulty="easy",
    ),
    FinanceScenario(
        id="tech_rally_sector_breach",
        description="Tech sector rallies past 40% ceiling",
        source_event="Tech recovery Jan 2023 + AI narrative",
        price_changes={
            "AAPL": 238.85,     # +30%
            "MSFT": 473.97,     # +30%
            "NVDA":  76.86,     # +60%
        },
        difficulty="easy",
    ),
    FinanceScenario(
        id="cash_depleted",
        description="Recent large purchases settled; cash reserve violated",
        source_event="Simulated post-settlement cash drain",
        cash_override=25_000.0,
        difficulty="easy",
    ),
    FinanceScenario(
        id="nvda_concentration",
        description="NVDA earnings surge; single stock exceeds 20% ceiling",
        source_event="NVDA earnings H1 2024 (+100%)",
        price_changes={
            "NVDA":  96.28,     # +100%
        },
        difficulty="easy",
    ),
    FinanceScenario(
        id="energy_rally_tech_drop",
        description="Oil spike + tech risk-off; energy ceiling + tech sector weakens",
        source_event="2022 energy rally: XOM/CVX +110%, tech -20%",
        price_changes={
            "XOM":  199.19,     # +110%
            "CVX":  284.82,     # +110%
            "AAPL": 146.98,     # -20%
            "MSFT": 291.67,     # -20%
            "NVDA":  38.51,     # -20%
        },
        difficulty="easy",
    ),

    # ═══════════════════════ MEDIUM (10) ═════════════════════════════
    # Requires 4-6 steps, some tight at max_steps=6

    # --- BIDIR medium ---
    FinanceScenario(
        id="tech_selloff_2022_rotation",
        description="Tech dumps + energy rallies; tech floor risk + energy ceiling approach",
        source_event="Jan 2022 selloff: tech -15~40%, energy +25~35%",
        price_changes={
            "AAPL": 156.17,     # -15%
            "MSFT": 309.90,     # -15%
            "NVDA":  28.88,     # -40%
            "XOM":  118.56,     # +25%
            "CVX":  183.10,     # +35%
        },
        cash_override=30_000.0,
        difficulty="medium",
    ),
    FinanceScenario(
        id="health_crash_tech_boom",
        description="Health crash + tech boom; health floor + tech ceiling + cash",
        source_event="COVID 2020 health -25~40% + pandemic tech boom",
        price_changes={
            "JNJ":  112.25,     # -25%
            "PFE":   15.42,     # -40%
            "UNH":  308.56,     # -40%
            "AAPL": 220.48,     # +20%
            "MSFT": 437.51,     # +20%
            "NVDA":  67.40,     # +40%
        },
        cash_override=30_000.0,
        difficulty="medium",
    ),
    FinanceScenario(
        id="nvda_spike_energy_lag",
        description="NVDA surges + energy collapses; position ceiling + energy floor",
        source_event="NVDA H1 2024 (+120%) + energy rotation out -45%",
        price_changes={
            "NVDA": 105.91,     # +120%
            "XOM":  52.17,      # -45%
            "CVX":  74.60,      # -45%
        },
        difficulty="medium",
    ),
    FinanceScenario(
        id="energy_collapse_broad",
        description="Energy collapses + broad market stress; energy floor + cash",
        source_event="COVID energy: XOM -50%, CVX -55% + cash drain",
        price_changes={
            "XOM":  47.43,      # -50%
            "CVX":  61.03,      # -55%
            "AAPL": 165.36,     # -10%
            "MSFT": 328.13,     # -10%
        },
        cash_override=28_000.0,
        difficulty="medium",
    ),
    FinanceScenario(
        id="triple_sector_stress",
        description="Tech rallies, health flat, energy crashes; multi-sector rebalance",
        source_event="2022 H1 divergence: tech +15%, energy -35%",
        price_changes={
            "AAPL": 211.29,     # +15%
            "MSFT": 419.28,     # +15%
            "NVDA":  62.58,     # +30%
            "XOM":  61.65,      # -35%
            "CVX":  88.16,      # -35%
        },
        difficulty="medium",
    ),

    # --- Non-BIDIR medium ---
    FinanceScenario(
        id="nvda_surge_low_cash",
        description="NVDA rallies sharply while cash is tight",
        source_event="NVDA earnings Feb 2024 (+100%) + low cash",
        price_changes={
            "NVDA":  96.28,     # +100%
        },
        cash_override=22_000.0,
        difficulty="medium",
    ),
    FinanceScenario(
        id="broad_decline_2022",
        description="Sustained bear market decline; cash strain",
        source_event="H1 2022 bear market: broad -15% to -30%",
        price_changes={
            "AAPL": 152.50,     # -17%
            "MSFT": 291.67,     # -20%
            "NVDA":  33.70,     # -30%
            "JNJ":  134.69,     # -10%
            "PFE":   21.85,     # -15%
            "UNH":  452.55,     # -12%
            "XOM":   85.37,     # -10%
            "CVX":  118.00,     # -13%
        },
        cash_override=30_000.0,
        difficulty="medium",
    ),
    FinanceScenario(
        id="tech_selloff_2024_health_rally",
        description="Jul 2024 tech selloff + health defensive rally",
        source_event="Jul 2024: NVDA -17%, MSFT -10%, UNH +20%",
        price_changes={
            "AAPL": 171.60,     #  -6.6%
            "MSFT": 326.04,     # -10.3%
            "NVDA":  40.05,     # -16.8%
            "JNJ":  172.11,     # +15%
            "PFE":   29.56,     # +15%
            "UNH":  617.11,     # +20%
        },
        cash_override=35_000.0,
        difficulty="medium",
    ),
    FinanceScenario(
        id="dual_spike_energy_drop",
        description="NVDA + AAPL spike + energy drops; position ceilings + energy floor",
        source_event="AI + iPhone 2024: NVDA +80%, AAPL +30%, energy -45%",
        price_changes={
            "NVDA":  86.65,     # +80%
            "AAPL": 238.85,     # +30%
            "XOM":  52.17,      # -45%
            "CVX":  74.60,      # -45%
        },
        difficulty="medium",
    ),
    FinanceScenario(
        id="japan_carry_unwind",
        description="Global flash crash from carry trade unwind; broad drop + cash",
        source_event="Aug 2024 Japan carry trade: broad -8% to -15%",
        price_changes={
            "AAPL": 165.36,     # -10%
            "MSFT": 309.90,     # -15%
            "NVDA":  40.92,     # -15%
            "JNJ":  137.69,     #  -8%
            "PFE":   23.65,     #  -8%
            "UNH":  473.12,     #  -8%
            "XOM":   85.37,     # -10%
            "CVX":  122.07,     # -10%
        },
        cash_override=25_000.0,
        difficulty="medium",
    ),

    # ═══════════════════════ HARD (12) ═══════════════════════════════
    # Many require 6+ steps; some unsolvable in 6 steps with $15K limit

    # --- BIDIR hard ---
    FinanceScenario(
        id="covid_full_rotation",
        description="COVID crash + flight to tech; energy floor + health floor + tech ceiling",
        source_event="2020.02-03 actual returns + tech safe haven",
        price_changes={
            "AAPL": 220.48,     # +20% (pandemic tech)
            "MSFT": 437.51,     # +20%
            "NVDA":  72.21,     # +50%
            "JNJ":  112.25,     # -25%
            "PFE":   15.42,     # -40%
            "UNH":  308.56,     # -40%
            "XOM":  47.43,      # -50%
            "CVX":  67.82,      # -50%
        },
        cash_override=20_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="tech_bubble_burst_defensive",
        description="Tech collapses + defensive rotation; NVDA floor + health/energy ceiling",
        source_event="2022 tech bust: NVDA -70%, defensives +30%",
        price_changes={
            "AAPL": 137.80,     # -25%
            "MSFT": 236.98,     # -35%
            "NVDA":  14.44,     # -70%
            "JNJ":  194.56,     # +30%
            "UNH":  668.54,     # +30%
            "XOM":  123.31,     # +30%
            "CVX":  176.32,     # +30%
        },
        cash_override=20_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="cascade_tech_energy",
        description="Tech surges + energy crashes + low cash; ceiling + floor + cash cascade",
        source_event="NVDA 2024 rally + energy rotation + margin call",
        price_changes={
            "NVDA": 120.35,     # +150%
            "AAPL": 235.00,     # +28%
            "MSFT": 460.00,     # +26%
            "XOM":  47.43,      # -50%
            "CVX":  67.82,      # -50%
        },
        cash_override=15_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="worst_case_all_six",
        description="All 6 constraints violated: ceiling + floor + cash compound",
        source_event="Composite worst-case from 2020+2022+2024 events",
        price_changes={
            "NVDA": 120.35,     # +150% (position + sector ceiling)
            "AAPL": 146.98,     # -20%
            "MSFT": 291.67,     # -20%
            "JNJ":   74.83,     # -50% (position floor)
            "PFE":   12.85,     # -50% (position floor)
            "UNH":  257.13,     # -50% (position floor)
            "XOM":  47.43,      # -50% (sector floor)
            "CVX":  67.82,      # -50%
        },
        cash_override=10_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="energy_spike_health_crash",
        description="Oil crisis: energy ceiling + health floor + position floors",
        source_event="Composite: 2022 energy rally + health selloff",
        price_changes={
            "XOM":  189.70,     # +100%
            "CVX":  271.26,     # +100%
            "JNJ":   89.80,     # -40%
            "PFE":   15.42,     # -40%
            "UNH":  308.56,     # -40%
        },
        cash_override=18_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="health_surge_tech_energy_crash",
        description="Health rally + tech/energy crash; health ceiling + energy floor + tech floor risk",
        source_event="Defensive rotation: health +40%, tech -30%, energy -50%",
        price_changes={
            "JNJ":  209.52,     # +40%
            "UNH":  719.96,     # +40%
            "PFE":   35.98,     # +40%
            "AAPL": 128.61,     # -30%
            "MSFT": 255.21,     # -30%
            "NVDA":  33.70,     # -30%
            "XOM":  47.43,      # -50%
            "CVX":  67.82,      # -50%
        },
        cash_override=20_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="nvda_crash_energy_rally",
        description="NVDA crashes from peak + energy surges; NVDA floor + energy/position ceiling",
        source_event="NVDA -65% correction + oil spike",
        price_changes={
            "NVDA":  16.85,     # -65%
            "XOM":  161.25,     # +70%
            "CVX":  230.57,     # +70%
        },
        cash_override=22_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="three_way_rotation",
        description="Tech up, health down, energy down; 3-sector rebalance nightmare",
        source_event="Composite: tech AI rally + defensive selloff + energy slump",
        price_changes={
            "AAPL": 238.85,     # +30%
            "MSFT": 473.97,     # +30%
            "NVDA": 105.91,     # +120%
            "JNJ":   89.80,     # -40%
            "PFE":   12.85,     # -50%
            "UNH":  257.13,     # -50%
            "XOM":  47.43,      # -50%
            "CVX":  67.82,      # -50%
        },
        cash_override=12_000.0,
        difficulty="hard",
    ),

    # --- Non-BIDIR hard ---
    FinanceScenario(
        id="liquidity_crisis",
        description="Low-volume stocks drop sharply + cash depleted",
        source_event="Composite: bank crisis Mar 2023 + rate stress",
        price_changes={
            "UNH":  308.56,     # -40%
            "JNJ":   89.80,     # -40%
            "XOM":   56.91,     # -40%
            "CVX":   81.38,     # -40%
        },
        cash_override=12_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="triple_concentration_health_drop",
        description="NVDA + AAPL + XOM spike + health drops; ceilings + health floor",
        source_event="Composite: AI + oil spike + health selloff -40%",
        price_changes={
            "NVDA": 105.91,     # +120%
            "AAPL": 275.60,     # +50%
            "XOM":  170.73,     # +80%
            "JNJ":   89.80,     # -40%
            "PFE":   15.42,     # -40%
            "UNH":  308.56,     # -40%
        },
        cash_override=20_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="deep_bear_market",
        description="2008-style broad crash; massive drawdown + cash crisis",
        source_event="Composite: 2008/2020 magnitude broad decline -30~50%",
        price_changes={
            "AAPL": 110.24,     # -40%
            "MSFT": 218.75,     # -40%
            "NVDA":  19.26,     # -60%
            "JNJ":   89.80,     # -40%
            "PFE":   12.85,     # -50%
            "UNH":  257.13,     # -50%
            "XOM":  47.43,      # -50%
            "CVX":  67.82,      # -50%
        },
        cash_override=8_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="whipsaw_aftermath",
        description="Post-flash-crash: some stocks rebounded, others didn't; messy state",
        source_event="Composite: partial recovery after Aug 2024 flash crash",
        price_changes={
            "NVDA":  86.65,     # +80% (V-recovery)
            "AAPL": 238.85,     # +30% (recovered)
            "MSFT": 291.67,     # -20% (still down)
            "JNJ":  104.76,     # -30% (still down)
            "PFE":   15.42,     # -40% (still down)
            "UNH":  308.56,     # -40% (still down)
            "XOM":  66.40,      # -30% (still down)
            "CVX":  94.94,      # -30% (still down)
        },
        cash_override=15_000.0,
        difficulty="hard",
    ),
]

_SCENARIO_MAP = {s.id: s for s in SCENARIOS}


class FinanceScenarioLoader:
    """Load and apply portfolio stress scenarios to a FinanceManager."""

    def load(self, scenario_id: str) -> FinanceScenario:
        if scenario_id not in _SCENARIO_MAP:
            raise KeyError(f"Unknown scenario: {scenario_id}")
        return _SCENARIO_MAP[scenario_id]

    def load_all(self) -> list[FinanceScenario]:
        return list(SCENARIOS)

    def setup_episode(self, manager: FinanceManager, scenario: FinanceScenario) -> None:
        """Inject stress into the portfolio.

        1. Apply price shocks (historical return magnitudes)
        2. Override positions if specified
        3. Override cash if specified
        4. solve() to recompute derived metrics
        """
        for symbol, new_price in scenario.price_changes.items():
            manager.set_price(symbol, new_price)

        for symbol, new_qty in scenario.position_overrides.items():
            if symbol in manager._positions:
                manager._positions[symbol] = new_qty

        if scenario.cash_override is not None:
            manager._cash = scenario.cash_override

        manager.solve()
