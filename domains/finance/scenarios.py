"""Portfolio stress scenarios grounded in real US equity market events.

Each scenario applies historical return magnitudes (sourced from Yahoo
Finance 2019-2024, see data/close_prices.csv) to the 2024-01-02 baseline
prices.  The percentage moves are taken from actual market events; only
the absolute price levels are rebased to the common baseline date.

15 scenarios across 3 difficulty levels.

Compliance thresholds (defaults):
  - Position concentration: single stock ≤ 20%
  - Sector exposure: single sector ≤ 40%
  - Drawdown from peak: ≤ 8%
  - Cash reserve: ≥ 5% of portfolio
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .manager import FinanceManager


@dataclass
class FinanceScenario:
    """Portfolio stress scenario definition."""

    id: str
    description: str
    source_event: str                                    # Historical reference
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

SCENARIOS = [
    # ─────────────────────────── Easy (5) ────────────────────────────
    # Single constraint violation

    FinanceScenario(
        id="nvda_ai_surge",
        description="NVDA surges on AI demand; position concentration breached",
        source_event="NVDA Jan-Jul 2024 (+185%): 48.14 → 137.25",
        # NVDA +185% (actual 2024 move), rest slightly up
        price_changes={
            "NVDA": 137.25,     # 48.14 × 2.85
            "AAPL": 192.0,      # +4.5%
            "MSFT": 381.0,      # +4.5%
        },
        difficulty="easy",
    ),
    FinanceScenario(
        id="rate_hike_drawdown",
        description="Broad market decline on aggressive rate hike; drawdown exceeded",
        source_event="Fed 75bp hike Jun 2022: broad -10% to -16%",
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
        difficulty="easy",
    ),
    FinanceScenario(
        id="energy_covid_crash",
        description="Energy sector collapses as in COVID-era; other sectors exceed limits",
        source_event="COVID crash 2020.02-03: XOM -48%, CVX -51%",
        price_changes={
            "XOM":  49.32,      # -48.0%
            "CVX":  66.46,      # -51.0%
        },
        difficulty="easy",
    ),
    FinanceScenario(
        id="tech_rally_sector",
        description="Tech sector rallies; sector exposure breached",
        source_event="Tech recovery Jan 2023 + AI narrative",
        price_changes={
            "AAPL": 238.85,     # +30%
            "MSFT": 473.97,     # +30%
            "NVDA":  76.86,     # +60% (AI premium)
        },
        difficulty="easy",
    ),
    FinanceScenario(
        id="cash_overcommit",
        description="Recent large purchases settled; cash reserve depleted",
        source_event="Simulated post-settlement cash drain",
        cash_override=30_000.0,    # cash drops to ~2.7%
        difficulty="easy",
    ),

    # ─────────────────────────── Medium (5) ──────────────────────────
    # Two constraints or multi-step resolution

    FinanceScenario(
        id="tech_selloff_rotation",
        description="Tech dumps while energy rallies; drawdown + cash strain from margin",
        source_event="Jan 2022 selloff (amplified): tech -15~40%, energy +23%",
        price_changes={
            "AAPL": 156.17,     # -15%
            "MSFT": 309.90,     # -15%
            "NVDA":  28.88,     # -40%
            "PFE":   23.39,     #  -9%
            "UNH":  473.12,     #  -8%
            "XOM":  116.67,     # +23%
            "CVX":  155.97,     # +15%
        },
        cash_override=30_000.0,   # margin call during selloff
        difficulty="medium",
    ),
    FinanceScenario(
        id="nvda_surge_low_cash",
        description="NVDA rallies sharply while cash is already tight",
        source_event="NVDA earnings Feb 2024 (+25%) + low cash",
        price_changes={
            "NVDA": 137.25,     # +185% (full 2024 surge)
        },
        cash_override=22_000.0,
        difficulty="medium",
    ),
    FinanceScenario(
        id="health_crash_2020",
        description="Health sector crash; drawdown + sector imbalance",
        source_event="COVID crash 2020: JNJ -25%, PFE -21%, UNH -36%",
        price_changes={
            "JNJ":  112.25,     # -25.0%
            "PFE":   20.30,     # -21.0%
            "UNH":  329.13,     # -36.0%
        },
        difficulty="medium",
    ),
    FinanceScenario(
        id="tech_selloff_2024",
        description="Tech rotation: tech down, health rallies; sector breach + cash",
        source_event="Jul 2024 selloff (amplified): tech -7~17%, health +15~35%",
        price_changes={
            "AAPL": 171.60,     #  -6.6%
            "MSFT": 326.04,     # -10.3%
            "NVDA":  40.05,     # -16.8%
            "JNJ":  172.11,     # +15% (rotation into health)
            "PFE":   29.56,     # +15%
            "UNH":  694.25,     # +35% (defensive flight)
        },
        cash_override=40_000.0,   # selling tech generated cash then redeployed
        difficulty="medium",
    ),
    FinanceScenario(
        id="broad_decline_2022",
        description="Sustained market decline; drawdown + cash strain",
        source_event="H1 2022 bear market: broad -15% to -30%",
        price_changes={
            "AAPL": 152.50,     # -17%
            "MSFT": 291.67,     # -20%
            "NVDA":  33.70,     # -30%
            "JNJ":  134.69,     # -10%
            "PFE":   21.85,     # -15%
            "UNH":  452.55,     # -12%
            "XOM":  85.37,      # -10%
            "CVX":  118.00,     # -13%
        },
        cash_override=35_000.0,
        difficulty="medium",
    ),

    # ─────────────────────────── Hard (5) ────────────────────────────
    # Three+ constraints or cascading effects

    FinanceScenario(
        id="covid_full_crash",
        description="COVID-magnitude market crash; drawdown + cash + sector all breached",
        source_event="2020.02.19-03.23 actual returns applied",
        price_changes={
            "AAPL": 127.29,     # -30.7%
            "MSFT": 264.57,     # -27.4%
            "NVDA":  32.53,     # -32.4%
            "JNJ":  112.38,     # -24.9%
            "PFE":   20.18,     # -21.4%
            "UNH":  329.85,     # -35.9%
            "XOM":  49.32,      # -48.0%
            "CVX":  66.46,      # -51.0%
        },
        cash_override=20_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="tech_bubble_burst",
        description="Tech bubble collapses while defensives rally; drawdown + sector + cash",
        source_event="2022 tech bust: NVDA -70%, energy/health +5~30%",
        price_changes={
            "AAPL": 137.80,     # -25%
            "MSFT": 236.98,     # -35%
            "NVDA":  14.44,     # -70% (actual 2022 magnitude)
            "JNJ":  157.14,     #  +5% (defensive rotation)
            "UNH":  539.97,     #  +5%
            "XOM":  123.31,     # +30% (2022 energy rally)
            "CVX":  176.32,     # +30%
        },
        cash_override=20_000.0,   # margin call during crash
        difficulty="hard",
    ),
    FinanceScenario(
        id="liquidity_crisis",
        description="Low-volume stocks drop sharply + cash depleted",
        source_event="Composite: bank crisis Mar 2023 + rate stress",
        price_changes={
            "UNH":  308.56,     # -40% (concentrated large-cap health)
            "JNJ":   89.80,     # -40%
            "XOM":  56.91,      # -40%
            "CVX":  81.38,      # -40%
        },
        cash_override=12_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="cascade_rebalance",
        description="Selling NVDA to fix concentration depletes cash, triggers cascade",
        source_event="NVDA full-year 2024 rally + tight cash",
        price_changes={
            "NVDA": 137.25,     # +185%
            "AAPL": 235.00,     # +28%
            "MSFT": 460.00,     # +26%
        },
        cash_override=15_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="worst_case_composite",
        description="All 4 constraints violated: compound historical stress",
        source_event="Composite worst-case from 2020+2022 events",
        price_changes={
            "NVDA": 137.25,     # +185% (2024 AI surge → concentration)
            "AAPL": 146.98,     # -20% (COVID-like for non-NVDA)
            "MSFT": 291.67,     # -20%
            "JNJ":  104.76,     # -30%
            "PFE":   17.99,     # -30%
            "UNH":  359.98,     # -30%
            "XOM":  66.40,      # -30%
            "CVX":  94.94,      # -30%
        },
        cash_override=10_000.0,
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
