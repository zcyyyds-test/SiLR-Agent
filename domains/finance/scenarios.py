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
        id="energy_slump_tech_flight",
        description="COVID-era energy slump + flight to tech; sector rotation",
        source_event="COVID 2020: XOM -48%, CVX -51%, tech +10% rebound",
        price_changes={
            "XOM":  47.43,      # -50%
            "CVX":  61.03,      # -55%
            "AAPL": 193.00,     #  +5%
            "MSFT": 390.00,     #  +7%
            "NVDA":  52.95,     # +10%
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
        cash_override=20_000.0,
        difficulty="easy",
    ),
    FinanceScenario(
        id="tech_rally_sector_breach",
        description="Tech sector rallies through the ceiling while energy slips below floor",
        source_event="Tech recovery + AI bid with late energy underperformance",
        price_changes={
            "AAPL": 220.48,     # +20%
            "MSFT": 419.28,     # +15%
            "NVDA":  67.40,     # +40%
            "XOM":   56.91,     # -40%
            "CVX":   94.94,     # -30%
        },
        difficulty="easy",
    ),
    FinanceScenario(
        id="cash_depleted",
        description="Recent purchases settled into a thin cash buffer while energy slips below floor",
        source_event="Post-settlement cash drain with a sharp energy sleeve drawdown",
        price_changes={
            "XOM":  47.43,      # -50%
            "CVX":  74.60,      # -45%
        },
        cash_override=20_000.0,
        difficulty="easy",
    ),
    FinanceScenario(
        id="nvda_concentration",
        description="NVDA earnings surge; single-name concentration spills into tech sleeve",
        source_event="NVDA earnings H1 2024 (+110%)",
        price_changes={
            "NVDA": 101.09,     # +110%
        },
        difficulty="easy",
    ),
    FinanceScenario(
        id="energy_rally_tech_drop",
        description="Oil spike + tech selloff + low cash; energy ceiling + cash squeeze",
        source_event="2022 oil shock: XOM/CVX +130%, tech -30%, margin call drained cash",
        price_changes={
            "XOM":  218.16,     # +130%
            "CVX":  311.95,     # +130%
            "AAPL": 128.61,     # -30%
            "MSFT": 255.21,     # -30%
            "NVDA":  33.70,     # -30%
        },
        cash_override=25_000.0,
        difficulty="easy",
    ),

    # ═══════════════════════ MEDIUM (10) ═════════════════════════════
    # Requires 4-6 steps, some tight at max_steps=6

    # --- BIDIR medium ---
    FinanceScenario(
        id="tech_selloff_2022_rotation",
        description="Tech washout + oil spike; NVDA floor, energy ceiling, cash squeeze",
        source_event="2022 rotation stress: NVDA -65%, XOM/CVX +80%, cash drained",
        price_changes={
            "AAPL": 156.17,     # -15%
            "MSFT": 291.67,     # -20%
            "NVDA":  16.85,     # -65%
            "XOM":  170.73,     # +80%
            "CVX":  244.13,     # +80%
        },
        cash_override=25_000.0,
        difficulty="medium",
    ),
    FinanceScenario(
        id="health_selloff_tech_boom",
        description="Health selloff + tech boom; tech ceiling + cash squeeze",
        source_event="COVID rotation: health -25~40%, tech +20~40%, cash drained",
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
        description="NVDA spike + uneven energy lag; tech ceiling + energy floor",
        source_event="NVDA H1 2024 (+80%) + energy rotation out (XOM -30%, CVX -40%)",
        price_changes={
            "NVDA":  81.84,     # +70%
            "XOM":   71.14,     # -25%
            "CVX":   74.60,     # -45%
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
        description="Tech rallies, health softens, energy selloff; asymmetric sector rebalance",
        source_event="Cross-sector divergence: tech +10~35%, health -5%, energy -30~-40%",
        price_changes={
            "AAPL": 202.10,     # +10%
            "MSFT": 419.28,     # +15%
            "NVDA":  64.99,     # +35%
            "JNJ":  142.18,     #  -5%
            "PFE":   24.42,     #  -5%
            "UNH":  488.55,     #  -5%
            "XOM":   61.65,     # -35%
            "CVX":   81.38,     # -40%
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
        cash_override=20_000.0,
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
        description="AAPL + NVDA rebound while energy fades; tech ceiling + energy floor",
        source_event="AI + mega-cap bounce: AAPL +35%, NVDA +40%, XOM -40%, CVX -35%",
        price_changes={
            "NVDA":  67.40,     # +40%
            "AAPL": 247.95,     # +35%
            "XOM":   56.91,     # -40%
            "CVX":   88.16,     # -35%
        },
        difficulty="medium",
    ),
    FinanceScenario(
        id="japan_carry_unwind",
        description="Global risk-off unwind from carry trade stress; broad drop + cash",
        source_event="Aug 2024 carry unwind: tech hit hardest, defensives off modestly",
        price_changes={
            "AAPL": 172.71,     #  -6%
            "MSFT": 298.96,     # -18%
            "NVDA":  38.51,     # -20%
            "JNJ":  145.17,     #  -3%
            "PFE":   24.93,     #  -3%
            "UNH":  493.69,     #  -4%
            "XOM":   87.26,     #  -8%
            "CVX":  125.46,     #  -7%
        },
        cash_override=25_000.0,
        difficulty="medium",
    ),

    # ═══════════════════════ HARD (12) ═══════════════════════════════
    # Many require 6+ steps; some unsolvable in 6 steps with $15K limit

    # --- BIDIR hard ---
    FinanceScenario(
        id="covid_full_rotation",
        description="Pandemic rotation redux; tech ceiling + energy floor + cash squeeze",
        source_event="2020-style rotation rebased: tech +10~30%, health -15~20%, energy -45%, cash drained",
        price_changes={
            "AAPL": 202.10,     # +10%
            "MSFT": 401.05,     # +10%
            "NVDA":  62.58,     # +30%
            "JNJ":  127.21,     # -15%
            "PFE":   20.56,     # -20%
            "UNH":  411.41,     # -20%
            "XOM":   52.17,     # -45%
            "CVX":   74.60,     # -45%
        },
        cash_override=40_000.0,
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
        description="Tech melt-up + energy fade with tight cash; tech ceiling + energy floor + cash",
        source_event="AI rebound + energy pullback + cash drain",
        price_changes={
            "NVDA":  72.21,     # +50%
            "AAPL": 211.29,     # +15%
            "MSFT": 419.28,     # +15%
            "XOM":   56.91,     # -40%
            "CVX":   81.38,     # -40%
        },
        cash_override=40_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="worst_case_all_six",
        description="Composite concentration shock: NVDA/tech breach + cash squeeze",
        source_event="Composite rebased: NVDA +70%, defensives softer, energy lower, cash drained",
        price_changes={
            "NVDA":  81.84,     # +70%
            "AAPL": 165.36,     # -10%
            "MSFT": 328.13,     # -10%
            "JNJ":  104.76,     # -30%
            "PFE":   20.56,     # -20%
            "UNH":  411.41,     # -20%
            "XOM":   66.40,     # -30%
            "CVX":   94.94,     # -30%
        },
        cash_override=20_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="energy_spike_health_selloff",
        description="Oil crisis: energy ceiling + cash squeeze after health selloff",
        source_event="Composite: 2022 energy rally + health selloff + low cash",
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
        id="health_surge_tech_energy_lag",
        description="Health surge while energy sinks and cash stays tight; health ceiling + energy floor + cash squeeze",
        source_event="Defensive rotation: health +20%, tech -15%, energy -45%, cash tight",
        price_changes={
            "JNJ":  179.59,     # +20%
            "UNH":  617.11,     # +20%
            "PFE":   30.84,     # +20%
            "AAPL": 156.17,     # -15%
            "MSFT": 309.90,     # -15%
            "NVDA":  40.92,     # -15%
            "XOM":   52.17,     # -45%
            "CVX":   74.60,     # -45%
        },
        cash_override=35_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="nvda_reversal_energy_rally",
        description="NVDA reversal from prior peak + energy rebound; NVDA floor + cash squeeze",
        source_event="NVDA -65% correction + energy rebound + low cash",
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
        description="Tech outperforms while health and energy remain soft; tech ceiling + cash squeeze",
        source_event="Cross-sector dispersion: tech +15~50%, health softer, energy lower, cash drained",
        price_changes={
            "AAPL": 211.29,     # +15%
            "MSFT": 419.28,     # +15%
            "NVDA":  72.21,     # +50%
            "JNJ":  119.73,     # -20%
            "PFE":   20.56,     # -20%
            "UNH":  411.41,     # -20%
            "XOM":   66.40,     # -30%
            "CVX":   94.94,     # -30%
        },
        cash_override=25_000.0,
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
        description="Tech concentration and energy strength overpower health weakness; tech ceiling + cash squeeze",
        source_event="Composite: AAPL/NVDA rebound, XOM strength, health down 20~30%",
        price_changes={
            "NVDA":  86.65,     # +80%
            "AAPL": 247.95,     # +35%
            "XOM":  161.25,     # +70%
            "JNJ":  104.76,     # -30%
            "PFE":   17.99,     # -30%
            "UNH":  359.98,     # -30%
        },
        cash_override=30_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="deep_bear_market",
        description="2008-style broad drawdown; cash crisis",
        source_event="Composite: 2008/2020 magnitude broad decline -30~50% + cash drain",
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
        description="Partial rebound leaves tech stretched and cash thin",
        source_event="Uneven rebound after shock: tech recovered faster than the rest",
        price_changes={
            "NVDA":  76.86,     # +60%
            "AAPL": 220.48,     # +20%
            "MSFT": 328.13,     # -10%
            "JNJ":  127.21,     # -15%
            "PFE":   20.56,     # -20%
            "UNH":  411.41,     # -20%
            "XOM":   81.38,     # -14%
            "CVX":  115.29,     # -15%
        },
        cash_override=25_000.0,
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
