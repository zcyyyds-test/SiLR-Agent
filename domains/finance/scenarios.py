"""Portfolio stress scenarios for SiLR evaluation.

15 scenarios across 3 difficulty levels.  Each scenario injects price
changes and/or position mutations that put the portfolio into a
non-compliant state.  The agent must restore compliance.

Constraints (defaults):
  - Position concentration: single stock ≤ 25%
  - Sector exposure: single sector ≤ 45%
  - Drawdown from peak: ≤ 10%
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
    price_changes: dict[str, float] = field(default_factory=dict)
    position_overrides: dict[str, int] = field(default_factory=dict)
    cash_override: float | None = None
    difficulty: str = "easy"


# ── Easy: single constraint violation ────────────────────────────

SCENARIOS = [
    # --- Easy (5) ---
    FinanceScenario(
        id="crash_aapl",
        description="AAPL drops 40%, triggering drawdown violation",
        price_changes={"AAPL": 90.0},  # 150 → 90 (-40%)
        difficulty="easy",
    ),
    FinanceScenario(
        id="crash_nvda",
        description="NVDA drops 50%, triggering drawdown violation",
        price_changes={"NVDA": 400.0},  # 800 → 400 (-50%)
        difficulty="easy",
    ),
    FinanceScenario(
        id="cash_low_buy",
        description="Heavy buying depletes cash below reserve",
        position_overrides={"AAPL": 1500, "MSFT": 700},
        cash_override=15_000.0,  # ~1.3% of portfolio
        difficulty="easy",
    ),
    FinanceScenario(
        id="single_concentration",
        description="NVDA rallies 150%, one stock exceeds 25% weight",
        price_changes={"NVDA": 2000.0},  # 800 → 2000 (+150%)
        difficulty="easy",
    ),
    FinanceScenario(
        id="energy_crash",
        description="Energy sector crashes, other sectors passively exceed limits",
        price_changes={"XOM": 33.0, "CVX": 46.5},  # both -70%
        difficulty="easy",
    ),

    # --- Medium (5): two constraints or multi-step ---
    FinanceScenario(
        id="tech_bubble",
        description="All tech stocks rally, sector exposure + position concentration",
        price_changes={"AAPL": 300.0, "MSFT": 550.0, "NVDA": 1600.0},
        difficulty="medium",
    ),
    FinanceScenario(
        id="broad_decline_mild",
        description="Market-wide 12% decline, drawdown + need selective sell",
        price_changes={
            "AAPL": 132.0, "MSFT": 264.0, "NVDA": 704.0,
            "JNJ": 140.8, "PFE": 24.6, "UNH": 440.0,
            "XOM": 96.8, "CVX": 136.4,
        },
        difficulty="medium",
    ),
    FinanceScenario(
        id="sector_rotation",
        description="Tech crashes while energy rallies, multi-sector imbalance",
        price_changes={
            "AAPL": 90.0, "MSFT": 180.0, "NVDA": 480.0,
            "XOM": 220.0, "CVX": 310.0,
        },
        difficulty="medium",
    ),
    FinanceScenario(
        id="cash_and_concentration",
        description="Cash depleted + single stock concentrated",
        price_changes={"NVDA": 1800.0},
        cash_override=20_000.0,
        difficulty="medium",
    ),
    FinanceScenario(
        id="health_crash",
        description="Health sector crash, drawdown + sector rebalance needed",
        price_changes={"JNJ": 64.0, "PFE": 11.2, "UNH": 200.0},
        difficulty="medium",
    ),

    # --- Hard (5): three+ constraints or cascading ---
    FinanceScenario(
        id="flash_crash",
        description="Market-wide 20% crash, drawdown + cash + sector all violated",
        price_changes={
            "AAPL": 120.0, "MSFT": 240.0, "NVDA": 640.0,
            "JNJ": 128.0, "PFE": 22.4, "UNH": 400.0,
            "XOM": 88.0, "CVX": 124.0,
        },
        cash_override=25_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="bubble_burst",
        description="Tech bubble then crash: position + sector + drawdown",
        price_changes={
            "AAPL": 280.0, "MSFT": 520.0, "NVDA": 1500.0,
            "JNJ": 100.0, "PFE": 16.0, "UNH": 300.0,
        },
        difficulty="hard",
    ),
    FinanceScenario(
        id="liquidity_crisis",
        description="Low-volume stocks drop + cash depleted",
        price_changes={
            "UNH": 250.0, "JNJ": 80.0,
            "XOM": 55.0, "CVX": 77.5,
        },
        cash_override=10_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="cascade_rebalance",
        description="Fixing one constraint triggers another: concentration → cash → sector",
        price_changes={"NVDA": 2200.0, "AAPL": 250.0},
        cash_override=18_000.0,
        difficulty="hard",
    ),
    FinanceScenario(
        id="worst_case",
        description="All 4 checkers violated simultaneously",
        price_changes={
            "NVDA": 2000.0,
            "AAPL": 250.0, "MSFT": 500.0,
            "JNJ": 80.0, "PFE": 14.0, "UNH": 250.0,
            "XOM": 55.0, "CVX": 77.5,
        },
        cash_override=12_000.0,
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

        1. Apply price shocks
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
