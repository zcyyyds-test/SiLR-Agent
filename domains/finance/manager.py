"""FinanceManager: portfolio simulator for SiLR framework.

Universe: 8 stocks across 3 sectors (tech/health/energy).
Initial portfolio: ~$1M in stocks + $100K cash.

Actions mutate positions/cash.  solve() recalculates derived
metrics (weights, sector exposure, drawdown).  Constraints are
checked against the derived state, not raw positions.

No external dependencies — all prices and volumes are synthetic.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from silr.core.interfaces import BaseSystemManager


# ── Universe definition ──────────────────────────────────────────

STOCKS = {
    "AAPL": {"price": 150.0, "sector": "tech", "daily_volume": 50_000_000},
    "MSFT": {"price": 300.0, "sector": "tech", "daily_volume": 25_000_000},
    "NVDA": {"price": 800.0, "sector": "tech", "daily_volume": 40_000_000},
    "JNJ":  {"price": 160.0, "sector": "health", "daily_volume": 8_000_000},
    "PFE":  {"price": 28.0,  "sector": "health", "daily_volume": 30_000_000},
    "UNH":  {"price": 500.0, "sector": "health", "daily_volume": 3_000_000},
    "XOM":  {"price": 110.0, "sector": "energy", "daily_volume": 15_000_000},
    "CVX":  {"price": 155.0, "sector": "energy", "daily_volume": 10_000_000},
}

# Initial position: ~$125K notional per stock → ~$1M stock + $100K cash
_DEFAULT_POSITIONS = {
    "AAPL": 833,
    "MSFT": 417,
    "NVDA": 156,
    "JNJ":  781,
    "PFE":  4464,
    "UNH":  250,
    "XOM":  1136,
    "CVX":  806,
}

_DEFAULT_CASH = 100_000.0


class FinanceManager(BaseSystemManager):
    """Portfolio simulator.  Pure Python, no external dependencies.

    Implements BaseSystemManager for SiLR verification compatibility.
    """

    def __init__(self):
        self._time: float = 0.0
        self._positions: dict[str, int] = dict(_DEFAULT_POSITIONS)
        self._prices: dict[str, float] = {s: d["price"] for s, d in STOCKS.items()}
        self._cash: float = _DEFAULT_CASH

        # Derived (recomputed by solve())
        self._portfolio_value: float = 0.0
        self._peak_value: float = 0.0
        self._weights: dict[str, float] = {}
        self._sector_exposure: dict[str, float] = {}
        self._drawdown_pct: float = 0.0

        # Initial solve to populate derived fields
        self.solve()
        self._peak_value = self._portfolio_value

    # ── BaseSystemManager interface ──────────────────────────────

    @property
    def sim_time(self) -> float:
        return self._time

    @property
    def base_mva(self) -> float:
        return 1.0  # not applicable for finance

    @property
    def system_state(self) -> dict:
        """Domain state for constraint checkers."""
        positions = {}
        for symbol, qty in self._positions.items():
            price = self._prices[symbol]
            notional = qty * price
            positions[symbol] = {
                "qty": qty,
                "price": price,
                "sector": STOCKS[symbol]["sector"],
                "notional": notional,
                "weight": self._weights.get(symbol, 0.0),
            }
        return {
            "positions": positions,
            "cash": self._cash,
            "portfolio_value": self._portfolio_value,
            "peak_value": self._peak_value,
            "drawdown_pct": self._drawdown_pct,
            "sector_exposure": dict(self._sector_exposure),
        }

    def create_shadow_copy(self) -> FinanceManager:
        """Create independent copy for SiLR verification."""
        shadow = FinanceManager.__new__(FinanceManager)
        shadow._time = self._time
        shadow._positions = dict(self._positions)
        shadow._prices = dict(self._prices)
        shadow._cash = self._cash
        shadow._portfolio_value = self._portfolio_value
        shadow._peak_value = self._peak_value
        shadow._weights = dict(self._weights)
        shadow._sector_exposure = dict(self._sector_exposure)
        shadow._drawdown_pct = self._drawdown_pct
        return shadow

    def solve(self) -> bool:
        """Recalculate portfolio metrics from current positions and prices.

        Always returns True — portfolio arithmetic never fails to converge.
        """
        # Total stock value
        stock_value = sum(
            self._positions[s] * self._prices[s] for s in self._positions
        )
        self._portfolio_value = stock_value + self._cash

        # Per-stock weights (fraction of total portfolio)
        if self._portfolio_value > 0:
            self._weights = {
                s: (self._positions[s] * self._prices[s]) / self._portfolio_value
                for s in self._positions
            }
        else:
            self._weights = {s: 0.0 for s in self._positions}

        # Sector exposure
        self._sector_exposure = {}
        for symbol, weight in self._weights.items():
            sector = STOCKS[symbol]["sector"]
            self._sector_exposure[sector] = self._sector_exposure.get(sector, 0.0) + weight

        # Peak and drawdown
        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value
        if self._peak_value > 0:
            self._drawdown_pct = (
                (self._peak_value - self._portfolio_value) / self._peak_value * 100
            )
        else:
            self._drawdown_pct = 0.0

        self._time += 1.0
        return True

    # ── Domain-specific operations ───────────────────────────────

    def adjust_position(self, symbol: str, qty_delta: int) -> bool:
        """Buy (positive) or sell (negative) shares.

        Returns False if the operation would result in negative shares or
        insufficient cash.
        """
        if symbol not in self._positions:
            return False
        new_qty = self._positions[symbol] + qty_delta
        if new_qty < 0:
            return False
        cost = qty_delta * self._prices[symbol]
        if cost > 0 and cost > self._cash:
            return False
        self._positions[symbol] = new_qty
        self._cash -= cost
        return True

    def liquidate_position(self, symbol: str) -> bool:
        """Sell entire position in a stock. Returns False if no position."""
        if symbol not in self._positions:
            return False
        qty = self._positions[symbol]
        if qty == 0:
            return False
        proceeds = qty * self._prices[symbol]
        self._positions[symbol] = 0
        self._cash += proceeds
        return True

    def rebalance_sector(self, sector: str, target_weight: float) -> bool:
        """Adjust all positions in a sector to achieve target total weight.

        Distributes the target weight equally among stocks in the sector.
        Returns False if target_weight is out of range or sector unknown.
        """
        if target_weight < 0 or target_weight > 1.0:
            return False

        sector_symbols = [s for s in STOCKS if STOCKS[s]["sector"] == sector]
        if not sector_symbols:
            return False

        # Target notional per stock
        target_per_stock = (target_weight * self._portfolio_value) / len(sector_symbols)

        total_delta_cash = 0.0
        new_positions = {}
        for symbol in sector_symbols:
            price = self._prices[symbol]
            target_qty = math.floor(target_per_stock / price) if price > 0 else 0
            new_positions[symbol] = max(target_qty, 0)
            delta = new_positions[symbol] - self._positions[symbol]
            total_delta_cash -= delta * price

        # Check cash feasibility (total_delta_cash > 0 means selling, < 0 buying)
        if self._cash + total_delta_cash < 0:
            return False

        for symbol, qty in new_positions.items():
            delta = qty - self._positions[symbol]
            self._cash -= delta * self._prices[symbol]
            self._positions[symbol] = qty
        return True

    def set_price(self, symbol: str, price: float) -> None:
        """Override price for a stock (used by scenario injection)."""
        if symbol in self._prices:
            self._prices[symbol] = price

    # ── Helpers for tool schemas ─────────────────────────────────

    def get_symbols(self) -> list[str]:
        """Return all stock symbols."""
        return sorted(self._positions.keys())

    def get_sectors(self) -> list[str]:
        """Return all sectors."""
        return sorted(set(STOCKS[s]["sector"] for s in STOCKS))
