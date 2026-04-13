"""Portfolio compliance tools for SiLR verification.

3 tools: 1 observe + 2 action.
Per-trade notional capped at MAX_TRADE_VALUE to force multi-step resolution.
All inherit from BaseTool for framework compatibility.
"""

from __future__ import annotations

from silr.tools.base import BaseTool
from silr.exceptions import DeviceNotFoundError, ValidationError

from .manager import STOCKS, MAX_TRADE_VALUE


class GetPortfolioStatusTool(BaseTool):
    """Observe portfolio state: positions, weights, sector exposure, drawdown."""

    name = "get_portfolio_status"
    description = "Get current portfolio positions, weights, sector exposure, cash, and drawdown"

    def _validate_params(self, **kwargs) -> None:
        pass

    def _run(self, **kwargs) -> dict:
        mgr = self.manager
        state = mgr.system_state
        positions = []
        for symbol in sorted(state["positions"].keys()):
            pos = state["positions"][symbol]
            positions.append({
                "symbol": symbol,
                "qty": pos["qty"],
                "price": pos["price"],
                "sector": pos["sector"],
                "notional": round(pos["notional"], 2),
                "weight_pct": round(pos["weight"] * 100, 2),
            })
        return {
            "positions": positions,
            "cash": round(state["cash"], 2),
            "portfolio_value": round(state["portfolio_value"], 2),
            "drawdown_pct": round(state["drawdown_pct"], 2),
            "sector_exposure": {
                s: round(w * 100, 2)
                for s, w in state["sector_exposure"].items()
            },
            "max_trade_value": MAX_TRADE_VALUE,
        }


class AdjustPositionTool(BaseTool):
    """Buy or sell shares of a stock, subject to per-trade notional limit."""

    name = "adjust_position"
    description = (
        f"Buy (positive qty_delta) or sell (negative qty_delta) shares of a stock. "
        f"Single trade notional capped at ${MAX_TRADE_VALUE:,.0f}."
    )

    def _validate_params(self, symbol: str = "", qty_delta: int = 0, **kwargs) -> None:
        if not symbol:
            raise ValidationError("symbol is required")
        if symbol not in STOCKS:
            raise DeviceNotFoundError(
                f"Unknown symbol: {symbol}. "
                f"Available: {sorted(STOCKS.keys())}"
            )
        if qty_delta == 0:
            raise ValidationError("qty_delta must be non-zero")
        trade_value = abs(qty_delta) * self.manager._prices.get(symbol, 0)
        if trade_value > MAX_TRADE_VALUE:
            max_qty = int(MAX_TRADE_VALUE / self.manager._prices[symbol])
            raise ValidationError(
                f"Trade value ${trade_value:,.0f} exceeds per-trade limit "
                f"${MAX_TRADE_VALUE:,.0f}. Max qty for {symbol}: {max_qty}"
            )

    def _run(self, symbol: str = "", qty_delta: int = 0, **kwargs) -> dict:
        mgr = self.manager
        success = mgr.adjust_position(symbol, qty_delta)
        action = "buy" if qty_delta > 0 else "sell"
        return {
            "symbol": symbol,
            "qty_delta": qty_delta,
            "action": action,
            "success": success,
            "message": (
                f"{action.title()} {abs(qty_delta)} shares of {symbol}"
                if success
                else f"Failed to {action} {abs(qty_delta)} shares of {symbol} (insufficient cash or shares)"
            ),
        }


class LiquidatePositionTool(BaseTool):
    """Sell entire position, only if notional is within per-trade limit."""

    name = "liquidate_position"
    description = (
        f"Sell all shares of a stock. Only works if position notional "
        f"≤ ${MAX_TRADE_VALUE:,.0f}. For larger positions, use adjust_position "
        f"to sell in chunks."
    )

    def _validate_params(self, symbol: str = "", **kwargs) -> None:
        if not symbol:
            raise ValidationError("symbol is required")
        if symbol not in STOCKS:
            raise DeviceNotFoundError(
                f"Unknown symbol: {symbol}. "
                f"Available: {sorted(STOCKS.keys())}"
            )
        qty = self.manager._positions.get(symbol, 0)
        if qty == 0:
            raise ValidationError(f"No position in {symbol}")
        notional = qty * self.manager._prices.get(symbol, 0)
        if notional > MAX_TRADE_VALUE:
            max_qty = int(MAX_TRADE_VALUE / self.manager._prices[symbol])
            raise ValidationError(
                f"Position notional ${notional:,.0f} exceeds per-trade limit "
                f"${MAX_TRADE_VALUE:,.0f}. Use adjust_position to sell "
                f"up to {max_qty} shares at a time."
            )

    def _run(self, symbol: str = "", **kwargs) -> dict:
        mgr = self.manager
        qty_before = mgr._positions.get(symbol, 0)
        success = mgr.liquidate_position(symbol)
        return {
            "symbol": symbol,
            "qty_sold": qty_before if success else 0,
            "success": success,
            "message": (
                f"Liquidated {qty_before} shares of {symbol}"
                if success
                else f"No position in {symbol} to liquidate"
            ),
        }


def create_finance_toolset(manager) -> dict:
    """Create toolset for the portfolio compliance domain."""
    tools = [
        GetPortfolioStatusTool(manager),
        AdjustPositionTool(manager),
        LiquidatePositionTool(manager),
    ]
    return {t.name: t for t in tools}
