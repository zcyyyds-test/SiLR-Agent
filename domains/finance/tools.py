"""Portfolio compliance tools for SiLR verification.

4 tools: 1 observe + 3 action.
All inherit from BaseTool for framework compatibility.
"""

from __future__ import annotations

from silr.tools.base import BaseTool
from silr.exceptions import DeviceNotFoundError, ValidationError

from .manager import STOCKS


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
        }


class AdjustPositionTool(BaseTool):
    """Buy or sell shares of a stock."""

    name = "adjust_position"
    description = "Buy (positive qty_delta) or sell (negative qty_delta) shares of a stock"

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
    """Sell entire position in a stock."""

    name = "liquidate_position"
    description = "Sell all shares of a stock to fully exit the position"

    def _validate_params(self, symbol: str = "", **kwargs) -> None:
        if not symbol:
            raise ValidationError("symbol is required")
        if symbol not in STOCKS:
            raise DeviceNotFoundError(
                f"Unknown symbol: {symbol}. "
                f"Available: {sorted(STOCKS.keys())}"
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


class RebalanceSectorTool(BaseTool):
    """Rebalance all positions in a sector to a target weight."""

    name = "rebalance_sector"
    description = "Adjust all positions in a sector to achieve a target portfolio weight"

    def _validate_params(self, sector: str = "", target_weight: float = 0.0,
                         **kwargs) -> None:
        if not sector:
            raise ValidationError("sector is required")
        valid_sectors = sorted(set(STOCKS[s]["sector"] for s in STOCKS))
        if sector not in valid_sectors:
            raise DeviceNotFoundError(
                f"Unknown sector: {sector}. Available: {valid_sectors}"
            )
        if target_weight < 0 or target_weight > 1.0:
            raise ValidationError(
                f"target_weight must be between 0.0 and 1.0, got {target_weight}"
            )

    def _run(self, sector: str = "", target_weight: float = 0.0,
             **kwargs) -> dict:
        mgr = self.manager
        success = mgr.rebalance_sector(sector, target_weight)
        return {
            "sector": sector,
            "target_weight": target_weight,
            "success": success,
            "message": (
                f"Rebalanced {sector} sector to {target_weight:.1%} target weight"
                if success
                else f"Failed to rebalance {sector} (insufficient cash)"
            ),
        }


def create_finance_toolset(manager) -> dict:
    """Create toolset for the portfolio compliance domain."""
    tools = [
        GetPortfolioStatusTool(manager),
        AdjustPositionTool(manager),
        LiquidatePositionTool(manager),
        RebalanceSectorTool(manager),
    ]
    return {t.name: t for t in tools}
