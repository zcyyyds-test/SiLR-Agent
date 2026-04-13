"""OpenAI function-calling tool schemas for portfolio compliance domain."""

from __future__ import annotations

from ..manager import STOCKS, MAX_TRADE_VALUE


def build_finance_tool_schemas(manager) -> list[dict]:
    """Build OpenAI function-calling schemas for portfolio tools.

    Args:
        manager: FinanceManager instance (for dynamic enums).

    Returns:
        List of schema dicts in OpenAI function-calling format.
    """
    symbols = manager.get_symbols()

    return [
        {
            "type": "function",
            "function": {
                "name": "get_portfolio_status",
                "description": (
                    "Get current portfolio positions, weights, sector exposure, "
                    "cash, and drawdown."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "adjust_position",
                "description": (
                    "Buy (positive qty_delta) or sell (negative qty_delta) "
                    f"shares of a stock. Single trade notional capped at "
                    f"${MAX_TRADE_VALUE:,.0f}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol.",
                            "enum": symbols,
                        },
                        "qty_delta": {
                            "type": "integer",
                            "description": (
                                "Number of shares to buy (positive) or sell (negative). "
                                f"Trade notional must not exceed ${MAX_TRADE_VALUE:,.0f}."
                            ),
                        },
                    },
                    "required": ["symbol", "qty_delta"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "liquidate_position",
                "description": (
                    f"Sell all shares of a stock. Only works if position notional "
                    f"≤ ${MAX_TRADE_VALUE:,.0f}. For larger positions, use "
                    f"adjust_position to sell in chunks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol to liquidate.",
                            "enum": symbols,
                        },
                    },
                    "required": ["symbol"],
                },
            },
        },
    ]
