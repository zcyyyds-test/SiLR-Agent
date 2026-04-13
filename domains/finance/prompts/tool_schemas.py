"""OpenAI function-calling tool schemas for portfolio compliance domain."""

from __future__ import annotations

from ..manager import STOCKS


def build_finance_tool_schemas(manager) -> list[dict]:
    """Build OpenAI function-calling schemas for all 4 portfolio tools.

    Args:
        manager: FinanceManager instance (for dynamic enums).

    Returns:
        List of schema dicts in OpenAI function-calling format.
    """
    symbols = manager.get_symbols()
    sectors = manager.get_sectors()

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
                    "shares of a stock."
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
                                "Must be non-zero."
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
                "description": "Sell all shares of a stock to fully exit the position.",
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
        {
            "type": "function",
            "function": {
                "name": "rebalance_sector",
                "description": (
                    "Adjust all positions in a sector to achieve a target "
                    "portfolio weight. The weight is distributed equally among "
                    "stocks in the sector."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sector": {
                            "type": "string",
                            "description": "Sector to rebalance.",
                            "enum": sectors,
                        },
                        "target_weight": {
                            "type": "number",
                            "description": (
                                "Target total weight for the sector as a decimal "
                                "(e.g., 0.30 for 30%)."
                            ),
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["sector", "target_weight"],
                },
            },
        },
    ]
