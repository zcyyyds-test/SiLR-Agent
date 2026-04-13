"""System prompt builder for the portfolio compliance domain."""

from __future__ import annotations

from ..manager import STOCKS, MAX_TRADE_VALUE


def build_finance_system_prompt(manager, tool_schemas: list[dict]) -> str:
    """Build a system prompt describing the portfolio and compliance rules.

    Args:
        manager: FinanceManager instance.
        tool_schemas: List of OpenAI function-calling schema dicts.

    Returns:
        System prompt string for LLM consumption.
    """
    state = manager.system_state

    # Universe summary
    universe_lines = []
    for sym in sorted(STOCKS.keys()):
        info = STOCKS[sym]
        price = manager._prices[sym]
        universe_lines.append(
            f"  {sym}: ${price:.2f}, sector={info['sector']}, "
            f"daily_vol={info['daily_volume']:,}"
        )
    universe_block = "\n".join(universe_lines)

    # Current portfolio summary
    pv = state["portfolio_value"]
    cash = state["cash"]
    cash_pct = (cash / pv * 100) if pv > 0 else 0
    dd_pct = state["drawdown_pct"]

    sector_lines = []
    for sector, weight in sorted(state["sector_exposure"].items()):
        sector_lines.append(f"  {sector}: {weight * 100:.1f}%")
    sector_block = "\n".join(sector_lines)

    tool_names = [s["function"]["name"] for s in tool_schemas]
    max_trade = MAX_TRADE_VALUE

    prompt = f"""\
You are a portfolio compliance agent. Your job is to restore a portfolio to a
compliant state after market stress events, using the minimum number of actions.

## Stock Universe

{universe_block}

## Current Portfolio

  Total value: ${pv:,.2f}
  Cash: ${cash:,.2f} ({cash_pct:.1f}%)
  Drawdown from peak: {dd_pct:.1f}%

Sector exposure:
{sector_block}

## Compliance Rules (all must hold simultaneously)

1. **Position ceiling**: No single stock may exceed 20% of portfolio value.
2. **Position floor**: Every held stock must be at least 4% of portfolio value.
3. **Sector ceiling**: No single sector may exceed 40% of portfolio value.
4. **Sector floor**: Every sector must be at least 15% of portfolio value.
5. **Cash reserve**: Cash must be at least 5% of portfolio value.
6. **Drawdown limit**: Portfolio drawdown from peak must not exceed 8% (monitored).

## Available Actions

{', '.join(tool_names)}

## Response Format

Respond with a JSON object containing exactly ONE action:
{{"tool_name": "<action>", "params": {{...}}}}

If all compliance rules are satisfied and no action is needed:
{{"tool_name": "none", "params": {{}}}}

## Trading Constraint

  Per-trade notional limit: ${max_trade:,.0f}
  Large positions must be sold in multiple steps using adjust_position.
  liquidate_position only works if position notional ≤ ${max_trade:,.0f}.

## Strategy Guidelines

1. **Read the observation first**: Check which constraints are violated before acting.
   Both ceiling violations (overweight) and floor violations (underweight) must be fixed.
2. **Sell overweight, buy underweight**: Fixing ceilings requires selling; fixing floors
   requires buying. You often need both in the same episode.
3. **Sell before buy**: Selling generates cash; buying consumes it. Always sell overweight
   positions first to build cash, then buy underweight positions.
4. **Respect the per-trade limit**: Calculate how many shares fit within ${max_trade:,.0f}.
   Large positions require multiple sells across several steps.
5. **One action per step**: Plan a multi-step sequence carefully. Fix the most critical
   violation first, but consider the cash needed for later buys.
6. **Cascading effects**: Selling stock changes weights for ALL positions. Selling a
   tech stock may push an energy stock below its 4% floor, requiring a subsequent buy.
"""
    return prompt
