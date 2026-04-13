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

1. **Position concentration**: No single stock may exceed 20% of portfolio value.
2. **Sector exposure**: No single sector may exceed 40% of portfolio value.
3. **Drawdown limit**: Portfolio drawdown from peak must not exceed 8%.
4. **Cash reserve**: Cash must be at least 5% of portfolio value.

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
2. **Sell overweight positions**: If a stock or sector exceeds limits, sell shares
   to reduce weight. Prefer selling the most overweight positions first.
3. **Respect the per-trade limit**: Calculate how many shares you can sell in one
   step. For example, at $137/share with an $80,000 limit, sell at most 583 shares.
4. **Mind cash impact**: Selling increases cash; buying decreases it. If cash is
   already low, sell first before buying.
5. **One action per step**: The verifier checks each action independently. Plan
   a multi-step sequence: fix the most critical violation first.
6. **Cascading effects**: Selling stock to fix concentration may improve sector
   exposure and increase cash simultaneously. Think about side effects.
"""
    return prompt
