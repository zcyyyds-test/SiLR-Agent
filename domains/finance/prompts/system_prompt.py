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

    # Render a compact action schema inline so the model knows the exact
    # parameter names (Kimi often substitutes `delta_qty`/`qty_change` for
    # `qty_delta` if the schema isn't shown). This is cheaper than trusting
    # the parser's alias fallback.
    action_schema_lines: list[str] = []
    for schema in tool_schemas:
        fn = schema.get("function", schema)
        name = fn.get("name", "?")
        props = (fn.get("parameters") or {}).get("properties", {})
        required = (fn.get("parameters") or {}).get("required", [])
        if props:
            param_desc = []
            for p, spec in props.items():
                req_marker = "" if p in required else "?"
                ptype = spec.get("type", "any")
                param_desc.append(f"{p}{req_marker}: {ptype}")
            action_schema_lines.append(f"  {name}({', '.join(param_desc)})")
        else:
            action_schema_lines.append(f"  {name}()")
    action_schema_block = "\n".join(action_schema_lines)

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

## Compliance Rules

Constraints 1-5 are ACTIONABLE — you must fix all of them through trades
before issuing ``none``. Constraint 6 is MONITOR-ONLY — the observation
reports it for context but it does NOT need to be zero to stop trading, and
it cannot be fixed by trades anyway (it recovers as market prices recover).

1. **Position ceiling**: No single stock may exceed 20% of portfolio value.
2. **Position floor**: Every held stock must be at least 4% of portfolio value.
3. **Sector ceiling**: No single sector may exceed 40% of portfolio value.
4. **Sector floor**: Every sector must be at least 15% of portfolio value.
5. **Cash reserve**: Cash must be at least 5% of portfolio value.
6. **Drawdown limit** (monitor-only): Portfolio drawdown from peak should
   stay under 8%. Reported for awareness; does not block the ``none`` action.

``n_violations`` in the observation counts only the actionable ones (1-5).
When ``n_violations == 0`` you MUST respond with
``{{"tool_name":"none","params":{{}}}}``.

## Available Actions

{action_schema_block}

Use these exact parameter names in your JSON — ``qty_delta`` is an integer
(positive = buy, negative = sell), ``symbol`` is a ticker from the universe
above. Do NOT rename fields (e.g. ``delta_qty`` or ``qty_change`` will fail).

## Response Format — CRITICAL

Your Thought must be ONE short declarative sentence (≤30 words, ~40 tokens).
No preamble, no priority lists, no markdown bullets, no double-checking.
State the violation being fixed and the trade, then emit the JSON action.

GOOD examples:

  Thought: NVDA at 22% exceeds the 20% ceiling; sell 155 NVDA (~$14.9K, under cap).

  {{"tool_name": "adjust_position", "params": {{"symbol": "NVDA", "qty_delta": -155}}}}

  Thought: Energy at 14% is below the 15% floor; buy 100 XOM (~$5.7K) to lift energy.

  {{"tool_name": "adjust_position", "params": {{"symbol": "XOM", "qty_delta": 100}}}}

BAD example (do NOT write like this):

  Thought: There are three violations. Let me analyze each one. First, tech sector
  is at 44.4%, which exceeds the 40% ceiling. NVDA is the largest tech position
  at 22.2%, so I should focus on it first. Per-trade limit is $15,000...

If all compliance rules are satisfied:

  Thought: All actionable constraints satisfied.

  {{"tool_name": "none", "params": {{}}}}

Aim for ≤50 tokens per Thought. Verbose reasoning wastes the 8-step budget.

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
6. **Cascading effects**: Selling stock changes weights for ALL positions.
   Selling from an overweight sector can push another sector below its
   15% floor (or an individual stock below its 4% floor), triggering a
   buy in a subsequent step.
7. **Retries, if any, stay silent**: If the system rejects your previous
   proposal (e.g. trade over $15K), treat the next step as a fresh
   observation. Do NOT mention the rejected attempt in your Thought —
   the rejected action is not part of the trajectory the student model
   will see at inference, so referencing it creates a missing-context
   artifact. Reason only from the observation you were just given.
"""
    return prompt
