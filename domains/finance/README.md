# Portfolio Compliance Domain

A reference SiLR domain for mandate-gated portfolio rebalancing, with a complete SFT → GRPO post-training pipeline using dense reward shaping for observer-only constraints.

## Overview

This domain models an 8-stock equity portfolio across 3 sectors where an LLM agent must restore compliance after market stress events. Every proposed trade is verified on a shadow copy of the portfolio state before execution — trades that would exceed per-trade notional limits, deplete cash below reserve requirements, or violate position/sector constraints are rejected before reaching the real system.

The finance domain uses **observer-only constraints**: all compliance metrics are evaluated globally by the observer rather than per-action by verifier checkers. A $15K per-trade cap forces multi-step resolution — the agent must plan a sequence of smaller adjustments to fix violations.

## Portfolio Universe

8 US equities across 3 sectors (Tech, Health, Energy). Baseline prices from Yahoo Finance (2024-01-02). See [`data/README.md`](data/README.md) for provenance.

## Constraints

| Constraint | Type | Threshold |
|------------|------|-----------|
| Position concentration | Ceiling | ≤ 20% per stock |
| Sector exposure | Ceiling | ≤ 40% per sector |
| Cash reserve | Floor | ≥ 5% of portfolio |
| Position minimum | Floor | ≥ 4% per stock |
| Sector minimum | Floor | ≥ 15% per sector |
| Drawdown | Monitor-only | ≤ 8% |

## Tools

- `adjust_position(symbol, qty_delta)` — buy or sell shares
- `liquidate_position(symbol)` — sell entire holding

## Failure Scenarios

30 training + 10 held-out scenarios across 4 difficulty tiers (easy/medium/hard/extreme), derived from real historical events (COVID crash, 2022 tech selloff, Japan carry trade unwind, etc.). Max 8 steps per episode. Scenario specs live in [`scenarios.py`](scenarios.py).

## Training Pipeline

**SFT stage** — Teacher-generated recovery trajectories train a Qwen3-14B + LoRA student via QLoRA.

**GRPO stage** — Step-level GRPO with **dense reward**: violation-count delta provides per-step gradient signal, with a terminal recovery bonus. All scenarios are rolled out every iteration (no curriculum narrowing) to maintain generalization.

## Results

| Model | Recovery Rate (120 episodes) |
|-------|----------------------------|
| Qwen3-14B + SFT | 85.0% (102/120) |
| **Qwen3-14B + SFT + GRPO** | **92.5% (111/120)** |

Eval protocol: 3 repeats × 40 scenarios (30 train + 10 held-out), temperature=0.3, max 8 steps.

**Key improvements**:
- SFT → GRPO: **+7.5pp** recovery (+9 episodes)
- Held-out generalization: **100%** (30/30) vs SFT 90% (27/30)

## Application Context

The constraint structure and scenario design reflect compliance patterns in institutional equity portfolio management, where mandate violations must be corrected through incremental rebalancing under trade-size constraints.
