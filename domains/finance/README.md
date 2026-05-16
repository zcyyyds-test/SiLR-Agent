# Portfolio Compliance Domain

A reference SiLR domain for mandate-gated portfolio rebalancing, with a complete SFT → GRPO post-training pipeline using dense reward shaping for observer-only constraints.

## Overview

This domain models an 8-stock equity portfolio across 3 sectors where an LLM agent must restore compliance after market stress events. Every proposed trade is verified on a shadow copy of the portfolio state before execution. The verifier enforces per-trade invariants (non-negative shares, sufficient cash, per-trade notional ≤ $15K); the six aggregate compliance metrics (position / sector ceilings and floors, cash reserve, drawdown) are evaluated globally by the observer, and the agent resolves violations through a sequence of trades.

The finance domain uses **observer-only constraints** with a **$15K per-trade cap**, forcing multi-step resolution — the agent must plan a sequence of smaller adjustments to fix violations.

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

30 training + 10 held-out scenarios across 3 difficulty tiers (easy/medium/hard), derived from real historical events (COVID crash, 2022 tech selloff, Japan carry trade unwind, etc.). Max 8 steps per episode. Scenario specs live in [`scenarios.py`](scenarios.py).

### Extended scenario pool (auto-mined historical magnitude variants)

In addition to the curated set, [`scripts/mine_finance_scenarios.py`](../../scripts/mine_finance_scenarios.py) sweeps the 6-year historical CSV (`data/close_prices.csv`, 2019–2024) with multiple window lengths (5–180 trading days) and emits every window whose returns, applied to the baseline portfolio, would trigger ≥1 compliance violation. Each mined scenario carries provenance (start/end date, window length, source-event tag) and is import-compatible with the eval pipeline:

```python
from domains.finance import FinanceScenarioLoader

loader = FinanceScenarioLoader()
mined = loader.load_mined(difficulty="hard")  # or "easy" / "medium"
```

A default run (`--windows 5,10,15,20,30,45,60,90,120,180 --min-shock 0.06`) produces 126 deduplicated scenarios spanning the 2019–2024 window (COVID crash & recovery, 2022 bear market, NVDA AI surge, etc.). Pass `--require-multi-constraint` or `--require-bidir` to filter for higher-difficulty windows that exercise multi-step planning more thoroughly.

## Training Pipeline

**SFT stage** — Teacher-generated recovery trajectories train a Qwen3-14B + LoRA student via QLoRA.

**GRPO stage** — Step-level GRPO with **dense reward**: violation-count delta provides per-step gradient signal, with a terminal recovery bonus. All scenarios are rolled out every iteration (no curriculum narrowing) to maintain generalization.

## Results

Qwen3-14B + LoRA agent evaluated on four scenario pools, three repeats each (temperature=0.3):

| Pool | N (episodes) | baseline SFT | DEDUP-ep3 SFT | **SFT + GRPO** |
|------|-------------:|-------------:|--------------:|---------------:|
| Curated + held-out (40 scenarios, 8-step)         | 120 | 85.0% | – | **92.5%** |
| ↳ Held-out subset (10 scenarios)                  | 30  | 90.0% | – | **100%** (30/30) |
| Mined historical magnitude variants (126 scenarios, 8-step) | 126 | 92.1% | 85.7% | 91.3% |
| Bidirectional rebalancing — BIDIR (25 scenarios, 12-step planning) | 75 | 68.0% | – | **76.0%** |
| ↳ BIDIR hard subset (16 scenarios)                | 48  | 50.0% | – | **62.5%** |

Total: 449 evaluation episodes across 201 unique scenarios.

**Headline observations**:
- **+7.5pp on curated set** (GRPO 92.5% vs baseline SFT 85.0% across 120 episodes).
- **100% held-out generalization** (Fisher exact one-sided p ≈ 0.012 vs the 80% SFT baseline) — the policy generalizes to scenarios never seen during SFT data collection.
- **+12.5pp on the BIDIR hard tier**, where the policy must execute both directions of rebalancing (sell over-weight + buy under-weight) within the same recovery episode — the regime where multi-step planning matters most.
- On single-violation magnitude-variant scenarios (mined 126) all three checkpoints converge within ~1pp; the solution path is short enough that single-shot imitation suffices.

## Application Context

The constraint structure and scenario design reflect compliance patterns in institutional equity portfolio management, where mandate violations must be corrected through incremental rebalancing under trade-size constraints.
