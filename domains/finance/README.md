# Portfolio Compliance Domain

A reference SiLR domain for mandate-gated portfolio rebalancing, with a complete SFT → GRPO post-training pipeline using violation-count dense reward shaping.

## Overview

This domain models an 8-stock equity portfolio across 3 sectors where an LLM agent must restore compliance after market stress events. Every proposed trade is verified on a shadow copy of the portfolio state before execution — trades that would exceed per-trade notional limits, deplete cash below reserve requirements, or violate position/sector constraints are rejected before reaching the real system.

Unlike the cluster and grid domains where the verifier's constraint checkers gate individual actions, the finance domain uses **observer-only constraints**: all 6 compliance metrics (position concentration, sector exposure, cash reserve, drawdown, position floor, sector floor) are evaluated globally by the observer. This design choice reflects the reality that portfolio compliance is a multi-trade objective — a single trade rarely fixes a violation, but the $15K per-trade cap forces the agent to plan multi-step trade sequences.

## Portfolio Universe

| Symbol | Sector | Baseline Price (2024-01-02) |
|--------|--------|----------------------------|
| AAPL, MSFT, NVDA | Tech | $185.64, $376.04, $48.16 |
| JNJ, PFE, UNH | Health | $156.74, $28.74, $527.01 |
| XOM, CVX | Energy | $99.57, $149.43 |

Baseline prices sourced from Yahoo Finance. See [`data/README.md`](data/README.md) for provenance.

## Constraints

| Constraint | Type | Threshold |
|------------|------|-----------|
| Position concentration | Ceiling | ≤ 20% per stock |
| Sector exposure | Ceiling | ≤ 40% per sector |
| Cash reserve | Floor | ≥ 5% of portfolio value |
| Position minimum | Floor | ≥ 4% per stock |
| Sector minimum | Floor | ≥ 15% per sector |
| Drawdown | Monitor-only | ≤ 8% (not fixable by trading) |

**Per-trade limit**: $15,000 notional cap on any single trade. This forces multi-step resolution — the agent cannot fix a 25% position in one trade but must plan a sequence of smaller adjustments.

## Tools

- `adjust_position(symbol, qty_delta)` — buy or sell shares (positive = buy, negative = sell)
- `liquidate_position(symbol)` — sell entire holding to zero

## Failure Scenarios

30 training scenarios + 10 held-out across 4 difficulty tiers:

| Difficulty | Count | Examples |
|------------|-------|---------|
| Easy | 5 | Single stock concentration, simple sector breach |
| Medium | 10 | Dual-stock spikes, sector rotation, cash depletion |
| Hard | 12 | Three-way rotations, cascade failures, worst-case all-six violations |
| Extreme | 3 | Full market crash, compound multi-sector stress |

Scenario return magnitudes are derived from real historical events (COVID crash, 2022 tech selloff, Japan carry trade unwind, etc.). Max 8 steps per episode.

Scenario specs live in [`scenarios.py`](scenarios.py).

## Training Pipeline

**SFT stage** — A teacher model (Gemini Flash) generates recovery trajectories; duplicates are removed by (scenario_id, action_sequence) deduplication. The cleaned dataset (140 unique conversations) trains a Qwen3-14B + LoRA student via QLoRA with domain system prompt injection. See `scripts/train_sft.py`.

**GRPO stage** — Step-level GRPO with **dense reward shaping** designed for observer-only constraint domains:

```
step_reward = -0.02 (step cost)
            + 0.5 × (prev_violations − curr_violations)   # per-step progress
            + 5.0 × recovered                              # terminal bonus
            − 0.1 × rejected                               # soft penalty
```

Unlike the cluster domain's sparse accept/reject reward, the violation-count delta provides per-step gradient signal, preventing the "trivially legal loop" failure mode where the agent repeatedly executes valid-but-useless actions. No curriculum filtering — all 30 scenarios are rolled out every iteration to prevent catastrophic forgetting.

## Results

| Model | Train (30) | Held-out (10) | All (40) | Wilson 95% CI |
|-------|-----------|---------------|----------|---------------|
| Qwen3-14B + SFT (baseline) | 93.3% | 80.0% | 90.0% | [77, 96] |
| Qwen3-14B + SFT (DEDUP-ep3) | 83.3% | 90.0% | 85.0% | [71, 93] |
| **Qwen3-14B + SFT + GRPO** | **90.0%** | **100.0%** | **92.5%** | **[86, 96]** |

Eval protocol: 3 repeats × 40 scenarios, temperature=0.3, max 8 steps per episode.

**Key improvements**:
- GRPO fixed 5 of 6 previously-failing scenarios (`cash_depleted`, `dual_spike_energy_drop`, `health_selloff_tech_boom`, `liquidity_crisis`, `ood_quadruple_shock`)
- Held-out generalization: 100% (30/30 across 3 repeats) — the dense reward teaches a general "reduce violations" strategy rather than scenario-specific memorization

## Application Context

The portfolio model, constraint structure, and scenario design reflect compliance patterns in institutional equity portfolio management, where mandate violations (concentration limits, sector caps, cash reserves) must be corrected through incremental rebalancing under trade-size constraints.
