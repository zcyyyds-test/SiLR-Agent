"""Mine portfolio stress scenarios from historical CSV data.

Reads `domains/finance/data/close_prices.csv` (8 tickers × 6 years of daily
closes), slides multiple window sizes across the timeline, and emits each
window whose price moves would push the baseline portfolio out of compliance
as a `FinanceScenario`. Returns are applied to the 2024-01-02 baseline prices
defined in `domains/finance/manager.py`.

The output module is import-compatible with `domains.finance.scenarios` —
each entry uses the same `FinanceScenario` dataclass — so it can be fed to
the existing eval / SFT / GRPO pipelines without code changes.

Usage:
    python scripts/mine_finance_scenarios.py \\
        --csv domains/finance/data/close_prices.csv \\
        --output domains/finance/scenarios_mined.py \\
        --windows 10,20,30,60 \\
        --min-shock 0.12

Determinism: same CSV + same flags → same output (sorted by window start date).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# Mirror manager.py — kept inline so this script has zero project deps
# and can run on stock Python (no silr import path needed at mine time).
BASELINE_PRICES = {
    "AAPL": 183.73, "MSFT": 364.59, "NVDA":  48.14,
    "JNJ":  149.66, "PFE":   25.70, "UNH":  514.26,
    "XOM":   94.85, "CVX":  135.63,
}
SECTORS = {
    "AAPL": "tech",   "MSFT": "tech",   "NVDA": "tech",
    "JNJ":  "health", "PFE":  "health", "UNH":  "health",
    "XOM":  "energy", "CVX":  "energy",
}
DEFAULT_POSITIONS = {
    "AAPL": 680, "MSFT": 343, "NVDA": 2596,
    "JNJ":  835, "PFE":  4864, "UNH":  243,
    "XOM":  1318, "CVX":  921,
}
DEFAULT_CASH = 100_000.0

# Compliance thresholds (mirror checkers.py)
POS_CEILING = 0.20
POS_FLOOR = 0.04
SECTOR_CEILING = 0.40
SECTOR_FLOOR = 0.15
CASH_FLOOR = 0.05


@dataclass
class MinedScenario:
    id: str
    description: str
    source_event: str
    price_changes: dict
    difficulty: str
    window_start: str
    window_end: str
    max_abs_return: float
    triggers: tuple
    cash_override: float | None = None


def load_csv(path: Path) -> list[tuple[str, dict]]:
    """Return [(date_str, {symbol: close_price})] sorted by date."""
    rows = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        symbols = [c for c in reader.fieldnames if c != "Date"]
        for r in reader:
            prices = {s: float(r[s]) for s in symbols}
            rows.append((r["Date"], prices))
    rows.sort(key=lambda x: x[0])
    return rows


def compute_violations(price_changes: dict, cash_override: float | None = None) -> list:
    """Apply returns to baseline portfolio, return list of violation tags.

    Mirrors manager.solve() + checkers in pure stdlib so we can filter
    scenarios at mining time without importing silr.

    `cash_override` lets the miner stress different cash buffers — lower
    cash makes `cash_floor` easier to trigger and biases the portfolio
    toward multi-violation scenarios.
    """
    cash = DEFAULT_CASH if cash_override is None else cash_override
    new_prices = {s: BASELINE_PRICES[s] * (1 + price_changes.get(s, 0.0))
                  for s in BASELINE_PRICES}
    notionals = {s: DEFAULT_POSITIONS[s] * new_prices[s] for s in BASELINE_PRICES}
    stock_value = sum(notionals.values())
    pv = stock_value + cash
    if pv <= 0:
        return []
    weights = {s: notionals[s] / pv for s in BASELINE_PRICES}
    sectors = {}
    for s, w in weights.items():
        sectors[SECTORS[s]] = sectors.get(SECTORS[s], 0.0) + w

    triggers = []
    for s, w in weights.items():
        if w > POS_CEILING:
            triggers.append(f"pos_ceiling:{s}")
        elif 0 < w < POS_FLOOR:
            triggers.append(f"pos_floor:{s}")
    for sec, w in sectors.items():
        if w > SECTOR_CEILING:
            triggers.append(f"sector_ceiling:{sec}")
        elif w < SECTOR_FLOOR:
            triggers.append(f"sector_floor:{sec}")
    if cash / pv < CASH_FLOOR:
        triggers.append("cash_floor")
    return triggers


def classify_difficulty(returns: dict, triggers: list) -> str:
    max_abs = max(abs(r) for r in returns.values()) if returns else 0.0
    n_triggers = len(triggers)
    bidir = (any(t.startswith("pos_ceiling") or t.startswith("sector_ceiling")
                 for t in triggers) and
             any(t.startswith("pos_floor") or t.startswith("sector_floor")
                 for t in triggers))
    if max_abs < 0.35 and n_triggers <= 3 and not bidir:
        return "easy"
    if max_abs < 0.65 and n_triggers <= 5:
        return "medium"
    return "hard"


def returns_signature(returns: dict, cash_override: float | None = None,
                      bucket: float = 0.025) -> tuple:
    """Hash key for dedup — round each return to nearest `bucket`.

    bucket=0.025 means two windows whose returns differ by < ~2.5pp on every
    name collapse into one scenario. Tighter bucket → more scenarios kept.
    Cash override is part of the key — same returns at different cash
    levels are distinct scenarios.
    """
    cash_key = "default" if cash_override is None else int(cash_override // 5000)
    return ("cash", cash_key) + tuple(sorted(
        (s, round(returns[s] / bucket) * bucket)
        for s in returns
    ))


def mine(rows: list, windows: list, min_shock: float, stride_frac: float,
         cash_levels: list | None = None,
         require_multi_constraint: bool = False,
         require_bidir: bool = False) -> list:
    """Slide each window length across `rows`, keeping windows that trigger ≥1 violation.

    `cash_levels` (list of floats or None=default $100K) enumerates baseline cash
    levels to stress. Each (window, cash_level) combo is a separate scenario,
    de-duplicated by (cash_bucket, returns) signature.

    `require_multi_constraint=True` only keeps windows that trigger ≥2 distinct
    violation types (e.g. sector_ceiling + cash_floor) — used to build a pool
    where pure "sell the over-weight tech name" policies don't suffice.
    """
    if cash_levels is None:
        cash_levels = [None]  # default DEFAULT_CASH
    by_sig = {}
    for window in windows:
        if window >= len(rows):
            continue
        stride = max(1, int(window * stride_frac))
        for i in range(0, len(rows) - window, stride):
            d_start, p_start = rows[i]
            d_end, p_end = rows[i + window]
            returns = {s: p_end[s] / p_start[s] - 1 for s in p_start}
            max_abs = max(abs(r) for r in returns.values())
            if max_abs < min_shock:
                continue
            kept = {s: round(r, 4) for s, r in returns.items() if abs(r) >= 0.05}
            if not kept:
                continue
            for cash_lvl in cash_levels:
                triggers = compute_violations(kept, cash_override=cash_lvl)
                if not triggers:
                    continue
                if require_multi_constraint:
                    trigger_types = {t.split(":")[0] for t in triggers}
                    if len(trigger_types) < 2:
                        continue
                if require_bidir:
                    has_ceiling = any(t.startswith("pos_ceiling")
                                      or t.startswith("sector_ceiling")
                                      for t in triggers)
                    has_floor = any(t.startswith("pos_floor")
                                    or t.startswith("sector_floor")
                                    for t in triggers)
                    if not (has_ceiling and has_floor):
                        continue
                sig = returns_signature(kept, cash_override=cash_lvl)
                existing = by_sig.get(sig)
                if existing is not None and existing.window_end <= d_end:
                    continue
                difficulty = classify_difficulty(kept, triggers)
                cash_tag = "" if cash_lvl is None else f"_cash{int(cash_lvl/1000)}k"
                scen = MinedScenario(
                    id=f"mined_{d_start}_to_{d_end}{cash_tag}".replace("-", ""),
                    description=_describe(kept, triggers),
                    source_event=f"Historical window {d_start} → {d_end} ({window}d)"
                                 + (f", cash=${cash_lvl:,.0f}" if cash_lvl else ""),
                    price_changes={s: round(BASELINE_PRICES[s] * (1 + r), 2)
                                   for s, r in kept.items()},
                    difficulty=difficulty,
                    window_start=d_start,
                    window_end=d_end,
                    max_abs_return=round(max_abs, 4),
                    triggers=tuple(sorted(set(triggers))),
                    cash_override=cash_lvl,
                )
                by_sig[sig] = scen
    out = list(by_sig.values())
    out.sort(key=lambda s: (s.window_start, s.window_end))
    return out


def _describe(returns: dict, triggers: list) -> str:
    movers = sorted(returns.items(), key=lambda kv: -abs(kv[1]))[:3]
    mover_str = ", ".join(f"{s} {r*100:+.0f}%" for s, r in movers)
    trig_str = ", ".join(sorted({t.split(':')[0] for t in triggers}))
    return f"{mover_str}; triggers: {trig_str}"


def render_module(scenarios: list, source_csv: str, args, var_name: str = "MINED_SCENARIOS") -> str:
    header = f'''"""Auto-mined portfolio stress scenarios.

Generated by `scripts/mine_finance_scenarios.py` from `{source_csv}`.
Do not edit by hand — re-run the miner to refresh.

Settings: windows={args.windows} min_shock={args.min_shock} stride_frac={args.stride_frac} cash_levels={args.cash_levels} require_multi_constraint={args.require_multi_constraint}
Total scenarios: {len(scenarios)}
"""

from .scenarios import FinanceScenario

{var_name} = [
'''
    body_lines = []
    for s in scenarios:
        pc = ", ".join(f'"{sym}": {p:.2f}' for sym, p in s.price_changes.items())
        cash_line = (f'\n        cash_override={s.cash_override},'
                     if s.cash_override is not None else "")
        body_lines.append(
            f'    FinanceScenario(\n'
            f'        id="{s.id}",\n'
            f'        description="{s.description}",\n'
            f'        source_event="{s.source_event}",\n'
            f'        price_changes={{{pc}}},{cash_line}\n'
            f'        difficulty="{s.difficulty}",\n'
            f'    ),'
        )
    return header + "\n".join(body_lines) + "\n]\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="domains/finance/data/close_prices.csv")
    parser.add_argument("--output", default="domains/finance/scenarios_mined.py")
    parser.add_argument("--windows", default="10,20,30,60",
                        help="Comma-separated window sizes in trading days")
    parser.add_argument("--min-shock", type=float, default=0.12,
                        help="Min |return| of any single name to keep window")
    parser.add_argument("--stride-frac", type=float, default=0.5,
                        help="Window stride = window * this (avoid near-duplicate adjacents)")
    parser.add_argument("--max-output", type=int, default=0,
                        help="If >0, cap output count (deterministic by date order)")
    parser.add_argument("--cash-levels", default="",
                        help="Comma-separated cash overrides (e.g. '20000,40000,70000'). "
                             "Empty = use only default $100K. Lower cash makes "
                             "cash_floor easier to trigger and biases toward "
                             "multi-violation scenarios.")
    parser.add_argument("--require-multi-constraint", action="store_true",
                        help="Only keep scenarios that trigger ≥2 distinct "
                             "violation types (sector_ceiling + cash_floor, etc.)")
    parser.add_argument("--require-bidir", action="store_true",
                        help="Only keep scenarios that trigger BOTH a ceiling "
                             "(sector_ceiling or pos_ceiling) AND a floor "
                             "(sector_floor or pos_floor) — requires opposite-"
                             "direction trades within the same recovery episode.")
    parser.add_argument("--var-name", default="MINED_SCENARIOS",
                        help="Module variable name in output file")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1
    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows: {rows[0][0]} → {rows[-1][0]}", file=sys.stderr)

    windows = [int(w) for w in args.windows.split(",")]
    cash_levels = ([float(c) for c in args.cash_levels.split(",") if c.strip()]
                   if args.cash_levels else None)
    if cash_levels and None not in cash_levels:
        # Always include the default cash level alongside overrides
        cash_levels = [None] + cash_levels
    scenarios = mine(rows, windows, args.min_shock, args.stride_frac,
                     cash_levels=cash_levels,
                     require_multi_constraint=args.require_multi_constraint,
                     require_bidir=args.require_bidir)
    tag_parts = []
    if args.require_multi_constraint: tag_parts.append("multi-constraint")
    if args.require_bidir: tag_parts.append("BIDIR")
    tag = f" ({' + '.join(tag_parts)})" if tag_parts else ""
    print(f"Mined {len(scenarios)} unique stress scenarios{tag}", file=sys.stderr)

    by_diff = {}
    for s in scenarios:
        by_diff[s.difficulty] = by_diff.get(s.difficulty, 0) + 1
    for d in ("easy", "medium", "hard"):
        print(f"  {d}: {by_diff.get(d, 0)}", file=sys.stderr)

    if args.max_output > 0 and len(scenarios) > args.max_output:
        scenarios = scenarios[:args.max_output]
        print(f"Capped to {len(scenarios)} (deterministic prefix)", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_module(scenarios, str(csv_path), args, var_name=args.var_name))
    print(f"Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
