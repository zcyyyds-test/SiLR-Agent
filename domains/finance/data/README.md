# Market Data Source

Historical US equity OHLCV data used to design portfolio stress scenarios.

## Files

- `close_prices.csv` — Daily closing prices, 2019-01-02 to 2024-12-30
- `volumes.csv` — Daily trading volumes, same date range

## Source

Yahoo Finance via `yfinance` Python package.  Downloaded 2026-04-13.

## Tickers

| Symbol | Sector | Description |
|--------|--------|-------------|
| AAPL | Tech | Apple Inc. |
| MSFT | Tech | Microsoft Corp. |
| NVDA | Tech | NVIDIA Corp. |
| JNJ | Health | Johnson & Johnson |
| PFE | Health | Pfizer Inc. |
| UNH | Health | UnitedHealth Group |
| XOM | Energy | Exxon Mobil Corp. |
| CVX | Energy | Chevron Corp. |

## Usage

The runtime domain code does **not** read these CSVs directly.  Baseline
prices (2024-01-02 close) and scenario return magnitudes are hardcoded in
`manager.py` and `scenarios.py` for zero-dependency execution.  These files
serve as the reproducible provenance record.

## Key Historical Events Used in Scenarios

| Scenario | Source Event | Date Range |
|----------|-------------|------------|
| nvda_ai_surge | NVDA AI rally | Jan–Jul 2024 |
| rate_hike_drawdown | Fed 75bp hike | Jun 2022 |
| energy_covid_crash | COVID energy collapse | Feb–Mar 2020 |
| tech_selloff_rotation | Tech/energy rotation | Jan 2022 |
| covid_full_crash | COVID market crash | Feb–Mar 2020 |
| tech_bubble_burst | 2022 tech bear market | Jan–Oct 2022 |
