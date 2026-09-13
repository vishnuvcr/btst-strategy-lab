# BTST Strategy Lab

A research framework for discovering and validating robust Buy-Today-Sell-Tomorrow (BTST) strategies for Indian equities.

## Objective

This project is deliberately separate from the ML signal engine in `research-ML-trading`. It does not assume that machine learning is the best approach. It compares rule-based, factor, regime, ranking, and ML strategies under the same leakage-safe portfolio backtest.

### Strategy families

- Momentum / relative strength
- Mean reversion
- Closing-range strength
- Volume / accumulation
- Gap continuation and gap fade
- Market and India VIX regime filters
- Sector-relative strength
- ML probability/ranking models
- Hybrid filter + ranker strategies

## Research rules

1. Features at day *t* may only use information available by the BTST entry cutoff.
2. Entry is at the next session open unless explicitly configured otherwise.
3. Exit is the following session close by default, with optional stop/target rules.
4. Parameters are selected using walk-forward validation, not the final test period.
5. The final OOS period is never used for parameter selection.
6. Transaction costs and slippage are included.
7. Strategy ranking considers return, Sharpe, drawdown, profit factor, expectancy, turnover, trade count, and stability.
8. Survivorship bias is reported when historical constituent membership is unavailable.
9. No claim of profitability is made until the complete historical and OOS runs finish.

## Data sources

The configuration supports:

- NIFTY-50 daily stock data
- NIFTY-100 5-minute stock data
- NIFTY / NIFTY Bank / sector / India VIX market context
- Local CSV/Parquet datasets
- yfinance for recent validation snapshots

The repository does not commit large market datasets. GitHub Actions downloads configured datasets on demand.

## Outputs

The research pipeline produces:

- `docs/strategy_leaderboard.csv`
- `docs/strategy_leaderboard.json`
- `docs/oos_trades.csv`
- `docs/equity_curves.csv`
- `docs/research_manifest.json`

## Status

Initial framework: data adapters, feature generation, strategy competition, walk-forward validation, portfolio backtest, and GitHub Actions orchestration.

**This is a research system, not investment advice or an automated broker.**
