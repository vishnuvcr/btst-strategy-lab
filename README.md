# BTST Strategy Lab

A research framework for discovering and validating robust Buy-Today-Sell-Tomorrow (BTST) strategies for Indian equities.

## Objective

This project is deliberately separate from the ML signal engine in `research-ML-trading`. It does not assume that machine learning is the best approach. It compares rule-based, factor, regime, ranking, and ML strategies under the same portfolio backtest.

## Strategy families

- Momentum / relative strength
- Mean reversion
- Closing-range strength
- Volume / accumulation
- Gap continuation and gap fade
- Market-regime filters
- Sector-relative strength
- ML probability/ranking models
- Hybrid filter + ranker models

## Research v2 methodology

The default runner is `src/research_v2.py`.

1. Daily features at day *t* use only information available by the BTST entry cutoff.
2. Entry is the next trading session open.
3. Exit is the next session close unless the configured stop/target is reached.
4. Rule-based families tune `top_n` and minimum score on validation data, then freeze those parameters for the following OOS fold.
5. ML families fit only on the training window; probability thresholds are selected on validation data and then frozen for OOS.
6. Walk-forward folds use an embargo between validation and test windows.
7. OOS folds are concatenated only after each fold has been evaluated with parameters chosen without that fold's outcomes.
8. Costs and slippage are applied on both entry and exit.
9. Daily OHLC cannot reveal the intraday order of stop and target hits; the simulator uses stop-first ordering as the conservative assumption.
10. A strategy is marked `eligible` only when it clears the configured OOS trade-count, parameter-stability, and positive-evidence gates.
11. An infinite profit factor is valid when there are no losing trades; invalid/NaN PF is rejected.
12. The final leaderboard is a research ranking, not a claim that the top row will be profitable in live trading.

## Important data caveats

- The configured NIFTY-50 daily dataset may not provide point-in-time historical constituent membership. The manifest therefore reports a survivorship-bias warning unless a historical universe is supplied.
- `sector_relative` currently requires a real sector mapping to become genuinely sector-relative; without one it is a global relative-strength proxy and is explicitly flagged in the manifest.
- Daily data cannot model the exact intraday path of a stop/target. Use the optional 5-minute dataset for a later execution-refinement stage.

## Data sources

The configuration supports:

- NIFTY-50 daily stock data
- NIFTY-100 5-minute stock data
- NIFTY / NIFTY Bank / sector / India VIX market context
- Local CSV datasets
- yfinance for recent validation snapshots

Large market datasets are not committed to the repository. GitHub Actions downloads configured datasets on demand.

## Outputs

The research pipeline produces:

- `docs/strategy_leaderboard.csv`
- `docs/strategy_leaderboard.json`
- `docs/oos_trades.csv`
- `docs/research_manifest.json`
- `docs/data_audit.json`

## How to run

1. Open GitHub → **Actions** → **BTST Strategy Research**.
2. Run it manually with `max_symbols=200` for the full configured universe.
3. Keep `include_intraday=false` for the first daily BTST tournament; the large 5-minute dataset is reserved for execution refinement.
4. Inspect `docs/strategy_leaderboard.csv` and `docs/research_manifest.json` after completion.
5. Do not promote a strategy to paper trading until it survives cost sensitivity, regime/year splits, and a genuinely point-in-time universe check.

## Status

The research engine, compatibility layer, tests, data audit, forensic audit, and GitHub Actions orchestration are implemented. A corrected full historical tournament is being rerun after hardening the OOS eligibility gate. No performance result is considered valid until that corrected run and its forensic checks finish.

**This is a research system, not investment advice or an automated broker.**
