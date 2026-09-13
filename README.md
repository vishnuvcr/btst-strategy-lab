# NSE Daily Swing Strategy Lab

A leakage-safe research framework for discovering and validating daily swing strategies across the broader NSE equity universe.

The original BTST engine remains available for comparison. The new primary research path uses daily OHLCV and tests 1, 2, 3, 5, 10 and 20-session holding periods.

## Why the pivot

Reliable swing research can be performed from daily OHLCV. True intraday/day-trading research needs reliable intraday data; the project therefore does not pretend that daily data is intraday data.

## NSE-wide universe

The new runner downloads a dataset covering stocks currently listed on NSE and applies minimum history, price and rolling median turnover filters. NSE publishes daily security files, bhavcopy/market reports, and security-wise price/volume data.

The current public dataset used by the swing workflow is the Kaggle dataset `paramamithra/historical-data-of-stocks-listed-on-nse`, which documents daily OHLCV plus `Adj Close` and `Symbol` fields.

**Important:** this is a current-listed-stock universe, not point-in-time historical NSE membership. The manifest explicitly records the survivorship-bias limitation. We will not claim an unbiased historical NSE universe until historical membership is supplied.

## Price architecture

- Features/research returns: use `Adj Close` when available.
- Execution: use raw tradable `Open/High/Low/Close`.
- Stops/targets: use raw execution prices.
- Daily stop/target ambiguity: conservative stop-first ordering.

This prevents adjusted prices from accidentally becoming simulated execution prices.

## Holding periods

- 1 session — BTST benchmark
- 2 sessions
- 3 sessions
- 5 sessions — primary short swing horizon
- 10 sessions
- 20 sessions

## Strategy families

- Momentum
- Mean reversion
- Closing strength
- Volume
- Breakout
- Market regime
- Relative strength

ML ranking is retained as a later comparison layer; it will not be assumed superior to deterministic strategies.

## Walk-forward protocol

Each strategy/horizon combination uses chronological train/validation/test windows, an embargo, validation-based top-N selection, and untouched OOS test windows. Costs and slippage are applied to execution. Portfolio gross exposure is capped.

## Outputs

- `docs/swing_strategy_leaderboard.csv`
- `docs/swing_oos_trades.csv`
- `docs/swing_research_manifest.json`

No performance result is considered valid until the workflow completes its data checks and the resulting artifacts are inspected.

**This is a research system, not investment advice or an automated broker.**
