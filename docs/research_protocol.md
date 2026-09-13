# BTST Research Protocol

## Primary question

Which BTST strategy produces the most repeatable risk-adjusted out-of-sample performance in Indian equities after realistic costs and execution assumptions?

## Candidate families

1. Momentum
2. Mean reversion
3. Closing strength
4. Volume / accumulation
5. Gap continuation
6. Market regime
7. Sector-relative strength
8. ML ranking
9. Hybrid filter + ML ranker

## Label

For a signal formed at the close of session **t**, the base BTST return is:

`next_session_close / next_session_open - 1`

The signal cannot use any field from the next session.

## Validation

The intended production research loop is chronological:

`train -> validation -> untouched OOS test -> roll forward`

Validation selects thresholds/parameters. OOS is used only for evaluation.

## Execution model

- Entry: next session open
- Slippage: configurable per side
- Transaction cost: configurable per side
- Exit: next session close by default
- Optional stop and target based on ATR or fixed percentage
- If both stop and target are touched within a daily bar, the current conservative simulator assumes the stop is hit first because intraday ordering is unknown from daily OHLC.
- Maximum positions and gross exposure are enforced at the portfolio layer.

## Important caveats

The configured NIFTY-50 historical stock dataset must be audited for constituent survivorship. If it represents today's constituents over the full historical period rather than point-in-time membership, historical results can be upward biased.

The NIFTY-100 5-minute dataset is an optional refinement source. It should not be treated as a replacement for a point-in-time historical universe.

A strategy is not considered a candidate for deployment merely because it has the highest backtest return. It should also demonstrate sufficient trade count, acceptable drawdown, positive rolling stability, and performance that is not concentrated in one short period.
