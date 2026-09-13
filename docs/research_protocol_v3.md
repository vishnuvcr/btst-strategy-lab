# BTST Research Protocol V3

## Purpose
V3 hardens the research process before any strategy optimization or deployment claim.

## Required gates
1. Trading prices must be chronologically ordered before return diagnostics.
2. Stock OHLC must satisfy positive-price and OHLC consistency checks.
3. Corporate-action discontinuities must be explicitly audited; raw close is not assumed to be economically comparable across splits/bonus events.
4. Adjusted-price availability must be reported separately from execution prices. Adjusted prices may be used for signal construction only when their provenance is known; actual execution remains based on tradable OHLC.
5. Historical universe membership must not be assumed from today's NIFTY-50 membership when evaluating long historical periods. Any unavailable historical constituent reconstruction is reported as a limitation rather than silently treated as solved.
6. Sector-relative strategies require an explicit sector mapping; otherwise they are labelled proxy implementations and cannot be presented as true sector-relative results.
7. Hybrid strategies must combine independently defined rule and ML components; a model fallback alone is not a hybrid.
8. Portfolio constraints (risk per trade, ADV participation, gross exposure and sector cap) must be enforced by the simulator, not merely stored in configuration.
9. Daily OHLC stop/target ordering is ambiguous. The simulator uses a conservative convention and sensitivity to alternative ordering must be tested before deployment.
10. Strategy selection must use walk-forward OOS evidence and retain a final untouched holdout after research choices are frozen.

## Interpretation rule
A strategy is not called "best" unless it clears the hard eligibility gate on OOS data and remains robust under the V3 sensitivity suite. Negative or unstable OOS evidence is reported as a rejection, not optimized away.

## Current baseline
The existing tournament is a diagnostic baseline. Its latest successful run has no eligible strategy; ML Ranker is the least negative candidate but has negative OOS return, negative Sharpe and PF below 1. This is insufficient evidence for deployment.
