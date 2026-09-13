"""Forensic diagnostics and benchmark helpers for BTST research."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trade_return_sanity(trades: pd.DataFrame) -> dict:
    """Return deterministic sanity diagnostics; raise only on malformed inputs."""
    if trades.empty:
        return {"trades": 0, "finite_returns": True, "min_return": None, "max_return": None}
    required = {"return", "weight", "weighted_return", "signal_date"}
    missing = sorted(required - set(trades.columns))
    if missing:
        raise ValueError(f"Missing forensic trade columns: {missing}")
    r = pd.to_numeric(trades["return"], errors="coerce")
    w = pd.to_numeric(trades["weight"], errors="coerce")
    wr = pd.to_numeric(trades["weighted_return"], errors="coerce")
    finite = bool(np.isfinite(r).all() and np.isfinite(w).all() and np.isfinite(wr).all())

    # oos_trades.csv contains several strategies. Portfolio gross exposure must
    # therefore be checked within strategy/date, not by summing every strategy
    # into one fictional portfolio. If a single strategy is supplied, this
    # reduces to the ordinary signal-date exposure check.
    group_cols = ["signal_date"]
    if "strategy" in trades.columns:
        group_cols.insert(0, "strategy")
    daily_weight = w.groupby([trades[c] for c in group_cols]).sum()
    daily = wr.groupby([trades[c] for c in group_cols]).sum()
    combined_daily = wr.groupby(trades["signal_date"]).sum()
    return {
        "trades": int(len(trades)),
        "finite_returns": finite,
        "min_return": float(r.min()),
        "max_return": float(r.max()),
        "mean_return": float(r.mean()),
        "median_return": float(r.median()),
        "max_abs_weight": float(w.abs().max()),
        "max_abs_weighted_return": float(wr.abs().max()),
        "max_gross_weight": float(daily_weight.max()),
        "max_combined_gross_weight_across_strategies": float(combined_daily.max()),
        "min_daily_return": float(daily.min()),
        "max_daily_return": float(daily.max()),
        "pct_daily_below_minus_5pct": float((daily < -0.05).mean()),
        "pct_daily_below_minus_10pct": float((daily < -0.10).mean()),
    }


def equal_weight_buy_and_hold(df: pd.DataFrame) -> dict:
    """Equal-weight daily universe benchmark using close-to-close returns.

    This is a diagnostic universe benchmark, not a NIFTY index replacement.
    The source stock files are not adjusted-price data, so extreme corporate
    action/data errors can make a compounded benchmark meaningless. In that
    case the result is retained for diagnosis but explicitly marked invalid.
    """
    required = {"date", "symbol", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing benchmark columns: {missing}")
    x = df[["date", "symbol", "close"]].copy().sort_values(["symbol", "date"])
    x["ret"] = x.groupby("symbol")["close"].pct_change()
    daily = x.groupby("date")["ret"].mean().dropna().sort_index()
    if daily.empty:
        return {"days": 0, "total_return": 0.0, "mean_daily_return": 0.0, "valid": False}

    extreme_fraction = float((daily.abs() > 0.50).mean())
    valid = bool(extreme_fraction == 0.0 and abs(float(daily.mean())) < 0.10)
    equity = (1.0 + daily).cumprod()
    result = {
        "days": int(len(daily)),
        "date_start": str(pd.to_datetime(daily.index.min()).date()),
        "date_end": str(pd.to_datetime(daily.index.max()).date()),
        "total_return": float(equity.iloc[-1] - 1.0) if valid else None,
        "mean_daily_return": float(daily.mean()),
        "median_daily_return": float(daily.median()),
        "max_abs_daily_return": float(daily.abs().max()),
        "extreme_daily_return_fraction": extreme_fraction,
        "valid": valid,
    }
    if not valid:
        result["warning"] = "Close-to-close universe benchmark is not trusted because the raw stock data contain extreme daily returns; adjusted prices or constituent-aware data are required for a valid buy-and-hold benchmark."
    return result


def random_entry_benchmark(df: pd.DataFrame, top_n: int, repeats: int = 100, seed: int = 42) -> dict:
    """Monte-Carlo random next-session entries from the same eligible universe.

    Selection is performed independently by signal date, without using future returns.
    The returned distribution is useful for checking whether a ranked strategy beats
    random selection at comparable portfolio breadth.
    """
    required = {"date", "symbol", "next_open", "next_close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing random benchmark columns: {missing}")
    x = df[["date", "symbol", "next_open", "next_close"]].dropna().copy()
    x["next_return"] = x["next_close"] / x["next_open"] - 1.0
    groups = [g["next_return"].to_numpy(dtype=float) for _, g in x.groupby("date", sort=True)]
    rng = np.random.default_rng(seed)
    totals = np.empty(int(repeats), dtype=float)
    valid = 0
    for _ in range(int(repeats)):
        log_growth = 0.0
        used = 0
        for returns in groups:
            n = min(int(top_n), len(returns))
            if n <= 0:
                continue
            if n == len(returns):
                sample = returns
            else:
                sample = returns[rng.choice(len(returns), size=n, replace=False)]
            daily_return = float(sample.mean())
            if daily_return <= -1.0:
                log_growth = -np.inf
                break
            log_growth += np.log1p(daily_return)
            used += 1
        if used:
            totals[valid] = np.expm1(log_growth)
            valid += 1
    if valid == 0:
        return {"repeats": 0, "median_total_return": 0.0}
    a = totals[:valid]
    return {
        "repeats": int(valid),
        "median_total_return": float(np.median(a)),
        "p05_total_return": float(np.quantile(a, 0.05)),
        "p95_total_return": float(np.quantile(a, 0.95)),
        "mean_total_return": float(a.mean()),
        "seed": int(seed),
        "top_n": int(top_n),
    }
