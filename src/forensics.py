"""Forensic diagnostics and benchmark helpers for BTST research.

These helpers are deliberately independent of strategy selection. They are used to
answer three questions before trusting a tournament result:
1. Are trade/accounting returns internally sane?
2. Is the strategy actually adding value versus simple baselines?
3. Is the reported equity curve dominated by exposure or a small number of bad days?
"""
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
    daily = wr.groupby(trades["signal_date"]).sum()
    return {
        "trades": int(len(trades)),
        "finite_returns": finite,
        "min_return": float(r.min()),
        "max_return": float(r.max()),
        "mean_return": float(r.mean()),
        "median_return": float(r.median()),
        "max_abs_weight": float(w.abs().max()),
        "max_abs_weighted_return": float(wr.abs().max()),
        "max_gross_weight": float(w.groupby(trades["signal_date"]).sum().max()),
        "min_daily_return": float(daily.min()),
        "max_daily_return": float(daily.max()),
        "pct_daily_below_minus_5pct": float((daily < -0.05).mean()),
        "pct_daily_below_minus_10pct": float((daily < -0.10).mean()),
    }


def equal_weight_buy_and_hold(df: pd.DataFrame) -> dict:
    """Equal-weight daily universe benchmark using close-to-close returns.

    This is a universe benchmark, not an investable NIFTY index replacement.
    """
    required = {"date", "symbol", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing benchmark columns: {missing}")
    x = df[["date", "symbol", "close"]].copy().sort_values(["symbol", "date"])
    x["ret"] = x.groupby("symbol")["close"].pct_change()
    daily = x.groupby("date")["ret"].mean().dropna().sort_index()
    if daily.empty:
        return {"days": 0, "total_return": 0.0, "mean_daily_return": 0.0}
    equity = (1.0 + daily).cumprod()
    return {
        "days": int(len(daily)),
        "date_start": str(pd.to_datetime(daily.index.min()).date()),
        "date_end": str(pd.to_datetime(daily.index.max()).date()),
        "total_return": float(equity.iloc[-1] - 1.0),
        "mean_daily_return": float(daily.mean()),
        "median_daily_return": float(daily.median()),
    }


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
    x = df[list(required)].dropna().copy()
    x["next_return"] = x["next_close"] / x["next_open"] - 1.0
    rng = np.random.default_rng(seed)
    daily_returns = []
    dates = sorted(x["date"].unique())
    for _ in range(int(repeats)):
        path = []
        for d in dates:
            day = x[x["date"] == d]
            if day.empty:
                continue
            n = min(int(top_n), len(day))
            picks = rng.choice(len(day), size=n, replace=False)
            path.append(float(day.iloc[picks]["next_return"].mean()))
        if path:
            daily_returns.append(float((1.0 + pd.Series(path)).prod() - 1.0))
    if not daily_returns:
        return {"repeats": 0, "median_total_return": 0.0}
    a = np.asarray(daily_returns, dtype=float)
    return {
        "repeats": int(len(a)),
        "median_total_return": float(np.median(a)),
        "p05_total_return": float(np.quantile(a, 0.05)),
        "p95_total_return": float(np.quantile(a, 0.95)),
        "mean_total_return": float(a.mean()),
        "seed": int(seed),
        "top_n": int(top_n),
    }
