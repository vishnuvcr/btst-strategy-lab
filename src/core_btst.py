from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


@dataclass
class BacktestResult:
    strategy: str
    trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    turnover: float
    stability: float
    score: float


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_col(c: str) -> str:
    return str(c).replace("\ufeff", "").strip().lower().replace(" ", "_").replace("-", "_")


def read_market_files(pattern: str, max_files: int | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(pattern, recursive=True))
    if max_files:
        files = files[:max_files]
    frames: List[pd.DataFrame] = []
    skipped = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            skipped.append(f"{fp}: read error: {exc}")
            continue
        df.columns = [_norm_col(c) for c in df.columns]
        date_col = next((c for c in ["date", "datetime", "timestamp", "time"] if c in df.columns), None)
        if date_col is None:
            skipped.append(f"{fp}: no date column; columns={list(df.columns)[:12]}")
            continue
        aliases = {"open": ["o"], "high": ["h"], "low": ["l"], "close": ["adj_close", "price", "c"], "volume": ["vol", "v"]}
        rename = {}
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                for a in aliases[c]:
                    if a in df.columns:
                        rename[a] = c
                        break
        df = df.rename(columns=rename)
        if not all(c in df.columns for c in ["open", "high", "low", "close"]):
            skipped.append(f"{fp}: missing OHLC after normalization; columns={list(df.columns)[:12]}")
            continue
        raw_date = df[date_col]
        # Handle both normal date strings and common YYYYMMDD integer/string formats.
        parsed = pd.to_datetime(raw_date, errors="coerce", utc=True)
        numeric = pd.to_numeric(raw_date, errors="coerce")
        mask = parsed.isna() & numeric.notna()
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(numeric.loc[mask].astype("Int64").astype(str), format="%Y%m%d", errors="coerce", utc=True)
        df["date"] = parsed.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=["date", "open", "high", "low", "close"])
        if df.empty:
            skipped.append(f"{fp}: no rows with valid dates/OHLC")
            continue
        symbol = Path(fp).stem.upper().replace("-", "_")
        df["symbol"] = symbol
        keep = ["date", "symbol", "open", "high", "low", "close"]
        if "volume" in df.columns:
            keep.append("volume")
        frames.append(df[keep])
    if not frames:
        detail = " | ".join(skipped[:5])
        raise RuntimeError(f"No usable CSV files found for {pattern}. Candidates={len(files)}. {detail}")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol", "date"])
    g = df.groupby("symbol", group_keys=False)
    df["ret_1"] = g["close"].pct_change()
    for n in [2, 3, 5, 10, 20, 60]:
        df[f"ret_{n}"] = g["close"].pct_change(n)
    df["gap"] = df["open"] / g["close"].shift(1) - 1.0
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["close_location"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["body_pct"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["atr_pct"] = g["range_pct"].transform(lambda s: s.rolling(14, min_periods=10).mean())
    df["sma20_gap"] = df["close"] / g["close"].transform(lambda s: s.rolling(20, min_periods=15).mean()) - 1
    df["sma50_gap"] = df["close"] / g["close"].transform(lambda s: s.rolling(50, min_periods=30).mean()) - 1
    df["vol20"] = g["ret_1"].transform(lambda s: s.rolling(20, min_periods=15).std())
    if "volume" in df.columns:
        mean = g["volume"].transform(lambda s: s.rolling(20, min_periods=15).mean())
        std = g["volume"].transform(lambda s: s.rolling(20, min_periods=15).std())
        df["volume_z"] = (df["volume"] - mean) / std.replace(0, np.nan)
        df["turnover"] = df["close"] * df["volume"]
    else:
        df["volume_z"] = 0.0
        df["turnover"] = np.nan
    df["btst_return"] = g["close"].shift(-1) / g["open"].shift(-1) - 1.0
    df["label"] = (df["btst_return"] > 0).astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


def add_market_context(stock: pd.DataFrame, market: pd.DataFrame | None) -> pd.DataFrame:
    if market is None or market.empty:
        return stock
    m = market.copy().sort_values(["symbol", "date"])
    m = m[m["symbol"].str.contains("NIFTY_50|NIFTY50", regex=True, na=False)]
    if m.empty:
        return stock
    m = m.groupby("date", as_index=False).agg(market_close=("close", "last"))
    m["market_ret_1"] = m["market_close"].pct_change()
    m["market_ret_5"] = m["market_close"].pct_change(5)
    m["market_vol20"] = m["market_ret_1"].rolling(20, min_periods=15).std()
    return stock.merge(m[["date", "market_ret_1", "market_ret_5", "market_vol20"]], on="date", how="left")


def cross_sectional_rank(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].rank(pct=True, method="average")


def strategy_score(df: pd.DataFrame, family: str) -> pd.Series:
    if family == "momentum":
        return 0.45 * df["ret_20"].rank(pct=True) + 0.35 * df["ret_5"].rank(pct=True) + 0.20 * df["sma50_gap"].rank(pct=True)
    if family == "mean_reversion":
        return 0.55 * (-df["ret_5"]).rank(pct=True) + 0.45 * (-df["sma20_gap"]).rank(pct=True)
    if family == "closing_strength":
        return 0.60 * df["close_location"] + 0.40 * df["body_pct"].rank(pct=True)
    if family == "volume":
        return 0.60 * df["volume_z"].rank(pct=True) + 0.40 * df["ret_5"].rank(pct=True)
    if family == "gap":
        return 0.50 * (-df["gap"]).rank(pct=True) + 0.50 * df["ret_5"].rank(pct=True)
    if family == "regime":
        trend = (df["sma20_gap"] > 0).astype(float)
        return 0.60 * trend + 0.40 * df["market_ret_5"].rank(pct=True)
    if family == "sector_relative":
        return 0.60 * df["ret_20"].rank(pct=True) + 0.40 * df["ret_5"].rank(pct=True)
    raise ValueError(f"Unknown rule family: {family}")


def top_n_signals(df: pd.DataFrame, score: pd.Series, top_n: int, min_score: float) -> pd.DataFrame:
    x = df.copy()
    x["score_raw"] = score
    x["score_rank"] = x.groupby("date")["score_raw"].rank(pct=True, method="first")
    x["signal"] = (x["score_rank"] >= (1.0 - top_n / x.groupby("date")["symbol"].transform("count"))) & (x["score_raw"] >= min_score)
    return x


def _metric_series(returns: pd.Series) -> Tuple[float, float, float, float, float]:
    r = returns.dropna()
    if r.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    total = float((1.0 + r).prod() - 1.0)
    ann = float((1.0 + total) ** (252 / max(len(r), 1)) - 1.0)
    sharpe = float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0
    equity = (1.0 + r).cumprod()
    mdd = float((equity / equity.cummax() - 1.0).min())
    turnover = float(r.abs().sum())
    return total, ann, sharpe, mdd, turnover


def backtest_signals(df: pd.DataFrame, signal_col: str, costs: dict, initial_capital: float = 1_000_000.0) -> BacktestResult:
    x = df.copy().sort_values(["date", "symbol"])
    x["entry"] = x["open"]
    x["exit"] = x.groupby("symbol")["close"].shift(-1)
    x["gross_ret"] = x["exit"] / x["entry"] - 1.0
    cost = 2.0 * (float(costs.get("slippage_bps_per_side", 0)) + float(costs.get("transaction_cost_bps_per_side", 0))) / 10_000.0
    x["net_ret"] = x["gross_ret"] - cost
    trades = x[x[signal_col].fillna(False) & x["net_ret"].notna()].copy()
    if trades.empty:
        return BacktestResult("unknown", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    daily = trades.groupby("date")["net_ret"].mean()
    total, ann, sharpe, mdd, turnover = _metric_series(daily)
    wins = trades.loc[trades["net_ret"] > 0, "net_ret"]
    losses = trades.loc[trades["net_ret"] < 0, "net_ret"]
    pf = float(wins.sum() / abs(losses.sum())) if not losses.empty and losses.sum() != 0 else (float("inf") if not wins.empty else 0.0)
    exp = float(trades["net_ret"].mean())
    return BacktestResult("unknown", len(trades), float((trades["net_ret"] > 0).mean()), pf, exp, total, ann, sharpe, mdd, turnover, 0.0, 0.0)
