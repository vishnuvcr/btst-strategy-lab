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


def _parse_dates(raw: pd.Series) -> pd.Series:
    s = raw.astype("string").str.replace("\ufeff", "", regex=False).str.strip()
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    compact = s.str.fullmatch(r"\d{8}")
    if compact.any():
        out.loc[compact] = pd.to_datetime(s.loc[compact], format="%Y%m%d", errors="coerce", utc=True)
    remaining = out.isna()
    if remaining.any():
        out.loc[remaining] = pd.to_datetime(s.loc[remaining], errors="coerce", utc=True, format="mixed")
    return out


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
        if date_col is None and "price" in df.columns:
            price_as_date = _parse_dates(df["price"])
            if price_as_date.notna().mean() >= 0.80:
                date_col = "price"
        if date_col is None:
            skipped.append(f"{fp}: no date column; columns={list(df.columns)[:12]}")
            continue
        aliases = {"open": ["o"], "high": ["h"], "low": ["l"], "close": ["adj_close", "price", "c"], "volume": ["vol", "v"]}
        rename = {}
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                for a in aliases[c]:
                    if a in df.columns and a != date_col:
                        rename[a] = c
                        break
        df = df.rename(columns=rename)
        if not all(c in df.columns for c in ["open", "high", "low", "close"]):
            skipped.append(f"{fp}: missing OHLC after normalization; columns={list(df.columns)[:12]}")
            continue
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        parsed = _parse_dates(df[date_col])
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
    m = market.sort_values("date").drop_duplicates("date")
    m["mkt_ret_1"] = m["close"].pct_change()
    m["mkt_ret_5"] = m["close"].pct_change(5)
    m["mkt_vol_20"] = m["mkt_ret_1"].rolling(20, min_periods=15).std()
    return stock.merge(m[["date", "mkt_ret_1", "mkt_ret_5", "mkt_vol_20"]], on="date", how="left")


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Keep this list synchronized with every cross-sectional feature used by
    # the multi-method tournament. Missing ranks caused otherwise-valid runs
    # to fail when a strategy referenced a feature not ranked here.
    for c in ["ret_1", "ret_5", "ret_20", "ret_60", "gap", "range_pct", "close_location", "volume_z", "sma20_gap"]:
        if c in df.columns:
            df[f"rank_{c}"] = df.groupby("date")[c].rank(pct=True)
    return df


def strategy_scores(df: pd.DataFrame, family: str) -> pd.Series:
    if family == "momentum":
        return 0.45 * df.rank_ret_5 + 0.35 * df.rank_ret_20 + 0.20 * df.rank_close_location
    if family == "mean_reversion":
        return 0.55 * (1 - df.rank_sma20_gap) + 0.25 * (1 - df.rank_ret_5) + 0.20 * df.rank_close_location
    if family == "closing_strength":
        return 0.65 * df.rank_close_location + 0.35 * df.rank_ret_1
    if family == "volume":
        return 0.45 * df.rank_volume_z + 0.35 * df.rank_ret_5 + 0.20 * df.rank_close_location
    if family == "gap":
        return 0.55 * df.rank_ret_1 + 0.25 * df.rank_close_location + 0.20 * (1 - df.rank_gap.clip(0, 1))
    if family == "regime":
        base = 0.55 * df.rank_ret_20 + 0.45 * df.rank_close_location
        good = (df["mkt_ret_5"] > 0).astype(float) if "mkt_ret_5" in df else 0.0
        return base * (0.5 + good)
    if family == "sector_relative":
        return 0.65 * df.rank_ret_20 + 0.35 * df.rank_ret_5
    raise ValueError(f"Unknown strategy family: {family}")


def select_top(scores: pd.DataFrame, top_n: int = 10, min_score: float = 0.60) -> pd.DataFrame:
    required = ["score", "btst_return", "next_open", "next_close"]
    x = scores.dropna(subset=required).copy()
    x = x[x.score >= min_score]
    x["rank"] = x.groupby("date").score.rank(method="first", ascending=False)
    return x[x["rank"] <= top_n]


def prepare_entries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("symbol", group_keys=False)
    for c in ["open", "high", "low", "close"]:
        df[f"next_{c}"] = g[c].shift(-1)
    return df


def simulate_trades(trades: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, float]:
    if trades.empty:
        return trades.copy(), 0.0
    sl_bps = float(cfg["costs"]["slippage_bps_per_side"]) / 10000
    tc_bps = float(cfg["costs"]["transaction_cost_bps_per_side"]) / 10000
    stop_mode = str(cfg["execution"].get("stop_mode", "ATR")).upper()
    stop_pct = float(cfg["execution"].get("stop_pct", 2.0)) / 100
    target_pct = float(cfg["execution"].get("target_pct", 5.0)) / 100
    stop_mult = float(cfg["execution"].get("stop_atr_mult", 1.5))
    target_mult = float(cfg["execution"].get("target_atr_mult", 2.0))
    max_gross = float(cfg["portfolio"].get("max_gross_exposure", 0.95))
    if not (0 < max_gross <= 1):
        raise ValueError("portfolio.max_gross_exposure must be in (0, 1]")
    rows = []
    for _, r in trades.iterrows():
        entry = float(r.next_open) * (1 + sl_bps)
        stop = entry * (1 - stop_pct)
        target = entry * (1 + target_pct)
        if stop_mode == "ATR" and pd.notna(r.get("atr_pct")):
            a = min(max(float(r.atr_pct), 0.005), 0.20)
            stop = entry * (1 - stop_mult * a)
            target = entry * (1 + target_mult * a)
        exit_px = float(r.next_close)
        reason = "time"
        if pd.notna(r.get("future_low_1")) and float(r.future_low_1) <= stop:
            exit_px, reason = stop, "stop"
        elif pd.notna(r.get("future_high_1")) and float(r.future_high_1) >= target:
            exit_px, reason = target, "target"
        ret = exit_px * (1 - sl_bps) / entry - 1 - 2 * tc_bps
        rows.append({"signal_date": r.date, "symbol": r.symbol, "return": ret, "reason": reason})
    o = pd.DataFrame(rows)
    if o.empty:
        return o, 0.0
    n = o.groupby("signal_date").symbol.transform("count")
    o["weight"] = max_gross / n.clip(lower=1)
    o["weighted_return"] = o.return * o.weight
    return o, float(o.weighted_return.sum())
