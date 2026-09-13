from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
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
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def read_market_files(pattern: str, max_files: int | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(pattern, recursive=True))
    if max_files:
        files = files[:max_files]
    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        df.columns = [_norm_col(c) for c in df.columns]
        date_col = next((c for c in ["date", "datetime", "timestamp", "time"] if c in df), None)
        if date_col is None:
            continue
        rename = {}
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                aliases = {"open": ["o"], "high": ["h"], "low": ["l"], "close": ["adj_close", "price", "c"], "volume": ["vol", "v"]}[c]
                for a in aliases:
                    if a in df.columns:
                        rename[a] = c
                        break
        df = df.rename(columns=rename)
        if not all(c in df.columns for c in ["open", "high", "low", "close"]):
            continue
        df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=["date", "open", "high", "low", "close"])
        symbol = Path(fp).stem.upper().replace("-", "_")
        df["symbol"] = symbol
        frames.append(df[["date", "symbol", "open", "high", "low", "close"] + (["volume"] if "volume" in df else [])])
    if not frames:
        raise RuntimeError(f"No usable CSV files found for {pattern}")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
    return out


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
    if "volume" in df:
        df["volume_z"] = g["volume"].transform(lambda s: (s - s.rolling(20, min_periods=15).mean()) / s.rolling(20, min_periods=15).std().replace(0, np.nan))
        df["turnover"] = df["close"] * df["volume"]
    else:
        df["volume_z"] = 0.0
        df["turnover"] = np.nan
    # BTST label: return from next session open to next session close.
    next_open = g["open"].shift(-1)
    next_close = g["close"].shift(-1)
    df["btst_return"] = next_close / next_open - 1.0
    df["label"] = (df["btst_return"] > 0).astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


def add_market_context(stock: pd.DataFrame, market: pd.DataFrame | None) -> pd.DataFrame:
    if market is None or market.empty:
        return stock
    m = market.copy().sort_values(["date"])
    m = m.drop_duplicates("date")
    m["mkt_ret_1"] = m["close"].pct_change()
    m["mkt_ret_5"] = m["close"].pct_change(5)
    m["mkt_vol_20"] = m["mkt_ret_1"].rolling(20, min_periods=15).std()
    cols = ["date", "mkt_ret_1", "mkt_ret_5", "mkt_vol_20"]
    return stock.merge(m[cols], on="date", how="left")


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["ret_5", "ret_20", "close_location", "volume_z", "sma20_gap"]:
        if c in df:
            df[f"rank_{c}"] = df.groupby("date")[c].rank(pct=True)
    return df


def strategy_scores(df: pd.DataFrame, family: str) -> pd.Series:
    eps = 1e-9
    if family == "momentum":
        return 0.45 * df["rank_ret_5"] + 0.35 * df["rank_ret_20"] + 0.20 * df["rank_close_location"]
    if family == "mean_reversion":
        return 0.55 * (1 - df["rank_sma20_gap"]) + 0.25 * (1 - df["rank_ret_5"]) + 0.20 * df["rank_close_location"]
    if family == "closing_strength":
        return 0.65 * df["rank_close_location"] + 0.35 * df["rank_ret_1"]
    if family == "volume":
        return 0.45 * df["rank_volume_z"] + 0.35 * df["rank_ret_5"] + 0.20 * df["rank_close_location"]
    if family == "gap":
        # Strong prior-day close with a controlled gap is used as a continuation signal.
        return 0.55 * df["rank_ret_1"] + 0.25 * df["rank_close_location"] + 0.20 * (1 - df["rank_gap"].clip(0, 1))
    if family == "regime":
        base = 0.55 * df["rank_ret_20"] + 0.45 * df["rank_close_location"]
        good_regime = (df["mkt_ret_5"] > 0).astype(float)
        return base * (0.5 + good_regime)
    if family == "sector_relative":
        return 0.65 * df["rank_ret_20"] + 0.35 * df["rank_ret_5"]
    raise ValueError(f"Unknown non-ML strategy family: {family}")


def select_top(scores: pd.DataFrame, top_n: int = 10, min_score: float = 0.65) -> pd.DataFrame:
    x = scores.dropna(subset=["score", "btst_return", "next_open", "next_close"]).copy()
    x = x[x["score"] >= min_score]
    x["rank"] = x.groupby("date")["score"].rank(method="first", ascending=False)
    return x[x["rank"] <= top_n]


def prepare_entries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("symbol", group_keys=False)
    df["next_open"] = g["open"].shift(-1)
    df["next_high"] = g["high"].shift(-1)
    df["next_low"] = g["low"].shift(-1)
    df["next_close"] = g["close"].shift(-1)
    return df


def simulate_trades(trades: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, float]:
    if trades.empty:
        return trades.assign(pnl=[]), 0.0
    sl_bps = float(cfg["costs"]["slippage_bps_per_side"]) / 10000
    tc_bps = float(cfg["costs"]["transaction_cost_bps_per_side"]) / 10000
    stop_mode = cfg["execution"].get("stop_mode", "ATR")
    stop_pct = float(cfg["execution"].get("stop_pct", 2.0)) / 100
    target_pct = float(cfg["execution"].get("target_pct", 5.0)) / 100
    risk = float(cfg["portfolio"]["risk_per_trade"])
    rows = []
    for _, r in trades.iterrows():
        entry = float(r["next_open"]) * (1 + sl_bps)
        stop = entry * (1 - stop_pct)
        target = entry * (1 + target_pct)
        if stop_mode.upper() == "ATR" and pd.notna(r.get("atr_pct")):
            a = max(float(r["atr_pct"]), 0.005)
            stop = entry * (1 - 1.5 * a)
            target = entry * (1 + 2.0 * a)
        lo, hi = float(r["next_low"]), float(r["next_high"])
        exit_px = float(r["next_close"])
        exit_reason = "close"
        if lo <= stop:
            exit_px, exit_reason = stop, "stop"
        elif hi >= target:
            exit_px, exit_reason = target, "target"
        exit_px *= (1 - sl_bps)
        ret = exit_px / entry - 1
        net_ret = ret - 2 * tc_bps
        rows.append({"signal_date": r["date"], "symbol": r["symbol"], "entry": entry, "exit": exit_px, "return": net_ret, "reason": exit_reason, "score": r["score"]})
    out = pd.DataFrame(rows)
    return out, float(out["return"].sum()) if not out.empty else 0.0


def metrics(trades: pd.DataFrame, strategy: str, initial_capital: float, score: float = 0.0) -> BacktestResult:
    if trades.empty:
        return BacktestResult(strategy, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    r = trades["return"].astype(float)
    wins, losses = r[r > 0], r[r <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    sharpe = float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0
    total = float(eq.iloc[-1] - 1)
    years = max((pd.to_datetime(trades["signal_date"]).max() - pd.to_datetime(trades["signal_date"]).min()).days / 365.25, 1 / 365.25)
    ann = float((1 + total) ** (1 / years) - 1) if 1 + total > 0 else -1.0
    turnover = float((trades["entry"] * 2).sum() / initial_capital)
    return BacktestResult(strategy, len(r), float((r > 0).mean()), pf, float(r.mean()), total, ann, sharpe, float(dd.min()), turnover, 1.0, score)


def run_ml_walk_forward(df: pd.DataFrame, family: str, cfg: dict, top_n: int = 10) -> pd.DataFrame:
    features = ["ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60", "gap", "range_pct", "close_location", "body_pct", "atr_pct", "sma20_gap", "sma50_gap", "vol20", "volume_z", "mkt_ret_1", "mkt_ret_5", "mkt_vol_20"]
    features = [c for c in features if c in df.columns]
    work = df.dropna(subset=features + ["label", "btst_return", "next_open", "next_high", "next_low", "next_close"]).copy()
    dates = sorted(work["date"].unique())
    step = max(int(cfg["research"].get("step_months", 6)), 1)
    train_days = int(cfg["research"].get("train_years", 4) * 252)
    val_days = int(cfg["research"].get("validation_months", 12) * 21)
    test_days = int(cfg["research"].get("test_months", 12) * 21)
    outputs = []
    start = train_days + val_days
    while start < len(dates):
        test_end = min(start + test_days, len(dates))
        train_dates = dates[max(0, start - train_days - val_days): max(0, start - val_days)]
        val_dates = dates[max(0, start - val_days):start]
        test_dates = dates[start:test_end]
        train = work[work.date.isin(train_dates)]
        val = work[work.date.isin(val_dates)]
        test = work[work.date.isin(test_dates)].copy()
        if len(train) < cfg["research"].get("min_train_observations", 2000) or test.empty:
            start += test_days
            continue
        if family == "ml_ranker":
            model = LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=31, random_state=cfg["research"]["random_state"], verbosity=-1) if LGBMClassifier else RandomForestClassifier(n_estimators=300, min_samples_leaf=25, random_state=cfg["research"]["random_state"], n_jobs=-1)
        else:
            model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, C=0.25, random_state=cfg["research"]["random_state"]))])
        model.fit(train[features], train["label"])
        test["score"] = model.predict_proba(test[features])[:, 1]
        # Validation-derived threshold: maximize validation expectancy subject to minimum signal count.
        val["score"] = model.predict_proba(val[features])[:, 1]
        best_thr = 0.60
        best_exp = -np.inf
        for thr in np.arange(0.50, 0.81, 0.02):
            q = val[val.score >= thr]
            if len(q) < 30:
                continue
            e = float(q.btst_return.mean())
            if e > best_exp:
                best_exp, best_thr = e, float(thr)
        test = test[test.score >= best_thr]
        test["rank"] = test.groupby("date")["score"].rank(method="first", ascending=False)
        outputs.append(test[test["rank"] <= top_n])
        start += test_days
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def run(config_path: str) -> Dict[str, object]:
    cfg = load_config(config_path)
    max_symbols = int(cfg["data"].get("max_symbols", 200))
    daily = read_market_files(cfg["data"]["daily_glob"])
    # Prefer liquid and sufficiently long series before limiting the universe.
    counts = daily.groupby("symbol")["date"].nunique().sort_values(ascending=False)
    symbols = counts[counts >= int(cfg["data"]["min_history_days"])].index[:max_symbols]
    daily = daily[daily.symbol.isin(symbols)].copy()
    daily = prepare_entries(add_features(daily))
    market = None
    try:
        market = read_market_files(cfg["data"]["market_glob"], max_files=50)
    except Exception:
        pass
    daily = add_market_context(daily, market)
    daily = add_cross_sectional_features(daily)

    out_trades: List[pd.DataFrame] = []
    results: List[BacktestResult] = []
    families = cfg["strategy_search"]["families"]
    for family in families:
        if family in {"ml_ranker", "hybrid"}:
            picks = run_ml_walk_forward(daily, family, cfg)
        else:
            x = daily.copy()
            x["score"] = strategy_scores(x, family)
            # Require a positive market regime for the regime strategy; other families use score alone.
            picks = select_top(x, top_n=int(cfg["portfolio"]["max_positions"]), min_score=0.60)
        if picks.empty:
            continue
        trades, _ = simulate_trades(picks, cfg)
        if trades.empty:
            continue
        result = metrics(trades, family, float(cfg["portfolio"]["initial_capital"]))
        results.append(result)
        trades["strategy"] = family
        out_trades.append(trades)

    if not results:
        raise RuntimeError("No strategy produced trades. Check data paths, schema, and date coverage.")
    # Conservative composite score: reward risk-adjusted return and PF, penalize drawdown.
    def composite(r: BacktestResult) -> float:
        pf = min(r.profit_factor, 4.0) / 4.0 if np.isfinite(r.profit_factor) else 1.0
        dd_penalty = max(0.0, 1 + r.max_drawdown)
        return 0.30 * np.tanh(max(r.sharpe, -3) / 2) + 0.20 * pf + 0.15 * np.tanh(r.expectancy * 100) + 0.15 * dd_penalty + 0.20 * np.tanh(max(r.annualized_return, -1) * 3)
    ranked = sorted(results, key=composite, reverse=True)
    leaderboard = []
    for r in ranked:
        leaderboard.append({**r.__dict__, "score": composite(r)})
    all_trades = pd.concat(out_trades, ignore_index=True) if out_trades else pd.DataFrame()
    Path("docs").mkdir(exist_ok=True)
    pd.DataFrame(leaderboard).to_csv("docs/strategy_leaderboard.csv", index=False)
    all_trades.to_csv("docs/oos_trades.csv", index=False)
    with open("docs/strategy_leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, default=str)
    manifest = {"symbols": len(symbols), "date_start": str(daily.date.min().date()), "date_end": str(daily.date.max().date()), "strategies_tested": families, "data_glob": cfg["data"]["daily_glob"]}
    with open("docs/research_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"leaderboard": leaderboard, "manifest": manifest}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/btst.yaml")
    args = ap.parse_args()
    result = run(args.config)
    print(json.dumps(result, indent=2, default=str))
