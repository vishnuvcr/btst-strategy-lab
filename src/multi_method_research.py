from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core_btst import add_cross_sectional_features, add_features, add_market_context, load_config, metrics, read_market_files

FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60", "gap",
    "range_pct", "close_location", "body_pct", "atr_pct", "sma20_gap",
    "sma50_gap", "vol20", "volume_z", "mkt_ret_1", "mkt_ret_5", "mkt_vol_20",
]

METHODS = {
    "btst_momentum": ("momentum", 1, 1),
    "gap_fade": ("gap_fade", 1, 1),
    "opening_reversal": ("opening_reversal", 1, 1),
    "short_term_momentum": ("momentum", 3, 1),
    "short_mean_reversion": ("mean_reversion", 3, 1),
    "breakout": ("breakout", 5, 1),
    "volatility_breakout": ("vol_breakout", 10, 1),
    "trend_following": ("trend", 20, 1),
    "long_trend": ("trend", 60, 1),
    "cross_sectional_relative_strength": ("relative_strength", 20, 1),
    "long_short_cross_sectional": ("relative_strength", 20, 0),
}


def add_horizon_columns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    x = df.copy().sort_values(["symbol", "date"])
    g = x.groupby("symbol", group_keys=False)
    for h in horizons:
        x[f"future_close_{h}"] = g["close"].shift(-h)
        x[f"future_high_{h}"] = g["high"].transform(lambda s: s.shift(-1).rolling(h, min_periods=h).max())
        x[f"future_low_{h}"] = g["low"].transform(lambda s: s.shift(-1).rolling(h, min_periods=h).min())
        x[f"fwd_ret_{h}"] = x[f"future_close_{h}"] / g["open"].shift(-1).rolling(h, min_periods=h).first() - 1
    return x.replace([np.inf, -np.inf], np.nan)


def score(x: pd.DataFrame, family: str) -> pd.Series:
    if family == "momentum":
        return .45*x.rank_ret_5 + .35*x.rank_ret_20 + .20*x.rank_close_location
    if family == "mean_reversion":
        return .55*(1-x.rank_sma20_gap) + .25*(1-x.rank_ret_5) + .20*x.rank_close_location
    if family == "gap_fade":
        return .60*(1-x.rank_gap.clip(0, 1)) + .25*(1-x.rank_ret_1) + .15*x.rank_close_location
    if family == "opening_reversal":
        return .55*(1-x.rank_ret_1) + .30*x.rank_close_location + .15*(1-x.rank_gap.clip(0, 1))
    if family == "breakout":
        return .55*x.rank_ret_20 + .30*x.rank_close_location + .15*x.rank_volume_z
    if family == "vol_breakout":
        return .45*x.rank_volume_z + .35*x.rank_range_pct + .20*x.rank_close_location
    if family == "trend":
        return .50*x.rank_ret_60 + .30*x.rank_ret_20 + .20*x.rank_close_location
    if family == "relative_strength":
        return .65*x.rank_ret_20 + .35*x.rank_ret_5
    raise ValueError(f"unknown family {family}")


def folds(dates, train: int, val: int, test: int, step: int, embargo: int):
    i = train + val + embargo
    while i < len(dates):
        te = min(i + test, len(dates))
        train_end = i - val - embargo
        val_end = i - embargo
        yield dates[max(0, train_end-train):train_end], dates[train_end:val_end], dates[i:te]
        i += step


def simulate(picks: pd.DataFrame, horizon: int, cfg: dict, side_col: str = "side") -> pd.DataFrame:
    if picks.empty:
        return pd.DataFrame()
    slip = float(cfg["costs"]["slippage_bps_per_side"]) / 10000
    tc = float(cfg["costs"]["transaction_cost_bps_per_side"]) / 10000
    borrow = float(cfg["costs"].get("short_borrow_bps_per_day", 0)) / 10000
    sm = float(cfg["execution"].get("stop_atr_mult", 1.5))
    tm = float(cfg["execution"].get("target_atr_mult", 2.0))
    rows = []
    for _, r in picks.iterrows():
        side = int(r.get(side_col, 1))
        entry = float(r.next_open) * (1 + slip if side > 0 else 1 - slip)
        atr = float(r.atr_pct) if pd.notna(r.atr_pct) else .03
        atr = min(max(atr, .005), .20)
        stop = entry * (1 - sm*atr) if side > 0 else entry * (1 + sm*atr)
        target = entry * (1 + tm*atr) if side > 0 else entry * (1 - tm*atr)
        hi, lo = float(r[f"future_high_{horizon}"]), float(r[f"future_low_{horizon}"])
        final = float(r[f"future_close_{horizon}"])
        reason = "time"
        if side > 0:
            if lo <= stop: final, reason = stop, "stop"
            elif hi >= target: final, reason = target, "target"
            net = final*(1-slip)/entry - 1 - 2*tc - (borrow*horizon)
        else:
            if hi >= stop: final, reason = stop, "stop"
            elif lo <= target: final, reason = target, "target"
            net = 1 - final*(1+slip)/entry - 2*tc - (borrow*horizon)
        rows.append({"signal_date": r.date, "symbol": r.symbol, "return": net, "side": side, "score": r.score, "reason": reason, "horizon": horizon})
    out = pd.DataFrame(rows)
    n = out.groupby("signal_date").symbol.transform("count")
    out["weight"] = float(cfg["portfolio"]["max_gross_exposure"]) / n.clip(lower=1)
    out["weighted_return"] = out["return"] * out["weight"]
    return out


def tune(val: pd.DataFrame, horizon: int, method: str, cfg: dict) -> tuple[int, float]:
    best, best_score = (10, .60), -1e9
    for top_n in [5, 10, 15, 20]:
        for threshold in np.arange(.50, .81, .05):
            q = val[val.score >= threshold].copy()
            q["rank"] = q.groupby("date").score.rank(method="first", ascending=False)
            q = q[q["rank"] <= top_n]
            t = simulate(q, horizon, cfg)
            if len(t) < 30: continue
            daily = t.groupby("signal_date").weighted_return.sum()
            if daily.std() == 0: continue
            s = float(daily.mean()/daily.std()*math.sqrt(252))
            if s > best_score: best, best_score = (top_n, float(threshold)), s
    return best


def run_method(df: pd.DataFrame, method: str, cfg: dict) -> pd.DataFrame:
    if method == "ml_ranker":
        return run_ml(df, cfg)
    family, horizon, direction = METHODS[method]
    dates = sorted(df.date.dropna().unique())
    r = cfg["research"]
    train_n, val_n = int(r["train_years"]*252), int(r["validation_months"]*21)
    test_n, step_n = int(r["test_months"]*21), int(r["step_months"]*21)
    all_oos = []
    for train_dates, val_dates, test_dates in folds(dates, train_n, val_n, test_n, step_n, int(r["embargo_days"])):
        val = df[df.date.isin(val_dates)].copy(); test = df[df.date.isin(test_dates)].copy()
        val["score"] = score(val, family); test["score"] = score(test, family)
        if direction == 0:
            val["side"] = np.where(val.score >= .5, 1, -1)
            test["side"] = np.where(test.score >= .5, 1, -1)
        else:
            val["side"] = direction; test["side"] = direction
        top_n, threshold = tune(val, horizon, method, cfg)
        test["rank"] = test.groupby("date").score.rank(method="first", ascending=(direction < 0))
        picks = test[(test.score >= threshold) & (test["rank"] <= top_n)]
        t = simulate(picks, horizon, cfg)
        if not t.empty: all_oos.append(t)
    return pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()


def run_ml(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    horizon = 5
    work = df.copy()
    work["label_5"] = (work["fwd_ret_5"] > 0).astype(int)
    feats = [c for c in FEATURES if c in work.columns]
    dates = sorted(work.date.dropna().unique()); r = cfg["research"]
    out=[]
    for tr, va, te in folds(dates, int(r["train_years"]*252), int(r["validation_months"]*21), int(r["test_months"]*21), int(r["step_months"]*21), int(r["embargo_days"])):
        train = work[work.date.isin(tr)].dropna(subset=feats+["label_5"])
        val = work[work.date.isin(va)].dropna(subset=feats).copy(); test = work[work.date.isin(te)].dropna(subset=feats).copy()
        train = train[train.date < (pd.Timestamp(min(va)) - pd.Timedelta(days=5))]
        if len(train) < int(r["min_train_observations"]): continue
        model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, C=.25, class_weight="balanced", random_state=int(r["random_state"])))])
        model.fit(train[feats], train.label_5)
        val["score"] = model.predict_proba(val[feats])[:,1]; test["score"] = model.predict_proba(test[feats])[:,1]
        threshold = max(.55, float(val.score.quantile(.80)))
        test["rank"] = test.groupby("date").score.rank(method="first", ascending=False)
        picks = test[(test.score >= threshold) & (test["rank"] <= int(cfg["portfolio"]["max_positions"]))]
        t=simulate(picks, horizon, cfg)
        if not t.empty: out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(config_path: str):
    cfg=load_config(config_path)
    df=read_market_files(cfg["data"]["daily_glob"])
    market=read_market_files(cfg["data"]["market_glob"])
    df=add_features(df); market=add_features(market); df=add_market_context(df, market); df=add_cross_sectional_features(df)
    df=add_horizon_columns(df, cfg["research"]["horizons"])
    methods=cfg["methods"]
    rows=[]; all_trades=[]
    for method in methods:
        print(f"Running {method}...")
        t=run_method(df, method, cfg)
        m=metrics(t, method, cfg["portfolio"]["initial_capital"]) if not t.empty else metrics(pd.DataFrame(), method, cfg["portfolio"]["initial_capital"])
        rows.append(m.__dict__)
        if not t.empty: all_trades.append(t.assign(method=method))
    Path("docs").mkdir(exist_ok=True)
    pd.DataFrame(rows).sort_values(["sharpe","profit_factor"], ascending=False).to_csv("docs/multi_method_leaderboard.csv", index=False)
    if all_trades: pd.concat(all_trades, ignore_index=True).to_csv("docs/multi_method_oos_trades.csv", index=False)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", default="config/multi_method.yaml"); a=p.parse_args(); main(a.config)
