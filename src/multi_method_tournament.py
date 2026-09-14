from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from core_btst import add_cross_sectional_features, add_features, add_market_context, load_config, metrics, prepare_entries, read_market_files

FEATURES = ["ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60", "gap", "range_pct", "close_location", "body_pct", "atr_pct", "sma20_gap", "sma50_gap", "vol20", "volume_z", "mkt_ret_1", "mkt_ret_5", "mkt_vol_20"]
METHODS = {"BTST momentum": ("momentum", 1, 1), "Gap fade": ("gap_fade", 1, 1), "Opening reversal": ("opening_reversal", 1, 1), "Short-term momentum": ("momentum", 3, 1), "Short mean reversion": ("mean_reversion", 3, 1), "Breakout": ("breakout", 5, 1), "Volatility breakout": ("vol_breakout", 10, 1), "Trend following": ("trend", 20, 1), "Long trend": ("trend", 60, 1), "Relative strength": ("relative", 20, 1)}


def add_forward(x, horizons):
    x = x.sort_values(["symbol", "date"]).copy()
    g = x.groupby("symbol", group_keys=False)
    for h in horizons:
        x[f"future_close_{h}"] = g.close.shift(-h)
        x[f"future_high_{h}"] = g.high.transform(lambda s: s.shift(-1).iloc[::-1].rolling(h, min_periods=h).max().iloc[::-1])
        x[f"future_low_{h}"] = g.low.transform(lambda s: s.shift(-1).iloc[::-1].rolling(h, min_periods=h).min().iloc[::-1])
        x[f"fwd_ret_{h}"] = x[f"future_close_{h}"] / g.open.shift(-1) - 1
    return x.replace([np.inf, -np.inf], np.nan)


def score(x, f):
    if f == "momentum": return .45 * x["rank_ret_5"] + .35 * x["rank_ret_20"] + .20 * x["rank_close_location"]
    if f == "mean_reversion": return .55 * (1 - x["rank_sma20_gap"]) + .25 * (1 - x["rank_ret_5"]) + .20 * x["rank_close_location"]
    if f == "gap_fade": return .60 * (1 - x["rank_gap"].clip(0, 1)) + .25 * (1 - x["rank_ret_1"]) + .15 * x["rank_close_location"]
    if f == "opening_reversal": return .55 * (1 - x["rank_ret_1"]) + .30 * x["rank_close_location"] + .15 * (1 - x["rank_gap"].clip(0, 1))
    if f == "breakout": return .55 * x["rank_ret_20"] + .30 * x["rank_close_location"] + .15 * x["rank_volume_z"]
    if f == "vol_breakout": return .45 * x["rank_volume_z"] + .35 * x["rank_range_pct"] + .20 * x["rank_close_location"]
    if f == "trend": return .50 * x["rank_ret_60"] + .30 * x["rank_ret_20"] + .20 * x["rank_close_location"]
    if f == "relative": return .65 * x["rank_ret_20"] + .35 * x["rank_ret_5"]
    raise ValueError(f)


def folds(dates, train, val, test, step, emb):
    i = train + val + emb
    while i < len(dates):
        tr_end = i - val - emb
        va_end = i - emb
        te = min(i + test, len(dates))
        yield dates[max(0, tr_end - train):tr_end], dates[tr_end:va_end], dates[i:te]
        i += step


def simulate(p, h, cfg):
    if p.empty:
        return pd.DataFrame()
    slip = cfg["costs"]["slippage_bps_per_side"] / 10000
    tc = cfg["costs"]["transaction_cost_bps_per_side"] / 10000
    borrow = cfg["costs"].get("short_borrow_bps_per_day", 0) / 10000
    sm = cfg["execution"].get("stop_atr_mult", 1.5)
    tm = cfg["execution"].get("target_atr_mult", 2.0)
    rows = []
    for _, r in p.iterrows():
        side = int(r["side"])
        next_open = r.get("next_open")
        if pd.isna(next_open):
            continue
        e = float(next_open) * (1 + slip if side > 0 else 1 - slip)
        a = min(max(float(r["atr_pct"]) if pd.notna(r["atr_pct"]) else .03, .005), .20)
        stop = e * (1 - sm * a) if side > 0 else e * (1 + sm * a)
        target = e * (1 + tm * a) if side > 0 else e * (1 - tm * a)
        hi = float(r[f"future_high_{h}"])
        lo = float(r[f"future_low_{h}"])
        f = float(r[f"future_close_{h}"])
        reason = "time"
        if side > 0:
            if lo <= stop:
                f, reason = stop, "stop"
            elif hi >= target:
                f, reason = target, "target"
            ret = f * (1 - slip) / e - 1 - 2 * tc
        else:
            if hi >= stop:
                f, reason = stop, "stop"
            elif lo <= target:
                f, reason = target, "target"
            ret = 1 - f * (1 + slip) / e - 2 * tc - borrow * h
        rows.append({"signal_date": r["date"], "symbol": r["symbol"], "return": ret, "side": side, "score": r["score"], "reason": reason, "horizon": h})
    o = pd.DataFrame(rows)
    if o.empty:
        return o
    n = o.groupby("signal_date").symbol.transform("count")
    o["weight"] = cfg["portfolio"]["max_gross_exposure"] / n.clip(lower=1)
    o["weighted_return"] = o["return"] * o["weight"]
    return o


def tune(v, h, cfg):
    best = (10, .60)
    best_s = -1e9
    for n in [5, 10, 15, 20]:
        for th in np.arange(.50, .81, .05):
            q = v[v.score >= th].copy()
            q["rank"] = q.groupby("date")["score"].rank(method="first", ascending=False)
            q = q[q["rank"] <= n]
            t = simulate(q, h, cfg)
            if len(t) < 30:
                continue
            d = t.groupby("signal_date").weighted_return.sum()
            s = d.mean() / d.std() * math.sqrt(252) if d.std() > 0 else -1e9
            if s > best_s:
                best, best_s = (n, float(th)), s
    return best


def run_rule(df, name, cfg):
    fam, h, side = METHODS[name]
    dates = sorted(df.date.unique())
    r = cfg["research"]
    out = []
    for tr, va, te in folds(dates, int(r["train_years"] * 252), int(r["validation_months"] * 21), int(r["test_months"] * 21), int(r["step_months"] * 21), int(r["embargo_days"])):
        v = df[df.date.isin(va)].copy()
        t = df[df.date.isin(te)].copy()
        v["score"] = score(v, fam)
        t["score"] = score(t, fam)
        v["side"] = side
        t["side"] = side
        n, th = tune(v, h, cfg)
        t["rank"] = t.groupby("date")["score"].rank(method="first", ascending=False)
        p = t[(t.score >= th) & (t["rank"] <= n)]
        z = simulate(p, h, cfg)
        if not z.empty:
            out.append(z)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_long_short(df, cfg):
    dates = sorted(df.date.unique())
    r = cfg["research"]
    out = []
    for tr, va, te in folds(dates, int(r["train_years"] * 252), int(r["validation_months"] * 21), int(r["test_months"] * 21), int(r["step_months"] * 21), int(r["embargo_days"])):
        t = df[df.date.isin(te)].copy()
        t["score"] = score(t, "relative")
        t["center"] = t.groupby("date").score.transform("mean")
        longs = t[t.score >= t.center].copy()
        shorts = t[t.score < t.center].copy()
        longs["side"] = 1
        shorts["side"] = -1
        longs["rank"] = longs.groupby("date")["score"].rank(method="first", ascending=False)
        shorts["rank"] = shorts.groupby("date")["score"].rank(method="first", ascending=True)
        p = pd.concat([longs[longs["rank"] <= 5], shorts[shorts["rank"] <= 5]])
        z = simulate(p, 20, cfg)
        if not z.empty:
            out.append(z)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_ml(df, cfg):
    h = 5
    r = cfg["research"]
    feats = [c for c in FEATURES if c in df.columns]
    x = df.copy()
    x["label"] = (x.fwd_ret_5 > 0).astype(int)
    dates = sorted(x.date.unique())
    out = []
    for tr, va, te in folds(dates, int(r["train_years"] * 252), int(r["validation_months"] * 21), int(r["test_months"] * 21), int(r["step_months"] * 21), int(r["embargo_days"])):
        train = x[x.date.isin(tr)].dropna(subset=feats + ["label"])
        val = x[x.date.isin(va)].dropna(subset=feats).copy()
        test = x[x.date.isin(te)].dropna(subset=feats).copy()
        if len(train) < int(r["min_train_observations"]):
            continue
        model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, C=.25, class_weight="balanced", random_state=int(r["random_state"])))])
        model.fit(train[feats], train.label)
        val["score"] = model.predict_proba(val[feats])[:, 1]
        test["score"] = model.predict_proba(test[feats])[:, 1]
        th = max(.55, float(val.score.quantile(.80)))
        test["rank"] = test.groupby("date")["score"].rank(method="first", ascending=False)
        p = test[(test.score >= th) & (test["rank"] <= 10)].copy()
        p["side"] = 1
        z = simulate(p, h, cfg)
        if not z.empty:
            out.append(z)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(path):
    cfg = load_config(path)
    df = read_market_files(cfg["data"]["daily_glob"])
    m = read_market_files(cfg["data"]["market_glob"])
    df = add_features(df)
    m = add_features(m)
    df = add_market_context(df, m)
    df = add_cross_sectional_features(df)
    # Build next-session execution fields explicitly before any simulation.
    # The prior version tried to read next_open without creating it.
    df = prepare_entries(df)
    df = add_forward(df, cfg["research"]["horizons"])
    rows = []
    trades = []
    for name in list(METHODS) + ["Long-short cross-sectional", "ML ranker"]:
        print(name)
        t = run_long_short(df, cfg) if name == "Long-short cross-sectional" else run_ml(df, cfg) if name == "ML ranker" else run_rule(df, name, cfg)
        z = metrics(t, name, cfg["portfolio"]["initial_capital"])
        rows.append(z.__dict__)
        if not t.empty:
            trades.append(t.assign(method=name))
    Path("docs").mkdir(exist_ok=True)
    pd.DataFrame(rows).sort_values(["sharpe", "profit_factor"], ascending=False).to_csv("docs/multi_method_leaderboard.csv", index=False)
    if trades:
        pd.concat(trades, ignore_index=True).to_csv("docs/multi_method_oos_trades.csv", index=False)
    else:
        pd.DataFrame().to_csv("docs/multi_method_oos_trades.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/multi_method.yaml")
    a = p.parse_args()
    main(a.config)
