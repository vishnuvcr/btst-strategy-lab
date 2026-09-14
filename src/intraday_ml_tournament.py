from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "range_pct", "body_pct",
    "close_location", "volume_z", "vwap_gap", "day_gap", "orb_gap",
    "rv_ratio", "time_sin", "time_cos"
]


def cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm(c: str) -> str:
    return str(c).replace("\ufeff", "").strip().lower().replace(" ", "_").replace("-", "_")


def load_5m(pattern: str, start: str, end: str, max_symbols: int) -> pd.DataFrame:
    files = sorted(glob.glob(pattern, recursive=True))
    frames: List[pd.DataFrame] = []
    for fp in files[:max_symbols]:
        try:
            x = pd.read_csv(fp)
        except Exception:
            continue
        x.columns = [norm(c) for c in x.columns]
        dt = next((c for c in ["datetime", "timestamp", "time", "date"] if c in x.columns), None)
        if dt is None:
            continue
        aliases = {"open": ["o"], "high": ["h"], "low": ["l"], "close": ["c", "price", "adj_close"], "volume": ["vol", "v"]}
        ren = {}
        for c, aa in aliases.items():
            if c not in x.columns:
                for a in aa:
                    if a in x.columns and a != dt:
                        ren[a] = c
                        break
        x = x.rename(columns=ren)
        if not all(c in x.columns for c in ["open", "high", "low", "close"]):
            continue
        x[dt] = pd.to_datetime(x[dt], errors="coerce", format="mixed")
        if getattr(x[dt].dt, "tz", None) is not None:
            x[dt] = x[dt].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        else:
            # Dataset timestamps are treated as India-local unless an offset was supplied.
            x[dt] = x[dt]
        for c in ["open", "high", "low", "close", "volume"]:
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        x = x.rename(columns={dt: "datetime"})
        x["date"] = x.datetime.dt.normalize()
        x["symbol"] = Path(fp).stem.upper().replace("-", "_")
        x = x.dropna(subset=["datetime", "open", "high", "low", "close"])
        frames.append(x[["datetime", "date", "symbol", "open", "high", "low", "close"] + (["volume"] if "volume" in x.columns else [])])
    if not frames:
        raise RuntimeError(f"No usable intraday files found for {pattern}")
    out = pd.concat(frames, ignore_index=True)
    out = out[(out.date >= pd.Timestamp(start)) & (out.date <= pd.Timestamp(end))]
    out = out.sort_values(["symbol", "datetime"]).drop_duplicates(["symbol", "datetime"], keep="last")
    return out.reset_index(drop=True)


def prepare(x: pd.DataFrame, min_bars: int) -> pd.DataFrame:
    x = x.copy().sort_values(["symbol", "datetime"])
    counts = x.groupby(["symbol", "date"])["datetime"].transform("size")
    x = x[counts >= min_bars].copy()
    g = x.groupby("symbol", group_keys=False)
    x["ret_1"] = g.close.pct_change()
    for n in [3, 6, 12, 24]:
        x[f"ret_{n}"] = g.close.pct_change(n)
    x["range_pct"] = (x.high - x.low) / x.close.replace(0, np.nan)
    x["body_pct"] = (x.close - x.open) / x.open.replace(0, np.nan)
    x["close_location"] = (x.close - x.low) / (x.high - x.low).replace(0, np.nan)
    if "volume" in x:
        gm = x.groupby("symbol").volume.transform(lambda s: s.rolling(78, min_periods=30).mean())
        gs = x.groupby("symbol").volume.transform(lambda s: s.rolling(78, min_periods=30).std())
        x["volume_z"] = (x.volume - gm) / gs.replace(0, np.nan)
    else:
        x["volume_z"] = 0.0
    # Session-relative VWAP and opening-range features. These use only current/past bars.
    pv = x.close * x.get("volume", pd.Series(1.0, index=x.index)).fillna(0)
    x["cum_pv"] = pv.groupby([x.symbol, x.date]).cumsum()
    x["cum_vol"] = x.get("volume", pd.Series(1.0, index=x.index)).fillna(1).groupby([x.symbol, x.date]).cumsum()
    x["vwap"] = x.cum_pv / x.cum_vol.replace(0, np.nan)
    x["vwap_gap"] = x.close / x.vwap.replace(0, np.nan) - 1
    first = x.groupby(["symbol", "date"])["close"].transform("first")
    x["day_gap"] = x.open / g.close.shift(1) - 1
    x["minutes"] = (x.datetime.dt.hour * 60 + x.datetime.dt.minute).astype(float)
    x["time_sin"] = np.sin(2 * np.pi * x.minutes / 1440)
    x["time_cos"] = np.cos(2 * np.pi * x.minutes / 1440)
    # Realized-volatility expansion relative to its recent intraday baseline.
    rv = x.groupby("symbol").ret_1.transform(lambda s: s.rolling(24, min_periods=12).std())
    rv0 = x.groupby("symbol").ret_1.transform(lambda s: s.rolling(240, min_periods=60).std())
    x["rv_ratio"] = rv / rv0.replace(0, np.nan)
    x["first_close"] = first
    x["session_bar"] = x.groupby(["symbol", "date"]).cumcount()
    x["orb_gap"] = np.nan
    for n in [3, 6, 12]:
        hi = x.groupby(["symbol", "date"]).high.transform(lambda s: s.iloc[:n].max())
        lo = x.groupby(["symbol", "date"]).low.transform(lambda s: s.iloc[:n].min())
        x[f"orb_hi_{n}"] = hi
        x[f"orb_lo_{n}"] = lo
        x[f"orb_gap_{n}"] = (x.close - (hi + lo) / 2) / ((hi - lo).replace(0, np.nan))
    # Forward return label is strictly future bars.
    future = g.close.shift(-3) / x.close - 1
    x["future_ret_3"] = future
    x["label"] = (future > 0).astype(float)
    return x.replace([np.inf, -np.inf], np.nan)


def signal_rule(x: pd.DataFrame, method: str, orb: int, horizon: int) -> pd.Series:
    b = x.session_bar >= orb
    if method == "opening_range_breakout":
        return np.where(b & (x.close > x[f"orb_hi_{orb}"]) & (x.volume_z > -1), 1.0, np.nan)
    if method == "opening_range_reversal":
        return np.where(b & (x.close < x[f"orb_lo_{orb}"]) & (x.vwap_gap < 0), 1.0, np.nan)
    if method == "vwap_reversion":
        return np.where(b & (x.vwap_gap < -0.003) & (x.close_location > 0.5), 1.0, np.nan)
    if method == "intraday_momentum":
        return np.where(b & (x.ret_6 > 0.004) & (x.ret_12 > 0.005), 1.0, np.nan)
    if method == "volatility_expansion":
        return np.where(b & (x.rv_ratio > 1.5) & (x.close_location > 0.6), 1.0, np.nan)
    raise ValueError(method)


def simulate(row: pd.Series, bars: pd.DataFrame, horizon: int, stop_pct: float, target_pct: float, sl: float, tc: float) -> Tuple[float, str]:
    i = int(row._row_pos)
    end = min(i + horizon, len(bars) - 1)
    if i >= end:
        return np.nan, "no_exit"
    entry = float(bars.iloc[i].open) * (1 + sl)
    stop = entry * (1 - stop_pct)
    target = entry * (1 + target_pct)
    for j in range(i, end + 1):
        hi, lo = float(bars.iloc[j].high), float(bars.iloc[j].low)
        if lo <= stop:
            return stop / entry - 1 - sl - tc, "stop"
        if hi >= target:
            return target / entry - 1 - sl - tc, "target"
    exit_px = float(bars.iloc[end].close)
    return exit_px / entry - 1 - sl - tc, "time"


def metrics(t: pd.DataFrame) -> Dict[str, float]:
    if t.empty:
        return {"trades": 0, "win_rate": np.nan, "profit_factor": np.nan, "total_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    r = t.weighted_return.astype(float)
    eq = (1 + r).cumprod()
    gains = r[r > 0].sum(); losses = -r[r < 0].sum()
    pf = gains / losses if losses > 0 else np.inf
    sh = np.sqrt(252 * 78) * r.mean() / r.std() if r.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    return {"trades": int(len(t)), "win_rate": float((r > 0).mean()), "profit_factor": float(pf), "total_return": float(eq.iloc[-1] - 1), "sharpe": float(sh), "max_drawdown": float(dd), "expectancy": float(r.mean())}


def run(cfg_path: str) -> None:
    c = cfg(cfg_path)
    x = prepare(load_5m(c["data"]["intraday_glob"], c["data"]["start_date"], c["data"]["end_date"], int(c["data"]["max_symbols"])), int(c["data"]["min_bars_per_day"]))
    x = x[x.close >= float(c["data"]["min_price"])].copy()
    dates = np.array(sorted(x.date.dropna().unique()))
    train_n, val_n, test_n, step = [int(c["research"][k]) for k in ["train_days", "validation_days", "test_days", "step_days"]]
    methods = list(c["methods"])
    all_trades = []
    rng = np.random.RandomState(int(c["research"]["random_state"]))
    for start in range(train_n, len(dates) - val_n - test_n + 1, step):
        train_dates = dates[start-train_n:start]
        val_dates = dates[start:start+val_n]
        test_dates = dates[start+val_n:start+val_n+test_n]
        train = x[x.date.isin(train_dates)].copy()
        val = x[x.date.isin(val_dates)].copy()
        test = x[x.date.isin(test_dates)].copy()
        # Fit one classifier per fold on past data only.
        clf = None
        tr = train.dropna(subset=FEATURES + ["label"])
        if len(tr) >= 500 and tr.label.nunique() == 2:
            clf = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))])
            clf.fit(tr[FEATURES], tr.label)
        for method in methods:
            if method in {"gap_ml", "ml_direction", "ml_ranker"} and clf is None:
                continue
            z = test.copy()
            if method == "gap_ml":
                z["score"] = np.nan
                q = z.day_gap.abs() > 0.004
                z.loc[q, "score"] = clf.predict_proba(z.loc[q, FEATURES].fillna(0))[:, 1]
                z = z[z.score >= 0.60]
            elif method == "ml_direction":
                z["score"] = np.nan
                q = z[FEATURES].notna().all(axis=1)
                z.loc[q, "score"] = clf.predict_proba(z.loc[q, FEATURES])[:, 1]
                z = z[z.score >= 0.65]
            elif method == "ml_ranker":
                z["score"] = np.nan
                q = z[FEATURES].notna().all(axis=1)
                z.loc[q, "score"] = clf.predict_proba(z.loc[q, FEATURES])[:, 1]
                z["rank"] = z.groupby("datetime").score.rank(pct=True)
                z = z[z.rank >= 0.90]
            else:
                # Validation chooses the opening-range/horizon pair; deterministic tie-breaking avoids leakage.
                best = (0.0, 3, 6)
                for orb in c["research"]["opening_range_bars"]:
                    for h in c["research"]["horizons_bars"]:
                        vv = val.copy()
                        vv["s"] = signal_rule(vv, method, int(orb), int(h))
                        rr = vv[vv.s.notna() & vv.future_ret_3.notna()].future_ret_3
                        score = float(rr.mean()) if len(rr) >= 30 else -1
                        if score > best[0]: best = (score, int(orb), int(h))
                _, orb, horizon = best
                z["score"] = signal_rule(z, method, orb, horizon)
            if z.empty:
                continue
            horizon = 6 if method in {"gap_ml", "ml_direction", "ml_ranker"} else horizon
            # Execute next bar open; simulate within each symbol independently.
            for sym, zz in z.groupby("symbol"):
                bars = test[test.symbol == sym].sort_values("datetime").reset_index(drop=True)
                idxmap = {d: i for i, d in enumerate(bars.datetime)}
                zz = zz.sort_values("datetime")
                used = set()
                for _, r in zz.iterrows():
                    pos = idxmap.get(r.datetime)
                    if pos is None or pos + 1 >= len(bars) or pos in used:
                        continue
                    # Avoid overlapping positions in the same symbol.
                    used.add(pos)
                    rr, reason = simulate(r.assign(_row_pos=pos + 1), bars, min(horizon, int(c["execution"]["max_holding_bars"])), float(c["execution"]["stop_pct"]), float(c["execution"]["target_pct"]), float(c["costs"]["slippage_bps_per_side"])/10000, float(c["costs"]["transaction_cost_bps_per_side"])/10000)
                    if pd.notna(rr):
                        all_trades.append({"signal_datetime": r.datetime, "exit_date": bars.iloc[min(pos+1+horizon, len(bars)-1)].date, "symbol": sym, "method": method, "return": rr, "weighted_return": rr / int(c["portfolio"]["max_positions"]), "score": float(r.score), "reason": reason, "horizon_bars": int(horizon)})
    trades = pd.DataFrame(all_trades)
    os.makedirs("docs", exist_ok=True)
    trades.to_csv("docs/intraday_ml_oos_trades.csv", index=False)
    rows = []
    for method, t in trades.groupby("method") if not trades.empty else []:
        m = metrics(t.sort_values("signal_datetime")); m["strategy"] = method; rows.append(m)
    lb = pd.DataFrame(rows).sort_values("profit_factor", ascending=False) if rows else pd.DataFrame()
    lb.to_csv("docs/intraday_ml_leaderboard.csv", index=False)
    if not trades.empty:
        monthly = trades.groupby(["method", trades.exit_date.dt.to_period("M")]).weighted_return.sum().reset_index(name="monthly_return")
        monthly.to_csv("docs/intraday_ml_monthly_returns.csv", index=False)
    print(lb.to_string(index=False) if not lb.empty else "No OOS trades produced")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/intraday_ml.yaml")
    run(p.parse_args().config)
