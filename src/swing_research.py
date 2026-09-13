from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_dates(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    compact = x.str.fullmatch(r"\d{8}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if compact.any():
        out.loc[compact] = pd.to_datetime(x.loc[compact], format="%Y%m%d", errors="coerce")
    rem = out.isna()
    if rem.any():
        out.loc[rem] = pd.to_datetime(x.loc[rem], errors="coerce", format="mixed")
    return out.dt.normalize()


def load_universe(cfg: dict) -> pd.DataFrame:
    files = sorted(Path("data/nse_all_daily").rglob("*.csv"))
    if not files:
        raise RuntimeError("No NSE-wide CSV files found under data/nse_all_daily")
    frames = []
    for fp in files:
        try:
            x = pd.read_csv(fp)
        except Exception:
            continue
        x.columns = [str(c).replace("\ufeff", "").strip().lower().replace(" ", "_") for c in x.columns]
        aliases = {
            "symbol": ["symbol", "ticker", "code"],
            "date": ["date", "datetime", "timestamp"],
            "open": ["open"], "high": ["high"], "low": ["low"], "close": ["close"],
            "adj_close": ["adj_close", "adjusted_close"], "volume": ["volume", "vol"]
        }
        col = {}
        for k, opts in aliases.items():
            col[k] = next((c for c in opts if c in x.columns), None)
        if not all(col[k] for k in ["symbol", "date", "open", "high", "low", "close"]):
            continue
        keep = {col[k]: k for k in col if col[k]}
        x = x.rename(columns=keep)
        if "symbol" not in x.columns:
            continue
        for c in ["open", "high", "low", "close", "adj_close", "volume"]:
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        x["date"] = parse_dates(x["date"])
        x["symbol"] = x["symbol"].astype(str).str.upper().str.replace(r"\.NS$", "", regex=True)
        x = x.dropna(subset=["date", "open", "high", "low", "close", "symbol"])
        if not x.empty:
            frames.append(x[[c for c in ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"] if c in x.columns]])
    if not frames:
        raise RuntimeError("No consolidated NSE OHLCV file with a symbol column was found")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")

    min_days = int(cfg["data"].get("min_history_days", 252))
    min_price = float(cfg["data"].get("min_price", 20))
    min_turnover = float(cfg["data"].get("min_turnover_inr", 1e7))
    g = df.groupby("symbol", group_keys=False)
    df["turnover"] = df["close"] * df.get("volume", pd.Series(np.nan, index=df.index))
    history = g["date"].transform("count")
    median_turnover = g["turnover"].transform(lambda s: s.rolling(20, min_periods=10).median())
    eligible = (history >= min_days) & (df["close"] >= min_price) & (median_turnover >= min_turnover)
    df = df.loc[eligible].copy()
    max_symbols = int(cfg["data"].get("max_symbols", 5000))
    symbols = sorted(df["symbol"].unique())[:max_symbols]
    return df[df.symbol.isin(symbols)].copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", group_keys=False)
    px = "adj_close" if "adj_close" in df.columns and df["adj_close"].notna().mean() > 0.95 else "close"
    df["feature_price"] = df[px]
    df["ret_1"] = g["feature_price"].pct_change()
    for n in [5, 10, 20, 60]:
        df[f"ret_{n}"] = g["feature_price"].pct_change(n)
    prev_close = g["close"].shift(1)
    df["gap"] = df["open"] / prev_close - 1
    df["range_pct"] = (df.high - df.low) / df.close.replace(0, np.nan)
    df["close_location"] = (df.close - df.low) / (df.high - df.low).replace(0, np.nan)
    df["body_pct"] = (df.close - df.open) / df.open.replace(0, np.nan)
    df["atr_pct"] = g["range_pct"].transform(lambda s: s.rolling(14, min_periods=10).mean())
    df["sma20_gap"] = df.feature_price / g["feature_price"].transform(lambda s: s.rolling(20, min_periods=15).mean()) - 1
    df["sma50_gap"] = df.feature_price / g["feature_price"].transform(lambda s: s.rolling(50, min_periods=30).mean()) - 1
    if "volume" in df:
        vm = g.volume.transform(lambda s: s.rolling(20, min_periods=15).mean())
        vs = g.volume.transform(lambda s: s.rolling(20, min_periods=15).std())
        df["volume_z"] = (df.volume - vm) / vs.replace(0, np.nan)
    else:
        df["volume_z"] = 0.0
    for c in ["ret_1", "ret_5", "ret_20", "gap", "close_location", "volume_z", "sma20_gap"]:
        df[f"rank_{c}"] = df.groupby("date")[c].rank(pct=True)
    return df.replace([np.inf, -np.inf], np.nan)


def score(x: pd.DataFrame, family: str) -> pd.Series:
    if family == "momentum":
        return .45*x.rank_ret_5 + .35*x.rank_ret_20 + .20*x.rank_close_location
    if family == "mean_reversion":
        return .55*(1-x.rank_sma20_gap) + .25*(1-x.rank_ret_5) + .20*x.rank_close_location
    if family == "closing_strength":
        return .65*x.rank_close_location + .35*x.rank_ret_1
    if family == "volume":
        return .45*x.rank_volume_z + .35*x.rank_ret_5 + .20*x.rank_close_location
    if family == "breakout":
        return .55*x.rank_ret_20 + .25*x.rank_ret_5 + .20*x.rank_close_location
    if family == "regime":
        return .65*x.rank_ret_20 + .35*x.rank_close_location
    if family == "relative_strength":
        return .70*x.rank_ret_20 + .30*x.rank_ret_5
    raise ValueError(f"Unknown family: {family}")


def future_maps(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("symbol", group_keys=False)
    out["entry_open"] = g.open.shift(-1)
    out["exit_close"] = g.close.shift(-horizon)
    out["future_date"] = g.date.shift(-horizon)
    out["realized_return"] = out.exit_close / out.entry_open - 1
    return out


def simulate(selected: pd.DataFrame, cfg: dict, horizon: int) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    stop_mult = float(cfg["execution"]["stop_atr_mult"])
    target_mult = float(cfg["execution"]["target_atr_mult"])
    stop_pct = float(cfg["execution"]["stop_pct"])/100
    target_pct = float(cfg["execution"]["target_pct"])/100
    sl = float(cfg["costs"]["slippage_bps_per_side"])/10000
    tc = float(cfg["costs"]["transaction_cost_bps_per_side"])/10000
    rows = []
    grouped = {s: g for s, g in selected.groupby("symbol")}
    for _, r in selected.iterrows():
        sym = r.symbol
        # Use the full symbol history to model daily stop/target checks during the holding window.
        hist = grouped.get(sym)
        if hist is None:
            continue
        entry_date = r.date
        path = hist[hist.date > entry_date].sort_values("date").head(horizon)
        if len(path) < horizon:
            continue
        entry = float(r.entry_open) * (1 + sl)
        atr = float(r.atr_pct) if pd.notna(r.atr_pct) else stop_pct/stop_mult
        atr = min(max(atr, .005), .20)
        stop = entry*(1-stop_mult*atr) if str(cfg["execution"].get("stop_mode","ATR")).upper()=="ATR" else entry*(1-stop_pct)
        target = entry*(1+target_mult*atr) if str(cfg["execution"].get("target_mode","ATR")).upper()=="ATR" else entry*(1+target_pct)
        exit_px = float(path.iloc[-1].close)
        reason = "time"
        for _, d in path.iterrows():
            if float(d.low) <= stop:
                exit_px, reason = stop, "stop"
                break
            if float(d.high) >= target:
                exit_px, reason = target, "target"
                break
        exit_px *= (1-sl)
        net = (exit_px/entry-1) - 2*tc
        rows.append({"signal_date": entry_date, "symbol": sym, "horizon": horizon, "entry": entry, "exit": exit_px, "return": net, "reason": reason, "score": r.score})
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades
    max_gross = float(cfg["portfolio"]["max_gross_exposure"])
    counts = trades.groupby("signal_date").symbol.transform("count")
    trades["weight"] = max_gross / counts.clip(lower=1)
    trades["weighted_return"] = trades["return"] * trades["weight"]
    return trades


def stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades":0,"total_return":0.0,"annualized_return":0.0,"sharpe":0.0,"max_drawdown":0.0,"profit_factor":0.0,"expectancy":0.0,"win_rate":0.0}
    daily = trades.groupby("signal_date").weighted_return.sum().sort_index()
    eq = (1+daily).cumprod()
    dd = eq/eq.cummax()-1
    wins = trades.loc[trades.return>0,"return"].sum()
    losses = abs(trades.loc[trades.return<=0,"return"].sum())
    total = float(eq.iloc[-1]-1)
    days = max((daily.index.max()-daily.index.min()).days, 1)
    years = max(days/365.25, 1/365.25)
    ann = (1+total)**(1/years)-1 if 1+total>0 else -1
    sd = daily.std()
    return {"trades":int(len(trades)),"total_return":total,"annualized_return":float(ann),"sharpe":float(daily.mean()/sd*math.sqrt(252)) if sd>0 else 0.0,"max_drawdown":float(dd.min()),"profit_factor":float(wins/losses) if losses else float("inf"),"expectancy":float(trades.return.mean()),"win_rate":float((trades.return>0).mean())}


def walk_forward(df: pd.DataFrame, family: str, horizon: int, cfg: dict) -> tuple[pd.DataFrame, dict]:
    dates = sorted(df.date.unique())
    r = cfg["research"]
    train_n = int(r["train_years"]*252)
    val_n = int(r["validation_months"]*21)
    test_n = int(r["test_months"]*21)
    step_n = int(r["step_months"]*21)
    embargo = int(r["embargo_days"])
    folds=[]; params=[]
    start=train_n+val_n+embargo
    while start < len(dates):
        tr=dates[start-val_n-embargo-train_n:start-val_n-embargo]
        va=dates[start-val_n-embargo:start-embargo]
        te=dates[start:min(start+test_n,len(dates))]
        if len(tr)<train_n or len(va)<20 or len(te)<20: break
        train=df[df.date.isin(tr)].copy()
        val=df[df.date.isin(va[:-1])].copy()
        test=df[df.date.isin(te)].copy()
        val=future_maps(val,horizon); test=future_maps(test,horizon)
        val=val.dropna(subset=["entry_open","exit_close"]); test=test.dropna(subset=["entry_open","exit_close"])
        best=(10,-1e9)
        for top_n in [10,20,30]:
            if family in {"ml_ranker","hybrid"}:
                continue
            v=val.copy(); v["score"]=score(v,family)
            picks=v.sort_values(["date","score"],ascending=[True,False]).groupby("date").head(top_n)
            sc=float(picks.realized_return.mean()) if not picks.empty else -1e9
            if sc>best[1]: best=(top_n,sc)
        top_n=best[0]
        params.append(top_n)
        if family in {"ml_ranker","hybrid"}:
            from sklearn.linear_model import LogisticRegression
            feats=[c for c in ["ret_1","ret_5","ret_20","gap","range_pct","close_location","body_pct","atr_pct","sma20_gap","sma50_gap","volume_z"] if c in train]
            train2=train.dropna(subset=feats).copy()
            val2=val.dropna(subset=feats).copy(); test2=test.dropna(subset=feats).copy()
            if len(train2)>=int(r["min_train_observations"]):
                y=(train2.feature_price.shift(-horizon)/train2.feature_price-1>0).astype(int)
                y=y.fillna(0)
                model=LogisticRegression(max_iter=1000,C=.25,class_weight="balanced",random_state=int(r["random_state"]))
                model.fit(train2[feats],y)
                val2["score"]=model.predict_proba(val2[feats])[:,1]
                test2["score"]=model.predict_proba(test2[feats])[:,1]
                threshold=.60
                q=val2[val2.score>=threshold]
                if not q.empty: threshold=float(np.clip(q.score.quantile(.50),.50,.80))
                test=test2[test2.score>=threshold].copy()
                test=test.sort_values(["date","score"],ascending=[True,False]).groupby("date").head(int(cfg["portfolio"]["max_positions"]))
            else:
                test=pd.DataFrame()
        else:
            test["score"]=score(test,family)
            test=test.sort_values(["date","score"],ascending=[True,False]).groupby("date").head(top_n)
        trades=simulate(test,cfg,horizon)
        if not trades.empty:
            trades["fold"]=len(folds)+1; folds.append(trades)
        start += step_n
    out=pd.concat(folds,ignore_index=True) if folds else pd.DataFrame()
    stability=float(pd.Series(params).value_counts(normalize=True).iloc[0]) if params else 0.0
    return out,{"folds":len(folds),"parameter_stability":stability,"params":params}


def main(config_path: str) -> None:
    cfg=yaml.safe_load(open(config_path,encoding="utf-8"))
    df=add_features(load_universe(cfg))
    families=cfg["strategy_search"]["families"]
    horizons=cfg["research"]["horizons_days"]
    rows=[]; all_trades=[]; diagnostics={}
    for h in horizons:
        for family in families:
            if family in {"ml_ranker","hybrid"}:
                pass
            trades,diag=walk_forward(df,family,int(h),cfg)
            s=stats(trades)
            eligible=(s["trades"]>=int(cfg["research"]["min_trades_oos"]) and diag["parameter_stability"]>=.34 and s["expectancy"]>0 and s["profit_factor"]>1 and s["sharpe"]>0 and s["total_return"]>0 and s["max_drawdown"]>-0.50)
            rows.append({"horizon_days":int(h),"strategy":family,**s,"parameter_stability":diag["parameter_stability"],"eligible":bool(eligible)})
            diagnostics[f"{family}_{h}"]=diag
            if not trades.empty: all_trades.append(trades.assign(strategy=family))
    lb=pd.DataFrame(rows).sort_values(["eligible","sharpe","total_return"],ascending=[False,False,False])
    Path("docs").mkdir(exist_ok=True)
    lb.to_csv("docs/swing_strategy_leaderboard.csv",index=False)
    pd.concat(all_trades,ignore_index=True).to_csv("docs/swing_oos_trades.csv",index=False) if all_trades else pd.DataFrame().to_csv("docs/swing_oos_trades.csv",index=False)
    manifest={"engine":"nse_daily_swing_v1","universe":cfg["universe"],"symbols_tested":int(df.symbol.nunique()),"date_start":str(df.date.min().date()),"date_end":str(df.date.max().date()),"horizons_days":horizons,"strategies":families,"adjusted_feature_price":bool("adj_close" in df.columns and df.adj_close.notna().mean()>.95),"execution_prices":"raw OHLC","survivorship_warning":True,"diagnostics":diagnostics}
    json.dump(manifest,open("docs/swing_research_manifest.json","w",encoding="utf-8"),indent=2,default=str)
    print(lb.to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/swing.yaml"); args=p.parse_args(); main(args.config)
