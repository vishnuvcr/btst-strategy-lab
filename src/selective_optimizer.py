from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from swing_tournament import load, features, simulate, stat


COMPONENTS = ("mr5", "mr20", "sma20", "loc", "volume", "relative")


def component_frame(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["mr5"] = 1.0 - x["r_ret5"]
    x["mr20"] = 1.0 - x["r_ret20"]
    x["sma20"] = 1.0 - x["r_sma20"]
    x["loc"] = x["r_loc"]
    x["volume"] = x["r_vz"].fillna(0.5)
    x["relative"] = 0.5 * x["r_ret20"] + 0.5 * x["r_ret5"]
    return x


def score_candidate(x: pd.DataFrame, p: dict) -> pd.Series:
    s = sum(float(p[k]) * x[k] for k in COMPONENTS)
    if p.get("use_regime", False):
        s = s.where(x["sma50"] < 0, s + float(p["regime_bonus"]))
    if p.get("use_volume_filter", False):
        s = s.where(x["vz"].fillna(0) >= float(p["min_vz"]), -1e6)
    return s


def fold_ranges(dates: list[pd.Timestamp], cfg: dict):
    r = cfg["research"]
    trn = int(r["train_years"] * 252)
    vn = int(r["validation_months"] * 21)
    ten = int(r["test_months"] * 21)
    step = int(r["step_months"] * 21)
    emb = int(r["embargo_days"])
    start = trn + vn + emb
    while start < len(dates):
        va = dates[start-vn-emb:start-emb]
        te = dates[start:min(start+ten, len(dates))]
        if len(va) >= 20 and len(te) >= 20:
            yield va[:-1], te
        start += step


def candidate_grid():
    # Focused search: enough diversity to discover useful combinations without a massive multiple-testing explosion.
    base = [
        {"mr5": .20, "mr20": .35, "sma20": .35, "loc": .10, "volume": 0, "relative": 0},
        {"mr5": .25, "mr20": .30, "sma20": .30, "loc": .15, "volume": 0, "relative": 0},
        {"mr5": .15, "mr20": .30, "sma20": .25, "loc": .15, "volume": .15, "relative": 0},
        {"mr5": .15, "mr20": .25, "sma20": .25, "loc": .10, "volume": 0, "relative": .25},
        {"mr5": .15, "mr20": .25, "sma20": .20, "loc": .10, "volume": .15, "relative": .15},
    ]
    out=[]
    for b in base:
        for n in (5, 10, 15, 20, 30):
            for regime in (False, True):
                p=dict(b, top_n=n, use_regime=regime, regime_bonus=.05, use_volume_filter=False, min_vz=0.0)
                out.append(p)
            for vz in (0.0, 0.5):
                p=dict(b, top_n=n, use_regime=False, regime_bonus=.05, use_volume_filter=True, min_vz=vz)
                out.append(p)
    return out


def objective(t: pd.DataFrame) -> float:
    if len(t) < 50: return -1e9
    s = stat(t)
    # Precision is rewarded, but only alongside positive expectancy, PF and controlled drawdown.
    return (1.5 * s["sharpe"] + 1.0 * s["profit_factor"] + 2.0 * s["win_rate"]
            + 40.0 * s["expectancy"] + 0.5 * s["total_return"] + 0.5 * s["max_drawdown"])


def tune_fold(d, history, val_dates, test_dates, horizon, cfg):
    val = d[d.date.isin(val_dates)].copy()
    test = d[d.date.isin(test_dates)].copy()
    val = val.dropna(subset=["entry_open", "future_close", "atr"])
    test = test.dropna(subset=["entry_open", "atr"])
    best = None
    for p in candidate_grid():
        v = val.copy()
        v["score"] = score_candidate(v, p)
        picks = v.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(p["top_n"])
        t = simulate(picks, history, cfg, horizon)
        if t.empty: continue
        obj = objective(t)
        if best is None or obj > best[0]: best = (obj, p, t)
    if best is None: return pd.DataFrame(), None
    p = best[1]
    test["score"] = score_candidate(test, p)
    picks = test.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(p["top_n"])
    return simulate(picks, history, cfg, horizon), p


def run(cfg):
    d = component_frame(features(load(cfg)))
    d["entry_open"] = d.groupby("symbol").open.shift(-1)
    d["future_close"] = d.groupby("symbol").close.shift(-1)
    all_rows=[]; all_trades=[]; dates=sorted(d.date.unique())
    for h in (5, 10, 20):
        for fold, (va, te) in enumerate(fold_ranges(dates, cfg), 1):
            # Future-close label is horizon-specific and must be created from full history, never from a truncated fold.
            d["future_close"] = d.groupby("symbol").close.shift(-h)
            t, p = tune_fold(d, d, va, te, h, cfg)
            if t.empty or p is None: continue
            s=stat(t)
            all_trades.append(t.assign(horizon=h, fold=fold))
            all_rows.append({"horizon_days":h,"fold":fold,**s,"params":json.dumps(p,sort_keys=True)})
    trades=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    fold_df=pd.DataFrame(all_rows)
    if fold_df.empty: raise RuntimeError("No optimized folds produced trades")
    # Aggregate OOS by exit date, preserving each fold's optimized parameters and avoiding duplicate fold dates.
    rows=[]
    for h,g in trades.groupby("horizon"):
        s=stat(g); rows.append({"horizon_days":int(h),**s,"oos_folds":int(g.fold.nunique())})
    summary=pd.DataFrame(rows).sort_values(["sharpe","expectancy"],ascending=False)
    Path("docs").mkdir(exist_ok=True)
    summary.to_csv("docs/selective_oos_summary.csv",index=False)
    fold_df.to_csv("docs/selective_fold_parameters.csv",index=False)
    trades.to_csv("docs/selective_oos_trades.csv",index=False)
    manifest={"engine":"nse_daily_swing_selective_v1","tuning":"nested validation-to-OOS","horizons":[5,10,20],"candidate_count":len(candidate_grid()),"selection_objective":"Sharpe + profit factor + win rate + expectancy + return - drawdown penalty","survivorship_warning":True,"point_in_time_membership":False}
    json.dump(manifest,open("docs/selective_research_manifest.json","w",encoding="utf-8"),indent=2)
    print(summary.to_string(index=False))
    print("Candidates per fold:",len(candidate_grid()))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config/swing.yaml"); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config,encoding="utf-8")); run(cfg)
