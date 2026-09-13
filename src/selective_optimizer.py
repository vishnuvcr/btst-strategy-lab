from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import yaml

from swing_tournament import load, features, add_forward_fields, simulate, stat

COMPONENTS = ("mr5", "mr20", "sma20", "loc", "volume", "relative")

_WORKER_D = None
_WORKER_CFG = None
_WORKER_DATES = None


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
    base = [
        {"mr5": .20, "mr20": .35, "sma20": .35, "loc": .10, "volume": 0, "relative": 0},
        {"mr5": .25, "mr20": .30, "sma20": .30, "loc": .15, "volume": 0, "relative": 0},
        {"mr5": .15, "mr20": .30, "sma20": .25, "loc": .15, "volume": .15, "relative": 0},
        {"mr5": .15, "mr20": .25, "sma20": .25, "loc": .10, "volume": 0, "relative": .25},
        {"mr5": .15, "mr20": .25, "sma20": .20, "loc": .10, "volume": .15, "relative": .15},
    ]
    out = []
    for b in base:
        for n in (5, 10, 15, 20, 30):
            out.append(dict(b, top_n=n, use_regime=False, regime_bonus=.05, use_volume_filter=False, min_vz=0.0))
            out.append(dict(b, top_n=n, use_regime=True, regime_bonus=.05, use_volume_filter=False, min_vz=0.0))
            out.append(dict(b, top_n=n, use_regime=False, regime_bonus=.05, use_volume_filter=True, min_vz=0.0))
            out.append(dict(b, top_n=n, use_regime=False, regime_bonus=.05, use_volume_filter=True, min_vz=.5))
    assert len(out) == 100
    return out


def objective(t: pd.DataFrame) -> float:
    if len(t) < 50:
        return -1e9
    s = stat(t)
    return (1.5*s["sharpe"] + s["profit_factor"] + 2*s["win_rate"] +
            40*s["expectancy"] + .5*s["total_return"] + .5*s["max_drawdown"])


def validation_proxy(v: pd.DataFrame, p: dict):
    """Cheap first-pass selector. Exact OHLC stop/target simulation is run only on finalists."""
    x = v.copy()
    x["score"] = score_candidate(x, p)
    picks = x.sort_values(["date", "score"], ascending=[True, False]).groupby("date", sort=False).head(p["top_n"])
    if picks.empty:
        return -1e9
    r = picks["future_close"] / picks["entry_open"] - 1
    return float(r.mean() + .25*(r > 0).mean())


def tune_fold(d, history, val_dates, test_dates, horizon, cfg):
    val = d[d.date.isin(val_dates)].dropna(subset=["entry_open", "future_close", "atr"]).copy()
    test = d[d.date.isin(test_dates)].dropna(subset=["entry_open", "atr"]).copy()
    ranked = sorted(((validation_proxy(val, p), p) for p in candidate_grid()), key=lambda z: z[0], reverse=True)
    best = None
    for _, p in ranked[:5]:
        v = val.copy(); v["score"] = score_candidate(v, p)
        picks = v.sort_values(["date", "score"], ascending=[True, False]).groupby("date", sort=False).head(p["top_n"])
        t = simulate(picks, history, cfg, horizon)
        if t.empty:
            continue
        obj = objective(t)
        if best is None or obj > best[0]:
            best = (obj, p)
    if best is None:
        return pd.DataFrame(), None
    p = best[1]
    test["score"] = score_candidate(test, p)
    picks = test.sort_values(["date", "score"], ascending=[True, False]).groupby("date", sort=False).head(p["top_n"])
    return simulate(picks, history, cfg, horizon), p


def _run_horizon_worker(horizon: int):
    """Run one complete nested walk-forward horizon in an isolated process."""
    d = _WORKER_D
    cfg = _WORKER_CFG
    dates = _WORKER_DATES
    d = add_forward_fields(d, horizon)
    all_rows, all_trades = [], []
    for fold, (va, te) in enumerate(fold_ranges(dates, cfg), 1):
        t, p = tune_fold(d, d, va, te, horizon, cfg)
        if t.empty or p is None:
            continue
        all_trades.append(t.assign(horizon=horizon, fold=fold))
        all_rows.append({"horizon_days": horizon, "fold": fold, **stat(t), "params": json.dumps(p, sort_keys=True)})
    return horizon, all_rows, all_trades


def run(cfg, horizons=None):
    global _WORKER_D, _WORKER_CFG, _WORKER_DATES

    horizons = tuple(sorted(set(horizons or (5, 10, 20))))
    if not horizons or not all(h in (5, 10, 20) for h in horizons):
        raise ValueError("horizons must be a non-empty subset of 5, 10, 20")

    d = component_frame(features(load(cfg)))
    _WORKER_D = d
    _WORKER_CFG = cfg
    _WORKER_DATES = sorted(d.date.unique())

    workers = min(len(horizons), max(1, (os.cpu_count() or 2) - 1))
    print(f"Parallel selective optimization: {workers} workers across horizons {horizons}")

    all_rows, all_trades = [], []
    if workers > 1 and os.name == "posix":
        ctx = get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for h, rows, trades in ex.map(_run_horizon_worker, horizons):
                all_rows.extend(rows)
                all_trades.extend(trades)
    else:
        for h in horizons:
            _, rows, trades = _run_horizon_worker(h)
            all_rows.extend(rows)
            all_trades.extend(trades)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    fold_df = pd.DataFrame(all_rows)
    if fold_df.empty:
        raise RuntimeError("No optimized folds produced trades")
    rows = []
    for h, g in trades.groupby("horizon"):
        rows.append({"horizon_days": int(h), **stat(g), "oos_folds": int(g.fold.nunique())})
    summary = pd.DataFrame(rows).sort_values(["sharpe", "expectancy"], ascending=False)
    Path("docs").mkdir(exist_ok=True)
    summary.to_csv("docs/selective_oos_summary.csv", index=False)
    fold_df.sort_values(["horizon_days", "fold"]).to_csv("docs/selective_fold_parameters.csv", index=False)
    trades.sort_values(["horizon", "signal_date", "symbol"]).to_csv("docs/selective_oos_trades.csv", index=False)
    manifest = {
        "engine": "nse_daily_swing_selective_v4_session_aligned",
        "tuning": "nested validation-to-OOS",
        "horizons": list(horizons),
        "candidate_count": len(candidate_grid()),
        "exact_simulation_finalists_per_fold": 5,
        "parallel_horizon_workers": workers,
        "selection_objective": "Sharpe + profit factor + win rate + expectancy + return - drawdown penalty",
        "survivorship_warning": True,
        "point_in_time_membership": False,
        "session_alignment": "global trading calendar; incomplete symbol paths rejected",
    }
    json.dump(manifest, open("docs/selective_research_manifest.json", "w", encoding="utf-8"), indent=2)
    print(summary.to_string(index=False))
    print("Candidates per fold:", len(candidate_grid()), "exact finalists:", 5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/swing.yaml")
    ap.add_argument("--horizon", type=int, choices=(5, 10, 20), action="append", help="Run only selected horizon(s); omit to run all three")
    args = ap.parse_args()
    run(yaml.safe_load(open(args.config, encoding="utf-8")), horizons=args.horizon)
