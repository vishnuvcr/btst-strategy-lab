from __future__ import annotations

import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

FEATURES = [
    "gap", "day_return", "range_pct", "realized_vol", "volume_z",
    "close_location", "vwap_gap", "ret_1", "ret_5", "ret_20",
]


def load_intraday(cfg):
    files = sorted(glob.glob(cfg["data"]["intraday_glob"], recursive=True))[:int(cfg["data"]["max_symbols"])]
    parts = []
    for fp in files:
        try:
            x = pd.read_csv(fp)
        except Exception:
            continue
        x.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in x.columns]
        dt = next((c for c in ("datetime", "timestamp", "time", "date") if c in x.columns), None)
        if dt is None:
            continue
        ren = {}
        for dst, srcs in {"open":["o"], "high":["h"], "low":["l"], "close":["c","price","adj_close"], "volume":["vol","v"]}.items():
            if dst not in x.columns:
                for s in srcs:
                    if s in x.columns:
                        ren[s] = dst
                        break
        x = x.rename(columns=ren)
        if not all(c in x.columns for c in ("open","high","low","close")):
            continue
        x[dt] = pd.to_datetime(x[dt], errors="coerce", format="mixed")
        if getattr(x[dt].dt, "tz", None) is not None:
            x[dt] = x[dt].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        x = x.rename(columns={dt: "datetime"})
        for c in ("open","high","low","close","volume"):
            if c in x:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        x = x.dropna(subset=["datetime","open","high","low","close"])
        t = x.datetime.dt.time
        x = x[(t >= pd.Timestamp("09:15").time()) & (t <= pd.Timestamp("15:30").time())]
        x["date"] = x.datetime.dt.normalize()
        x["symbol"] = Path(fp).stem.upper().replace("-", "_")
        cols = ["datetime","date","symbol","open","high","low","close"] + (["volume"] if "volume" in x else [])
        parts.append(x[cols])
    if not parts:
        raise RuntimeError("No usable intraday files")
    x = pd.concat(parts, ignore_index=True).sort_values(["symbol","datetime"])
    x = x.drop_duplicates(["symbol","datetime"], keep="last")
    x = x[(x.date >= pd.Timestamp(cfg["data"]["start_date"])) & (x.date <= pd.Timestamp(cfg["data"]["end_date"]))]
    counts = x.groupby(["symbol","date"])["datetime"].transform("size")
    x = x[counts >= int(cfg["data"]["min_bars_per_day"])].copy()
    x = x[x.close >= float(cfg["data"]["min_price"])]
    return x


def make_daily(x):
    first = x.groupby(["symbol","date"], sort=True).first(numeric_only=True)
    last = x.groupby(["symbol","date"], sort=True).last(numeric_only=True)
    daily = pd.DataFrame({
        "open": first.open,
        "high": x.groupby(["symbol","date"]).high.max(),
        "low": x.groupby(["symbol","date"]).low.min(),
        "close": last.close,
    }).reset_index()
    if "volume" in x:
        daily["volume"] = x.groupby(["symbol","date"]).volume.sum().to_numpy()
    daily = daily.sort_values(["symbol","date"])
    dg = daily.groupby("symbol", group_keys=False)
    daily["prev_close"] = dg.close.shift(1)
    daily["gap"] = daily.open / daily.prev_close - 1
    daily["day_return"] = daily.close / daily.open - 1
    daily["range_pct"] = (daily.high - daily.low) / daily.open.replace(0, np.nan)
    intraday = x.copy()
    intraday["bar_ret"] = intraday.groupby("symbol").close.pct_change()
    rv = intraday.groupby(["symbol","date"]).bar_ret.std().rename("realized_vol").reset_index()
    daily = daily.merge(rv, on=["symbol","date"], how="left")
    if "volume" in daily:
        vm = daily.groupby("symbol").volume.transform(lambda s: s.rolling(20, min_periods=10).mean())
        vs = daily.groupby("symbol").volume.transform(lambda s: s.rolling(20, min_periods=10).std())
        daily["volume_z"] = (daily.volume - vm) / vs.replace(0, np.nan)
    else:
        daily["volume_z"] = 0.0
    daily["close_location"] = (daily.close - daily.low) / (daily.high - daily.low).replace(0, np.nan)
    if "volume" in x:
        v = x.volume.fillna(0)
        vwap_num = (x.close * v).groupby([x.symbol,x.date]).sum()
        vwap_den = v.groupby([x.symbol,x.date]).sum()
        vwap = (vwap_num / vwap_den.replace(0, np.nan)).rename("vwap").reset_index()
        daily = daily.merge(vwap, on=["symbol","date"], how="left")
        daily["vwap_gap"] = daily.close / daily.vwap - 1
    else:
        daily["vwap_gap"] = 0.0
    daily["ret_1"] = dg.close.pct_change(1)
    daily["ret_5"] = dg.close.pct_change(5)
    daily["ret_20"] = dg.close.pct_change(20)
    daily["next_open_return"] = dg.open.shift(-1) / daily.close - 1
    daily["next_close_return"] = dg.close.shift(-1) / daily.close - 1
    return daily.replace([np.inf, -np.inf], np.nan)


def fit_model(train, k, seed, model_type):
    z = train.dropna(subset=FEATURES).copy()
    if len(z) > 100000:
        z = z.sample(100000, random_state=seed)
    scaler = RobustScaler().fit(z[FEATURES])
    X = scaler.transform(z[FEATURES])
    if model_type == "gmm":
        model = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, reg_covar=1e-5)
    else:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    model.fit(X)
    return scaler, model


def assign(model, scaler, frame):
    z = frame.dropna(subset=FEATURES).copy()
    X = scaler.transform(z[FEATURES])
    z["cluster"] = model.predict(X)
    # Confidence is derived only from the fitted unsupervised model, never from future returns.
    if isinstance(model, KMeans):
        dist = model.transform(X)
        z["cluster_confidence"] = -dist[np.arange(len(z)), z["cluster"].to_numpy()]
    else:
        probs = model.predict_proba(X)
        z["cluster_confidence"] = probs[np.arange(len(z)), z["cluster"].to_numpy()]
    return z


def choose_model(train, val, cfg):
    seed = int(cfg["research"]["random_state"])
    candidates = []
    for model_type in cfg["research"]["models"]:
        for k in cfg["research"]["n_clusters"]:
            k = int(k)
            scaler, model = fit_model(train, k, seed, model_type)
            v = assign(model, scaler, val)
            if len(v) < k * 20:
                continue
            grp = v.groupby("cluster").next_close_return.agg(["count","mean","median"])
            eligible = grp[grp["count"] >= int(cfg["research"]["min_cluster_obs"])]
            if eligible.empty:
                score = -1e9
            else:
                long_edge = float(eligible["mean"].max())
                short_edge = float(-eligible["mean"].min())
                score = max(long_edge, short_edge)
            candidates.append({"model_type":model_type,"k":k,"validation_best_edge":score})
    if not candidates:
        raise RuntimeError("No valid clustering candidates")
    return max(candidates, key=lambda r: r["validation_best_edge"]), pd.DataFrame(candidates)


def run(cfg):
    daily = make_daily(load_intraday(cfg))
    dates = np.array(sorted(daily.date.unique()))
    tn, vn, te, step = [int(cfg["research"][k]) for k in ("train_days","validation_days","test_days","step_days")]
    seed = int(cfg["research"]["random_state"])
    max_positions = int(cfg["portfolio"]["max_positions"])
    max_gross = float(cfg["portfolio"]["max_gross_exposure"])
    cost = 2.0 * (float(cfg["costs"]["slippage_bps_per_side"]) + float(cfg["costs"]["transaction_cost_bps_per_side"])) / 10000.0
    all_oos, model_rows = [], []
    for s in range(tn, len(dates) - vn - te + 1, step):
        fold = len(model_rows) + 1
        train_dates, val_dates, test_dates = dates[s-tn:s], dates[s:s+vn], dates[s+vn:s+vn+te]
        train = daily[daily.date.isin(train_dates)]
        val = daily[daily.date.isin(val_dates)]
        test = daily[daily.date.isin(test_dates)]
        best, cand = choose_model(train, val, cfg)
        scaler, model = fit_model(train, int(best["k"]), seed, best["model_type"])
        v = assign(model, scaler, val)
        vg = v.groupby("cluster").next_close_return.agg(["count","mean"])
        eligible = vg[vg["count"] >= int(cfg["research"]["min_cluster_obs"])]
        long_cluster = int(eligible["mean"].idxmax()) if not eligible.empty else None
        short_cluster = int(eligible["mean"].idxmin()) if not eligible.empty else None
        long_edge = float(eligible.loc[long_cluster, "mean"]) if long_cluster is not None else 0.0
        short_edge = float(eligible.loc[short_cluster, "mean"]) if short_cluster is not None else 0.0
        t = assign(model, scaler, test)
        t["side"] = 0
        t.loc[t.cluster == long_cluster, "side"] = 1
        t.loc[t.cluster == short_cluster, "side"] = -1
        t = t[t.side != 0].copy()
        if not t.empty:
            rows = []
            for d, day in t.groupby("date", sort=True):
                # IMPORTANT: never rank test observations by realized/future return.
                # Rank only by model-derived cluster confidence.
                longs = day[day.side == 1].nlargest(max_positions, "cluster_confidence")
                shorts = day[day.side == -1].nlargest(max_positions, "cluster_confidence")
                selected = pd.concat([longs, shorts], ignore_index=True)
                if len(selected) > max_positions:
                    selected = selected.nlargest(max_positions, "cluster_confidence")
                if selected.empty:
                    continue
                w = max_gross / len(selected)
                selected["weight"] = w
                selected["net_return"] = selected.side * selected.next_close_return - cost
                selected["weighted_return"] = selected.weight * selected.net_return
                rows.append(selected[["date","symbol","cluster","side","next_open_return","next_close_return","cluster_confidence","weight","net_return","weighted_return"]])
            if rows:
                all_oos.append(pd.concat(rows, ignore_index=True))
        model_rows.append({"fold":fold,"model_type":best["model_type"],"k":int(best["k"]),"validation_long_edge":long_edge,"validation_short_edge":short_edge,"long_cluster":long_cluster,"short_cluster":short_cluster})
        print(f"fold {fold} complete; model={best['model_type']}; k={best['k']}; OOS positions={sum(len(x) for x in all_oos)}", flush=True)

    os.makedirs("docs", exist_ok=True)
    pd.DataFrame(model_rows).to_csv("docs/unsupervised_model_selection.csv", index=False)
    out = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame(columns=["date","symbol","cluster","side","next_open_return","next_close_return","cluster_confidence","weight","net_return","weighted_return"])
    out.to_csv("docs/unsupervised_pattern_oos.csv", index=False)
    if out.empty:
        metrics = {"trades":0,"active_days":0,"total_return":0.0,"cagr":0.0,"win_rate":0.0,"mean_trade_return":0.0,"sharpe_daily":0.0,"max_drawdown":0.0,"best_month":0.0,"worst_month":0.0,"positive_month_fraction":0.0,"months_ge_30pct":0}
    else:
        daily_ret = out.groupby("date").weighted_return.sum().sort_index()
        equity = (1 + daily_ret).cumprod()
        dd = equity / equity.cummax() - 1
        monthly = daily_ret.groupby(daily_ret.index.to_period("M")).apply(lambda s: float((1+s).prod()-1))
        years = max((daily_ret.index[-1] - daily_ret.index[0]).days / 365.25, 1/365.25)
        metrics = {"trades":int(len(out),),"active_days":int(len(daily_ret)),"total_return":float(equity.iloc[-1]-1),"cagr":float(equity.iloc[-1] ** (1/years) - 1),"win_rate":float((out.net_return > 0).mean()),"mean_trade_return":float(out.net_return.mean()),"sharpe_daily":float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if daily_ret.std() > 0 else 0.0,"max_drawdown":float(dd.min()),"best_month":float(monthly.max()),"worst_month":float(monthly.min()),"positive_month_fraction":float((monthly > 0).mean()),"months_ge_30pct":int((monthly >= 0.30).sum())}
    pd.DataFrame([metrics]).to_csv("docs/unsupervised_pattern_metrics.csv", index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/unsupervised_pattern.yaml")
    args = ap.parse_args()
    run(yaml.safe_load(open(args.config, "r", encoding="utf-8")))
