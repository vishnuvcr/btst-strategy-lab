from __future__ import annotations

import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

FEATURES = [
    "gap", "day_return", "range_pct", "realized_vol", "volume_z",
    "close_location", "vwap_gap", "ret_1", "ret_5", "ret_20",
]


def load_daily(cfg):
    files = sorted(glob.glob(cfg["data"]["intraday_glob"], recursive=True))[:int(cfg["data"]["max_symbols"])]
    parts = []
    for fp in files:
        try: x = pd.read_csv(fp)
        except Exception: continue
        x.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in x.columns]
        dt = next((c for c in ["datetime", "timestamp", "time", "date"] if c in x.columns), None)
        if not dt: continue
        ren = {}
        for dst, srcs in {"open":["o"], "high":["h"], "low":["l"], "close":["c","price","adj_close"], "volume":["vol","v"]}.items():
            if dst not in x.columns:
                for s in srcs:
                    if s in x.columns: ren[s] = dst; break
        x = x.rename(columns=ren)
        if not all(c in x.columns for c in ["open","high","low","close"]): continue
        x[dt] = pd.to_datetime(x[dt], errors="coerce", format="mixed")
        if getattr(x[dt].dt, "tz", None) is not None:
            x[dt] = x[dt].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        x = x.rename(columns={dt:"datetime"})
        for c in ["open","high","low","close","volume"]:
            if c in x: x[c] = pd.to_numeric(x[c], errors="coerce")
        x = x.dropna(subset=["datetime","open","high","low","close"])
        x = x[(x.datetime.dt.time >= pd.Timestamp("09:15").time()) & (x.datetime.dt.time <= pd.Timestamp("15:30").time())]
        x["date"] = x.datetime.dt.normalize(); x["symbol"] = Path(fp).stem.upper().replace("-", "_")
        parts.append(x[["datetime","date","symbol","open","high","low","close"] + (["volume"] if "volume" in x else [])])
    if not parts: raise RuntimeError("No usable intraday files")
    x = pd.concat(parts, ignore_index=True).sort_values(["symbol","datetime"])
    x = x.drop_duplicates(["symbol","datetime"], keep="last")
    x = x[(x.date >= pd.Timestamp(cfg["data"]["start_date"])) & (x.date <= pd.Timestamp(cfg["data"]["end_date"]))]
    counts = x.groupby(["symbol","date"])["datetime"].transform("size")
    x = x[counts >= int(cfg["data"]["min_bars_per_day"])].copy()
    x = x[x.close >= float(cfg["data"]["min_price"])]
    return x


def make_daily(x):
    g = x.groupby("symbol", group_keys=False)
    first = x.groupby(["symbol","date"], sort=True).first(numeric_only=True)
    last = x.groupby(["symbol","date"], sort=True).last(numeric_only=True)
    daily = pd.DataFrame({
        "open": first.open, "high": x.groupby(["symbol","date"]).high.max(),
        "low": x.groupby(["symbol","date"]).low.min(), "close": last.close,
    }).reset_index()
    if "volume" in x:
        daily["volume"] = x.groupby(["symbol","date"]).volume.sum().values
    daily = daily.sort_values(["symbol","date"])
    dg = daily.groupby("symbol", group_keys=False)
    daily["prev_close"] = dg.close.shift(1)
    daily["gap"] = daily.open / daily.prev_close - 1
    daily["day_return"] = daily.close / daily.open - 1
    daily["range_pct"] = (daily.high - daily.low) / daily.open.replace(0, np.nan)
    intraday = x.copy(); intraday["bar_ret"] = intraday.groupby("symbol").close.pct_change()
    rv = intraday.groupby(["symbol","date"]).bar_ret.std().rename("realized_vol").reset_index()
    daily = daily.merge(rv, on=["symbol","date"], how="left")
    if "volume" in daily:
        m = dg.volume.transform(lambda s: s.rolling(20, min_periods=10).mean())
        sd = dg.volume.transform(lambda s: s.rolling(20, min_periods=10).std())
        daily["volume_z"] = (daily.volume - m) / sd.replace(0, np.nan)
    else: daily["volume_z"] = 0.0
    daily["close_location"] = (daily.close - daily.low) / (daily.high - daily.low).replace(0, np.nan)
    # Session VWAP from actual intraday bars; this uses only the completed session.
    if "volume" in x:
        v = x.volume.fillna(0); xv = (x.close * v).groupby([x.symbol,x.date]).sum(); vv = v.groupby([x.symbol,x.date]).sum()
        vwap = (xv / vv.replace(0,np.nan)).rename("vwap").reset_index(); daily = daily.merge(vwap, on=["symbol","date"], how="left")
        daily["vwap_gap"] = daily.close / daily.vwap - 1
    else: daily["vwap_gap"] = 0.0
    daily["ret_1"] = dg.close.pct_change(1); daily["ret_5"] = dg.close.pct_change(5); daily["ret_20"] = dg.close.pct_change(20)
    daily["next_open_return"] = dg.open.shift(-1) / daily.close - 1
    daily["next_day_return"] = dg.close.shift(-1) / daily.close - 1
    daily["next_day_range"] = (dg.high.shift(-1) - dg.low.shift(-1)) / daily.close
    return daily.replace([np.inf,-np.inf], np.nan)


def fit_clusters(train, k, seed):
    z = train.dropna(subset=FEATURES).copy()
    if len(z) > 100000: z = z.sample(100000, random_state=seed)
    scaler = RobustScaler().fit(z[FEATURES])
    xx = scaler.transform(z[FEATURES])
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    model.fit(xx)
    return scaler, model, z


def evaluate(train, val, test, cfg):
    rows, oos = [], []
    seed = int(cfg["research"]["random_state"])
    ks = [int(k) for k in cfg["research"]["n_clusters"]]
    # Select k only on validation using stability + separation, never test returns.
    best = None
    for k in ks:
        scaler, model, z = fit_clusters(train, k, seed)
        v = val.dropna(subset=FEATURES).copy()
        if len(v) < k * 20: continue
        labels = model.predict(scaler.transform(v[FEATURES])); v["cluster"] = labels
        sample = v.sample(min(30000, len(v)), random_state=seed)
        sil = silhouette_score(scaler.transform(sample[FEATURES]), sample.cluster) if sample.cluster.nunique() > 1 else -1
        # Economic selection happens only on validation: choose clusters with positive next-day return and enough observations.
        grp = v.groupby("cluster").next_day_return.agg(["count","mean","median",lambda s: float((s>0).mean())]).rename(columns={"<lambda_0>":"hit_rate"})
        eligible = grp[(grp["count"] >= int(cfg["research"]["min_cluster_obs"])) & (grp["mean"] > 0)]
        econ = float(eligible["mean"].mean()) if not eligible.empty else -1e9
        score = econ + 0.05 * sil
        rows.append({"k":k,"silhouette":sil,"validation_selected_cluster_mean":econ,"validation_positive_clusters":len(eligible)})
        if best is None or score > best[0]: best = (score,k)
    if best is None: raise RuntimeError("No valid clustering model")
    k = best[1]; scaler, model, _ = fit_clusters(train, k, seed)
    for name, frame in [("validation",val),("test",test)]:
        z = frame.dropna(subset=FEATURES).copy(); z["cluster"] = model.predict(scaler.transform(z[FEATURES])); z["anomaly_score"] = IsolationForest(n_estimators=100, contamination=0.01, random_state=seed).fit_predict(scaler.transform(z[FEATURES]))
        grp = z.groupby("cluster").next_day_return.agg(["count","mean","median"]).reset_index(); grp["hit_rate"] = z.groupby("cluster").next_day_return.apply(lambda s: float((s>0).mean())).values; grp["period"] = name; grp["k"] = k
        rows.extend(grp.rename(columns={"mean":"next_day_return_mean","median":"next_day_return_median"}).to_dict("records"))
        if name == "test":
            # Pattern strategy: trade only clusters proven positive on validation.
            vz = val.dropna(subset=FEATURES).copy(); vz["cluster"] = model.predict(scaler.transform(vz[FEATURES]))
            vg = vz.groupby("cluster").next_day_return.mean(); good = set(vg[vg > 0].index)
            z["pattern_selected"] = z.cluster.isin(good)
            oos.append(z[["date","symbol","cluster","anomaly_score","next_day_return","next_open_return","pattern_selected"]])
    return pd.DataFrame(rows), pd.concat(oos, ignore_index=True), k


def run(cfg):
    daily = make_daily(load_daily(cfg)); dates = np.array(sorted(daily.date.unique()))
    tn,vn,te,step = [int(cfg["research"][k]) for k in ["train_days","validation_days","test_days","step_days"]]
    summaries, trades = [], []; fold = 0
    for s in range(tn, len(dates)-vn-te+1, step):
        fold += 1; trd, vad, ted = dates[s-tn:s], dates[s:s+vn], dates[s+vn:s+vn+te]
        train=daily[daily.date.isin(trd)]; val=daily[daily.date.isin(vad)]; test=daily[daily.date.isin(ted)]
        summ, oos, k = evaluate(train,val,test,cfg); summ["fold"]=fold; summaries.append(summ); oos["fold"]=fold; oos["k"]=k; trades.append(oos)
        print(f"fold {fold} complete; k={k}; test rows={len(oos)}", flush=True)
    os.makedirs("docs", exist_ok=True)
    summary=pd.concat(summaries,ignore_index=True); summary.to_csv("docs/unsupervised_cluster_summary.csv",index=False)
    out=pd.concat(trades,ignore_index=True); out.to_csv("docs/unsupervised_pattern_oos.csv",index=False)
    selected=out[out.pattern_selected].copy();
    if selected.empty: metrics=pd.DataFrame([{"trades":0,"total_return":0.0,"win_rate":0.0,"mean_next_day_return":0.0}])
    else:
        r=selected.next_day_return.astype(float); metrics=pd.DataFrame([{"trades":len(r),"total_return":float((1+r).prod()-1),"win_rate":float((r>0).mean()),"mean_next_day_return":float(r.mean()),"median_next_day_return":float(r.median()),"best_day":float(r.max()),"worst_day":float(r.min()),"positive_month_fraction":float(selected.assign(month=pd.to_datetime(selected.date).dt.to_period('M')).groupby('month').next_day_return.sum().gt(0).mean())}])
    metrics.to_csv("docs/unsupervised_pattern_metrics.csv",index=False)
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config/unsupervised_pattern.yaml"); args=ap.parse_args(); run(yaml.safe_load(open(args.config,"r",encoding="utf-8")))
