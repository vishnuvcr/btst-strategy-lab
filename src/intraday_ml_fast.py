from __future__ import annotations

import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["ret_1","ret_3","ret_6","ret_12","range_pct","body_pct","close_location","volume_z","vwap_gap","day_gap","rv_ratio","time_sin","time_cos"]


def load(cfg):
    files = sorted(glob.glob(cfg["data"]["intraday_glob"], recursive=True))[:int(cfg["data"]["max_symbols"])]
    out=[]
    for fp in files:
        try: x=pd.read_csv(fp)
        except Exception: continue
        x.columns=[str(c).strip().lower().replace(" ","_").replace("-","_") for c in x.columns]
        dt=next((c for c in ["datetime","timestamp","time","date"] if c in x.columns),None)
        if not dt: continue
        ren={}
        for dst,srcs in {"open":["o"],"high":["h"],"low":["l"],"close":["c","price","adj_close"],"volume":["vol","v"]}.items():
            if dst not in x.columns:
                for s in srcs:
                    if s in x.columns: ren[s]=dst; break
        x=x.rename(columns=ren)
        if not all(c in x.columns for c in ["open","high","low","close"]): continue
        x[dt]=pd.to_datetime(x[dt],errors="coerce",format="mixed")
        if getattr(x[dt].dt,"tz",None) is not None: x[dt]=x[dt].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        x=x.rename(columns={dt:"datetime"})
        for c in ["open","high","low","close","volume"]:
            if c in x: x[c]=pd.to_numeric(x[c],errors="coerce")
        x=x.dropna(subset=["datetime","open","high","low","close"])
        x["symbol"]=Path(fp).stem.upper().replace("-","_")
        x["date"]=x.datetime.dt.normalize()
        x=x[(x.datetime.dt.time>=pd.Timestamp("09:15").time())&(x.datetime.dt.time<=pd.Timestamp("15:30").time())]
        out.append(x[["datetime","date","symbol","open","high","low","close"]+(["volume"] if "volume" in x else [])])
    if not out: raise RuntimeError("No usable intraday files")
    x=pd.concat(out,ignore_index=True).sort_values(["symbol","datetime"])
    x=x.drop_duplicates(["symbol","datetime"],keep="last")
    x=x[(x.date>=pd.Timestamp(cfg["data"]["start_date"]))&(x.date<=pd.Timestamp(cfg["data"]["end_date"]))]
    counts=x.groupby(["symbol","date"])["datetime"].transform("size")
    x=x[counts>=int(cfg["data"]["min_bars_per_day"])].copy()
    x=x[x.close>=float(cfg["data"]["min_price"])].copy()
    return x.reset_index(drop=True)


def features(x):
    x=x.copy(); g=x.groupby("symbol",group_keys=False)
    for n in [1,3,6,12,24]: x[f"ret_{n}"]=g.close.pct_change(n)
    x["range_pct"]=(x.high-x.low)/x.close.replace(0,np.nan)
    x["body_pct"]=(x.close-x.open)/x.open.replace(0,np.nan)
    x["close_location"]=(x.close-x.low)/(x.high-x.low).replace(0,np.nan)
    if "volume" in x:
        m=g.volume.transform(lambda s:s.rolling(78,min_periods=20).mean()); sd=g.volume.transform(lambda s:s.rolling(78,min_periods=20).std())
        x["volume_z"]=(x.volume-m)/sd.replace(0,np.nan)
    else: x["volume_z"]=0.0
    vol=x.get("volume",pd.Series(1.0,index=x.index)).fillna(1.0)
    x["vwap"]=(x.close*vol).groupby([x.symbol,x.date]).cumsum()/vol.groupby([x.symbol,x.date]).cumsum().replace(0,np.nan)
    x["vwap_gap"]=x.close/x.vwap-1
    # Previous session close, not previous bar close.
    session_close=x.groupby(["symbol","date"]).close.transform("last")
    prev_close=x.groupby("symbol").apply(lambda s:s.groupby(x.loc[s.index,"date"]).last().close.shift(1)).reset_index(level=[0,1],drop=True)
    x["prev_session_close"]=prev_close.reindex(x.index).values
    x["day_gap"]=x.open/x.prev_session_close-1
    x["session_bar"]=x.groupby(["symbol","date"]).cumcount()
    x["minutes"]=(x.datetime.dt.hour*60+x.datetime.dt.minute).astype(float)
    x["time_sin"]=np.sin(2*np.pi*x.minutes/1440); x["time_cos"]=np.cos(2*np.pi*x.minutes/1440)
    r=g.ret_1; x["rv_ratio"]=r.transform(lambda s:s.rolling(24,min_periods=12).std())/r.transform(lambda s:s.rolling(240,min_periods=60).std()).replace(0,np.nan)
    for n in [3,6,12]:
        x[f"orb_hi_{n}"]=x.groupby(["symbol","date"]).high.transform(lambda s:s.iloc[:n].max())
        x[f"orb_lo_{n}"]=x.groupby(["symbol","date"]).low.transform(lambda s:s.iloc[:n].min())
    # Future close returns, computed without crossing symbol boundaries.
    for n in [3,6,12,24]: x[f"fret_{n}"]=g.close.shift(-n)/x.close-1
    x=x.replace([np.inf,-np.inf],np.nan)
    return x


def rule_score(z,method,orb):
    b=z.session_bar>=orb
    if method=="opening_range_breakout": return np.where(b&(z.close>z[f"orb_hi_{orb}"])&(z.volume_z>-1),(z.close/z[f"orb_hi_{orb}"]-1),np.nan)
    if method=="opening_range_reversal": return np.where(b&(z.close<z[f"orb_lo_{orb}"])&(z.vwap_gap<0),(z[f"orb_lo_{orb}"]/z.close-1),np.nan)
    if method=="vwap_reversion": return np.where(b&(z.vwap_gap<-0.003)&(z.close_location>0.5),-z.vwap_gap,np.nan)
    if method=="intraday_momentum": return np.where(b&(z.ret_6>0.004)&(z.ret_12>0.005),0.5*z.ret_6+0.5*z.ret_12,np.nan)
    if method=="volatility_expansion": return np.where(b&(z.rv_ratio>1.5)&(z.close_location>0.6),z.rv_ratio*z.close_location,np.nan)
    raise ValueError(method)


def fit_model(train):
    t=train.dropna(subset=FEATURES+["fret_3"]).copy()
    if len(t)>250000:
        # Deterministic capped training set: keeps runtime bounded while retaining both classes.
        t=t.sort_values("datetime"); pos=t[t.fret_3>0]; neg=t[t.fret_3<=0]
        n=min(125000,len(pos),len(neg)); t=pd.concat([pos.tail(n),neg.tail(n)])
    if len(t)<1000 or t.fret_3.gt(0).nunique()<2: return None
    y=(t.fret_3>0).astype(int)
    clf=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(max_iter=200,class_weight="balanced",random_state=42))])
    clf.fit(t[FEATURES],y); return clf


def simulate_signals(sig,bars,horizon,stop,target,slip,tc):
    # Signal rows are already capped. Loop only over actual trades, not every bar.
    idx={d:i for i,d in enumerate(bars.datetime)}; used_until=-1; rows=[]
    for r in sig.sort_values("datetime").itertuples():
        p=idx.get(r.datetime)
        if p is None or p+1>=len(bars) or p<=used_until: continue
        eidx=p+1; entry=float(bars.iloc[eidx].open)*(1+slip); st=entry*(1-stop); tg=entry*(1+target); end=min(eidx+horizon,len(bars)-1); reason="time"; ex=float(bars.iloc[end].close)
        for j in range(eidx,end+1):
            hi=float(bars.iloc[j].high); lo=float(bars.iloc[j].low)
            if lo<=st: ex=st; reason="stop"; end=j; break
            if hi>=tg: ex=tg; reason="target"; end=j; break
        ret=ex/entry-1-slip-tc
        rows.append((r.datetime,bars.iloc[end].date,r.symbol,ret,float(r.score),reason,horizon))
        used_until=end
    return rows


def run(path):
    c=yaml.safe_load(open(path,"r",encoding="utf-8")); x=features(load(c)); dates=np.array(sorted(x.date.unique()))
    tn,vn,te,step=[int(c["research"][k]) for k in ["train_days","validation_days","test_days","step_days"]]
    methods=c["methods"]; allrows=[]; max_per_ts=5; max_train_days=tn
    folds=0
    for s in range(tn,len(dates)-vn-te+1,step):
        folds+=1; tr_dates=dates[s-tn:s]; va_dates=dates[s:s+vn]; ts_dates=dates[s+vn:s+vn+te]
        train=x[x.date.isin(tr_dates)]; val=x[x.date.isin(va_dates)]; test=x[x.date.isin(ts_dates)]
        clf=fit_model(train)
        for method in methods:
            if method in {"gap_ml","ml_direction","ml_ranker"}:
                if clf is None: continue
                z=test.copy(); q=z[FEATURES].notna().all(axis=1); z["score"]=np.nan; z.loc[q,"score"]=clf.predict_proba(z.loc[q,FEATURES])[:,1]
                if method=="gap_ml": z=z[(z.day_gap.abs()>0.004)&(z.score>=0.60)]
                elif method=="ml_direction": z=z[z.score>=0.65]
                else:
                    z["rank"]=z.groupby("datetime").score.rank(pct=True); z=z[z["rank"]>=0.90]
                horizon=6
            else:
                best=(-1e9,3,6)
                for orb in c["research"]["opening_range_bars"]:
                    for h in c["research"]["horizons_bars"]:
                        q=rule_score(val,method,int(orb)); rr=val.loc[pd.notna(q),f"fret_{int(h)}"]
                        rr=rr.replace([np.inf,-np.inf],np.nan).dropna()
                        sc=float(rr.mean()) if len(rr)>=30 else -1e9
                        if sc>best[0]: best=(sc,int(orb),int(h))
                _,orb,horizon=best; z=test.copy(); z["score"]=rule_score(z,method,orb); z=z[z.score.notna()]
            if z.empty: continue
            # Keep only the strongest few candidates at each timestamp to control compute and model portfolio concentration.
            z=z.sort_values(["datetime","score"],ascending=[True,False]).groupby("datetime",sort=False).head(max_per_ts)
            for sym,zz in z.groupby("symbol",sort=False):
                bars=test[test.symbol==sym].sort_values("datetime").reset_index(drop=True)
                for row in simulate_signals(zz,bars,min(int(horizon),int(c["execution"]["max_holding_bars"])),float(c["execution"]["stop_pct"]),float(c["execution"]["target_pct"]),float(c["costs"]["slippage_bps_per_side"])/10000,float(c["costs"]["transaction_cost_bps_per_side"])/10000):
                    allrows.append(row+(method,))
        print(f"fold {folds}/{max(1,(len(dates)-vn-te-tn)//step+1)} complete; trades={len(allrows)}",flush=True)
    cols=["signal_datetime","exit_date","symbol","return","score","reason","horizon_bars","method"]
    t=pd.DataFrame(allrows,columns=cols); os.makedirs("docs",exist_ok=True)
    if t.empty: raise RuntimeError("No OOS trades produced")
    # Portfolio weighting: 95% gross, max 10 entries per timestamp across the complete method tournament.
    t=t.sort_values(["signal_datetime","score"],ascending=[True,False]); t["rank_ts"]=t.groupby("signal_datetime").cumcount(); t=t[t.rank_ts<int(c["portfolio"]["max_positions"])].copy(); t["weighted_return"]=t["return"]*float(c["portfolio"]["max_gross_exposure"])/t.groupby("signal_datetime")["return"].transform("size")
    t.to_csv("docs/intraday_ml_oos_trades.csv",index=False)
    rows=[]
    for method,g in t.groupby("method"):
        r=g.weighted_return.astype(float); eq=(1+r).cumprod(); gains=r[r>0].sum(); losses=-r[r<0].sum(); rows.append({"strategy":method,"trades":len(g),"win_rate":float((r>0).mean()),"profit_factor":float(gains/losses) if losses else np.inf,"total_return":float(eq.iloc[-1]-1),"sharpe":float(np.sqrt(252*78)*r.mean()/r.std()) if r.std()>0 else 0,"max_drawdown":float((eq/eq.cummax()-1).min()),"expectancy":float(r.mean())})
    lb=pd.DataFrame(rows).sort_values("profit_factor",ascending=False); lb.to_csv("docs/intraday_ml_leaderboard.csv",index=False)
    m=t.groupby(["method",t.exit_date.dt.to_period("M")]).weighted_return.sum().reset_index(name="monthly_return"); m.to_csv("docs/intraday_ml_monthly_returns.csv",index=False)
    print(lb.to_string(index=False)); print("folds",folds,"trades",len(t))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/intraday_ml.yaml"); run(p.parse_args().config)
