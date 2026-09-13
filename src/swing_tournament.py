from __future__ import annotations

import argparse, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

FAMILIES=("momentum","mean_reversion","closing_strength","volume","breakout","regime","relative_strength")


def dates(s):
    s=s.astype("string").str.strip(); out=pd.Series(pd.NaT,index=s.index,dtype="datetime64[ns]")
    m=s.str.fullmatch(r"\d{8}"); out.loc[m]=pd.to_datetime(s.loc[m],format="%Y%m%d",errors="coerce")
    m=out.isna(); out.loc[m]=pd.to_datetime(s.loc[m],errors="coerce",format="mixed"); return out.dt.normalize()


def _filename_symbol(fp: Path) -> str | None:
    """Infer a ticker only when the file appears to be a single-stock file."""
    stem=fp.stem.upper().strip()
    stem=re.sub(r"\.(NS|NSE)$", "", stem)
    stem=re.sub(r"^(STOCK_|STOCK-|NSE_|NSE-)", "", stem)
    stem=re.sub(r"(_DATA|_HISTORICAL|_HISTORY|_DAILY|_OHLCV)$", "", stem)
    generic={"ALL_STOCKS","ALL_STOCK","NSE_STOCKS","NSE_STOCK","HISTORICAL_DATA","DATA","STOCKS","NSE_DATA"}
    if not stem or stem in generic or len(stem)>40 or not re.fullmatch(r"[A-Z0-9&_-]+",stem):
        return None
    return stem


def load(cfg, root="data/nse_all_daily"):
    """Load NSE daily CSVs, supporting both consolidated files and one-file-per-symbol layouts."""
    frames=[]; scanned=0; skipped=0; missing_schema={}
    root_path=Path(root)
    for fp in sorted(root_path.rglob("*.csv")):
        scanned+=1
        try: x=pd.read_csv(fp)
        except Exception: skipped+=1; continue
        x.columns=[str(c).replace("\ufeff","").strip().lower().replace(" ","_") for c in x.columns]
        def pick(*a): return next((c for c in a if c in x.columns),None)
        mapping={"symbol":pick("symbol","ticker","code","scrip","security_code"),"date":pick("date","datetime","timestamp","price"),"open":pick("open"),"high":pick("high"),"low":pick("low"),"close":pick("close","last_close"),"adj_close":pick("adj_close","adjusted_close","adjclose"),"volume":pick("volume","vol","shares_traded")}
        if not all(mapping[k] for k in ("date","open","high","low","close")):
            skipped+=1; missing_schema[fp.name]=[k for k in ("date","open","high","low","close") if not mapping[k]]; continue
        if mapping["symbol"]:
            x=x.rename(columns={v:k for k,v in mapping.items() if v})
        else:
            sym=_filename_symbol(fp)
            if sym is None:
                skipped+=1; missing_schema[fp.name]=["symbol_or_single_stock_filename"]; continue
            x=x.rename(columns={v:k for k,v in mapping.items() if v})
            x["symbol"]=sym
        for c in ("open","high","low","close","adj_close","volume"):
            if c in x: x[c]=pd.to_numeric(x[c],errors="coerce")
        x["date"]=dates(x["date"]); x["symbol"]=x.symbol.astype(str).str.upper().str.replace(r"\.NS$","",regex=True).str.strip()
        x=x.dropna(subset=["date","symbol","open","high","low","close"])
        if not x.empty: frames.append(x[[c for c in ("date","symbol","open","high","low","close","adj_close","volume") if c in x]])
    if not frames:
        sample={k:v for k,v in list(missing_schema.items())[:10]}
        raise RuntimeError(f"No usable NSE daily OHLCV CSVs found; scanned={scanned}, skipped={skipped}, examples={sample}")
    d=pd.concat(frames,ignore_index=True).sort_values(["symbol","date"]).drop_duplicates(["symbol","date"],keep="last")
    d["turnover"]=d.close*d.get("volume",pd.Series(np.nan,index=d.index))
    g=d.groupby("symbol",group_keys=False)
    # Point-in-time eligibility: only information available up to each row is used.
    hist=g.cumcount()+1
    med=g.turnover.transform(lambda s:s.rolling(20,min_periods=10).median())
    ok=(hist>=int(cfg["data"]["min_history_days"]))&(d.close>=float(cfg["data"]["min_price"]))&(med>=float(cfg["data"]["min_turnover_inr"]))
    d=d.loc[ok].copy()
    max_symbols=cfg["data"].get("max_symbols")
    if max_symbols is not None:
        syms=sorted(d.symbol.unique())[:int(max_symbols)]
        d=d[d.symbol.isin(syms)].copy()
    print(f"Loaded NSE daily data: files_scanned={scanned}, files_used={len(frames)}, symbols={d.symbol.nunique()}, rows={len(d)}")
    return d


def features(d):
    d=d.sort_values(["symbol","date"]).copy(); g=d.groupby("symbol",group_keys=False); p="adj_close" if "adj_close" in d and d.adj_close.notna().mean()>.95 else "close"; d["fp"]=d[p]
    d["ret1"]=g.fp.pct_change()
    for n in (5,10,20,60): d[f"ret{n}"]=g.fp.pct_change(n)
    d["gap"]=d.open/g.close.shift(1)-1; d["range"]=((d.high-d.low)/d.close).replace([np.inf,-np.inf],np.nan); d["loc"]=(d.close-d.low)/(d.high-d.low).replace(0,np.nan); d["body"]=(d.close-d.open)/d.open.replace(0,np.nan)
    d["atr"]=g.range.transform(lambda s:s.rolling(14,min_periods=10).mean()); d["sma20"]=d.fp/g.fp.transform(lambda s:s.rolling(20,min_periods=15).mean())-1; d["sma50"]=d.fp/g.fp.transform(lambda s:s.rolling(50,min_periods=30).mean())-1
    if "volume" in d:
        vm=g.volume.transform(lambda s:s.rolling(20,min_periods=15).mean()); vs=g.volume.transform(lambda s:s.rolling(20,min_periods=15).std()); d["vz"]=(d.volume-vm)/vs.replace(0,np.nan)
    else: d["vz"]=0
    for c in ("ret1","ret5","ret20","gap","loc","vz","sma20"): d[f"r_{c}"]=d.groupby("date")[c].rank(pct=True)
    return d.replace([np.inf,-np.inf],np.nan)


def score(d,f):
    if f=="momentum": return .45*d.r_ret5+.35*d.r_ret20+.20*d.r_loc
    if f=="mean_reversion": return .55*(1-d.r_sma20)+.25*(1-d.r_ret5)+.20*d.r_loc
    if f=="closing_strength": return .65*d.r_loc+.35*d.r_ret1
    if f=="volume": return .45*d.r_vz+.35*d.r_ret5+.20*d.r_loc
    if f=="breakout": return .55*d.r_ret20+.25*d.r_ret5+.20*d.r_loc
    if f=="regime": return .65*d.r_ret20+.35*d.r_loc
    if f=="relative_strength": return .70*d.r_ret20+.30*d.r_ret5
    raise ValueError(f)


def simulate(picks,history,cfg,h):
    if picks.empty:return pd.DataFrame()
    sl=float(cfg["costs"]["slippage_bps_per_side"])/10000; tc=float(cfg["costs"]["transaction_cost_bps_per_side"])/10000; sm=float(cfg["execution"]["stop_atr_mult"]); tm=float(cfg["execution"]["target_atr_mult"]); maxg=float(cfg["portfolio"]["max_gross_exposure"])
    if not 0 < maxg <= 1: raise ValueError("max_gross_exposure must be in (0, 1]")
    groups={s:g.sort_values("date") for s,g in history.groupby("symbol")}; rows=[]
    for _,r in picks.iterrows():
        path=groups.get(r.symbol,pd.DataFrame()); path=path[path.date>r.date].head(h)
        if len(path)<h: continue
        entry=float(r.entry_open)*(1+sl); atr=min(max(float(r.atr) if pd.notna(r.atr) else .02,.005),.20); stop=entry*(1-sm*atr); target=entry*(1+tm*atr); exit_px=float(path.iloc[-1].close); reason="time"; exit_date=path.iloc[-1].date
        for _,bar in path.iterrows():
            if bar.low<=stop: exit_px,reason,exit_date=stop,"stop",bar.date; break
            if bar.high>=target: exit_px,reason,exit_date=target,"target",bar.date; break
        exit_px*=1-sl; ret=(exit_px/entry-1)-2*tc
        rows.append((r.date,r.symbol,entry,exit_px,ret,reason,r.score,h,exit_date))
    t=pd.DataFrame(rows,columns=["signal_date","symbol","entry","exit","return","reason","score","horizon","exit_date"])
    if t.empty:return t
    active=list(zip(t.signal_date,t.exit_date)); max_concurrent=1
    for dt in sorted(set(t.signal_date)|set(t.exit_date)):
        n=sum(a<=dt<=b for a,b in active); max_concurrent=max(max_concurrent,n)
    t["weight"]=maxg/max_concurrent; t["weighted_return"]=t["return"]*t.weight
    return t


def stat(t):
    if t.empty:return dict(trades=0,total_return=0,annualized_return=0,sharpe=0,max_drawdown=0,profit_factor=0,expectancy=0,win_rate=0)
    day=t.groupby("exit_date").weighted_return.sum().sort_index(); eq=(1+day).cumprod(); dd=eq/eq.cummax()-1; total=float(eq.iloc[-1]-1); days=max((day.index.max()-day.index.min()).days,1); years=max(days/365.25,1/365.25); sd=day.std(); wins=t.loc[t['return']>0,'return'].sum(); loss=abs(t.loc[t['return']<=0,'return'].sum())
    return dict(trades=len(t),total_return=total,annualized_return=float((1+total)**(1/years)-1) if 1+total>0 else -1,sharpe=float(day.mean()/sd*math.sqrt(252)) if sd>0 else 0,max_drawdown=float(dd.min()),profit_factor=float(wins/loss) if loss else float('inf'),expectancy=float(t['return'].mean()),win_rate=float((t['return']>0).mean()))


def run_fold(d,f,h,cfg):
    ds=sorted(d.date.unique()); r=cfg["research"]; trn=int(r["train_years"]*252); vn=int(r["validation_months"]*21); ten=int(r["test_months"]*21); step=int(r["step_months"]*21); emb=int(r["embargo_days"]); start=trn+vn+emb; folds=[]; params=[]
    work=d.copy(); work["entry_open"]=work.groupby("symbol").open.shift(-1); work["future_close"]=work.groupby("symbol").close.shift(-h)
    while start<len(ds):
        va=ds[start-vn-emb:start-emb]; te=ds[start:min(start+ten,len(ds))]
        if len(va)<20 or len(te)<20: break
        val=work[work.date.isin(va[:-1])].copy(); test=work[work.date.isin(te)].copy(); val=val.dropna(subset=["entry_open","future_close"]); test=test.dropna(subset=["entry_open"])
        best=(20,-1e99)
        for n in (10,20,30):
            v=val.copy(); v["score"]=score(v,f); v["future_ret"]=v.future_close/v.entry_open-1; q=v.sort_values(["date","score"],ascending=[True,False]).groupby("date").head(n); s=float(q.future_ret.mean()) if not q.empty else -1e99
            if s>best[1]:best=(n,s)
        n=best[0]; params.append(n); test["score"]=score(test,f); picks=test.sort_values(["date","score"],ascending=[True,False]).groupby("date").head(n); trades=simulate(picks,d,cfg,h)
        if not trades.empty: folds.append(trades)
        start+=step
    out=pd.concat(folds,ignore_index=True) if folds else pd.DataFrame(); stability=float(pd.Series(params).value_counts(normalize=True).iloc[0]) if params else 0; return out,stability


def main(cfg_path):
    cfg=yaml.safe_load(open(cfg_path,encoding="utf-8")); d=features(load(cfg)); rows=[]; all_t=[]
    for h in cfg["research"]["horizons_days"]:
        for f in FAMILIES:
            t,st=run_fold(d,f,int(h),cfg); s=stat(t); eligible=bool(s["trades"]>=cfg["research"]["min_trades_oos"] and st>=.34 and s["expectancy"]>0 and s["profit_factor"]>1 and s["sharpe"]>0 and s["total_return"]>0 and s["max_drawdown"]>-.50); rows.append({"horizon_days":h,"strategy":f,**s,"parameter_stability":st,"eligible":eligible});
            if not t.empty: all_t.append(t.assign(strategy=f))
    lb=pd.DataFrame(rows).sort_values(["eligible","sharpe","total_return"],ascending=[False,False,False]); Path("docs").mkdir(exist_ok=True); lb.to_csv("docs/swing_strategy_leaderboard.csv",index=False); (pd.concat(all_t,ignore_index=True) if all_t else pd.DataFrame()).to_csv("docs/swing_oos_trades.csv",index=False)
    manifest={"engine":"nse_daily_swing_v2","universe":cfg["universe"],"symbols_tested":int(d.symbol.nunique()),"date_start":str(d.date.min().date()),"date_end":str(d.date.max().date()),"horizons_days":cfg["research"]["horizons_days"],"execution_prices":"raw OHLC","feature_price":"adjusted close when available","survivorship_warning":True,"point_in_time_membership":False,"gross_exposure_control":"max concurrent open positions"}
    json.dump(manifest,open("docs/swing_research_manifest.json","w",encoding="utf-8"),indent=2); print(lb.to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/swing.yaml"); a=p.parse_args(); main(a.config)
