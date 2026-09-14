from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

STRATS = ["trend", "mean_reversion", "breakout", "vwap_reversion", "orb_breakout"]
BASE = ["ret_1","ret_3","ret_6","range_pct","body_pct","close_location","volume_z","vwap_gap","realized_vol","accel","session_frac","cluster_conf"]

def load(cfg):
    parts=[]
    for fp in sorted(glob.glob(cfg['data']['intraday_glob'], recursive=True))[:int(cfg['data']['max_symbols'])]:
        try: x=pd.read_csv(fp)
        except Exception: continue
        x.columns=[str(c).strip().lower().replace(' ','_').replace('-','_') for c in x.columns]
        dc=next((c for c in ['datetime','timestamp','time','date'] if c in x),None)
        if dc is None or not all(c in x for c in ['open','high','low','close']): continue
        x[dc]=pd.to_datetime(x[dc], errors='coerce', format='mixed')
        if getattr(x[dc].dt,'tz',None) is not None: x[dc]=x[dc].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        x=x.rename(columns={dc:'datetime'})
        for c in ['open','high','low','close','volume']:
            if c in x: x[c]=pd.to_numeric(x[c],errors='coerce')
        x=x.dropna(subset=['datetime','open','high','low','close'])
        t=x.datetime.dt.time; x=x[(t>=pd.Timestamp('09:15').time())&(t<=pd.Timestamp('15:30').time())].copy()
        x['date']=x.datetime.dt.normalize(); x['symbol']=Path(fp).stem.upper()
        n=x.groupby('date').datetime.transform('size'); x=x[n>=int(cfg['data']['min_bars_per_day'])]; x=x[x.close>=float(cfg['data']['min_price'])]
        keep=['datetime','date','symbol','open','high','low','close']+(['volume'] if 'volume' in x else []); parts.append(x[keep])
    if not parts: raise RuntimeError('No usable intraday files')
    return pd.concat(parts,ignore_index=True).sort_values(['symbol','datetime']).drop_duplicates(['symbol','datetime']).reset_index(drop=True)

def features(x,horizons):
    x=x.copy(); g=x.groupby('symbol',group_keys=False); gd=x.groupby(['symbol','date'],group_keys=False)
    x['ret_1']=g.close.pct_change(); x['ret_3']=g.close.pct_change(3); x['ret_6']=g.close.pct_change(6)
    x['range_pct']=(x.high-x.low)/x.open.replace(0,np.nan); x['body_pct']=(x.close-x.open)/x.open.replace(0,np.nan); x['close_location']=(x.close-x.low)/(x.high-x.low).replace(0,np.nan)
    if 'volume' in x:
        vm=g.volume.transform(lambda s:s.rolling(20,min_periods=10).mean()); vs=g.volume.transform(lambda s:s.rolling(20,min_periods=10).std()); x['volume_z']=(x.volume-vm)/vs.replace(0,np.nan)
        den=(x.volume.fillna(0)*1.0).groupby([x.symbol,x.date]).transform('sum').replace(0,np.nan); vw=(x.close*x.volume.fillna(0)).groupby([x.symbol,x.date]).transform('sum')/den; x['vwap_gap']=x.close/vw-1
    else: x['volume_z']=0.; x['vwap_gap']=0.
    x['realized_vol']=g.ret_1.transform(lambda s:s.rolling(12,min_periods=6).std()); x['accel']=g.ret_3.transform(lambda s:s-s.shift(3)); x['session_frac']=gd.cumcount()/gd.datetime.transform('size')
    x['atr_pct']=g.range_pct.transform(lambda s:s.rolling(20,min_periods=10).mean()); x['next_open']=gd.open.shift(-1)
    for h in horizons: x[f'fwd_{h}']=gd.close.shift(-h)/x.next_open-1
    x['orb6_hi']=gd.high.transform(lambda s:s.shift(1).rolling(6,min_periods=6).max()); x['orb6_lo']=gd.low.transform(lambda s:s.shift(1).rolling(6,min_periods=6).min())
    x['orb12_hi']=gd.high.transform(lambda s:s.shift(1).rolling(12,min_periods=12).max()); x['orb12_lo']=gd.low.transform(lambda s:s.shift(1).rolling(12,min_periods=12).min())
    return x.replace([np.inf,-np.inf],np.nan)

def signals(x):
    out=[]
    for sid,name in enumerate(STRATS):
        z=x.copy(); z['strategy']=name; z['strategy_id']=sid
        if name=='trend': a=np.where(z.ret_6>0,1,np.where(z.ret_6<0,-1,0))
        elif name=='mean_reversion': a=np.where(z.ret_1<-0.004,1,np.where(z.ret_1>0.004,-1,0))
        elif name=='breakout': a=np.where(z.close>z.orb6_hi,1,np.where(z.close<z.orb6_lo,-1,0))
        elif name=='vwap_reversion': a=np.where(z.vwap_gap<-0.003,1,np.where(z.vwap_gap>0.003,-1,0))
        else: a=np.where(z.close>z.orb12_hi,1,np.where(z.close<z.orb12_lo,-1,0))
        z['action']=a; out.append(z[z.action!=0])
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def cluster_fit(tr,va,te,k,seed):
    cf=BASE[:-1]; sc=RobustScaler().fit(tr[cf]); km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(sc.transform(tr[cf])); centers=km.cluster_centers_
    for z in (tr,va,te):
        a=sc.transform(z[cf]); z['cluster']=km.predict(a); d=((a-centers[z.cluster.to_numpy()])**2).sum(1); z['cluster_conf']=1/(1+d)
    return tr,va,te

def frame(z):
    q=z.copy()
    for i in range(len(STRATS)): q[f'strat_{i}']=(q.strategy_id==i).astype(int)
    q['side']=q.action; q['ret_x_vol']=q.ret_6/(q.realized_vol.abs()+1e-6); q['cluster_x_side']=q.cluster*q.action
    cols=BASE+[f'strat_{i}' for i in range(len(STRATS))]+['side','ret_x_vol','cluster_x_side']; return q,cols

def simulate(selected,h,cfg):
    if selected.empty:return pd.DataFrame()
    cost=2*(float(cfg['costs']['slippage_bps_per_side'])+float(cfg['costs']['transaction_cost_bps_per_side']))/10000
    maxpos=int(cfg['portfolio']['max_positions']); gross=float(cfg['portfolio']['max_gross_exposure']); stop=float(cfg['execution']['stop_atr']); target=float(cfg['execution']['target_atr'])
    p=selected.sort_values(['datetime','score'],ascending=[True,False]).groupby('datetime',sort=False).head(maxpos).copy()
    p['weight']=gross/maxpos
    a=p.atr_pct.fillna(.01).clip(.005,.20)
    raw=p.action*p[f'fwd_{h}']; raw=np.clip(raw,-stop*a,target*a); p['ret']=raw-cost; p['weighted']=p.weight*p.ret
    return p

def metrics(p):
    rows=[]
    for h,g in p.groupby('horizon'):
        daily=g.groupby('date').weighted.sum().sort_index(); eq=(1+daily).cumprod(); dd=eq/eq.cummax()-1
        months=daily.groupby(daily.index.to_period('M')).apply(lambda z:(1+z).prod()-1)
        years=max((daily.index[-1]-daily.index[0]).days/365.25,1/365.25)
        rows.append({'horizon':int(h),'trades':len(g),'total_return':eq.iloc[-1]-1,'cagr':eq.iloc[-1]**(1/years)-1,'sharpe':np.sqrt(252)*daily.mean()/daily.std() if daily.std()>0 else 0,'max_drawdown':dd.min(),'best_month':months.max(),'worst_month':months.min(),'positive_month_fraction':(months>0).mean(),'months_ge_30pct':int((months>=.30).sum())})
    return pd.DataFrame(rows)

def run(cfg):
    hs=[int(v) for v in cfg['research']['horizons_bars']]; x=features(load(cfg),hs); dates=np.array(sorted(x.date.unique())); r=cfg['research']; tn,vn,te,step=[int(r[k]) for k in ['train_days','validation_days','test_days','step_days']]; seed=int(r['random_state']); sample=int(r['sample_per_fold']); k=int(r['n_clusters'][1]); allp=[]; folds=[]
    for s in range(tn,len(dates)-vn-te+1,step):
        tr=x[x.date.isin(dates[s-tn:s])].dropna(subset=BASE[:-1]+[f'fwd_{max(hs)}']).copy(); va=x[x.date.isin(dates[s:s+vn])].dropna(subset=BASE[:-1]+[f'fwd_{max(hs)}']).copy(); tef=x[x.date.isin(dates[s+vn:s+vn+te])].dropna(subset=BASE[:-1]).copy()
        if len(tr)>sample: tr=tr.sample(sample,random_state=seed)
        tr,va,tef=cluster_fit(tr,va,tef,k,seed); tr,va,tef=signals(tr),signals(va),signals(tef)
        tr=tr.dropna(subset=[f'fwd_{max(hs)}']); va=va.dropna(subset=[f'fwd_{max(hs)}'])
        if tr.empty or va.empty or tef.empty: continue
        tr,cols=frame(tr); va,_=frame(va); tef,_=frame(tef)
        model=LGBMRegressor(n_estimators=int(r['max_estimators']),learning_rate=.05,num_leaves=31,max_depth=6,subsample=.8,colsample_bytree=.8,reg_lambda=2,random_state=seed,n_jobs=2,verbosity=-1)
        model.fit(tr[cols],tr[f'fwd_{max(hs)}']); va['score']=model.predict(va[cols]); tef['score']=model.predict(tef[cols])
        q=float(va.score.quantile(.75));
        if int((va.score>=q).sum())<int(r['min_signal_obs']): q=float(va.score.quantile(.60))
        selected=tef[tef.score>=q].copy()
        for h in hs:
            p=simulate(selected,h,cfg); p['horizon']=h; p['fold']=len(folds)+1; allp.append(p)
        folds.append({'fold':len(folds)+1,'k':k,'validation_threshold':q,'test_candidates':len(selected)})
    out=pd.concat(allp,ignore_index=True) if allp else pd.DataFrame(); os.makedirs('docs',exist_ok=True); out.to_csv('docs/intraday_meta_v3_oos.csv',index=False); pd.DataFrame(folds).to_csv('docs/intraday_meta_v3_folds.csv',index=False); m=metrics(out); m.to_csv('docs/intraday_meta_v3_metrics.csv',index=False); print(m.to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/intraday_meta_v2.yaml'); a=ap.parse_args(); run(yaml.safe_load(open(a.config)))
