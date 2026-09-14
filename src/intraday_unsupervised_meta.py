from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

FEATURES = ["ret_1","ret_3","ret_6","range_pct","body_pct","close_location","volume_z","vwap_gap","realized_vol","accel","session_frac"]


def load(cfg):
    parts=[]
    for fp in sorted(glob.glob(cfg['data']['intraday_glob'], recursive=True))[:int(cfg['data']['max_symbols'])]:
        try: x=pd.read_csv(fp)
        except Exception: continue
        x.columns=[str(c).strip().lower().replace(' ','_').replace('-','_') for c in x.columns]
        dt=next((c for c in ['datetime','timestamp','time','date'] if c in x),None)
        if not dt or not all(c in x for c in ['open','high','low','close']): continue
        x[dt]=pd.to_datetime(x[dt],errors='coerce',format='mixed')
        if getattr(x[dt].dt,'tz',None) is not None: x[dt]=x[dt].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        x=x.rename(columns={dt:'datetime'})
        for c in ['open','high','low','close','volume']:
            if c in x: x[c]=pd.to_numeric(x[c],errors='coerce')
        x=x.dropna(subset=['datetime','open','high','low','close'])
        t=x.datetime.dt.time
        x=x[(t>=pd.Timestamp('09:15').time())&(t<=pd.Timestamp('15:30').time())].copy()
        x['date']=x.datetime.dt.normalize(); x['symbol']=Path(fp).stem.upper()
        counts=x.groupby('date').datetime.transform('size'); x=x[counts>=int(cfg['data']['min_bars_per_day'])]
        x=x[x.close>=float(cfg['data']['min_price'])]
        parts.append(x[['datetime','date','symbol','open','high','low','close']+(['volume'] if 'volume' in x else [])])
    if not parts: raise RuntimeError('No usable intraday files')
    return pd.concat(parts,ignore_index=True).sort_values(['symbol','datetime']).drop_duplicates(['symbol','datetime'])


def features(x):
    x=x.copy(); g=x.groupby('symbol',group_keys=False); gs=x.groupby(['symbol','date'],group_keys=False)
    x['ret_1']=g.close.pct_change(); x['ret_3']=g.close.pct_change(3); x['ret_6']=g.close.pct_change(6)
    x['range_pct']=(x.high-x.low)/x.open.replace(0,np.nan); x['body_pct']=(x.close-x.open)/x.open.replace(0,np.nan)
    x['close_location']=(x.close-x.low)/(x.high-x.low).replace(0,np.nan)
    if 'volume' in x:
        vm=g.volume.transform(lambda s:s.rolling(20,min_periods=10).mean()); vs=g.volume.transform(lambda s:s.rolling(20,min_periods=10).std()); x['volume_z']=(x.volume-vm)/vs.replace(0,np.nan)
        vden=x.volume.fillna(0).groupby([x.symbol,x.date]).transform('sum').replace(0,np.nan)
        x['vwap_gap']=((x.close*x.volume.fillna(0)).groupby([x.symbol,x.date]).transform('sum')/vden)/x.close-1
    else: x['volume_z']=0.; x['vwap_gap']=0.
    x['realized_vol']=g['ret_1'].transform(lambda s:s.rolling(12,min_periods=6).std())
    x['accel']=g['ret_3'].transform(lambda s:s-s.shift(3))
    x['session_frac']=gs.cumcount()/gs.datetime.transform('size')
    # Keep forward returns inside the same trading session; no overnight leakage.
    for h in [3,6,12,24]: x[f'fwd_{h}']=gs.close.shift(-h)/x.close-1
    x['fwd_6_label']=(x.fwd_6>0).astype(float)
    return x.replace([np.inf,-np.inf],np.nan)


def simulate(day, h, cost, max_pos, gross):
    picks=day[day.action!=0].copy()
    if picks.empty:return None
    picks['strength']=picks.confidence
    picks=picks.sort_values('strength',ascending=False).head(max_pos)
    w=gross/len(picks); picks['ret']=picks.action*picks[f'fwd_{h}']-cost; picks['weighted']=w*picks.ret
    return picks


def run(cfg):
    x=features(load(cfg)); dates=np.array(sorted(x.date.unique())); r=cfg['research']
    tn,vn,te,step=map(int,[r['train_days'],r['validation_days'],r['test_days'],r['step_days']]); seed=int(r['random_state'])
    cost=2*(float(cfg['costs']['slippage_bps_per_side'])+float(cfg['costs']['transaction_cost_bps_per_side']))/10000
    max_pos=int(cfg['portfolio']['max_positions']); gross=float(cfg['portfolio']['max_gross_exposure']); sample=int(r['sample_per_fold'])
    outputs=[]; selections=[]
    for s in range(tn,len(dates)-vn-te+1,step):
        train=x[x.date.isin(dates[s-tn:s])].dropna(subset=FEATURES+['fwd_6']).copy()
        val=x[x.date.isin(dates[s:s+vn])].dropna(subset=FEATURES+['fwd_6']).copy()
        test=x[x.date.isin(dates[s+vn:s+vn+te])].dropna(subset=FEATURES).copy()
        if len(train)>sample: train=train.sample(sample,random_state=seed)
        scaler=RobustScaler().fit(train[FEATURES]); km=KMeans(n_clusters=int(r['n_clusters'][1]),n_init=10,random_state=seed).fit(scaler.transform(train[FEATURES]))
        val['cluster']=km.predict(scaler.transform(val[FEATURES])); test['cluster']=km.predict(scaler.transform(test[FEATURES]))
        cg=val.groupby('cluster').fwd_6.agg(['count','mean','std']); eligible=cg[cg['count']>=int(r['min_pattern_obs'])]
        if eligible.empty: continue
        longc=int(eligible['mean'].idxmax()); shortc=int(eligible['mean'].idxmin())
        selections.append({'fold':len(selections)+1,'k':int(r['n_clusters'][1]),'long_cluster':longc,'short_cluster':shortc,'validation_long_edge':float(eligible.loc[longc,'mean']),'validation_short_edge':float(eligible.loc[shortc,'mean'])})
        centers=km.cluster_centers_
        for d,day in test.groupby('date',sort=True):
            day=day.copy(); day['action']=0; day.loc[day.cluster==longc,'action']=1; day.loc[day.cluster==shortc,'action']=-1
            z=scaler.transform(day[FEATURES]); dist=((z[:,None,:]-centers[None,:,:])**2).sum(axis=2)
            day['confidence']=1/(1+dist[np.arange(len(day)),day.cluster.to_numpy()])
            for h in [3,6,12,24]:
                q=simulate(day,h,cost,max_pos,gross)
                if q is not None:
                    q=q[['date','symbol','cluster','action','confidence',f'fwd_{h}','weight','ret','weighted']].copy(); q['horizon']=h; outputs.append(q)
    os.makedirs('docs',exist_ok=True); out=pd.concat(outputs,ignore_index=True) if outputs else pd.DataFrame()
    out.to_csv('docs/intraday_unsupervised_meta_oos.csv',index=False); pd.DataFrame(selections).to_csv('docs/intraday_unsupervised_meta_selection.csv',index=False)
    rows=[]
    for h,g in out.groupby('horizon') if not out.empty else []:
        dr=g.groupby('date').weighted.sum().sort_index(); eq=(1+dr).cumprod(); dd=eq/eq.cummax()-1; mo=dr.groupby(dr.index.to_period('M')).apply(lambda z:(1+z).prod()-1); years=max((dr.index[-1]-dr.index[0]).days/365.25,1/365.25)
        rows.append({'horizon':int(h),'trades':len(g),'total_return':eq.iloc[-1]-1,'cagr':eq.iloc[-1]**(1/years)-1,'sharpe':np.sqrt(252)*dr.mean()/dr.std() if dr.std()>0 else 0,'max_drawdown':dd.min(),'best_month':mo.max(),'worst_month':mo.min(),'positive_month_fraction':(mo>0).mean(),'months_ge_30pct':int((mo>=.30).sum())})
    pd.DataFrame(rows).to_csv('docs/intraday_unsupervised_meta_metrics.csv',index=False); print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/intraday_unsupervised_meta.yaml'); a=ap.parse_args(); run(yaml.safe_load(open(a.config)))