from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

BASE_FEATURES=['ret_1','ret_3','ret_6','range_pct','body_pct','close_location','volume_z','vwap_gap','realized_vol','accel','session_frac','cluster_conf']
STRATEGIES=['trend','mean_reversion','breakout','vwap_reversion','orb_breakout']

def load(cfg):
    parts=[]
    for fp in sorted(glob.glob(cfg['data']['intraday_glob'],recursive=True))[:int(cfg['data']['max_symbols'])]:
        try:x=pd.read_csv(fp)
        except Exception:continue
        x.columns=[str(c).strip().lower().replace(' ','_').replace('-','_') for c in x.columns]
        dt=next((c for c in ['datetime','timestamp','time','date'] if c in x),None)
        if not dt or not all(c in x for c in ['open','high','low','close']):continue
        x[dt]=pd.to_datetime(x[dt],errors='coerce',format='mixed')
        if getattr(x[dt].dt,'tz',None) is not None:x[dt]=x[dt].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        x=x.rename(columns={dt:'datetime'})
        for c in ['open','high','low','close','volume']:
            if c in x:x[c]=pd.to_numeric(x[c],errors='coerce')
        x=x.dropna(subset=['datetime','open','high','low','close'])
        t=x.datetime.dt.time;x=x[(t>=pd.Timestamp('09:15').time())&(t<=pd.Timestamp('15:30').time())].copy()
        x['date']=x.datetime.dt.normalize();x['symbol']=Path(fp).stem.upper();n=x.groupby('date').datetime.transform('size');x=x[n>=int(cfg['data']['min_bars_per_day'])];x=x[x.close>=float(cfg['data']['min_price'])]
        keep=['datetime','date','symbol','open','high','low','close']+(['volume'] if 'volume' in x else []);parts.append(x[keep])
    if not parts:raise RuntimeError('No usable intraday files')
    return pd.concat(parts,ignore_index=True).sort_values(['symbol','datetime']).drop_duplicates(['symbol','datetime'])

def make_features(x):
    x=x.copy();g=x.groupby('symbol',group_keys=False);gs=x.groupby(['symbol','date'],group_keys=False)
    x['ret_1']=g.close.pct_change();x['ret_3']=g.close.pct_change(3);x['ret_6']=g.close.pct_change(6);x['range_pct']=(x.high-x.low)/x.open.replace(0,np.nan);x['body_pct']=(x.close-x.open)/x.open.replace(0,np.nan);x['close_location']=(x.close-x.low)/(x.high-x.low).replace(0,np.nan)
    if 'volume' in x:
        vm=g.volume.transform(lambda s:s.rolling(20,min_periods=10).mean());vs=g.volume.transform(lambda s:s.rolling(20,min_periods=10).std());x['volume_z']=(x.volume-vm)/vs.replace(0,np.nan);den=x.volume.fillna(0).groupby([x.symbol,x.date]).transform('sum').replace(0,np.nan);v=(x.close*x.volume.fillna(0)).groupby([x.symbol,x.date]).transform('sum')/den;x['vwap_gap']=v/x.close-1
    else:x['volume_z']=0.;x['vwap_gap']=0.
    x['realized_vol']=g['ret_1'].transform(lambda s:s.rolling(12,min_periods=6).std());x['accel']=g['ret_3'].transform(lambda s:s-s.shift(3));x['session_frac']=gs.cumcount()/gs.datetime.transform('size');x['atr_pct']=g['range_pct'].transform(lambda s:s.rolling(20,min_periods=10).mean());x['next_open']=gs.open.shift(-1)
    for h in [6,12,24]:x[f'fwd_{h}']=gs.close.shift(-h)/x.next_open-1
    for b in [6,12]:
        x[f'orb{b}_hi']=gs.high.transform(lambda s:s.shift(1).rolling(b,min_periods=b).max());x[f'orb{b}_lo']=gs.low.transform(lambda s:s.shift(1).rolling(b,min_periods=b).min())
    return x.replace([np.inf,-np.inf],np.nan)

def signals(x):
    cols=['datetime','date','symbol','cluster']+BASE_FEATURES+['close','high','low','next_open','atr_pct','fwd_6','fwd_12','fwd_24','orb6_hi','orb6_lo','orb12_hi','orb12_lo'];out=[]
    for name in STRATEGIES:
        z=x[cols].copy();z['strategy']=name;z['strategy_id']=STRATEGIES.index(name)
        if name=='trend':z['action']=np.where(z.ret_6>0,1,np.where(z.ret_6<0,-1,0))
        elif name=='mean_reversion':z['action']=np.where(z.ret_1<-0.004,1,np.where(z.ret_1>0.004,-1,0))
        elif name=='breakout':z['action']=np.where(z.close>z.orb6_hi,1,np.where(z.close<z.orb6_lo,-1,0))
        elif name=='vwap_reversion':z['action']=np.where(z.vwap_gap<-0.003,1,np.where(z.vwap_gap>0.003,-1,0))
        else:z['action']=np.where(z.close>z.orb12_hi,1,np.where(z.close<z.orb12_lo,-1,0))
        out.append(z[z.action!=0])
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def add_clusters(train,val,test,k,seed):
    cf=BASE_FEATURES[:-1];sc=RobustScaler().fit(train[cf]);km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(sc.transform(train[cf]));centers=km.cluster_centers_
    for z in [train,val,test]:
        z['cluster']=km.predict(sc.transform(z[cf]));a=sc.transform(z[cf]);z['cluster_conf']=1/(1+((a-centers[z.cluster.to_numpy()])**2).sum(axis=1))
    return train,val,test

def model_frame(z):
    q=z.copy();
    for i in range(len(STRATEGIES)):q[f'strat_{i}']=(q.strategy_id==i).astype(int)
    q['side']=q.action;q['ret_x_vol']=q.ret_6/(q.realized_vol.abs()+1e-6);q['cluster_x_side']=q.cluster*q.action
    cols=BASE_FEATURES+[f'strat_{i}' for i in range(len(STRATEGIES))]+['side','ret_x_vol','cluster_x_side'];return q,cols

def portfolio(test,h,cfg):
    cost=2*(float(cfg['costs']['slippage_bps_per_side'])+float(cfg['costs']['transaction_cost_bps_per_side']))/10000;maxpos=int(cfg['portfolio']['max_positions']);gross=float(cfg['portfolio']['max_gross_exposure']);stop=float(cfg['execution']['stop_atr']);target=float(cfg['execution']['target_atr'])
    p=test.sort_values(['datetime','score'],ascending=[True,False]).groupby('datetime',sort=False).head(maxpos).copy();p['weight']=gross/maxpos;atr=p.atr_pct.fillna(.01).clip(.005,.20);raw=p.action*p[f'fwd_{h}'];raw=np.where(raw<=-stop*atr,-stop*atr,raw);raw=np.where(raw>=target*atr,target*atr,raw);p['ret']=raw-cost;p['weighted']=p.weight*p.ret;return p

def run(cfg):
    x=make_features(load(cfg));dates=np.array(sorted(x.date.unique()));r=cfg['research'];tn,vn,te,step=map(int,[r['train_days'],r['validation_days'],r['test_days'],r['step_days']]);seed=int(r['random_state']);sample=int(r['sample_per_fold']);outs=[];folds=[]
    for s in range(tn,len(dates)-vn-te+1,step):
        tr=x[x.date.isin(dates[s-tn:s])].dropna(subset=BASE_FEATURES[:-1]+['fwd_24']).copy();va=x[x.date.isin(dates[s:s+vn])].dropna(subset=BASE_FEATURES[:-1]+['fwd_24']).copy();tef=x[x.date.isin(dates[s+vn:s+vn+te])].dropna(subset=BASE_FEATURES[:-1]).copy()
        if len(tr)>sample:tr=tr.sample(sample,random_state=seed)
        tr,va,tef=add_clusters(tr,va,tef,int(r['n_clusters'][1]),seed);tr,va,tef=signals(tr),signals(va),signals(tef);tr=tr.dropna(subset=['fwd_24']);va=va.dropna(subset=['fwd_24'])
        if tr.empty or va.empty or tef.empty:continue
        tr,cols=model_frame(tr);va,_=model_frame(va);tef,_=model_frame(tef)
        mdl=LGBMRegressor(n_estimators=int(r['max_estimators']),learning_rate=.05,num_leaves=31,max_depth=6,subsample=.8,colsample_bytree=.8,reg_lambda=2,random_state=seed,n_jobs=2,verbosity=-1);mdl.fit(tr[cols],tr.fwd_24);va['score']=mdl.predict(va[cols]);tef['score']=mdl.predict(tef[cols])
        q=float(va.score.quantile(.75));tv=va[va.score>=q];
        if len(tv)<int(r['min_signal_obs']):q=float(va.score.quantile(.60))
        selected=tef[tef.score>=q].copy();
        if selected.empty:continue
        for h in r['horizons_bars']:
            p=portfolio(selected,int(h),cfg);p['horizon']=int(h);p['fold']=len(folds)+1;outs.append(p)
        folds.append({'fold':len(folds)+1,'k':int(r['n_clusters'][1]),'validation_threshold':q,'validation_candidates':len(tv),'test_candidates':len(selected)})
    out=pd.concat(outs,ignore_index=True) if outs else pd.DataFrame();os.makedirs('docs',exist_ok=True);out.to_csv('docs/intraday_meta_v2_oos.csv',index=False);pd.DataFrame(folds).to_csv('docs/intraday_meta_v2_folds.csv',index=False)
    rows=[]
    for h,g in out.groupby('horizon') if not out.empty else []:
        dr=g.groupby('date').weighted.sum().sort_index();eq=(1+dr).cumprod();dd=eq/eq.cummax()-1;mo=dr.groupby(dr.index.to_period('M')).apply(lambda z:(1+z).prod()-1);years=max((dr.index[-1]-dr.index[0]).days/365.25,1/365.25);rows.append({'horizon':int(h),'trades':len(g),'total_return':eq.iloc[-1]-1,'cagr':eq.iloc[-1]**(1/years)-1,'sharpe':np.sqrt(252)*dr.mean()/dr.std() if dr.std()>0 else 0,'max_drawdown':dd.min(),'best_month':mo.max(),'worst_month':mo.min(),'positive_month_fraction':(mo>0).mean(),'months_ge_30pct':int((mo>=.30).sum())})
    pd.DataFrame(rows).to_csv('docs/intraday_meta_v2_metrics.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='config/intraday_meta_v2.yaml');a=ap.parse_args();run(yaml.safe_load(open(a.config)))