from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from intraday_meta_v3 import load, features, signals, cluster_fit, frame, BASE, STRATS


def event_sim(selected, prices, h, cfg):
    if selected.empty:
        return pd.DataFrame()
    slip=float(cfg['costs']['slippage_bps_per_side'])/10000
    fee=float(cfg['costs']['transaction_cost_bps_per_side'])/10000
    maxpos=int(cfg['portfolio']['max_positions'])
    gross=float(cfg['portfolio']['max_gross_exposure'])
    stop_mult=float(cfg['execution']['stop_atr'])
    target_mult=float(cfg['execution']['target_atr'])
    sig=(selected.sort_values(['datetime','score'],ascending=[True,False])
         .drop_duplicates(['datetime','symbol']).copy())
    px=prices.sort_values(['symbol','datetime']).reset_index(drop=True)
    groups={s:g.reset_index(drop=True) for s,g in px.groupby('symbol',sort=False)}
    pos={s:{d:i for i,d in enumerate(g.datetime)} for s,g in groups.items()}
    entries={}
    for _,r in sig.iterrows():
        g=groups.get(r.symbol); mp=pos.get(r.symbol,{})
        i=mp.get(r.datetime)
        if g is None or i is None or i+1>=len(g): continue
        entries.setdefault(g.datetime.iloc[i+1],[]).append(r)
    timeline=sorted(px.datetime.unique())
    openp=[]; equity=1.0; trades=[]
    for dt in timeline:
        still=[]
        for p in openp:
            g=groups[p['symbol']]; i=pos[p['symbol']].get(dt)
            if i is None or i<p['entry_i']:
                still.append(p); continue
            b=g.iloc[i]; side=p['side']; ex=None; reason=None
            stop_hit=(b.low<=p['stop']) if side==1 else (b.high>=p['stop'])
            target_hit=(b.high>=p['target']) if side==1 else (b.low<=p['target'])
            if stop_hit:
                ex=p['stop']*(1-slip if side==1 else 1+slip); reason='stop'
            elif target_hit:
                ex=p['target']*(1-slip if side==1 else 1+slip); reason='target'
            elif i-p['entry_i']+1>=h:
                ex=b.close*(1-slip if side==1 else 1+slip); reason='time'
            if ex is None:
                still.append(p); continue
            ret=side*(ex/p['entry']-1)-fee
            pnl=p['notional']*ret
            equity+=pnl
            trades.append({**p,'exit_dt':dt,'exit':ex,'reason':reason,'return':ret,'pnl':pnl,'equity_after':equity})
        openp=still
        held={p['symbol'] for p in openp}
        for r in sorted(entries.get(dt,[]),key=lambda z:float(z.score),reverse=True):
            if len(openp)>=maxpos or r.symbol in held: continue
            g=groups.get(r.symbol); i=pos.get(r.symbol,{}).get(dt)
            if g is None or i is None: continue
            b=g.iloc[i]; side=int(r.action)
            entry=float(b.open)*(1+slip if side==1 else 1-slip)
            a=float(r.atr_pct) if pd.notna(r.atr_pct) else .01
            a=float(np.clip(a,.005,.20))
            stop=entry*(1-stop_mult*a) if side==1 else entry*(1+stop_mult*a)
            target=entry*(1+target_mult*a) if side==1 else entry*(1-target_mult*a)
            notional=equity*gross/maxpos
            openp.append({'symbol':r.symbol,'side':side,'entry_dt':dt,'entry_i':i,'entry':entry,
                          'stop':stop,'target':target,'notional':notional,'score':float(r.score),
                          'strategy':r.strategy,'horizon':h})
            held.add(r.symbol)
    if openp:
        dt=timeline[-1]
        for p in openp:
            g=groups[p['symbol']]; i=pos[p['symbol']].get(dt)
            if i is None: continue
            cp=float(g.iloc[i].close); side=p['side']; ex=cp*(1-slip if side==1 else 1+slip)
            ret=side*(ex/p['entry']-1)-fee; pnl=p['notional']*ret; equity+=pnl
            trades.append({**p,'exit_dt':dt,'exit':ex,'reason':'end_of_test','return':ret,'pnl':pnl,'equity_after':equity})
    return pd.DataFrame(trades)


def metrics(trades):
    rows=[]
    for h,g in trades.groupby('horizon'):
        pnl=g.groupby('exit_dt').pnl.sum().sort_index()
        eq=(1+pnl).cumprod()
        daily=pnl.copy(); dd=eq/eq.cummax()-1
        months=daily.groupby(daily.index.to_period('M')).apply(lambda z:(1+z).prod()-1)
        years=max((pnl.index[-1]-pnl.index[0]).days/365.25,1/365.25)
        rows.append({'horizon':int(h),'trades':len(g),'total_return':eq.iloc[-1]-1,
                     'cagr':eq.iloc[-1]**(1/years)-1,
                     'sharpe':np.sqrt(252)*daily.mean()/daily.std() if daily.std()>0 else 0,
                     'max_drawdown':dd.min(),'best_month':months.max(),'worst_month':months.min(),
                     'positive_month_fraction':(months>0).mean(),
                     'months_ge_30pct':int((months>=.30).sum())})
    return pd.DataFrame(rows)


def run(cfg):
    hs=[int(v) for v in cfg['research']['horizons_bars']]
    x=features(load(cfg),hs); dates=np.array(sorted(x.date.unique())); r=cfg['research']
    tn,vn,te,step=[int(r[k]) for k in ['train_days','validation_days','test_days','step_days']]
    seed=int(r['random_state']); sample=int(r['sample_per_fold']); k=int(r['n_clusters'][1])
    allp=[]; folds=[]
    for s in range(tn,len(dates)-vn-te+1,step):
        tr=x[x.date.isin(dates[s-tn:s])].dropna(subset=BASE[:-1]+[f'fwd_{max(hs)}']).copy()
        va=x[x.date.isin(dates[s:s+vn])].dropna(subset=BASE[:-1]+[f'fwd_{max(hs)}']).copy()
        tef=x[x.date.isin(dates[s+vn:s+vn+te])].dropna(subset=BASE[:-1]).copy()
        if len(tr)>sample: tr=tr.sample(sample,random_state=seed)
        tr,va,tef=cluster_fit(tr,va,tef,k,seed)
        tr,va,tes=signals(tr),signals(va),signals(tef)
        tr=tr.dropna(subset=[f'fwd_{max(hs)}']); va=va.dropna(subset=[f'fwd_{max(hs)}'])
        if tr.empty or va.empty or tes.empty: continue
        tr,cols=frame(tr); va,_=frame(va); tes,_=frame(tes)
        model=LGBMRegressor(n_estimators=int(r['max_estimators']),learning_rate=.05,num_leaves=31,max_depth=6,
                            subsample=.8,colsample_bytree=.8,reg_lambda=2,random_state=seed,n_jobs=2,verbosity=-1)
        model.fit(tr[cols],tr[f'fwd_{max(hs)}'])
        va['score']=model.predict(va[cols]); tes['score']=model.predict(tes[cols])
        q=float(va.score.quantile(.75))
        if int((va.score>=q).sum())<int(r['min_signal_obs']): q=float(va.score.quantile(.60))
        selected=tes[tes.score>=q].copy()
        for h in hs:
            p=event_sim(selected,tef,h,cfg)
            if not p.empty: p['fold']=len(folds)+1; allp.append(p)
        folds.append({'fold':len(folds)+1,'k':k,'validation_threshold':q,'test_candidates':len(selected)})
    out=pd.concat(allp,ignore_index=True) if allp else pd.DataFrame()
    os.makedirs('docs',exist_ok=True)
    out.to_csv('docs/intraday_meta_v3_oos.csv',index=False)
    pd.DataFrame(folds).to_csv('docs/intraday_meta_v3_folds.csv',index=False)
    m=metrics(out); m.to_csv('docs/intraday_meta_v3_metrics.csv',index=False); print(m.to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/intraday_meta_v2.yaml'); a=ap.parse_args(); run(yaml.safe_load(open(a.config)))
