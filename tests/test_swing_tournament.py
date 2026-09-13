import numpy as np
import pandas as pd

from src.swing_tournament import features, load, simulate


def make_data():
    dates = pd.date_range('2020-01-01', periods=40, freq='B')
    rows=[]
    for sym,base in [('AAA',100.0),('BBB',200.0),('CCC',300.0)]:
        for i,d in enumerate(dates):
            c=base*(1+0.001*i)
            rows.append({'date':d,'symbol':sym,'open':c,'high':c*1.01,'low':c*.99,'close':c*1.002,'adj_close':c*1.002,'volume':100000})
    return pd.DataFrame(rows)


def test_features_are_cross_sectional_and_finite():
    d=features(make_data())
    assert d['symbol'].nunique()==3
    assert d['r_ret20'].notna().any()


def test_simulator_uses_full_history_for_multi_day_horizon():
    d=features(make_data())
    picks=d[d.date==d.date.iloc[20]].copy()
    picks['entry_open']=picks['open'].shift(0)
    picks['score']=1.0
    cfg={'costs':{'slippage_bps_per_side':0,'transaction_cost_bps_per_side':0},'execution':{'stop_atr_mult':1.5,'target_atr_mult':3.0},'portfolio':{'max_gross_exposure':0.95}}
    t=simulate(picks,d,cfg,5)
    assert not t.empty
    assert t['horizon'].eq(5).all()
    assert np.isfinite(t['return']).all()
    assert t['weight'].sum() <= 0.95 + 1e-9


def test_load_supports_symbol_from_filename(tmp_path):
    root=tmp_path/'nse'
    root.mkdir()
    pd.DataFrame({
        'Date':pd.date_range('2024-01-01',periods=12,freq='B'),
        'Open':range(100,112), 'High':range(101,113), 'Low':range(99,111),
        'Close':range(100,112), 'Adj Close':range(100,112), 'Volume':[200000]*12,
    }).to_csv(root/'RELIANCE.csv',index=False)
    cfg={'data':{'min_history_days':10,'min_price':20,'min_turnover_inr':1_000_000,'max_symbols':5000}}
    d=load(cfg,root)
    assert d['symbol'].nunique()==1
    assert d['symbol'].iloc[0]=='RELIANCE'
    assert len(d)>0
