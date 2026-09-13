import numpy as np
import pandas as pd

from src.btst_lab import add_features, add_cross_sectional_features, prepare_entries, select_top, strategy_scores


def test_btst_label_and_next_session_alignment():
    rows = []
    dates = pd.date_range('2024-01-01', periods=80, freq='B')
    for symbol, base in [('AAA', 100.0), ('BBB', 150.0), ('CCC', 80.0)]:
        for i, d in enumerate(dates):
            close = base * (1 + 0.001 * i)
            rows.append({'date': d, 'symbol': symbol, 'open': close, 'high': close * 1.02, 'low': close * 0.98, 'close': close * 1.005, 'volume': 100000 + i * 100})
    df = pd.DataFrame(rows)
    df = prepare_entries(add_features(df))
    assert df['next_open'].notna().sum() > 0
    assert df['btst_return'].notna().sum() > 0
    assert np.isfinite(df.loc[df['btst_return'].notna(), 'btst_return']).all()


def test_cross_sectional_strategy_produces_ranked_candidates():
    dates = pd.date_range('2024-01-01', periods=80, freq='B')
    rows = []
    for symbol, base in [('AAA', 100), ('BBB', 150), ('CCC', 80)]:
        for i, d in enumerate(dates):
            p = base * (1 + 0.002 * i)
            rows.append({'date': d, 'symbol': symbol, 'open': p, 'high': p*1.02, 'low': p*0.98, 'close': p*1.01, 'volume': 100000})
    x = add_cross_sectional_features(prepare_entries(add_features(pd.DataFrame(rows))))
    x['score'] = strategy_scores(x, 'momentum')
    picks = select_top(x, top_n=2, min_score=0.0)
    assert not picks.empty
    assert picks.groupby('date')['symbol'].nunique().max() <= 2
