import numpy as np
import pandas as pd

from src.forensics import equal_weight_buy_and_hold, random_entry_benchmark, trade_return_sanity


def _universe():
    dates = pd.date_range('2025-01-01', periods=12, freq='B')
    rows = []
    for symbol, base in [('AAA', 100.0), ('BBB', 200.0), ('CCC', 300.0), ('DDD', 400.0)]:
        for i, d in enumerate(dates):
            close = base * (1 + 0.002 * i)
            rows.append({
                'date': d, 'symbol': symbol, 'close': close,
                'next_open': close, 'next_close': close * 1.01,
            })
    return pd.DataFrame(rows)


def test_trade_return_sanity_detects_exposure_and_extremes():
    dates = pd.date_range('2025-01-01', periods=2, freq='B')
    trades = pd.DataFrame({
        'signal_date': dates,
        'return': [0.01, -0.02],
        'weight': [0.475, 0.475],
        'weighted_return': [0.00475, -0.0095],
    })
    d = trade_return_sanity(trades)
    assert d['finite_returns']
    assert np.isclose(d['max_gross_weight'], 0.475)
    assert np.isclose(d['min_daily_return'], -0.0095)


def test_buy_and_hold_benchmark_is_positive_on_positive_universe():
    d = equal_weight_buy_and_hold(_universe())
    assert d['days'] > 0
    assert d['total_return'] > 0


def test_random_entry_benchmark_is_reproducible():
    x = _universe()
    a = random_entry_benchmark(x, top_n=2, repeats=20, seed=123)
    b = random_entry_benchmark(x, top_n=2, repeats=20, seed=123)
    assert a == b
    assert a['repeats'] == 20
    assert a['p05_total_return'] <= a['median_total_return'] <= a['p95_total_return']
