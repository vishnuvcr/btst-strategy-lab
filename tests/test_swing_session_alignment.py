import numpy as np
import pandas as pd

from src.swing_tournament import add_forward_fields, simulate


def _cfg():
    return {
        'costs': {'slippage_bps_per_side': 0, 'transaction_cost_bps_per_side': 0},
        'execution': {'stop_atr_mult': 1.5, 'target_atr_mult': 3.0},
        'portfolio': {'max_gross_exposure': 0.95},
    }


def _rows(symbol, dates):
    out = []
    for i, d in enumerate(dates):
        px = 100.0 + i
        out.append({'date': d, 'symbol': symbol, 'open': px, 'high': px * 1.01, 'low': px * 0.99,
                    'close': px * 1.005, 'atr': 0.01, 'score': 1.0})
    return out


def test_forward_fields_reject_symbol_gaps():
    dates = pd.date_range('2025-01-01', periods=6, freq='B')
    history = pd.DataFrame(_rows('AAA', dates.delete(2)))
    history = pd.concat([history, pd.DataFrame(_rows('BBB', dates))], ignore_index=True)
    out = add_forward_fields(history, 3)
    a = out[out.symbol == 'AAA'].set_index('date')
    assert a.loc[dates[0], 'entry_open'] == 101.0
    assert pd.isna(a.loc[dates[0], 'future_close'])


def test_simulate_rejects_incomplete_symbol_path():
    dates = pd.date_range('2025-01-01', periods=6, freq='B')
    history = pd.DataFrame(_rows('AAA', dates.delete(2)))
    history = pd.concat([history, pd.DataFrame(_rows('BBB', dates))], ignore_index=True)
    picks = pd.DataFrame([{'date': dates[0], 'symbol': 'AAA', 'entry_open': 101.0,
                           'atr': 0.01, 'score': 1.0}])
    trades = simulate(picks, history, _cfg(), 3)
    assert trades.empty


def test_simulate_respects_max_gross_with_overlapping_positions():
    dates = pd.date_range('2025-01-01', periods=8, freq='B')
    history = pd.DataFrame(_rows('AAA', dates))
    picks = pd.DataFrame([
        {'date': dates[0], 'symbol': 'AAA', 'entry_open': 101.0, 'atr': 0.01, 'score': 1.0},
        {'date': dates[1], 'symbol': 'AAA', 'entry_open': 102.0, 'atr': 0.01, 'score': 0.9},
    ])
    trades = simulate(picks, history, _cfg(), 3)
    assert len(trades) == 2
    assert np.isclose(trades['weight'].sum(), 0.95)
    assert trades['exit_date'].min() >= trades['signal_date'].min()
