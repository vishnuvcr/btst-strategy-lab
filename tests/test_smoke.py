import numpy as np
import pandas as pd

from src.btst_lab import add_features, add_cross_sectional_features, prepare_entries, select_top, strategy_scores
from src.core_btst import read_market_files, simulate_trades, metrics
from src.research_v2 import eligibility


def _sample_frame(periods=80):
    rows = []
    dates = pd.date_range('2024-01-01', periods=periods, freq='B')
    for symbol, base in [('AAA', 100.0), ('BBB', 150.0), ('CCC', 80.0)]:
        for i, d in enumerate(dates):
            close = base * (1 + 0.001 * i)
            rows.append({'date': d, 'symbol': symbol, 'open': close, 'high': close * 1.02, 'low': close * 0.98, 'close': close * 1.005, 'volume': 100000 + i * 100})
    return pd.DataFrame(rows)


def _cfg(stop_mode='PCT'):
    return {
        'costs': {'slippage_bps_per_side': 0, 'transaction_cost_bps_per_side': 0},
        'execution': {'stop_mode': stop_mode, 'stop_pct': 2, 'target_pct': 5, 'stop_atr_mult': 1.5, 'target_atr_mult': 2},
        'portfolio': {'max_gross_exposure': 0.95},
    }


def test_btst_label_and_next_session_alignment():
    df = prepare_entries(add_features(_sample_frame()))
    assert df['next_open'].notna().sum() > 0
    assert df['btst_return'].notna().sum() > 0
    assert pd.isna(df.sort_values(['symbol', 'date']).groupby('symbol').tail(1)['btst_return']).all()
    assert np.isfinite(df.loc[df['btst_return'].notna(), 'btst_return']).all()


def test_cross_sectional_strategy_produces_ranked_candidates():
    x = add_cross_sectional_features(prepare_entries(add_features(_sample_frame())))
    x['score'] = strategy_scores(x, 'momentum')
    picks = select_top(x, top_n=2, min_score=0.0)
    assert not picks.empty
    assert picks.groupby('date')['symbol'].nunique().max() <= 2


def test_all_rule_families_have_valid_scores():
    x = add_cross_sectional_features(prepare_entries(add_features(_sample_frame())))
    families = ['momentum', 'mean_reversion', 'closing_strength', 'volume', 'gap', 'regime', 'sector_relative']
    for family in families:
        score = strategy_scores(x, family)
        assert score.notna().any(), family
        assert np.isfinite(score.dropna()).all(), family


def test_numeric_coercion_and_simulation(tmp_path):
    path = tmp_path / 'ADANIENT_10yr_daily.csv'
    pd.DataFrame({
        'price': ['2015-01-01', '2015-01-02', '2015-01-03'],
        'close': ['100.0', '101.0', '102.0'],
        'high': ['102.0', '103.0', '104.0'],
        'low': ['99.0', '100.0', '101.0'],
        'open': ['100.5', '100.8', '101.5'],
        'volume': ['100000', '110000', '120000'],
    }).to_csv(path, index=False)
    out = read_market_files(str(tmp_path / '*.csv'))
    assert len(out) == 3
    assert pd.api.types.is_numeric_dtype(out['open'])
    assert pd.api.types.is_numeric_dtype(out['close'])
    assert pd.api.types.is_numeric_dtype(out['volume'])

    x = prepare_entries(add_features(out))
    x['score'] = 0.9
    picks = x.dropna(subset=['next_open', 'next_high', 'next_low', 'next_close']).copy()
    trades, _ = simulate_trades(picks, _cfg('ATR'))
    assert not trades.empty
    assert np.isfinite(trades['return']).all()
    assert trades['weighted_return'].abs().max() <= 0.95 + 1e-12


def test_kaggle_price_column_can_be_the_date_index(tmp_path):
    path = tmp_path / 'ADANIENT_10yr_daily.csv'
    pd.DataFrame({
        'price': ['2015-01-01', '2015-01-02'],
        'close': [100.0, 101.0],
        'high': [102.0, 103.0],
        'low': [99.0, 100.0],
        'open': [100.5, 100.8],
        'volume': [100000, 110000],
    }).to_csv(path, index=False)
    out = read_market_files(str(path))
    assert len(out) == 2
    assert out['date'].notna().all()
    assert out['open'].iloc[0] == 100.5
    assert out['close'].iloc[-1] == 101.0


def test_numeric_yyyymmdd_dates_are_not_parsed_as_epoch_nanoseconds(tmp_path):
    path = tmp_path / 'NUMERIC_DATE.csv'
    pd.DataFrame({
        'date': [20150105, 20150106, 20150107],
        'open': [100, 101, 102],
        'high': [102, 103, 104],
        'low': [99, 100, 101],
        'close': [101, 102, 103],
    }).to_csv(path, index=False)
    out = read_market_files(str(path))
    assert out['date'].dt.year.min() == 2015
    assert out['date'].dt.strftime('%Y%m%d').tolist() == ['20150105', '20150106', '20150107']


def test_portfolio_weighting_is_bounded_by_max_gross():
    dates = pd.date_range('2025-01-01', periods=3, freq='B')
    rows = []
    for d in dates:
        for symbol in ['AAA', 'BBB', 'CCC']:
            rows.append({'date': d, 'symbol': symbol, 'next_open': 100.0, 'next_high': 100.5, 'next_low': 100.0, 'next_close': 100.5, 'score': 0.9, 'atr_pct': 0.01})
    trades, total = simulate_trades(pd.DataFrame(rows), _cfg('PCT'))
    assert np.isclose(trades.groupby('signal_date')['weight'].sum().max(), 0.95)
    assert np.isclose(trades.groupby('signal_date')['weighted_return'].sum().max(), 0.95 * 0.005)
    assert np.isfinite(total)


def test_eligibility_rejects_negative_oos_evidence():
    dates = pd.date_range('2025-01-01', periods=40, freq='B')
    trades = pd.DataFrame({
        'signal_date': dates,
        'return': np.full(len(dates), -0.01),
        'weighted_return': np.full(len(dates), -0.0095),
        'weight': np.full(len(dates), 0.95),
    })
    result = metrics(trades, 'bad', 1_000_000)
    eligible, reasons = eligibility(result, trades, {'parameter_stability': 1.0}, 30)
    assert not eligible
    assert 'non_positive_expectancy' in reasons
    assert 'profit_factor<=1' in reasons
    assert 'non_positive_sharpe' in reasons
    assert 'non_positive_oos_return' in reasons


def test_eligibility_accepts_only_positive_evidence():
    dates = pd.date_range('2025-01-01', periods=40, freq='B')
    returns = np.where(np.arange(len(dates)) % 2 == 0, 0.005, 0.015)
    trades = pd.DataFrame({
        'signal_date': dates,
        'return': returns,
        'weighted_return': returns * 0.95,
        'weight': np.full(len(dates), 0.95),
    })
    result = metrics(trades, 'good', 1_000_000)
    eligible, reasons = eligibility(result, trades, {'parameter_stability': 1.0}, 30)
    assert eligible
    assert reasons == []
