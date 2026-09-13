from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core_btst import add_features, prepare_entries, read_market_files
from src.forensics import equal_weight_buy_and_hold, random_entry_benchmark, trade_return_sanity


if __name__ == '__main__':
    trades_path = ROOT / 'docs/oos_trades.csv'
    assert trades_path.is_file() and trades_path.stat().st_size > 0, 'Missing OOS trades'
    trades = pd.read_csv(trades_path)
    sanity = trade_return_sanity(trades)

    daily = read_market_files(str(ROOT / 'data/daily/**/*.csv'))
    daily = prepare_entries(add_features(daily))
    oos_dates = pd.to_datetime(trades['signal_date']).dt.normalize().unique()
    universe = daily[daily['date'].isin(oos_dates)].copy()

    top_n = 10
    random_benchmark = random_entry_benchmark(universe, top_n=top_n, repeats=100, seed=42)
    buy_hold = equal_weight_buy_and_hold(universe)

    by_strategy = {}
    for strategy, g in trades.groupby('strategy'):
        by_strategy[strategy] = trade_return_sanity(g)

    result = {
        'trade_sanity': sanity,
        'strategy_trade_sanity': by_strategy,
        'benchmarks': {
            'equal_weight_buy_and_hold': buy_hold,
            'random_entry': random_benchmark,
            'random_entry_definition': 'Randomly selects up to top_n eligible symbols per signal date and holds next session open-to-close; no strategy score is used.',
        },
        'warnings': [
            'Equal-weight buy-and-hold is a universe benchmark, not a NIFTY index replacement.',
            'Random-entry benchmark uses the same OOS signal dates and comparable top_n breadth but is not a substitute for historical constituent reconstruction.',
            'Daily OHLC cannot resolve stop/target ordering; strategy trade diagnostics therefore remain subject to the engine stop-first convention.',
        ],
    }
    docs = ROOT / 'docs'
    docs.mkdir(exist_ok=True)
    with open(docs / 'forensic_audit.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
