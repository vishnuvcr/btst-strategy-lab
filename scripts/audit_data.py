from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd


def _parse_dates(raw: pd.Series) -> pd.Series:
    s = raw.astype('string').str.replace('\ufeff', '', regex=False).str.strip()
    out = pd.Series(pd.NaT, index=raw.index, dtype='datetime64[ns, UTC]')
    compact = s.str.fullmatch(r'\d{8}')
    if compact.any():
        out.loc[compact] = pd.to_datetime(s.loc[compact], format='%Y%m%d', errors='coerce', utc=True)
    remaining = out.isna()
    if remaining.any():
        out.loc[remaining] = pd.to_datetime(s.loc[remaining], errors='coerce', utc=True, format='mixed')
    return out


def audit(root: str) -> dict:
    files = sorted(glob.glob(f"{root}/**/*.csv", recursive=True))
    rows = 0
    usable = 0
    min_date = None
    max_date = None
    symbols = set()
    problems = []
    duplicate_rows = 0
    unsorted_rows = 0
    invalid_ohlc_rows = 0
    extreme_return_rows = 0
    extreme_return_symbols = set()
    max_abs_return = 0.0
    max_return_symbol = None
    zero_volume_rows = 0
    volume_rows = 0
    for fp in files:
        try:
            df = pd.read_csv(fp)
            cols = {str(c).strip().lower().replace(' ', '_') for c in df.columns}
            if not {'open', 'high', 'low', 'close'}.issubset(cols):
                problems.append(f'missing OHLC: {fp}')
                continue
            usable += 1
            rows += len(df)
            date_col = next((c for c in df.columns if str(c).strip().lower() in {'date', 'datetime', 'timestamp', 'time'}), None)
            if date_col is None and 'price' in {str(c).strip().lower() for c in df.columns}:
                date_col = next(c for c in df.columns if str(c).strip().lower() == 'price')
            d = _parse_dates(df[date_col]) if date_col else pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns, UTC]')
            if date_col and d.notna().any():
                lo, hi = d.min(), d.max()
                min_date = lo if min_date is None else min(min_date, lo)
                max_date = hi if max_date is None else max(max_date, hi)
                duplicate_rows += int(d.duplicated().sum())
                # The old audit measured returns in file order. A valid return audit
                # must sort by parsed date first; otherwise an unsorted CSV can hide
                # extreme jumps while the backtest (which sorts) still sees them.
                valid_dates = d.notna()
                if valid_dates.any():
                    order = d.loc[valid_dates]
                    unsorted_rows += int((order.diff().dropna() < pd.Timedelta(0)).sum())
            else:
                problems.append(f'invalid dates: {fp}')

            numeric = {}
            for c in ['open', 'high', 'low', 'close', 'volume']:
                actual = next((x for x in df.columns if str(x).strip().lower().replace(' ', '_') == c), None)
                if actual is not None:
                    numeric[c] = pd.to_numeric(df[actual], errors='coerce')
            o, h, l, c = (numeric.get(k) for k in ['open', 'high', 'low', 'close'])
            if all(v is not None for v in [o, h, l, c]):
                bad = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | (h < l) | (h < o) | (h < c) | (l > o) | (l > c)
                invalid_ohlc_rows += int(bad.fillna(False).sum())
                ret_frame = pd.DataFrame({'date': d, 'close': c}).dropna().sort_values('date')
                close_ret = ret_frame['close'].pct_change()
                extreme = close_ret.abs() > 0.50
                n_extreme = int(extreme.fillna(False).sum())
                extreme_return_rows += n_extreme
                if n_extreme:
                    extreme_return_symbols.add(Path(fp).stem.upper())
                if close_ret.notna().any():
                    local_idx = close_ret.abs().idxmax()
                    local_max = float(abs(close_ret.loc[local_idx]))
                    if local_max > max_abs_return:
                        max_abs_return = local_max
                        max_return_symbol = Path(fp).stem.upper()
            if 'volume' in numeric:
                v = numeric['volume']
                volume_rows += int(v.notna().sum())
                zero_volume_rows += int((v.fillna(0) <= 0).sum())
            symbols.add(Path(fp).stem.upper())
        except Exception as e:
            problems.append(f'read error {fp}: {e}')

    return {
        'files': len(files),
        'usable_files': usable,
        'rows': rows,
        'symbols': len(symbols),
        'min_date': str(min_date),
        'max_date': str(max_date),
        'quality_checks': {
            'duplicate_date_rows': duplicate_rows,
            'unsorted_date_rows': unsorted_rows,
            'invalid_ohlc_rows': invalid_ohlc_rows,
            'extreme_close_return_rows_gt_50pct': extreme_return_rows,
            'extreme_return_symbols': sorted(extreme_return_symbols),
            'max_abs_close_return': max_abs_return,
            'max_abs_close_return_symbol': max_return_symbol,
            'zero_or_negative_volume_rows': zero_volume_rows,
            'volume_rows': volume_rows,
            'extreme_return_threshold': 0.50,
        },
        'problems': problems[:100],
    }


if __name__ == '__main__':
    result = {'daily': audit('data/daily'), 'market': audit('data/market')}
    Path('docs').mkdir(exist_ok=True)
    Path('docs/data_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
