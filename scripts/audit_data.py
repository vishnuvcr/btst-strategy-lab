from __future__ import annotations

import glob
import json
import re
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
            if date_col:
                d = _parse_dates(df[date_col])
                if d.notna().any():
                    lo, hi = d.min(), d.max()
                    min_date = lo if min_date is None else min(min_date, lo)
                    max_date = hi if max_date is None else max(max_date, hi)
                else:
                    problems.append(f'invalid dates: {fp}')
            symbols.add(Path(fp).stem.upper())
        except Exception as e:
            problems.append(f'read error {fp}: {e}')
    return {'files': len(files), 'usable_files': usable, 'rows': rows, 'symbols': len(symbols), 'min_date': str(min_date), 'max_date': str(max_date), 'problems': problems[:100]}


if __name__ == '__main__':
    result = {'daily': audit('data/daily'), 'market': audit('data/market')}
    Path('docs').mkdir(exist_ok=True)
    Path('docs/data_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
