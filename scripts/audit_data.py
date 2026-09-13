from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd


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
            cols = {str(c).strip().lower().replace(" ", "_") for c in df.columns}
            if not {"open", "high", "low", "close"}.issubset(cols):
                problems.append(f"missing OHLC: {fp}")
                continue
            usable += 1
            rows += len(df)
            date_col = next((c for c in df.columns if str(c).strip().lower() in {"date", "datetime", "timestamp", "time"}), None)
            if date_col:
                d = pd.to_datetime(df[date_col], errors="coerce", utc=True)
                if d.notna().any():
                    lo, hi = d.min(), d.max()
                    min_date = lo if min_date is None else min(min_date, lo)
                    max_date = hi if max_date is None else max(max_date, hi)
            symbols.add(Path(fp).stem.upper())
        except Exception as e:
            problems.append(f"read error {fp}: {e}")
    return {"files": len(files), "usable_files": usable, "rows": rows, "symbols": len(symbols), "min_date": str(min_date), "max_date": str(max_date), "problems": problems[:100]}


if __name__ == "__main__":
    result = {"daily": audit("data/daily"), "market": audit("data/market")}
    Path("docs").mkdir(exist_ok=True)
    Path("docs/data_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
