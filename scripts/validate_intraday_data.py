from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd


SESSION_START = 9 * 60 + 15
SESSION_END = 15 * 60 + 30
MIN_BARS = 70


def norm(c: str) -> str:
    return str(c).replace("\ufeff", "").strip().lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    files = sorted(glob.glob("data/intraday/**/*.csv", recursive=True))
    if not files:
        raise SystemExit("No intraday CSV files found")

    usable = 0
    bad_time = 0
    duplicate_rows = 0
    short_sessions = 0
    rows = 0
    session_days = 0
    symbols = set()

    for fp in files:
        try:
            x = pd.read_csv(fp)
        except Exception:
            continue
        x.columns = [norm(c) for c in x.columns]
        dt = next((c for c in ("datetime", "timestamp", "time", "date") if c in x.columns), None)
        if dt is None or not {"open", "high", "low", "close"}.issubset(x.columns):
            continue
        d = pd.to_datetime(x[dt], errors="coerce", format="mixed")
        if getattr(d.dt, "tz", None) is not None:
            d = d.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        x = x.loc[d.notna()].copy()
        x["datetime"] = d.loc[x.index]
        x = x.sort_values("datetime")
        rows += len(x)
        usable += 1
        symbols.add(Path(fp).stem.upper().replace("-", "_"))
        duplicate_rows += int(x.datetime.duplicated().sum())
        mins = x.datetime.dt.hour * 60 + x.datetime.dt.minute
        bad_time += int(((mins < SESSION_START) | (mins > SESSION_END)).sum())
        counts = x.groupby(x.datetime.dt.normalize()).size()
        session_days += len(counts)
        short_sessions += int((counts < MIN_BARS).sum())

    if usable == 0:
        raise SystemExit("No usable OHLC intraday files found")

    report = {
        "files_seen": len(files),
        "usable_files": usable,
        "symbols": len(symbols),
        "rows": rows,
        "session_days": session_days,
        "duplicate_timestamps": duplicate_rows,
        "out_of_session_rows": bad_time,
        "sessions_below_70_bars": short_sessions,
        "session_definition": "09:15-15:30 Asia/Kolkata",
        "minimum_session_bars_for_research": MIN_BARS,
        "survivorship_warning": "The configured intraday source is NIFTY 100 historical data and may not be point-in-time constituent membership.",
    }
    Path("docs").mkdir(exist_ok=True)
    Path("docs/intraday_data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    # Data-quality findings are diagnostic. The research loader removes duplicate
    # timestamps and restricts the actual research universe to regular-session bars.
    # Do not abort the entire ML run merely because the source contains diagnostic
    # rows outside the research session or duplicate records.
    if session_days == 0:
        raise SystemExit("No dated sessions found")


if __name__ == "__main__":
    main()
