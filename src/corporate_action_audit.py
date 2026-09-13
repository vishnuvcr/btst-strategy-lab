from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd


def _parse_dates(raw: pd.Series) -> pd.Series:
    s = raw.astype("string").str.replace("\ufeff", "", regex=False).str.strip()
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    compact = s.str.fullmatch(r"\d{8}")
    if compact.any():
        out.loc[compact] = pd.to_datetime(
            s.loc[compact], format="%Y%m%d", errors="coerce", utc=True
        )
    remaining = out.isna()
    if remaining.any():
        out.loc[remaining] = pd.to_datetime(
            s.loc[remaining], errors="coerce", utc=True, format="mixed"
        )
    return out


def _date_col(df: pd.DataFrame):
    names = {str(c).strip().lower(): c for c in df.columns}
    for key in ("date", "datetime", "timestamp", "time", "price"):
        if key in names:
            return names[key]
    return None


def audit_file(fp: str) -> dict:
    df = pd.read_csv(fp)
    lower = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    if not {"open", "high", "low", "close"}.issubset(lower):
        return {"file": fp, "usable": False, "reason": "missing OHLC"}
    dc = _date_col(df)
    if dc is None:
        return {"file": fp, "usable": False, "reason": "missing date"}
    d = _parse_dates(df[dc])
    close = pd.to_numeric(df[lower["close"]], errors="coerce")
    high = pd.to_numeric(df[lower["high"]], errors="coerce")
    low = pd.to_numeric(df[lower["low"]], errors="coerce")
    x = pd.DataFrame({"date": d, "close": close, "high": high, "low": low}).dropna()
    x = x.sort_values("date").drop_duplicates("date", keep="last")
    if x.empty:
        return {"file": fp, "usable": False, "reason": "no valid observations"}
    ret = x["close"].pct_change()
    abs_ret = ret.abs()
    jumps20 = abs_ret > 0.20
    jumps50 = abs_ret > 0.50
    valid_returns = abs_ret.dropna()
    max_idx = valid_returns.idxmax() if not valid_returns.empty else None
    return {
        "file": fp,
        "usable": True,
        "rows": int(len(x)),
        "date_start": str(x["date"].min()),
        "date_end": str(x["date"].max()),
        "jumps_gt_20pct": int(jumps20.fillna(False).sum()),
        "jumps_gt_50pct": int(jumps50.fillna(False).sum()),
        "max_abs_close_return": float(valid_returns.max()) if not valid_returns.empty else None,
        "max_jump_date": str(x.loc[max_idx, "date"]) if max_idx is not None else None,
        "has_adj_close_column": "adj_close" in lower,
    }


def audit(root: str) -> dict:
    results = [audit_file(fp) for fp in sorted(glob.glob(f"{root}/**/*.csv", recursive=True))]
    usable = [r for r in results if r.get("usable")]
    return {
        "files": len(results),
        "usable_files": len(usable),
        "files_with_adj_close": sum(bool(r.get("has_adj_close_column")) for r in usable),
        "files_with_gt_50pct_jump": sum(r.get("jumps_gt_50pct", 0) > 0 for r in usable),
        "results": results,
        "note": "Large returns are diagnostic only. They may reflect splits, bonuses, rights, symbol changes, bad ticks, or other data issues and require source-level verification before adjustment.",
    }


if __name__ == "__main__":
    out = audit("data/daily")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/corporate_action_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
