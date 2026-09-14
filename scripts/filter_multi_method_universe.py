from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def norm(c: str) -> str:
    return str(c).replace("\ufeff", "").strip().lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    with open("config/multi_method.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]
    min_history = int(dcfg.get("min_history_days", 252))
    min_price = float(dcfg.get("min_price", 20))
    min_turnover = float(dcfg.get("min_turnover_inr", 0))
    files = sorted(glob.glob(dcfg["daily_glob"], recursive=True))
    kept = 0
    removed = 0
    rows_before = 0
    rows_after = 0
    for fp in files:
        try:
            x = pd.read_csv(fp)
        except Exception:
            removed += 1
            continue
        x.columns = [norm(c) for c in x.columns]
        if not {"date", "open", "high", "low", "close"}.issubset(x.columns):
            removed += 1
            continue
        rows_before += len(x)
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        x = x.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
        valid_ohlc = (
            (x["open"] > 0) & (x["high"] > 0) & (x["low"] > 0) & (x["close"] > 0)
            & (x["high"] >= x[["open", "close"]].max(axis=1))
            & (x["low"] <= x[["open", "close"]].min(axis=1))
        )
        x = x[valid_ohlc].copy()
        if len(x) < min_history:
            removed += 1
            continue
        x["turnover_20"] = np.nan
        if "volume" in x.columns:
            x["turnover_20"] = (x["close"] * x["volume"]).rolling(20, min_periods=20).median()
            eligible = (x["close"] >= min_price) & (x["turnover_20"] >= min_turnover)
        else:
            eligible = x["close"] >= min_price
        # Keep only dates on which the instrument satisfies the point-in-time filters.
        x = x[eligible].copy()
        if len(x) < max(60, min_history // 2):
            removed += 1
            continue
        x = x.drop(columns=["turnover_20"], errors="ignore")
        x.to_csv(fp, index=False)
        kept += 1
        rows_after += len(x)
    print(f"Universe filter: files={len(files)} kept={kept} removed={removed} rows={rows_before}->{rows_after}")


if __name__ == "__main__":
    main()
