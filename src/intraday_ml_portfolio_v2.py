from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import pandas as pd
import yaml

import intraday_ml_fast as engine


def replay_portfolio(trades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if trades.empty:
        raise RuntimeError("No OOS trades to replay")
    t = trades.copy()
    t["signal_datetime"] = pd.to_datetime(t["signal_datetime"])
    t["exit_date"] = pd.to_datetime(t["exit_date"]).dt.normalize()
    t["return"] = pd.to_numeric(t["return"], errors="coerce")
    t["score"] = pd.to_numeric(t["score"], errors="coerce").fillna(-np.inf)
    t = t.dropna(subset=["signal_datetime", "exit_date", "return"])
    t = t.sort_values(["signal_datetime", "score"], ascending=[True, False])

    max_pos = int(cfg["portfolio"]["max_positions"])
    gross = float(cfg["portfolio"]["max_gross_exposure"])
    selected = []
    open_until: list[pd.Timestamp] = []
    open_symbols: dict[str, pd.Timestamp] = {}

    for ts, group in t.groupby("signal_datetime", sort=True):
        # Conservative daily-level replay: positions are retained through their
        # reported exit date because the fast artifact does not expose exit time.
        open_until = [d for d in open_until if d >= ts.normalize()]
        active = len(open_until)
        slots = max(0, max_pos - active)
        if slots == 0:
            continue
        for r in group.itertuples():
            sym = str(r.symbol)
            old = open_symbols.get(sym)
            if old is not None and old >= ts.normalize():
                continue
            selected.append(r)
            open_until.append(pd.Timestamp(r.exit_date).normalize())
            open_symbols[sym] = pd.Timestamp(r.exit_date).normalize()
            slots -= 1
            if slots <= 0:
                break

    if not selected:
        raise RuntimeError("Portfolio replay selected no trades")
    out = pd.DataFrame(selected, columns=t.columns)
    out = out.sort_values("signal_datetime").reset_index(drop=True)
    # Equal-weight each newly opened position. This is intentionally conservative:
    # no leverage and no reuse of capital while a position remains open.
    out["portfolio_weight"] = gross / max_pos
    out["weighted_return"] = out["return"] * out["portfolio_weight"]
    return out


def metrics(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods_per_year = 252 * 78
    for method, g in t.groupby("method"):
        r = g["weighted_return"].astype(float)
        eq = (1.0 + r).cumprod()
        gains = r[r > 0].sum()
        losses = -r[r < 0].sum()
        rows.append({
            "strategy": method,
            "trades": len(g),
            "win_rate": float((r > 0).mean()),
            "profit_factor": float(gains / losses) if losses > 0 else np.inf,
            "total_return": float(eq.iloc[-1] - 1),
            "sharpe": float(np.sqrt(periods_per_year) * r.mean() / r.std()) if r.std() > 0 else 0.0,
            "max_drawdown": float((eq / eq.cummax() - 1).min()),
            "expectancy": float(r.mean()),
        })
    return pd.DataFrame(rows).sort_values("profit_factor", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/intraday_ml.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    # Enforce the configured embargo by removing the last embargo_days from the
    # training window. The original fast engine uses dates[s-train_days:s].
    embargo = int(cfg["research"].get("embargo_days", 0))
    train_days = int(cfg["research"]["train_days"])
    if embargo >= train_days:
        raise ValueError("embargo_days must be smaller than train_days")
    run_cfg = dict(cfg)
    run_cfg["research"] = dict(cfg["research"])
    run_cfg["research"]["train_days"] = train_days - embargo

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)
        tmp = f.name
    try:
        engine.run(tmp)
    finally:
        os.unlink(tmp)

    trades_path = "docs/intraday_ml_oos_trades.csv"
    t = pd.read_csv(trades_path)
    t = replay_portfolio(t, cfg)
    t.to_csv(trades_path, index=False)
    lb = metrics(t)
    lb.to_csv("docs/intraday_ml_leaderboard.csv", index=False)
    m = t.copy()
    m["month"] = pd.to_datetime(m["exit_date"]).dt.to_period("M").astype(str)
    m = m.groupby(["method", "month"], as_index=False)["weighted_return"].sum()
    m = m.rename(columns={"month": "period", "weighted_return": "monthly_return"})
    m.to_csv("docs/intraday_ml_monthly_returns.csv", index=False)
    print(lb.to_string(index=False))
    print(f"portfolio trades={len(t)} embargo_days={embargo}")


if __name__ == "__main__":
    main()
