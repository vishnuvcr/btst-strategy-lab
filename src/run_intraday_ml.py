from __future__ import annotations

import argparse
import importlib
import pandas as pd


def main(config: str) -> None:
    # The tournament source is deliberately kept as the signal engine; this runner
    # applies the final portfolio constraints before publishing OOS statistics.
    engine = importlib.import_module("intraday_ml_tournament")
    engine.FEATURES = [f for f in engine.FEATURES if f != "orb_gap"]
    engine.run(config)

    p = "docs/intraday_ml_oos_trades.csv"
    t = pd.read_csv(p, parse_dates=["signal_datetime", "exit_date"])
    if t.empty:
        raise SystemExit("Intraday tournament produced no OOS trades")

    # Entry-time portfolio cap: at most 10 simultaneous new allocations and
    # no more than 95% gross exposure. Within a timestamp, prefer the highest
    # model/rule score. This is intentionally conservative and deterministic.
    max_positions = int(engine.cfg(config)["portfolio"]["max_positions"])
    max_gross = float(engine.cfg(config)["portfolio"]["max_gross_exposure"])
    t = t.sort_values(["signal_datetime", "score"], ascending=[True, False]).copy()
    t["entry_rank"] = t.groupby("signal_datetime").cumcount()
    t = t[t.entry_rank < max_positions].copy()
    counts = t.groupby("signal_datetime")["symbol"].transform("count")
    t["weight"] = max_gross / counts
    t["weighted_return"] = t["return"] * t["weight"]
    t.to_csv(p, index=False)

    rows = []
    for method, g in t.groupby("method"):
        r = g.sort_values("signal_datetime")["weighted_return"].astype(float)
        eq = (1 + r).cumprod()
        gains = r[r > 0].sum(); losses = -r[r < 0].sum()
        pf = gains / losses if losses > 0 else float("inf")
        sharpe = (252 * 78) ** 0.5 * r.mean() / r.std() if r.std() > 0 else 0.0
        dd = (eq / eq.cummax() - 1).min()
        rows.append({
            "strategy": method,
            "trades": len(g),
            "win_rate": (r > 0).mean(),
            "profit_factor": pf,
            "expectancy": r.mean(),
            "total_return": eq.iloc[-1] - 1,
            "sharpe": sharpe,
            "max_drawdown": dd,
        })
    lb = pd.DataFrame(rows).sort_values(["profit_factor", "sharpe"], ascending=False)
    lb.to_csv("docs/intraday_ml_leaderboard.csv", index=False)

    monthly = t.assign(month=t.exit_date.dt.to_period("M")).groupby(["method", "month"], as_index=False).weighted_return.sum()
    monthly = monthly.rename(columns={"weighted_return": "monthly_return"})
    monthly.to_csv("docs/intraday_ml_monthly_returns.csv", index=False)
    print("\nFINAL PORTFOLIO-CONSTRAINED OOS LEADERBOARD")
    print(lb.to_string(index=False))
    print("\nBEST MONTH BY METHOD")
    print(monthly.groupby("method").monthly_return.max().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/intraday_ml.yaml")
    main(ap.parse_args().config)
