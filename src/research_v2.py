from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from btst_lab import (
    add_cross_sectional_features,
    add_features,
    add_market_context,
    load_config,
    metrics,
    prepare_entries,
    read_market_files,
    select_top,
    simulate_trades,
    strategy_scores,
)

RULE_FAMILIES = {
    "momentum", "mean_reversion", "closing_strength", "volume",
    "gap", "regime", "sector_relative"
}


def fold_dates(dates, train_days: int, val_days: int, test_days: int, step_days: int, embargo: int):
    start = train_days + val_days + embargo
    while start < len(dates):
        test_end = min(start + test_days, len(dates))
        train_end = start - val_days - embargo
        val_end = start - embargo
        train = dates[max(0, train_end - train_days):train_end]
        val = dates[train_end:val_end]
        test = dates[start:test_end]
        if len(train) and len(val) and len(test):
            yield train, val, test
        start += step_days


def _safe_validation_dates(val_dates) -> list:
    """Exclude the final validation date because its next-session outcome lies outside validation."""
    dates = list(val_dates)
    return dates[:-1] if len(dates) > 1 else []


def rule_score(val_trades: pd.DataFrame) -> float:
    if val_trades.empty:
        return -1e9
    r = val_trades["return"].astype(float)
    daily = val_trades.groupby("signal_date")["weighted_return"].sum()
    if len(r) < 20 or daily.empty:
        return float(r.mean())
    sd = daily.std()
    sharpe = daily.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
    return float(0.6 * sharpe + 0.4 * r.mean() * 100)


def tune_rule(x: pd.DataFrame, family: str, cfg: dict, val_dates) -> dict:
    top_candidates = [5, 10, 15]
    score_candidates = [0.55, 0.60, 0.65, 0.70]
    safe_val_dates = _safe_validation_dates(val_dates)
    val = x[x.date.isin(safe_val_dates)].copy()
    val["score"] = strategy_scores(val, family)
    best = None
    for top_n in top_candidates:
        for min_score in score_candidates:
            picks = select_top(val, top_n=top_n, min_score=min_score)
            trades, _ = simulate_trades(picks, cfg)
            score = rule_score(trades)
            if best is None or score > best["score"]:
                best = {"top_n": top_n, "min_score": min_score, "score": score}
    return best or {"top_n": 10, "min_score": 0.60, "score": -1e9}


def evaluate_rule_family(df: pd.DataFrame, family: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    dates = sorted(df.date.unique())
    r = cfg["research"]
    train_days = int(r.get("train_years", 4) * 252)
    val_days = int(r.get("validation_months", 12) * 21)
    test_days = int(r.get("test_months", 12) * 21)
    step_days = int(r.get("step_months", 6) * 21)
    embargo = int(r.get("embargo_days", 1))
    folds: List[pd.DataFrame] = []
    params = []
    for train_dates, val_dates, test_dates in fold_dates(dates, train_days, val_days, test_days, step_days, embargo):
        tuned = tune_rule(df, family, cfg, val_dates)
        params.append(tuned)
        test = df[df.date.isin(test_dates)].copy()
        test["score"] = strategy_scores(test, family)
        picks = select_top(test, top_n=tuned["top_n"], min_score=tuned["min_score"])
        trades, _ = simulate_trades(picks, cfg)
        if not trades.empty:
            trades["fold"] = len(folds) + 1
            folds.append(trades)
    if not folds:
        return pd.DataFrame(), {"folds": 0, "parameter_stability": 0.0, "params": []}
    out = pd.concat(folds, ignore_index=True)
    keys = [(p["top_n"], p["min_score"]) for p in params]
    stability = float(pd.Series(keys).value_counts(normalize=True).iloc[0]) if keys else 0.0
    return out, {"folds": len(folds), "parameter_stability": stability, "params": params}


def run_ml_family(df: pd.DataFrame, family: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        LGBMClassifier = None

    features = [
        "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60", "gap",
        "range_pct", "close_location", "body_pct", "atr_pct", "sma20_gap",
        "sma50_gap", "vol20", "volume_z", "mkt_ret_1", "mkt_ret_5", "mkt_vol_20"
    ]
    features = [c for c in features if c in df.columns]
    work = df.dropna(subset=features + ["label", "btst_return", "next_open", "next_high", "next_low", "next_close"]).copy()
    dates = sorted(work.date.unique())
    r = cfg["research"]
    folds = []
    thresholds = []
    train_days = int(r.get("train_years", 4) * 252)
    val_days = int(r.get("validation_months", 12) * 21)
    test_days = int(r.get("test_months", 12) * 21)
    step_days = int(r.get("step_months", 6) * 21)
    embargo = int(r.get("embargo_days", 1))
    for train_dates, val_dates, test_dates in fold_dates(dates, train_days, val_days, test_days, step_days, embargo):
        train = work[work.date.isin(train_dates)]
        safe_val_dates = _safe_validation_dates(val_dates)
        val = work[work.date.isin(safe_val_dates)]
        test = work[work.date.isin(test_dates)].copy()
        if len(train) < int(r.get("min_train_observations", 2000)) or val.empty or test.empty:
            continue
        if family == "ml_ranker" and LGBMClassifier is not None:
            model = LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=31, min_child_samples=40, random_state=int(r.get("random_state", 42)), verbosity=-1)
        else:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, C=0.25, class_weight="balanced", random_state=int(r.get("random_state", 42))))
            ])
        model.fit(train[features], train.label)
        val = val.copy()
        val["score"] = model.predict_proba(val[features])[:, 1]
        test["score"] = model.predict_proba(test[features])[:, 1]
        best = (0.60, -1e9)
        for threshold in np.arange(0.50, 0.81, 0.02):
            q = val[val.score >= threshold]
            if len(q) < 30:
                continue
            score = float(q.btst_return.mean())
            if score > best[1]:
                best = (float(threshold), score)
        threshold = best[0]
        thresholds.append(threshold)
        test = test[test.score >= threshold].copy()
        test["rank"] = test.groupby("date")["score"].rank(method="first", ascending=False)
        test = test[test["rank"] <= int(cfg["portfolio"]["max_positions"])]
        trades, _ = simulate_trades(test, cfg)
        if not trades.empty:
            trades["fold"] = len(folds) + 1
            folds.append(trades)
    if not folds:
        return pd.DataFrame(), {"folds": 0, "threshold_stability": 0.0, "thresholds": []}
    out = pd.concat(folds, ignore_index=True)
    stability = float(pd.Series(thresholds).round(2).value_counts(normalize=True).iloc[0]) if thresholds else 0.0
    return out, {"folds": len(folds), "threshold_stability": stability, "thresholds": thresholds}


def composite(r) -> float:
    pf = min(r.profit_factor, 4.0) / 4.0 if np.isfinite(r.profit_factor) else 1.0
    dd_quality = max(0.0, 1 + r.max_drawdown)
    return float(
        0.30 * np.tanh(max(r.sharpe, -3) / 2)
        + 0.20 * pf
        + 0.15 * np.tanh(r.expectancy * 100)
        + 0.15 * dd_quality
        + 0.15 * r.stability
        + 0.05 * np.tanh(max(r.annualized_return, -1) * 3)
    )


def eligibility(result, trades: pd.DataFrame, diag: dict, min_oos: int, min_stability: float = 0.34) -> tuple[bool, list[str]]:
    """Hard gate: eligibility means evidence is positive, not merely sufficiently numerous."""
    reasons = []
    stable = max(float(diag.get("parameter_stability", diag.get("threshold_stability", 0.0))), 0.0)
    if len(trades) < min_oos:
        reasons.append(f"oos_trades<{min_oos}")
    if stable < min_stability:
        reasons.append(f"stability<{min_stability:.2f}")
    if result.expectancy <= 0:
        reasons.append("non_positive_expectancy")
    # An infinite profit factor is valid when there are no losing trades. Reject
    # NaN/invalid values, but do not reject +inf merely because it is non-finite.
    if np.isnan(result.profit_factor) or result.profit_factor <= 1:
        reasons.append("profit_factor<=1")
    if result.sharpe <= 0:
        reasons.append("non_positive_sharpe")
    if result.total_return <= 0:
        reasons.append("non_positive_oos_return")
    if result.max_drawdown <= -0.50:
        reasons.append("max_drawdown<=-50pct")
    return not reasons, reasons


def run(config_path: str):
    cfg = load_config(config_path)
    daily = read_market_files(cfg["data"]["daily_glob"])
    market = read_market_files(cfg["data"]["market_glob"])
    daily = add_features(daily)
    market = add_features(market)
    daily = add_market_context(daily, market)
    daily = add_cross_sectional_features(daily)
    symbols = sorted(daily.symbol.dropna().unique())
    max_symbols = int(cfg["data"].get("max_symbols", 200))
    symbols = symbols[:max_symbols]
    daily = daily[daily.symbol.isin(symbols)].copy()
    daily = prepare_entries(daily)
    families = list(cfg["strategy_search"]["families"])
    min_oos = int(cfg["research"].get("min_trades_oos", 30))
    rows = []
    all_trades = []
    diagnostics = {}
    for family in families:
        if family in RULE_FAMILIES:
            trades, diag = evaluate_rule_family(daily, family, cfg)
        else:
            trades, diag = run_ml_family(daily, family, cfg)
        result = metrics(trades, family, cfg["portfolio"]["initial_capital"]) if not trades.empty else metrics(pd.DataFrame(), family, cfg["portfolio"]["initial_capital"])
        score = composite(result)
        eligible, rejection_reasons = eligibility(result, trades, diag, min_oos)
        stable = diag.get("parameter_stability", diag.get("threshold_stability", 0.0))
        rows.append({
            **result.__dict__,
            "score": score,
            "eligible": eligible,
            "eligibility_reasons": ";".join(rejection_reasons),
            "folds": diag.get("folds", 0),
            "parameter_stability": stable
        })
        if not trades.empty:
            all_trades.append(trades.assign(strategy=family))
        diagnostics[family] = diag
    leaderboard = pd.DataFrame(rows).sort_values(["eligible", "score"], ascending=[False, False])
    out_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    Path("docs").mkdir(exist_ok=True)
    leaderboard.to_csv("docs/strategy_leaderboard.csv", index=False)
    out_trades.to_csv("docs/oos_trades.csv", index=False)
    manifest = {
        "engine": "research_v2",
        "symbols_tested": int(len(symbols)),
        "date_start": str(pd.to_datetime(daily.date.min()).date()),
        "date_end": str(pd.to_datetime(daily.date.max()).date()),
        "strategies_tested": list(families),
        "min_oos_trades": min_oos,
        "eligibility_gate": {
            "min_oos_trades": min_oos,
            "min_parameter_stability": 0.34,
            "requires_positive_expectancy": True,
            "requires_profit_factor_gt_1": True,
            "requires_positive_sharpe": True,
            "requires_positive_oos_return": True,
            "rejects_max_drawdown_at_or_below_50pct": True,
        },
        "universe_warning": "Dataset universe may be survivorship-biased unless historical constituents are supplied.",
        "execution_warning": "Daily OHLC cannot determine intraday stop/target ordering; stop-first is used conservatively.",
        "sector_warning": "sector_relative is not truly sector-mapped unless a sector mapping is added.",
        "diagnostics": diagnostics,
    }
    with open("docs/research_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open("docs/strategy_leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    return {"leaderboard": rows, "manifest": manifest}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/btst.yaml")
    args = ap.parse_args()
    print(json.dumps(run(args.config), indent=2, default=str))