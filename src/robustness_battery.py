from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from swing_tournament import load, features, add_forward_fields, simulate, stat
from selective_optimizer import component_frame, fold_ranges, score_candidate

COMPONENTS = ("mr5", "mr20", "sma20", "loc", "volume", "relative")
DEFAULTS = {
    "mr5": 0.20,
    "mr20": 0.35,
    "sma20": 0.35,
    "loc": 0.10,
    "volume": 0.0,
    "relative": 0.0,
    "regime_bonus": 0.05,
    "min_vz": 0.0,
    "use_regime": False,
    "use_volume_filter": False,
}


def parse_params(value) -> dict:
    if isinstance(value, dict):
        p = dict(value)
    elif pd.isna(value):
        p = {}
    else:
        p = json.loads(str(value))
    missing = [k for k in COMPONENTS if k not in p]
    if missing:
        raise ValueError(f"Selected fold parameters missing required components: {missing}; params={p}")
    out = dict(DEFAULTS)
    out.update(p)
    return out


def perturb_one_component(p: dict, component: str, scale: float) -> dict:
    q = parse_params(p)
    q[component] = max(0.0, float(q[component]) * scale)
    total = sum(float(q[k]) for k in COMPONENTS)
    if total <= 0:
        raise ValueError("Perturbation produced zero total score weight")
    for k in COMPONENTS:
        q[k] = float(q[k]) / total
    return q


def evaluate(d, history, dates, cfg, horizon, fold_df, per_side_cost, sl_atr, tp_atr, top_n, perturb_component=None, perturb_scale=1.0):
    cfg2 = copy.deepcopy(cfg)
    # per_side_cost is total execution friction per side (slippage + transaction costs).
    txn = min(float(cfg2["costs"].get("transaction_cost_bps_per_side", 0.0)), per_side_cost)
    cfg2["costs"]["transaction_cost_bps_per_side"] = txn
    cfg2["costs"]["slippage_bps_per_side"] = max(0.0, per_side_cost - txn)
    cfg2["execution"]["stop_atr_mult"] = sl_atr
    cfg2["execution"]["target_atr_mult"] = tp_atr

    ranges = list(fold_ranges(dates, cfg))
    all_trades = []
    for _, fr in fold_df.iterrows():
        fold = int(fr["fold"])
        base_params = parse_params(fr["params"])
        params = base_params
        if perturb_component is not None:
            params = perturb_one_component(params, perturb_component, perturb_scale)
        params["top_n"] = top_n
        if fold < 1 or fold > len(ranges):
            continue
        _, test_dates = ranges[fold - 1]
        x = d[d.date.isin(test_dates)].dropna(subset=["entry_open", "atr"]).copy()
        if x.empty:
            continue
        x["score"] = score_candidate(x, params)
        picks = x.sort_values(["date", "score"], ascending=[True, False]).groupby("date", sort=False).head(top_n)
        t = simulate(picks, history, cfg2, horizon)
        if not t.empty:
            all_trades.append(t.assign(fold=fold, horizon_days=horizon))

    if not all_trades:
        return None
    trades = pd.concat(all_trades, ignore_index=True)
    return stat(trades), trades


def add_result(rows, h, per_side_cost, sl, tp, n, analysis, result, **extra):
    if result is None:
        return
    s, _ = result
    rows.append({"horizon_days": h, "per_side_cost_bps": per_side_cost, "stop_atr": sl, "target_atr": tp, "top_n": n, "analysis": analysis, **extra, **s})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/swing.yaml")
    ap.add_argument("--summary-dir", default="docs")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    docs = Path(args.summary_dir)

    raw = load(cfg)
    # Robustness must use the same component construction as the nested optimizer.
    base = component_frame(features(raw)).sort_values(["symbol", "date"]).reset_index(drop=True)
    dates = sorted(base.date.unique())
    scenarios = []

    for h in (10, 20):
        fold_path = docs / f"selective_{h}_fold_parameters.csv"
        if not fold_path.exists():
            fold_path = docs / "selective_fold_parameters.csv"
        f = pd.read_csv(fold_path)
        f = f[f.horizon_days == h].sort_values("fold")
        if len(f) == 0:
            raise ValueError(f"No selected fold parameters found for horizon {h}")
        # Validate every selected parameter record before starting expensive simulations.
        for value in f["params"]:
            parse_params(value)
        d = add_forward_fields(base, h)

        for per_side_cost in (13.0, 20.0, 30.0, 40.0):
            for sl, tp in ((1.0, 2.0), (1.5, 3.0), (2.0, 4.0)):
                for n in (5, 10, 15, 20):
                    result = evaluate(d, d, dates, cfg, h, f, per_side_cost, sl, tp, n)
                    add_result(scenarios, h, per_side_cost, sl, tp, n, "scenario", result)

        # True perturbation: change one selected component at a time, then renormalize.
        for component in COMPONENTS:
            for scale in (0.8, 0.9, 1.1, 1.2):
                result = evaluate(d, d, dates, cfg, h, f, 13.0, 1.5, 3.0, 10, component, scale)
                add_result(scenarios, h, 13.0, 1.5, 3.0, 10, "parameter_perturbation", result, parameter=component, parameter_scale=scale)

        base_result = evaluate(d, d, dates, cfg, h, f, 13.0, 1.5, 3.0, 10)
        if base_result:
            _, trades = base_result
            trades["year"] = pd.to_datetime(trades["signal_date"]).dt.year
            for year, g in trades.groupby("year"):
                add_result(scenarios, h, 13.0, 1.5, 3.0, 10, "year", (stat(g), g), year=int(year), parameter_scale=1.0)

    out = pd.DataFrame(scenarios)
    docs.mkdir(exist_ok=True)
    out.to_csv(docs / "swing_robustness_battery.csv", index=False)

    stress = out[(out.analysis == "scenario") & (out.per_side_cost_bps >= 30)]
    pert = out[out.analysis == "parameter_perturbation"]
    summary = []
    for h in (10, 20):
        hs, hp = stress[stress.horizon_days == h], pert[pert.horizon_days == h]
        summary.append({
            "horizon_days": h,
            "stress_scenarios": len(hs),
            "stress_positive_pf_fraction": float((hs.profit_factor > 1).mean()) if len(hs) else 0.0,
            "stress_positive_expectancy_fraction": float((hs.expectancy > 0).mean()) if len(hs) else 0.0,
            "perturbation_scenarios": len(hp),
            "perturbation_positive_pf_fraction": float((hp.profit_factor > 1).mean()) if len(hp) else 0.0,
            "perturbation_positive_expectancy_fraction": float((hp.expectancy > 0).mean()) if len(hp) else 0.0,
        })
    pd.DataFrame(summary).to_csv(docs / "swing_robustness_summary.csv", index=False)

    manifest = {
        "engine": "nse_daily_swing_robustness_v3",
        "primary_horizons": [10, 20],
        "per_side_cost_bps": [13, 20, 30, 40],
        "stop_target_atr": [[1.0, 2.0], [1.5, 3.0], [2.0, 4.0]],
        "top_n": [5, 10, 15, 20],
        "parameter_perturbation": {"components": list(COMPONENTS), "scales": [0.8, 0.9, 1.1, 1.2], "method": "one component at a time, followed by weight renormalization"},
        "method": "replay selected nested-OOS fold parameters on untouched test dates; vary only robustness controls",
        "survivorship_warning": True,
        "point_in_time_membership": False,
    }
    with open(docs / "swing_robustness_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(pd.DataFrame(summary).to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
