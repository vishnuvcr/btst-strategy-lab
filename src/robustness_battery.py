from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from swing_tournament import load, features, add_forward_fields, simulate, stat
from selective_optimizer import fold_ranges, score_candidate


def perturb_params(p: dict, scale: float) -> dict:
    q = dict(p)
    keys = ("mr5", "mr20", "sma20", "loc", "volume", "relative")
    vals = [max(0.0, float(q.get(k, 0.0)) * scale) for k in keys]
    total = sum(vals)
    if total > 0:
        for k, v in zip(keys, vals):
            q[k] = v / total
    return q


def evaluate(d, history, dates, cfg, horizon, fold_df, costs, sl_atr, tp_atr, top_n, perturb):
    cfg2 = copy.deepcopy(cfg)
    cfg2["costs"]["slippage_bps_per_side"] = costs / 2.0
    cfg2["costs"]["transaction_cost_bps_per_side"] = costs / 2.0
    cfg2["execution"]["stop_atr_mult"] = sl_atr
    cfg2["execution"]["target_atr_mult"] = tp_atr
    rows = []
    all_trades = []
    ranges = list(fold_ranges(dates, cfg))
    for _, fr in fold_df.iterrows():
        fold = int(fr["fold"])
        params = perturb_params(json.loads(fr["params"]), perturb)
        params["top_n"] = top_n
        params["use_regime"] = bool(params.get("use_regime", False))
        params["use_volume_filter"] = bool(params.get("use_volume_filter", False))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/swing.yaml")
    ap.add_argument("--summary-dir", default="docs")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    docs = Path(args.summary_dir)
    raw = load(cfg)
    base = features(raw).sort_values(["symbol", "date"]).reset_index(drop=True)
    dates = sorted(base.date.unique())
    scenarios = []
    for h in (10, 20):
        fold_path = docs / f"selective_{h}_fold_parameters.csv"
        if not fold_path.exists():
            fold_path = docs / "selective_fold_parameters.csv"
        f = pd.read_csv(fold_path)
        f = f[f.horizon_days == h].sort_values("fold")
        d = add_forward_fields(base, h)
        for costs in (13.0, 20.0, 30.0, 40.0):
            for sl, tp in ((1.0, 2.0), (1.5, 3.0), (2.0, 4.0)):
                for n in (5, 10, 15, 20):
                    result = evaluate(d, d, dates, cfg, h, f, costs, sl, tp, n, 1.0)
                    if result:
                        s, _ = result
                        scenarios.append({"horizon_days":h,"round_trip_cost_bps":costs,"stop_atr":sl,"target_atr":tp,"top_n":n,"parameter_scale":1.0,"analysis":"scenario",**s})
        for perturb in (0.8, 0.9, 1.1, 1.2):
            result = evaluate(d, d, dates, cfg, h, f, 13.0, 1.5, 3.0, 10, perturb)
            if result:
                s, _ = result
                scenarios.append({"horizon_days":h,"round_trip_cost_bps":13.0,"stop_atr":1.5,"target_atr":3.0,"top_n":10,"parameter_scale":perturb,"analysis":"parameter_perturbation",**s})
        base_result = evaluate(d, d, dates, cfg, h, f, 13.0, 1.5, 3.0, 10, 1.0)
        if base_result:
            _, trades = base_result
            trades["year"] = pd.to_datetime(trades["signal_date"]).dt.year
            for year, g in trades.groupby("year"):
                scenarios.append({"horizon_days":h,"round_trip_cost_bps":13.0,"stop_atr":1.5,"target_atr":3.0,"top_n":10,"parameter_scale":1.0,"analysis":"year","year":int(year),**stat(g)})
    out = pd.DataFrame(scenarios)
    docs.mkdir(exist_ok=True)
    out.to_csv(docs / "swing_robustness_battery.csv", index=False)
    stress = out[(out["round_trip_cost_bps"] >= 30) & (out["analysis"] == "scenario")]
    pert = out[(out["parameter_scale"] != 1.0) & (out["analysis"] == "parameter_perturbation")]
    summary=[]
    for h in (10,20):
        hs=stress[stress.horizon_days==h]; hp=pert[pert.horizon_days==h]
        summary.append({"horizon_days":h,"stress_scenarios":len(hs),"stress_positive_pf_fraction":float((hs.profit_factor>1).mean()) if len(hs) else 0.0,"stress_positive_expectancy_fraction":float((hs.expectancy>0).mean()) if len(hs) else 0.0,"perturbation_positive_pf_fraction":float((hp.profit_factor>1).mean()) if len(hp) else 0.0,"perturbation_positive_expectancy_fraction":float((hp.expectancy>0).mean()) if len(hp) else 0.0})
    pd.DataFrame(summary).to_csv(docs / "swing_robustness_summary.csv", index=False)
    json.dump({"engine":"nse_daily_swing_robustness_v1","primary_horizons":[10,20],"cost_round_trip_bps":[13,20,30,40],"stop_target_atr":[[1.0,2.0],[1.5,3.0],[2.0,4.0]],"top_n":[5,10,15,20],"parameter_scales":[0.8,0.9,1.0,1.1,1.2],"method":"replay selected nested-OOS fold parameters on untouched test dates; vary only robustness controls","survivorship_warning":True,"point_in_time_membership":False}, open(docs/"swing_robustness_manifest.json","w",encoding="utf-8"), indent=2)
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__":
    main()
