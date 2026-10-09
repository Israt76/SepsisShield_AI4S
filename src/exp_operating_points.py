"""AI4S experiment: operating-point / alert-burden trade-off of the shipped model (no retraining).

For a grid of alert thresholds on the calibrated risk, report on the held-out TEST set: patient sensitivity (alert in
the useful window), share of septic patients alerted >= 6 h before onset, patient specificity, challenge utility,
false-alert hours and new alert episodes per 100 non-septic patient-days. Validation utility is reported too, because
the shipped threshold (0.03) was chosen by maximizing utility on VALIDATION patients.

Output: results/exp_operating_points.json
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from metrics import patient_level, utility_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]


def episodes_per_100_days(d, pred, septic_ids):
    ns = d["patient_id"].map(lambda p: p not in septic_ids).to_numpy()
    x = pd.Series(pred.astype(int), index=d.index)
    prev = x.groupby(d["patient_id"]).shift(1).fillna(0).to_numpy()
    starts = (x.to_numpy() == 1) & (prev == 0)
    return float(starts[ns].sum() / (ns.sum() / 24) * 100)


def main():
    te = pd.read_parquet(RES / "test_predictions.parquet").sort_values(["patient_id", "hour"]).reset_index(drop=True)
    va = pd.read_parquet(RES / "val_predictions.parquet").sort_values(["patient_id", "hour"]).reset_index(drop=True)
    septic = set(te.loc[te["SepsisLabel"] == 1, "patient_id"])
    shipped = json.load(open(ROOT / "models" / "config.json"))["threshold"]
    rows = []
    for g in [shipped if abs(x - shipped) < 1e-9 else x for x in GRID]:   # the exact shipped value (isotonic ties)
        pred = (te["prob"] >= g).to_numpy()
        pl = patient_level(te, pred)
        rows.append({"threshold": g, "patient_sensitivity": pl["patient_sensitivity"],
                     "pct_detected_ge6h_early": pl["pct_detected_ge6h_early"],
                     "patient_specificity": pl["patient_specificity"],
                     "utility_test": utility_score(te, pred.astype(int)),
                     "utility_val": utility_score(va, (va["prob"] >= g).to_numpy().astype(int)),
                     "false_alert_hours_per_100_patient_days": pl["false_alert_hours_per_100_patient_days"],
                     "alert_episodes_per_100_nonseptic_patient_days": episodes_per_100_days(te, pred, septic),
                     "median_lead_time_h": pl["median_lead_time_h"]})
        print({k: round(v, 3) for k, v in rows[-1].items()})
    json.dump({"shipped_threshold": shipped, "rows": rows,
               "note": "test = held-out test set (8,068 patients); threshold was chosen on validation utility"},
              open(RES / "exp_operating_points.json", "w"), indent=1)


if __name__ == "__main__":
    main()
