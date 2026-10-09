"""Sensitivity check for the deliberate-edit ("masking_attack") benchmark scenario.

Question: the simulated edit subtracts fixed amounts (e.g. Resp - 8), which can push values below physiological
limits (Resp < 4). How much of the reported deliberate-edit coverage (42.5%, 334 / 786) depends on those impossible
values rather than on the manipulation-specific checks?

Method (does NOT change any reported result): rebuild the exact test benchmark sample and corruption windows used by
src/abstention.py (same seed and RNG stream), then score
  1. the original edit (must reproduce 334 / 786 exactly), and
  2. a "plausible" edit: identical, but every edited vital is kept inside normal adult ranges
     (HR >= 60, Resp >= 12, Temp >= 36.0, SBP <= 160, MAP <= 110), i.e. the kind of edit that "looks normal".
Writes results/masking_realism.json. Needs the raw data (data/processed/hourly.parquet).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abstention import KINDS, classify  # noqa: E402
from benchmark_integrity import corrupt  # noqa: E402
import pipeline  # noqa: E402

RES = ROOT / "results"
FLOORS = {"HR": (60, None), "Resp": (12, None), "Temp": (36.0, None), "SBP": (None, 160), "MAP": (None, 110)}


def summary(c):
    a = c["all"]
    return {"n": a["n"], "withheld": a["withheld"], "flagged": a["flagged"], "silent": a["silent"],
            "coverage": (a["withheld"] + a["flagged"]) / a["n"] if a["n"] else None}


def main(n_nonseptic=2500, seed=7):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    raw = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    test_ids = set(pd.read_parquet(RES / "test_patients.parquet")["patient_id"])
    raw = raw[raw["patient_id"].isin(test_ids)]
    sep = raw.groupby("patient_id")["SepsisLabel"].max()
    keep = set(sep[sep == 1].index) | set(rng.choice(sep[sep == 0].index.to_numpy(), n_nonseptic, replace=False))
    raw = raw[raw["patient_id"].isin(keep)].sort_values(["patient_id", "hour"]).reset_index(drop=True)
    clean, _ = pipeline.score(raw)
    for kind in KINDS[:4]:            # advance the RNG exactly as src/abstention.py does
        corrupt(raw, kind, rng)
    corr_raw, m = corrupt(raw, "masking_attack", rng)
    orig, _ = pipeline.score(corr_raw)
    c_orig = classify(clean, orig, m)

    plaus_raw = corr_raw.copy()
    for col, (lo, hi) in FLOORS.items():
        v = plaus_raw.loc[m, col]
        plaus_raw.loc[m, col] = v.clip(lower=lo, upper=hi)
    plaus, _ = pipeline.score(plaus_raw)
    c_plaus = classify(clean, plaus, m)

    # which checks fired on the covered hours of the original edit
    ca, xa = clean["alert"].to_numpy(), orig["alert"].to_numpy()
    covered = m & (ca != xa) & (orig["trust"].to_numpy() != "HIGH")
    impossible_resp = int((m & (corr_raw["Resp"] < 4)).sum())
    res = {
        "note": "Sensitivity analysis only; the reported benchmark numbers (results/abstention.json) are unchanged.",
        "plausible_edit_limits": {k: v for k, v in FLOORS.items()},
        "original_edit": summary(c_orig),
        "plausible_edit": summary(c_plaus),
        "original_edit_hours_with_resp_below_4": impossible_resp,
        "edited_hours": int(m.sum()),
        "original_covered_hours_with_implausible_flag": int((covered & orig["flag_implausible"].to_numpy()).sum()),
        "original_covered_hours_with_coordinated_shift_flag": int((covered & orig["flag_coordinated_shift"].to_numpy()).sum()),
        "runtime_s": round(time.time() - t0),
    }
    json.dump(res, open(RES / "masking_realism.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
