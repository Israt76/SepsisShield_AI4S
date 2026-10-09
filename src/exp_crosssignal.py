"""AI4S experiment: does a cross-signal consistency detector (src/crosssignal.py) catch the failures the current
integrity layer misses?

PRE-SPECIFIED PROTOCOL (written before the test-set run; do not edit after it):
  1. Fit the regression models on TRAINING patients only.
  2. Threshold = 99th percentile of the score on all clean VALIDATION patient-hours (adds at most ~1% flagged hours).
  3. Development check on the VALIDATION benchmark cohort (same construction as results/abstention_val.json).
  4. ONE evaluation on the held-out TEST benchmark cohort (identical patients and corruption windows as
     results/abstention.json), plus the clean cost on all 8,068 test patients.
  5. Action if shipped: a flag lowers trust to REDUCED only (warn), never LOW.
  6. SHIP CRITERION (both on test): plausible deliberate-edit coverage rises by >= 10 percentage points, AND the total
     share of clean test hours flagged or withheld rises by <= 2 percentage points. Otherwise: report as a negative or
     partial result and do not ship.

Output: results/exp_crosssignal.json
"""
from pathlib import Path
import json, sys, time

import numpy as np
import pandas as pd

from abstention import KINDS, classify
from benchmark_integrity import corrupt
from train import split_patients
import crosssignal
import pipeline

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "exp_crosssignal.json"
FLOORS = {"HR": (60, None), "Resp": (12, None), "Temp": (36.0, None), "SBP": (None, 160), "MAP": (None, 110)}
SHIP_EDIT_GAIN, SHIP_CLEAN_COST = 0.10, 0.02


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cohort(ids, n_nonseptic, seed=7):
    rng = np.random.default_rng(seed)
    raw = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    raw = raw[raw["patient_id"].isin(ids)]
    sep = raw.groupby("patient_id")["SepsisLabel"].max()
    keep = set(sep[sep == 1].index) | set(rng.choice(sep[sep == 0].index.to_numpy(), n_nonseptic, replace=False))
    raw = raw[raw["patient_id"].isin(keep)].sort_values(["patient_id", "hour"]).reset_index(drop=True)
    return raw, rng


def with_flag(scored, xflag):
    s = scored.copy()
    tr = s["trust"].to_numpy().copy()
    tr[(tr == "HIGH") & xflag] = "REDUCED"
    s["trust"] = tr
    return s


def cov(c):
    a = c["all"]
    return {"n": a["n"], "withheld": a["withheld"], "flagged": a["flagged"], "silent": a["silent"],
            "coverage": (a["withheld"] + a["flagged"]) / a["n"] if a["n"] else None}


def benchmark(raw, rng, thr):
    clean, _ = pipeline.score(raw)
    xclean = crosssignal.score(raw) > thr
    out = {"clean_cohort_flag_rate_added": float((xclean & (clean["trust"] == "HIGH").to_numpy()).mean())}
    runs = {}
    for kind in KINDS:
        cr, m = corrupt(raw, kind, rng)
        runs[kind] = (cr, m)
    cr, m = runs["masking_attack"]
    pl = cr.copy()
    for col, (lo, hi) in FLOORS.items():
        pl.loc[m, col] = pl.loc[m, col].clip(lower=lo, upper=hi)
    runs["masking_plausible"] = (pl, m)
    for kind, (cr, m) in runs.items():
        sc, _ = pipeline.score(cr)
        xf = crosssignal.score(cr) > thr
        before, after = cov(classify(clean, sc, m)), cov(classify(clean, with_flag(sc, xf), m))
        out[kind] = {"before": before, "after": after, "window_hours_flagged_by_detector": float(xf[m].mean())}
        log(kind, before["coverage"], "->", after["coverage"])
    acc = [k for k in KINDS[:4]]
    for label, kinds in (("accidental", acc),):
        for when in ("before", "after"):
            n = sum(out[k][when]["n"] for k in kinds)
            c = sum(out[k][when]["withheld"] + out[k][when]["flagged"] for k in kinds)
            out.setdefault(label, {})[when] = {"n": n, "covered": c, "coverage": c / n}
    return out


def main(stage):
    R = json.load(open(OUT)) if OUT.exists() else {"protocol": __doc__.split("PRE-SPECIFIED PROTOCOL")[1].split("Output:")[0].strip()}
    hourly = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    tr, va, te = split_patients(hourly)
    if "fit" in stage:
        t0 = time.time()
        trn = hourly[hourly["patient_id"].isin(tr)].sort_values(["patient_id", "hour"]).reset_index(drop=True)
        meta = crosssignal.fit(trn)
        log("fit", meta, f"{time.time() - t0:.0f}s")
        del trn
        vraw = hourly[hourly["patient_id"].isin(va)].sort_values(["patient_id", "hour"]).reset_index(drop=True)
        s = crosssignal.score(vraw)
        thr = float(np.quantile(s, 0.99))
        meta = json.load(open(crosssignal.DIR / "meta.json"))
        meta["threshold"] = thr
        meta["threshold_rule"] = "99th percentile of the score on clean validation patient-hours"
        json.dump(meta, open(crosssignal.DIR / "meta.json", "w"), indent=1)
        crosssignal._cache.clear()
        R["threshold"] = thr
        R["models"] = {k: meta[k] for k in ("scale", "n_train_rows")}
        log("threshold", thr)
        json.dump(R, open(OUT, "w"), indent=1)
    thr = json.load(open(crosssignal.DIR / "meta.json"))["threshold"]
    del hourly
    if "val" in stage:
        raw, rng = cohort(va, 1250)
        R["validation_benchmark"] = benchmark(raw, rng, thr)
        json.dump(R, open(OUT, "w"), indent=1)
    if "test" in stage:
        assert "test_benchmark" not in R, "the test set is evaluated once"
        raw, rng = cohort(te, 2500)
        T = benchmark(raw, rng, thr)
        full = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
        full = full[full["patient_id"].isin(te)].sort_values(["patient_id", "hour"]).reset_index(drop=True)
        sc, _ = pipeline.score(full)
        xf = crosssignal.score(full) > thr
        tr_ = sc["trust"].to_numpy()
        T["clean_full_test"] = {"hours": int(len(sc)),
                                "flagged_or_withheld_before": float((tr_ != "HIGH").mean()),
                                "flagged_or_withheld_after": float(((tr_ != "HIGH") | xf).mean()),
                                "detector_flag_rate": float(xf.mean())}
        gain = T["masking_plausible"]["after"]["coverage"] - T["masking_plausible"]["before"]["coverage"]
        cost = T["clean_full_test"]["flagged_or_withheld_after"] - T["clean_full_test"]["flagged_or_withheld_before"]
        T["decision"] = {"plausible_edit_gain": gain, "clean_cost_increase": cost,
                         "ship": bool(gain >= SHIP_EDIT_GAIN and cost <= SHIP_CLEAN_COST)}
        R["test_benchmark"] = T
        log("DECISION", T["decision"])
        json.dump(R, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["fit", "val"])
