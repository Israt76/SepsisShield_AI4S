"""AI4S experiments B-G: robustness sweeps, selective prediction, component ablation, simulated cohort shifts,
calibration before/after. Uses the SHIPPED models (no retraining) and the same held-out test cohort as the
abstention benchmark (all septic test patients + 2,500 random non-septic test patients, seed 7).

  missingness    delete 5-40% of the observed vital/lab values at random (whole stay)
  noise          Gaussian noise (k x population SD) on HR, SBP, Temp, Resp, Lactate, Creatinine (whole stay)
  extreme        physiologically impossible values (HR 900, Temp 70 C, SBP -20) injected into 1% of hours
  selective      risk-coverage on clean data (abstain on the most uncertain hours) and, on the accidental-fault
                 benchmark, how many dangerous failures each abstention signal catches at matched coverage
  ablation       component-by-component: what each layer adds (accidental-fault benchmark + clean cost)
  cohort_shift   simulated cohort-level shifts: does the shift indicator react, and what happens to AUROC?
  calibration    reliability curve of the ensemble before vs after isotonic calibration (held-out test)

Output: results/exp_robustness.json. Nothing in models/ is modified.
"""
from pathlib import Path
import json, time, sys
import numpy as np
import pandas as pd

from data import VITALS, LABS
from metrics import hourly, utility_score, patient_level, onset_hour
from benchmark_integrity import corrupt
from abstention import classify, KINDS, ACCIDENTAL
import pipeline
import shift

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "exp_robustness.json"
NOISE_VARS = ["HR", "SBP", "Temp", "Resp", "Lactate", "Creatinine"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cohort(seed=7, n_nonseptic=2500):
    """Identical to abstention.main(): same patients, same RNG stream."""
    rng = np.random.default_rng(seed)
    raw = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    test_ids = set(pd.read_parquet(RES / "test_patients.parquet")["patient_id"])
    raw = raw[raw["patient_id"].isin(test_ids)]
    sep = raw.groupby("patient_id")["SepsisLabel"].max()
    keep = set(sep[sep == 1].index) | set(rng.choice(sep[sep == 0].index.to_numpy(), n_nonseptic, replace=False))
    raw = raw[raw["patient_id"].isin(keep)].sort_values(["patient_id", "hour"]).reset_index(drop=True)
    return raw, rng


def run(raw):
    out, feats = pipeline.score(raw)
    s = shift.shift_score(feats, shift.load(), feats["patient_id"])
    out["shift_score"] = s.to_numpy()
    out["shift_state"] = shift.state_of(s, shift.load())
    return out


def perf(sc):
    y = sc["SepsisLabel"].to_numpy()
    h = hourly(y, sc["risk"].to_numpy())
    pred = sc["alert"].to_numpy().astype(int)
    pl = patient_level(sc, pred)
    tr, st = sc["trust"].to_numpy(), sc["shift_state"].to_numpy()
    return {"auroc": h["auroc"], "auprc": h["auprc"], "brier": h["brier"], "ece": h["ece"],
            "utility": utility_score(sc, pred), "patient_sensitivity": pl["patient_sensitivity"],
            "patient_specificity": pl["patient_specificity"],
            "trust_low_pct": float((tr == "LOW").mean()), "trust_reduced_pct": float((tr == "REDUCED").mean()),
            "shift_moderate_pct": float((st == "MODERATE").mean()), "shift_high_pct": float((st == "HIGH").mean())}


def flips(clean, corr, m):
    c = classify(clean, corr, m)["all"]
    c["coverage"] = (c["withheld"] + c["flagged"]) / c["n"] if c["n"] else None
    return c


# ------------------------------------------------------------------ B/C/D: missingness, noise, extreme values
def drop_values(raw, frac, rng):
    r = raw.copy()
    num = VITALS + LABS
    X = r[num].to_numpy(float, copy=True)
    obs = ~np.isnan(X)
    kill = obs & (rng.random(X.shape) < frac)
    X[kill] = np.nan
    r[num] = X
    return r, kill.any(1)


def add_noise(raw, k, rng):
    r = raw.copy()
    for c in NOISE_VARS:
        sd = r[c].std()
        v = r[c].to_numpy(float)
        r[c] = v + rng.normal(0, k * sd, len(v))
    if "Lactate" in r:
        r["Lactate"] = r["Lactate"].clip(lower=0.1)
    if "Creatinine" in r:
        r["Creatinine"] = r["Creatinine"].clip(lower=0.1)
    return r, np.ones(len(r), bool)


def inject_extreme(raw, rate, rng):
    r = raw.copy()
    hit = rng.random(len(r)) < rate
    which = rng.integers(0, 3, len(r))
    for j, (c, v) in enumerate((("HR", 900.0), ("Temp", 70.0), ("SBP", -20.0))):
        m = hit & (which == j)
        r[c] = r[c].astype(float)
        r.loc[m, c] = v
    return r, hit


def sweeps(raw, clean, rng):
    R = {"baseline_clean": perf(clean), "missingness": [], "noise": [], "extreme": {}}
    for f in (0.05, 0.10, 0.20, 0.30, 0.40):
        cr, m = drop_values(raw, f, rng)
        sc = run(cr)
        R["missingness"].append({"fraction_removed": f, **perf(sc), "decision_flips": flips(clean, sc, m)})
        log("missing", f, round(R["missingness"][-1]["auroc"], 4), R["missingness"][-1]["decision_flips"]["n"])
    for k in (0.05, 0.10, 0.20, 0.30, 0.50):
        cr, m = add_noise(raw, k, rng)
        sc = run(cr)
        R["noise"].append({"noise_sd_multiple": k, **perf(sc), "decision_flips": flips(clean, sc, m)})
        log("noise", k, round(R["noise"][-1]["auroc"], 4), R["noise"][-1]["decision_flips"]["n"])
    cr, hit = inject_extreme(raw, 0.01, rng)
    sc = run(cr)
    tr = sc["trust"].to_numpy()
    R["extreme"] = {"injected_hours": int(hit.sum()), "values": "HR=900, Temp=70 C, SBP=-20 (one per hit hour)",
                    "injected_hours_trust_low": int((hit & (tr == "LOW")).sum()),
                    "injected_hours_trust_low_pct": float((tr[hit] == "LOW").mean()),
                    **{k: v for k, v in perf(sc).items() if k in ("auroc", "utility", "trust_low_pct")}}
    log("extreme", R["extreme"])
    return R


# ------------------------------------------------------------------ E/F: selective prediction + component ablation
def accidental_benchmark(raw, rng, clean):
    """Re-create the four accidental faults of results/abstention.json exactly (same RNG stream)."""
    rows = []
    for kind in KINDS:
        cr, m = corrupt(raw, kind, rng)
        if kind not in ACCIDENTAL:
            continue
        sc = run(cr)
        ts = onset_hour(clean)
        t_s = clean["patient_id"].map(ts).to_numpy(float)
        h = clean["hour"].to_numpy()
        septic = ~np.isnan(t_s)
        in_win = septic & (h >= t_s - 12) & (h <= t_s + 3)
        ca, xa = clean["alert"].to_numpy(), sc["alert"].to_numpy()
        danger = m & ((in_win & ca & ~xa) | (~in_win & ~ca & xa))
        rows.append(pd.DataFrame({"kind": kind, "in_window": m, "danger": danger, "trust": sc["trust"].to_numpy(),
                                  "ens_std": sc["ens_std"].to_numpy(), "shift_score": sc["shift_score"].to_numpy(),
                                  "shift_state": sc["shift_state"].to_numpy(), "flag_low": (sc["trust"] == "LOW").to_numpy()}))
        log("benchmark", kind, int(danger.sum()))
    return pd.concat(rows, ignore_index=True)


def caught_at_coverage(score, danger, abstain_frac, rng=None):
    """Abstain on the top `abstain_frac` of hours by score; share of dangerous failures caught."""
    n = len(danger)
    k = int(round(abstain_frac * n))
    if k == 0:
        return 0.0
    if score is None:
        idx = rng.choice(n, k, replace=False)
    else:
        idx = np.argsort(-score, kind="stable")[:k]
    return float(danger[idx].sum() / danger.sum())


def selective(clean, B, std_thr):
    R = {}
    # clean data: risk-coverage by ensemble disagreement vs random abstention
    y, p, u = clean["SepsisLabel"].to_numpy(), clean["risk"].to_numpy(), clean["ens_std"].to_numpy()
    rng = np.random.default_rng(0)
    curve = []
    order = np.argsort(u)                     # most certain first
    rnd = rng.permutation(len(u))
    for cov in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7):
        k = int(round(cov * len(u)))
        a, r_ = order[:k], rnd[:k]
        ha, hr = hourly(y[a], p[a]), hourly(y[r_], p[r_])
        curve.append({"coverage": cov, "auroc_uncertainty": ha["auroc"], "brier_uncertainty": ha["brier"],
                      "auroc_random": hr["auroc"], "brier_random": hr["brier"]})
    R["clean_risk_coverage"] = curve
    # corrupted benchmark: which abstention signal catches the dangerous failures?
    d = B["danger"].to_numpy()
    trust_rank = np.select([B["trust"] == "LOW", B["trust"] == "REDUCED"], [2.0, 1.0], 0.0) + \
        1e-3 * np.tanh(B["ens_std"].to_numpy())
    signals = {"input_trust_policy": trust_rank, "ensemble_disagreement_only": B["ens_std"].to_numpy(),
               "shift_score_only": B["shift_score"].to_numpy(), "random": None}
    rows = []
    for frac in (0.05, 0.10, 0.20, 0.30, 0.40):
        row = {"abstain_fraction_of_hours": frac}
        for name, s in signals.items():
            row[name] = caught_at_coverage(s, d, frac, np.random.default_rng(1))
        rows.append(row)
    R["benchmark_signal_comparison"] = {"hours": int(len(B)), "dangerous_failures": int(d.sum()), "rows": rows,
                                        "note": "hours = all patient-hours of the four accidental-fault runs; "
                                                "dangerous failure = alert decision flipped by the fault AND wrong"}
    # natural operating point of the trust policy
    flagged = (B["trust"] != "HIGH").to_numpy()
    R["trust_policy_operating_point"] = {"hours_flagged_or_withheld_pct": float(flagged.mean()),
                                         "dangerous_caught": int((flagged & d).sum()), "dangerous": int(d.sum())}
    return R


def ablation(clean, B, std_thr, full_clean):
    d = B["danger"].to_numpy()
    n = int(d.sum())
    unc = B["ens_std"].to_numpy() > std_thr
    trust_flag = (B["trust"] != "HIGH").to_numpy()
    shift_flag = (B["shift_state"] != "LOW").to_numpy()
    fc_unc = (full_clean["ens_std"] > std_thr).mean()
    fc_tr = (full_clean["trust"] != "HIGH").mean()
    fc_low = (full_clean["trust"] == "LOW").mean()
    fc_shift = (full_clean["shift_state"] != "LOW").mean()
    E2 = json.load(open(RES / "experiments2.json"))["baselines"]
    get = {b["model"]: b for b in E2}
    single = get["Single LightGBM (uncalibrated)"]
    ens = get["5-model LightGBM ensemble (uncalibrated)"]
    cal = get["Ensemble + isotonic calibration (SepsisShield model)"]
    rows = [
        {"config": "1. Base: single LightGBM, raw probability", "auroc": single["auroc"], "ece": single["ece"],
         "dangerous_failures_caught": 0, "clean_hours_flagged_pct": 0.0, "clean_hours_withheld_pct": 0.0},
        {"config": "2. + 5-seed ensemble", "auroc": ens["auroc"], "ece": ens["ece"],
         "dangerous_failures_caught": 0, "clean_hours_flagged_pct": 0.0, "clean_hours_withheld_pct": 0.0},
        {"config": "3. + isotonic calibration", "auroc": cal["auroc"], "ece": cal["ece"],
         "dangerous_failures_caught": 0, "clean_hours_flagged_pct": 0.0, "clean_hours_withheld_pct": 0.0},
        {"config": "4. + ensemble-disagreement warning only", "auroc": cal["auroc"], "ece": cal["ece"],
         "dangerous_failures_caught": int((unc & d).sum()), "clean_hours_flagged_pct": float(fc_unc),
         "clean_hours_withheld_pct": 0.0},
        {"config": "5. + input-integrity checks (full trust layer: warn or withhold)", "auroc": cal["auroc"],
         "ece": cal["ece"], "dangerous_failures_caught": int((trust_flag & d).sum()),
         "dangerous_failures_withheld": int(((B["trust"] == "LOW").to_numpy() & d).sum()),
         "clean_hours_flagged_pct": float(fc_tr), "clean_hours_withheld_pct": float(fc_low)},
        {"config": "6. (counterfactual) also flag on shift MODERATE/HIGH", "auroc": cal["auroc"], "ece": cal["ece"],
         "dangerous_failures_caught": int(((trust_flag | shift_flag) & d).sum()),
         "clean_hours_flagged_pct": float(((full_clean["trust"] != "HIGH") | (full_clean["shift_state"] != "LOW")).mean()),
         "clean_hours_withheld_pct": float(fc_low),
         "note": "not deployed: shift stays advisory"},
    ]
    for r in rows:
        r["dangerous_failures"] = n
        r["coverage"] = r["dangerous_failures_caught"] / n
    return {"rows": rows, "shift_alone_catches": int((shift_flag & d).sum()),
            "shift_flags_on_clean_pct": float(fc_shift),
            "clean_reference": "full held-out test set (all hours of the 8,068 test patients)"}


# ------------------------------------------------------------------ G: simulated cohort shifts
def cohort_shift(raw, clean, rng):
    R = {"reference": perf(clean)}
    age = clean.merge(raw[["patient_id", "hour", "Age"]], on=["patient_id", "hour"])
    for name, mask in (("age_ge_80", age["Age"] >= 80), ("age_lt_45", age["Age"] < 45)):
        R[name] = {"patients": int(age.loc[mask, "patient_id"].nunique()), **perf(clean[mask.to_numpy()])}
    sims = {}
    r = raw.copy()
    r["HR"] += 20; r["Resp"] += 6; r["Temp"] += 1.0
    sims["shifted_vitals (HR+20, Resp+6, Temp+1 C)"] = r
    r = raw.copy()
    r[LABS] = r[LABS].where(rng.random((len(r), len(LABS))) >= 0.6)
    sims["missing_labs (60% of lab values removed)"] = r
    r = raw.copy()
    r["Age"] = (r["Age"] + 25).clip(upper=100)
    sims["older_cohort (Age +25 y, capped at 100)"] = r
    r, _ = add_noise(raw, 0.3, rng)
    sims["measurement_noise (0.3 SD)"] = r
    for name, rr in sims.items():
        R[name] = perf(run(rr))
        log("cohort", name, round(R[name]["auroc"], 4), round(R[name]["shift_moderate_pct"] + R[name]["shift_high_pct"], 4))
    # prevalence shift: keep all septic, subsample non-septic to triple prevalence (no input change)
    sep = clean.groupby("patient_id")["SepsisLabel"].transform("max") == 1
    ns_ids = clean.loc[~sep, "patient_id"].unique()
    keep_ns = set(rng.choice(ns_ids, len(ns_ids) // 3, replace=False))
    sub = clean[sep | clean["patient_id"].isin(keep_ns)]
    pr = perf(sub)
    pr["observed_rate"] = float(sub["SepsisLabel"].mean()); pr["mean_predicted"] = float(sub["risk"].mean())
    R["prevalence_x3 (non-septic patients subsampled)"] = pr
    R["reference"]["observed_rate"] = float(clean["SepsisLabel"].mean()); R["reference"]["mean_predicted"] = float(clean["risk"].mean())
    return R


# ------------------------------------------------------------------ calibration before / after
def calibration():
    te = pd.read_parquet(RES / "test_predictions.parquet")
    y = te["SepsisLabel"].to_numpy()
    out = {}
    for name, col in (("before", "prob_raw"), ("after", "prob")):
        p = te[col].to_numpy()
        q = np.quantile(p, np.linspace(0, 1, 16))
        idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 14)
        bins = [{"mean_predicted": float(p[idx == b].mean()), "observed_rate": float(y[idx == b].mean()),
                 "n": int((idx == b).sum())} for b in range(15) if (idx == b).any()]
        out[name] = {**hourly(y, p), "bins": bins}
    return out


def main(which):
    R = json.load(open(OUT)) if OUT.exists() else {}
    t0 = time.time()
    if "calibration" in which:
        R["calibration"] = calibration(); json.dump(R, open(OUT, "w"), indent=1)
        log("calibration", {k: round(R["calibration"][k]["ece"], 5) for k in ("before", "after")})
    raw, rng = cohort()
    clean = run(raw)
    log("cohort", raw["patient_id"].nunique(), "patients", len(raw), "hours")
    std_thr = pipeline.load()[3]["ens_std_thr"]
    if {"selective", "ablation"} & set(which):
        cache = ROOT / "data" / "processed"
        if (cache / "_bench.parquet").exists():
            B = pd.read_parquet(cache / "_bench.parquet")
            for kind in KINDS:                           # advance the RNG exactly as if the faults were re-created
                corrupt(raw, kind, rng)
        else:
            B = accidental_benchmark(raw, rng, clean)    # consumes the RNG exactly like abstention.py
            B.to_parquet(cache / "_bench.parquet")
        if (cache / "_full_clean.parquet").exists():
            full_clean = pd.read_parquet(cache / "_full_clean.parquet")
        else:
            full = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
            test_ids = set(pd.read_parquet(RES / "test_patients.parquet")["patient_id"])
            full = full[full["patient_id"].isin(test_ids)].sort_values(["patient_id", "hour"]).reset_index(drop=True)
            full_clean = run(full)
            del full
            full_clean.drop(columns=[c for c in full_clean if full_clean[c].dtype == object and c not in
                                     ("patient_id", "trust", "decision", "shift_state")]).to_parquet(cache / "_full_clean.parquet")
        R["selective"] = selective(clean, B, std_thr)
        R["selective"]["clean_risk_coverage_full_test"] = selective(full_clean, B, std_thr)["clean_risk_coverage"]
        R["ablation"] = ablation(clean, B, std_thr, full_clean)
        R["benchmark_check"] = {"dangerous_failures": int(B["danger"].sum()),
                                "caught": int(((B["trust"] != "HIGH") & B["danger"]).sum()),
                                "expected_from_abstention_json": [6481, 6790]}
        log("benchmark check", R["benchmark_check"])
        json.dump(R, open(OUT, "w"), indent=1)
    srng = np.random.default_rng(2026)                     # separate stream for the new perturbations
    if "sweeps" in which:
        R["sweeps"] = sweeps(raw, clean, srng); json.dump(R, open(OUT, "w"), indent=1)
    if "cohort_shift" in which:
        R["cohort_shift"] = cohort_shift(raw, clean, np.random.default_rng(2027)); json.dump(R, open(OUT, "w"), indent=1)
    R["cohort"] = {"patients": int(raw["patient_id"].nunique()), "hours": int(len(raw)),
                   "definition": "all septic held-out test patients + 2,500 random non-septic test patients (seed 7)"}
    R["runtime_s"] = time.time() - t0
    json.dump(R, open(OUT, "w"), indent=1, default=float)
    log("saved", OUT)


if __name__ == "__main__":
    main(sys.argv[1:] or ["calibration", "selective", "ablation", "sweeps", "cohort_shift"])
