"""Distribution-shift awareness: does this patient's recent input pattern look unlike the training data?

Method (pre-specified, transparent, no extra model):
  * Features: the 20 model features with the highest mean LightGBM gain across the 5 shipped models, excluding ICULOS
    (a pure time index whose range reflects stay length, not physiology or care practice).
  * Reference: for each feature, the 0.5th and 99.5th percentiles of its non-missing values on TRAINING patients only.
  * Hourly out-of-range fraction: among those features that are observed at an hour, the share outside that range.
  * Shift score S(t): mean of the hourly fraction over the last 6 hours up to and including t (causal: never uses
    later hours).
  * States: thresholds are quantiles of S on training patient-hours:
        LOW       S <= q95              (as typical as 95% of training patient-hours)
        MODERATE  q95 < S <= q99
        HIGH      S > q99               (more unusual than 99% of training patient-hours)
It is an awareness signal only. It never changes the risk, the alert, input trust or the final decision.

Run `python src/shift.py` (needs the raw data) to refit models/shift_reference.json and results/shift_evaluation.json.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "models" / "shift_reference.json"
TOP_K, WINDOW, LO_Q, HI_Q = 20, 6, 0.005, 0.995
STATES = ("LOW", "MODERATE", "HIGH")
TEXT = {
    "LOW": "This patient's recent inputs look like the training data.",
    "MODERATE": "Some recent inputs are unusual compared with the training data. Interpret the risk with extra care.",
    "HIGH": "Recent inputs are more unusual than 99% of training patient-hours. The model is working outside familiar data.",
}
_ref = None


def load():
    """The shipped reference (models/shift_reference.json), fitted on training patients only."""
    global _ref
    if _ref is None:
        _ref = json.load(open(REF))
    return _ref


def select_features(boosters, exclude=("ICULOS",), k=TOP_K):
    names = boosters[0].feature_name()
    gain = np.mean([b.feature_importance("gain") for b in boosters], axis=0)
    order = [names[i] for i in np.argsort(-gain) if names[i] not in exclude]
    return order[:k]


def hourly_fraction(feats: pd.DataFrame, ref: dict) -> pd.Series:
    cols = ref["features"]
    X = feats[cols].to_numpy(float)
    lo, hi = np.array(ref["lo"]), np.array(ref["hi"])
    obs = ~np.isnan(X)
    out = obs & ((X < lo) | (X > hi))
    n_obs = obs.sum(1)
    frac = np.where(n_obs > 0, out.sum(1) / np.maximum(n_obs, 1), 0.0)
    return pd.Series(frac, index=feats.index)


def shift_score(feats: pd.DataFrame, ref: dict, pid: pd.Series | None = None) -> pd.Series:
    """Causal rolling mean of the hourly out-of-range fraction (per patient if pid is given)."""
    f = hourly_fraction(feats, ref)
    if pid is None:
        return f.rolling(ref["window"], min_periods=1).mean()
    return f.groupby(pid.to_numpy()).transform(lambda s: s.rolling(ref["window"], min_periods=1).mean())


def state_of(score, ref: dict):
    s = np.asarray(score, float)
    return np.where(s > ref["thr_high"], "HIGH", np.where(s > ref["thr_moderate"], "MODERATE", "LOW"))


def assess(feats: pd.DataFrame, ref: dict | None = None) -> pd.DataFrame:
    """Per-hour score and state for ONE patient's feature frame (as returned by pipeline.score)."""
    ref = ref or load()
    s = shift_score(feats, ref)
    return pd.DataFrame({"shift_score": s.to_numpy(), "shift_state": state_of(s, ref)}, index=feats.index)


def explain(feats: pd.DataFrame, hour_pos: int, ref: dict | None = None, labeler=str, max_items=4) -> list[str]:
    """Plain-language list of the features outside the training range within the scoring window."""
    ref = ref or load()
    w = feats.iloc[max(0, hour_pos - ref["window"] + 1): hour_pos + 1]
    msgs = []
    for j, c in enumerate(ref["features"]):
        v = w[c].to_numpy(float)
        lo, hi = ref["lo"][j], ref["hi"][j]
        above, below = np.nansum(v > hi), np.nansum(v < lo)
        if above or below:
            n = int(above + below)
            side = "above" if above >= below else "below"
            val = np.nanmax(v) if side == "above" else np.nanmin(v)
            msgs.append((n, f"{labeler(c)} = {val:.4g}, {side} the training range "
                            f"({lo:.3g} to {hi:.3g}) in {n} of the last {len(w)} hours"))
    msgs.sort(key=lambda t: -t[0])
    return [m for _, m in msgs[:max_items]]


def fit_reference(train_feats: pd.DataFrame, features: list[str], pid: pd.Series) -> dict:
    X = train_feats[features]
    lo = [float(X[c].quantile(LO_Q)) for c in features]
    hi = [float(X[c].quantile(HI_Q)) for c in features]
    ref = {"features": features, "lo": lo, "hi": hi, "window": WINDOW, "lo_q": LO_Q, "hi_q": HI_Q}
    s = shift_score(train_feats, ref, pid)
    ref["thr_moderate"] = float(np.quantile(s, 0.95))
    ref["thr_high"] = float(np.quantile(s, 0.99))
    return ref


def _rates(states):
    st = pd.Series(states)
    return {k: float((st == k).mean()) for k in STATES}


if __name__ == "__main__":
    import sys
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, str(Path(__file__).parent))
    from train import split_patients

    cfg = json.load(open(ROOT / "models" / "config.json"))
    boosters = [lgb.Booster(model_file=str(ROOT / "models" / f"lgb_seed{s}.txt")) for s in cfg["seeds"]]
    features = select_features(boosters)
    raw = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet", columns=["patient_id", "hospital", "hour", "SepsisLabel"])
    tr, va, te = split_patients(raw)
    F = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet", columns=["patient_id", "hospital", "hour"] + features)
    F = F.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    trF = F[F["patient_id"].isin(tr)].reset_index(drop=True)
    ref = fit_reference(trF, features, trF["patient_id"])
    ref["fitted_on"] = {"cohort": "training", "patients": int(trF["patient_id"].nunique()), "hours": int(len(trF))}
    json.dump(ref, open(REF, "w"), indent=1)

    ev = {"method": "see src/shift.py docstring", "features": features,
          "thresholds": {"moderate": ref["thr_moderate"], "high": ref["thr_high"]}}
    # 1) held-out test cohort, reference = all training patients
    teF = F[F["patient_id"].isin(te)].reset_index(drop=True)
    te_s = shift_score(teF, ref, teF["patient_id"])
    te_state = state_of(te_s, ref)
    ev["train_rates"] = _rates(state_of(shift_score(trF, ref, trF["patient_id"]), ref))
    ev["test_rates"] = _rates(te_state)
    ev["test_rates_by_hospital"] = {h: _rates(te_state[teF["hospital"].to_numpy() == h]) for h in ("A", "B")}
    # 2) domain-shift check: reference refitted on ONE hospital's training patients, applied to both test hospitals
    ev["cross_hospital"] = {}
    for h, other in (("A", "B"), ("B", "A")):
        sub = trF[trF["hospital"] == h].reset_index(drop=True)
        r_h = fit_reference(sub, features, sub["patient_id"])
        st = state_of(shift_score(teF, r_h, teF["patient_id"]), r_h)
        hosp = teF["hospital"].to_numpy()
        ev["cross_hospital"][f"reference_{h}"] = {f"test_{h}_same": _rates(st[hosp == h]),
                                                  f"test_{other}_other": _rates(st[hosp == other])}
    # 3) exploratory: discrimination and calibration-in-the-large of the shipped model within each state (test)
    P = pd.read_parquet(ROOT / "results" / "test_predictions.parquet", columns=["patient_id", "hour", "SepsisLabel", "prob"])
    M = teF[["patient_id", "hour"]].assign(state=te_state).merge(P, on=["patient_id", "hour"], how="inner")
    ev["by_state_test"] = {}
    for k in STATES:
        d = M[M["state"] == k]
        y = d["SepsisLabel"].to_numpy()
        ev["by_state_test"][k] = {
            "hours": int(len(d)), "patients": int(d["patient_id"].nunique()),
            "auroc": float(roc_auc_score(y, d["prob"])) if 0 < y.sum() < len(y) else None,
            "observed_rate": float(y.mean()) if len(d) else None, "mean_predicted": float(d["prob"].mean()) if len(d) else None}
    # patient-level bootstrap (500 resamples) for the per-state AUROC: hours of one patient are correlated
    rng = np.random.default_rng(0)
    for k in STATES:
        d = M[M["state"] == k]
        g = {p: (x["SepsisLabel"].to_numpy(), x["prob"].to_numpy()) for p, x in d.groupby("patient_id")}
        ids, boots = np.array(list(g)), []
        for _ in range(500):
            pick = rng.choice(ids, len(ids), replace=True)
            y = np.concatenate([g[i][0] for i in pick]); q = np.concatenate([g[i][1] for i in pick])
            if 0 < y.sum() < len(y):
                boots.append(roc_auc_score(y, q))
        ev["by_state_test"][k]["auroc_ci95"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        ev["by_state_test"][k]["septic_hours"] = int(d["SepsisLabel"].sum())
    json.dump(ev, open(ROOT / "results" / "shift_evaluation.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in ref.items() if k not in ("lo", "hi")}, indent=1))
    print(json.dumps(ev, indent=1))
