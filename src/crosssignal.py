"""Cross-signal consistency detector (AI4S experiment; evaluated in src/exp_crosssignal.py).

Idea: a value can be individually plausible yet inconsistent with the rest of the patient's physiology. For each core
vital V we fit a regression model that predicts V at hour t from
  * the OTHER vitals at hour t,
  * the last observed laboratory values,
  * every vital's own baseline 7-24 h earlier (so the model knows where this patient was before the last 6 hours),
  * age, sex, ICU type and time in ICU.
The model never sees V at t or in the last 6 hours, so an edit that changes V (and its recent history) without the
rest of the physiology following leaves a large residual.

Score per hour: mean |standardized residual| over the vitals observed at that hour, where each residual is divided by
the robust (MAD) residual scale on training patients. The flag threshold is the 99th percentile of the score on clean
VALIDATION patient-hours. A flag can only lower trust to REDUCED (warn), never to LOW: it is statistical evidence,
not proof that a value is impossible.

All models are fitted on TRAINING patients only.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "models" / "crosssignal"
TARGETS = ["HR", "Resp", "Temp", "SBP", "MAP", "O2Sat"]
VITALS_ALL = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS_KEEP = ["Lactate", "WBC", "Creatinine", "pH", "HCO3", "BUN", "Platelets", "Hgb", "Glucose", "FiO2", "PaCO2",
             "Potassium", "Bilirubin_total"]
STATIC = ["Age", "Gender", "Unit1", "ICULOS"]
PARAMS = dict(objective="huber", alpha=1.0, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=2, verbose=-1, seed=11)


def base_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Shared inputs. raw sorted by (patient_id, hour)."""
    pid = raw["patient_id"]
    g = raw.groupby(pid, sort=False)
    X = pd.DataFrame(index=raw.index)
    for v in VITALS_ALL:
        X[v + "_now"] = raw[v].astype("float32")
        # baseline: mean over hours t-24 .. t-7 (causal, excludes the last 6 hours)
        s = g[v].transform(lambda x: x.shift(7).rolling(18, min_periods=1).mean())
        X[v + "_base"] = s.astype("float32")
    labs = g[LABS_KEEP].ffill()
    for c in LABS_KEEP:
        X[c] = labs[c].astype("float32")
    for c in STATIC:
        X[c] = raw[c].astype("float32")
    return X


def design(X: pd.DataFrame, target: str) -> pd.DataFrame:
    """Inputs for one target: everything except the target's own current value."""
    return X.drop(columns=[target + "_now"])


def fit(train_raw: pd.DataFrame, n_max=600_000, seed=0):
    X = base_frame(train_raw)
    rng = np.random.default_rng(seed)
    meta = {"targets": TARGETS, "scale": {}, "n_train_rows": {}}
    DIR.mkdir(parents=True, exist_ok=True)
    for t in TARGETS:
        y = train_raw[t].to_numpy(float)
        ok = np.where(~np.isnan(y))[0]
        if len(ok) > n_max:
            ok = np.sort(rng.choice(ok, n_max, replace=False))
        D = design(X, t).iloc[ok]
        # 10% of the sampled rows for early stopping (still training patients)
        cut = int(len(ok) * 0.9)
        dtr = lgb.Dataset(D.iloc[:cut], y[ok][:cut])
        dva = lgb.Dataset(D.iloc[cut:], y[ok][cut:], reference=dtr)
        m = lgb.train(PARAMS, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        res = y[ok] - m.predict(D)
        mad = float(np.median(np.abs(res - np.median(res))) * 1.4826)
        m.save_model(str(DIR / f"{t}.txt"), num_iteration=m.best_iteration)
        meta["scale"][t] = mad
        meta["n_train_rows"][t] = int(len(ok))
    json.dump(meta, open(DIR / "meta.json", "w"), indent=1)
    return meta


_cache = {}


def load():
    if "m" not in _cache:
        meta = json.load(open(DIR / "meta.json"))
        _cache["m"] = (meta, {t: lgb.Booster(model_file=str(DIR / f"{t}.txt")) for t in meta["targets"]})
    return _cache["m"]


def score(raw: pd.DataFrame) -> np.ndarray:
    """Hourly cross-signal inconsistency score (NaN-safe; 0 where no target vital is observed)."""
    meta, models = load()
    raw = raw.reset_index(drop=True)
    X = base_frame(raw)
    Z = np.full((len(raw), len(meta["targets"])), np.nan)
    for j, t in enumerate(meta["targets"]):
        y = raw[t].to_numpy(float)
        p = models[t].predict(design(X, t))
        Z[:, j] = np.abs(y - p) / meta["scale"][t]
    with np.errstate(all="ignore"):
        s = np.nanmean(np.clip(Z, 0, 10), axis=1)
    return np.nan_to_num(s, nan=0.0)


def flags(raw: pd.DataFrame, thr: float | None = None) -> np.ndarray:
    thr = thr if thr is not None else json.load(open(DIR / "meta.json"))["threshold"]
    return score(raw) > thr
