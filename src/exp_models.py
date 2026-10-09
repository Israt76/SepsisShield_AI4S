"""AI4S experiment A: compare more tabular model families on the SAME patient split as the deployed model.

Adds Random Forest, XGBoost and CatBoost to the existing comparison in experiments2.json (SIRS, qSOFA, logistic
regression, LightGBM). Every model gets the same treatment: fit on train patients, early stopping / isotonic
calibration / utility-optimal threshold on VALIDATION patients only, one evaluation on the held-out TEST patients.
These are comparison models only; the deployed SepsisShield model (models/) is not changed.

Output: results/exp_models.json
"""
from pathlib import Path
import json, time, sys
import numpy as np
import pandas as pd

from metrics import full_report, best_threshold
from experiments2 import iso_fit
from train import split_patients

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
KEYS = ("auroc", "auprc", "brier", "ece", "utility", "patient_sensitivity", "patient_specificity",
        "pct_detected_ge6h_early", "threshold")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def evaluate(name, dva, dte, pv, pt, fit_s, note=""):
    iso = iso_fit(pv, dva["SepsisLabel"].to_numpy())
    cv, ct = iso.predict(pv), iso.predict(pt)
    thr, _ = best_threshold(dva, cv)
    r = full_report(dte, ct, thr)
    row = {"model": name, **{k: r[k] for k in KEYS}, "train_seconds": round(fit_s), "note": note}
    log(name, {k: round(row[k], 4) for k in ("auroc", "auprc", "utility", "ece")})
    return row


def main(which):
    cols = json.load(open(ROOT / "models" / "config.json"))["features"]   # the 172 model features
    feats = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet", columns=["patient_id", "hospital", "hour", "SepsisLabel"] + cols)
    feats[cols] = feats[cols].astype(np.float32)
    tr, va, te = split_patients(feats)
    m_tr, m_va, m_te = (feats["patient_id"].isin(s).to_numpy() for s in (tr, va, te))
    dva, dte = feats.loc[m_va].reset_index(drop=True), feats.loc[m_te].reset_index(drop=True)
    y = feats["SepsisLabel"].to_numpy()
    Xtr, ytr = feats.loc[m_tr, cols].to_numpy(np.float32), y[m_tr]
    Xva, Xte = dva[cols].to_numpy(np.float32), dte[cols].to_numpy(np.float32)
    log(f"train {m_tr.sum():,} rows, val {m_va.sum():,}, test {m_te.sum():,}, {len(cols)} features")
    keep = ["patient_id", "hour", "SepsisLabel"]
    dva, dte = dva[keep], dte[keep]
    del feats                                           # keep peak memory low (2-core / 8 GB machine)
    out_path = RES / "exp_models.json"
    R = json.load(open(out_path)) if out_path.exists() else {"rows": []}
    done = {r["model"] for r in R["rows"]}

    if "rf" in which:
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(0)
        sub = rng.choice(len(Xtr), 400_000, replace=False)          # same sub-sample size as the LR baseline
        t = time.time()
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=100, max_features="sqrt",
                                    n_jobs=2, random_state=0).fit(Xtr[sub], ytr[sub])
        s = time.time() - t
        R["rows"] = [r for r in R["rows"] if not r["model"].startswith("Random forest")]
        R["rows"].append(evaluate("Random forest (+ isotonic)", dva, dte, rf.predict_proba(Xva)[:, 1],
                                  rf.predict_proba(Xte)[:, 1], s, "300 trees, min_samples_leaf=100, 400k-row train sub-sample"))
        del rf
        json.dump(R, open(out_path, "w"), indent=1)

    if "xgb" in which:
        import xgboost as xgb
        t = time.time()
        dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=256)
        dv = xgb.QuantileDMatrix(Xva, dva["SepsisLabel"].to_numpy(), ref=dtr)
        p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", eta=0.03, max_depth=6,
                 min_child_weight=50, subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0, nthread=2, seed=11)
        b = xgb.train(p, dtr, 3000, evals=[(dv, "val")], early_stopping_rounds=150, verbose_eval=False)
        s = time.time() - t
        it = (0, b.best_iteration + 1)
        R["rows"] = [r for r in R["rows"] if not r["model"].startswith("XGBoost")]
        R["rows"].append(evaluate("XGBoost (+ isotonic)", dva, dte, b.predict(dv, iteration_range=it),
                                  b.predict(xgb.QuantileDMatrix(Xte, ref=dtr), iteration_range=it), s, f"{b.best_iteration + 1} rounds"))
        del b, dtr
        json.dump(R, open(out_path, "w"), indent=1)

    if "cat" in which:
        from catboost import CatBoostClassifier
        t = time.time()
        cb = CatBoostClassifier(iterations=2000, learning_rate=0.06, depth=6, eval_metric="AUC", od_type="Iter",
                                od_wait=150, thread_count=2, random_seed=11, verbose=0, use_best_model=True)
        cb.fit(Xtr, ytr, eval_set=(Xva, dva["SepsisLabel"].to_numpy()))
        s = time.time() - t
        R["rows"] = [r for r in R["rows"] if not r["model"].startswith("CatBoost")]
        R["rows"].append(evaluate("CatBoost (+ isotonic)", dva, dte, cb.predict_proba(Xva)[:, 1],
                                  cb.predict_proba(Xte)[:, 1], s, f"{cb.get_best_iteration() + 1} iterations"))
        json.dump(R, open(out_path, "w"), indent=1)

    R["protocol"] = ("same 70/10/20 patient split as the deployed model (train.split_patients, seed 0); early stopping, "
                     "isotonic calibration and utility-optimal threshold chosen on validation patients only; "
                     "single evaluation on held-out test patients")
    json.dump(R, open(out_path, "w"), indent=1)
    log("saved", out_path)


if __name__ == "__main__":
    main(sys.argv[1:] or ["xgb", "rf", "cat"])
