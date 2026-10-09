#!/usr/bin/env bash
# Reproduce every number in the README from the raw PhysioNet files (~75 min on 2 CPU cores).
set -euo pipefail
cd "$(dirname "$0")/src"
python3 data.py                  # 40,336 PSV files -> data/processed/hourly.parquet
python3 features.py              # causal hourly features -> data/processed/features.parquet
python3 train.py main cross ablation subgroups   # models/ + results/results.json
python3 integrity.py             # fit integrity thresholds on training patients
python3 benchmark_integrity.py   # results/integrity_benchmark.json
python3 experiments2.py          # baselines, bootstrap CIs, threshold sweep, recalibration
python3 subgroup_ci.py           # subgroup audit with bootstrap CIs
python3 abstention.py            # trust-aware abstention, test cohort
python3 abstention.py val        # replication on the validation cohort (never used for integrity design)
python3 shift.py                 # distribution-shift reference + results/shift_evaluation.json
rm -f ../data/processed/_bench.parquet ../data/processed/_full_clean.parquet   # caches used by exp_robustness.py
python3 exp_models.py            # needs: pip install -r requirements-experiments.txt · AI4S: random forest, XGBoost, CatBoost on the same split (comparison only)
python3 exp_robustness.py        # AI4S: robustness sweeps, selective prediction, component ablation, cohort shifts
python3 exp_operating_points.py  # AI4S: threshold / alert-burden trade-off
python3 exp_crosssignal.py fit val && python3 exp_crosssignal.py test   # AI4S: cross-signal detector (pre-registered; not shipped)
python3 make_figures.py          # results/figures/*.png
python3 make_figures_ai4s.py     # results/figures/11-16
python3 prepare_demo.py          # app/demo_*.parquet (held-out test patients only)
echo "Done. Launch the dashboard with: streamlit run app/app.py"
