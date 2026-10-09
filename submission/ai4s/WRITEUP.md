# SepsisShield: Trustworthy AI for Early Sepsis Risk Detection Under Clinical Data Failures

**Category: End-to-End System**

**SepsisShield predicts sepsis risk early and separately checks whether the clinical data behind each prediction can be
trusted, so it can warn about or withhold a prediction built on unreliable inputs.**

> **Model uncertainty asks whether the models disagree. SepsisShield asks whether the data themselves deserve to be
> believed.** On 6,790 alert decisions that simulated accidental faults flipped to a wrong answer, ensemble disagreement
> alone caught 4.0%; the input-integrity layer raised coverage to 95.4%.

<!-- Links point to the main branch of the public GitHub repo: push before pasting into Kaggle, then test every link in a private window. -->

*Research prototype only. Not a medical device and not intended for clinical decision-making.*

## Project summary (≈270 words)

Sepsis models are usually tested on clean data. At the bedside, inputs can be wrong in ways a model cannot see: a
thermometer reporting °F into a °C field, a lab result in the wrong units, a frozen monitor feed, or a chart edited to
look normal. A model can stay confident on such inputs.

SepsisShield pairs a calibrated risk model (5-seed LightGBM, 172 causal features, isotonic calibration, TreeSHAP) with
an independent input-integrity layer. That layer grades each patient-hour's inputs HIGH, REDUCED or LOW and turns the
grade into an action: show the prediction, show it with a "verify inputs" warning, or withhold it. Only evidence from
the inputs can withhold. Model disagreement can only add a warning. A separate distribution-shift indicator is shown as
advisory context.

On 8,068 held-out PhysioNet 2019 patients the model reaches AUROC 0.852 and alerts 53.9% of septic patients at least
6 h before onset. Across four simulated accidental fault types, 95.4% of the alert decisions that faults flipped to a
wrong answer were flagged or withheld (validation replication 94.7%). Only 0.46% of clean hours were withheld. A
component ablation shows why the trust layer is needed:

- The base model, ensembling and calibration catch 0% of these failures.
- Ensemble disagreement alone catches 4.0%.
- Adding the input checks raises coverage to 95.4%.

We report the weak spots too:

- Deliberate edits: 42.5% caught, and 26.5% when the edits stay physiologically plausible.
- Random missingness changes decisions mostly without a flag.
- AUROC drops to 0.790 / 0.775 at an unseen hospital.
- A pre-registered cross-signal detector aimed at plausible edits added only 1.7 points, so it was not shipped.

A public, no-login demo shows six one-click scenarios on real held-out patients.

## Problem

A sepsis prediction can look normal even when the inputs behind it are wrong. Standard evaluation never tests this,
because validation data are assumed correct. The most dangerous outcome is a confident all-clear for a deteriorating
patient, or a false alarm caused by an impossible value.

## Innovation

- **Two signals:** *model confidence* (agreement among five models) and *input trust* (evidence from the raw inputs).
  Faults can leave the first unchanged while the second drops. In demo scenarios B and C the models stay confident
  while trust falls to LOW.
- **Asymmetric action rule:** only input evidence can withhold a prediction. Disagreement can only add a warning.
- **Explanations:** the trust decision is explained, not just the risk. The panel shows the SHAP drivers, the integrity
  check that fired, the shift state and the rule that was applied.
- **Evaluation:** a reproducible corruption benchmark measured by *dangerous failure coverage*, which counts only alert
  decisions flipped to a wrong answer.

**Same patient, same hour, same five models (demo scenario B).** A thermometer reports °F into a °C field. Sepsis risk
goes from 0.9% (no alert) to 3.9% (a false alert). Model confidence stays "Confident", input trust falls to LOW and the
prediction is withheld. All values are live outputs of the shipped pipeline on a held-out test patient.

![Clean vs corrupted: scenario B](https://raw.githubusercontent.com/Israt76/SepsisShield_GITHUB_Repo/main/results/screenshots/10_clean_vs_corrupted.png)

## System architecture

ICU data → **prediction path** (features → 5× LightGBM → isotonic calibration → SHAP) and **trust path** (plausibility,
blood-pressure consistency, jumps, coordinated "normalising" shift, frozen feed) → **decision layer**: SHOW / VERIFY
INPUTS / PREDICTION WITHHELD. The **distribution-shift indicator** is displayed beside the decision and never changes it.

![Architecture](https://raw.githubusercontent.com/Israt76/SepsisShield_GITHUB_Repo/main/results/figures/0_architecture.png)

## Results

| | |
|---|---|
| Held-out AUROC (8,068 patients) | **0.852** (95% CI 0.840–0.865) · XGBoost 0.853 · random forest 0.839 · CatBoost 0.839 · logistic regression 0.751 |
| Early warning | 53.9% of septic patients alerted ≥ 6 h before onset |
| Accidental faults flagged or withheld | **95.4%** (6,481 / 6,790); validation cohort 94.7% |
| Impossible values (HR 900, Temp 70 °C, SBP −20) | 1,232 / 1,232 injected hours withheld |
| Clean-data cost | 0.46% of clean hours withheld; 5.1% flagged or withheld in total |
| Component ablation | model + ensemble + calibration 0% → + disagreement 4.0% → + input checks 95.4% |
| Deliberate edits (limitation) | 42.5% (334 / 786); 26.5% (232 / 876) when edits stay physiologically plausible |
| Unseen hospital (limitation) | AUROC 0.790 / 0.775 |
| Operating point | Shipped 3%: 79.4% of septic patients alerted, 32.5 new alert episodes per 100 non-septic patient-days |
| Cross-signal detector (pre-registered) | Plausible edits 26.5% → 28.2% on test; below the +10-point bar, not shipped |

![Component ablation](https://raw.githubusercontent.com/Israt76/SepsisShield_GITHUB_Repo/main/results/figures/14_component_ablation.png)

## Demo video

*Add the YouTube link here.*

## Interactive demo

https://sepsisshieldapprepo-diqdknfnksrq5yyf6wkcrd.streamlit.app/ (no login). Scenarios:

- **A · Clean inputs:** the prediction is shown.
- **B · °F thermometer:** a false alert is withheld.
- **C · Edited chart:** an alert hidden by the edit is withheld instead of shown as an all-clear.
- **D · Unusual patient:** shift HIGH.
- **E · Early warning:** the alert is shown 8 h before onset.
- **F · Models disagree:** VERIFY INPUTS, never withheld.

## GitHub

https://github.com/Israt76/SepsisShield_GITHUB_Repo

## Technical report

[Technical report (PDF)](https://github.com/Israt76/SepsisShield_GITHUB_Repo/blob/main/docs/technical_report.pdf): 20 pages. It covers the dataset, architecture, model
comparison, trust layer, calibration, shift detection, robustness sweeps, selective prediction, ablations, cross-hospital
and subgroup results, failure cases and limitations.

## Reproducibility

- `make test` runs 75 automated tests and works without the raw data.
- `./run_all.sh` rebuilds every result from the raw PhysioNet files in about 75 min on 2 CPU cores.
- `tools/report/build_report.py` regenerates the report from the result files.
- Splits, seeds, thresholds, calibrators and integrity rules are fitted on training or validation patients only.
