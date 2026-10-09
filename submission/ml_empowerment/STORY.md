# SepsisShield AI · Devpost project story (ML Empowerment Build Challenge 3.0)

**Project title:** SepsisShield AI: Trust-Aware Early Warning for Sepsis

**Tagline:** Predicts sepsis early, and checks whether the data behind the prediction can be trusted.

Paste everything from "## Inspiration" down into the Devpost *About the project* field (Devpost Markdown).

---

## Inspiration

Sepsis early-warning evaluations typically focus on predictive performance, while input reliability receives much less
attention. Yet ICU data do go wrong. A thermometer reports Fahrenheit into a Celsius field. An external lab
sends SI units. A monitor feed freezes and copies the same values forward. A chart gets edited. A standard model can turn
each of these into a normal-looking risk score, sometimes with high confidence.

That gap is what I wanted to close: **model confidence is not the same as input trust.** Five models can agree
perfectly about data that cannot be right.

## What it does

SepsisShield predicts sepsis risk hour by hour from ICU time-series data, and performs **independent input-integrity
checks alongside the prediction path** to judge whether the inputs behind each prediction can be trusted. Model disagreement may reduce trust to REDUCED, but LOW trust and prediction withholding are triggered only by evidence from the clinical inputs.

* **Prediction path:** estimates the risk of sepsis in the next ~6–12 hours, with an alert threshold, a
  model-confidence signal (agreement between five models) and a SHAP explanation of what drove the risk.
* **Trust path:** an input-integrity layer that runs on the raw measurements, separately from the model. It checks
  physiological plausibility, blood-pressure consistency, abrupt jumps, frozen feeds and coordinated "normalising"
  shifts that look like an edited chart. It grades every patient-hour's inputs as **HIGH / REDUCED / LOW**.
* **Decision layer:** HIGH → **show the prediction** · REDUCED → **verify inputs** (shown with a warning) · LOW →
  **prediction withheld** until the data are checked. Every flag comes with a plain-language reason, for example
  "Physiologically implausible temperature — check units or sensor".

The Streamlit app opens on a **Judge Demo Mode**: three one-click scenarios, each a real held-out test patient scored
live.
* **A. Clean inputs (the normal case):** trust HIGH, prediction shown. This patient's first alert came 9 h before
  the recorded onset.
* **B. Accidental data fault** (°F thermometer): the models stay *Confident*, trust drops to LOW, and the false alert
  (28 h before onset, outside the useful window) is withheld.
* **C. Edited chart:** the model's alert disappears. The models stay *Confident*, the manipulation detector fires,
  and the prediction is withheld instead of giving a silent all-clear. This is a caught edit; most simulated edits are
  not caught (see limitations).

Four cards (Sepsis risk · Model confidence · Input trust · Final decision) make the difference between confidence
and trust visible in seconds.

A separate, advisory **distribution-shift** panel also flags when a patient's recent inputs differ from the training
data (LOW / MODERATE / HIGH). It never changes the prediction and does not establish that a prediction is wrong.

**Intended users (research setting):** ICU clinicians as decision *support* (not a replacement for clinical
judgement), clinical informatics teams checking data pipelines, and hospital ML / AI monitoring teams.

## How we built it

* **Data:** PhysioNet/CinC Challenge 2019. 40,336 ICU patients, 1,552,210 patient-hours and 2,932 sepsis cases from
  two U.S. hospital systems (de-identified, CC BY 4.0).
* **Prediction path:** 172 causal features that use only data up to the current hour (last values, informative
  missingness, 6 h / 12 h trends, SIRS/qSOFA/SOFA-style composites) → 5-seed LightGBM ensemble → isotonic calibration →
  alert threshold chosen on validation data to maximise the official challenge utility → TreeSHAP explanations.
* **Trust path:** every integrity threshold is fitted on clean training patients only. No corruption or attack data
  is used to fit it. LOW (withhold) is triggered only by input evidence. Unusual model disagreement can lower trust to
  REDUCED but can never cause a withhold.
* **Rigour:** patient-level train / validation / test split. The 8,068 test patients were never used for training,
  tuning, calibration or threshold selection. 62 automated tests check no look-ahead, split isolation, validation-only
  calibration and threshold, integrity behaviour, the utility metric, and that each Judge Demo scenario gives its
  documented decision. A browser test drives the live dashboard. All of it runs in GitHub Actions.
* **Judge mode:** `pip install -r requirements.txt` → `streamlit run app/app.py`. Models and demo patients ship with the
  repo, so no data download, retraining or credentials are needed. A public Streamlit Community Cloud deployment is
  linked under *Try it out*.
* **Stack:** Python · pandas · NumPy · LightGBM · scikit-learn · TreeSHAP contributions (LightGBM) · Plotly · Streamlit
  · pytest · GitHub Actions.

**Results (held-out test set, 95% patient-bootstrap CIs)**

| | |
|---|---|
| AUROC | **0.852** (0.840–0.865) |
| Sepsis patients alerted | **79.4%** (465 / 586) |
| Alerted ≥ 6 h before onset | **53.9%** (316 / 586) |
| Alert-changing wrong decisions from 4 simulated accidental fault types, flagged or withheld | **95.4%** (6,481 / 6,790); validation-cohort replication 94.7% |
| Clean patient-hours withheld | **0.46%** (1,438 / 309,270) |
| Deliberately edited inputs, flagged or withheld | **42.5%** (334 / 786), the main limitation; 26.5% (232 / 876) when the simulated edits stay inside normal physiological ranges |
| Cross-hospital AUROC (train on one hospital, test on the other) | **0.790 / 0.775** |

## Challenges we ran into

* **A trust layer that cries wolf is useless.** My first version marked a genuinely deteriorating patient "do not
  trust" at sepsis onset, because real deterioration produces abrupt changes. I made jumps *reduce* trust rather than
  veto the score, and measured model disagreement on the log-odds scale. Clean low-trust hours fell from 1.2% to 0.5%.
* **Measuring the right thing.** Counting "faults detected" rewards flagging everything. I measured *decisions*: of
  the patient-hours where a fault flipped the alert to a wrong one, how many were flagged or withheld? I then weighed
  that against the cost on clean data.
* **Auditing my own design.** The detector design was refined after looking at test-cohort results, so I re-ran the
  whole benchmark on validation patients that played no part in the design: 94.7% vs 95.4%.
* **Edited charts are hard.** An edit that stays inside normal physiology is nearly invisible in vitals alone. The
  benchmark gives 42.5%. When I noticed that the simulated edit sometimes produced physiologically implausible values, I re-ran it with
  the edits kept inside normal ranges, and coverage fell to 26.5%. I report both.

## Accomplishments that we're proud of

* Input trust built as its own **measured** component, with coverage on one side and cost on clean data on the other,
  not just a badge on a dashboard.
* A demo that proves its own point: in scenarios B and C the models are *Confident* and the inputs are wrong, and the
  dashboard shows both at once.
* On the *original* public data, the trust layer flagged 2,548 patient-hours with physiologically implausible values
  and 9,884 calcium values consistent with a possible unit mismatch.
* Honest reporting: confidence intervals, baselines, cross-hospital validation, subgroup results and a visible
  limitations panel in the app itself.

## What we learned

The accuracy number on the development hospital was the least informative number I produced. At an unseen hospital,
AUROC fell to 0.790 / 0.775 and the alert threshold stopped transferring. Model confidence never told me when the
inputs were broken. Trust has to be engineered, and validated, as its own component.

## What's next

* **Deliberate edits (42.5% in the benchmark, 26.5% for edits that stay physiologically plausible):** cross-signal consistency (does the charted heart rate still agree with lactate,
  white cell count and respiratory trends?) and EHR audit-trail provenance.
* **Cross-hospital shift:** local recalibration (already tested retrospectively) and multi-site training. The new
  distribution-shift panel makes shift visible, but it does not fix it: distribution-shift awareness indicates that an input pattern differs from the training distribution; it does not establish that a prediction is incorrect or clinically unsafe.
* **Real incidents:** evaluate the trust layer on logged real-world data-quality errors instead of simulated faults.
* **Clinical status:** SepsisShield is a research prototype, not a medical device, and has not been prospectively
  validated. The next step would be a silent, non-interventional evaluation with clinical partners.

**Disclaimer:** research prototype on retrospective, de-identified public data. Not a medical device and not for
clinical decision-making. The trust layer does not detect all bad data and does not prevent all wrong predictions.
