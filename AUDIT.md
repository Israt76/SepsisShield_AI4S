# Pre-submission audit

An independent re-check of every central claim before submission. Each item says what was checked, how, the result, and
what (if anything) changed. Scripts are in `tools/` and `tests/`, so anyone can re-run them.

## 1. Tests and no-leakage

| Check | Method | Result |
|---|---|---|
| Test suite from the submitted package | Unzipped the package into an empty folder, fresh virtualenv, `pip install -r requirements.txt`, `pytest` | 28 passed, 1 skipped (needs raw data) |
| Do the leakage tests have teeth? | **Mutation testing:** planted five leaks in `src/features.py` (back-fill, centred rolling window, stay-level lab count, stay-level lab max, diff against a future value) | All five caught (7 of 13 leakage tests fail each time); original code passes all 13. Re-run: `python tools/mutation_test.py` |
| Future values | Every one of the 34 raw variables perturbed after hour *t*, including filling future gaps | Features at hours ≤ *t* unchanged |
| Labels | Flip `SepsisLabel` | Features unchanged |
| Integrity layer causal | Trust flags on truncated vs full histories | Identical at shared hours |

*Changed:* the future-perturbation test now covers all raw variables (previously 8). Label-invariance and integrity-causality
tests were added (29 tests in total; 41 after section 14; 55 after section 15; 62 after the judge-facing panels).

## 2. The central metric: alert-changing accidental faults flagged or withheld

**Independent recomputation.** A fresh script (not importing the experiment's classification code) re-ran the pipeline on
clean and corrupted data and reproduced the originally reported 8,439 / 8,852 = 95.3% exactly.

**Definition weakness found and fixed.** The original count included 2,062 patient-hours where a fault created a *new
alert inside a septic patient's useful window* (12 h before to 3 h after onset). That decision is not wrong, so it
should not count as a dangerous failure. The definition is now:

> An **alert-changing fault** is a patient-hour inside the corruption window where the fault flipped the alert decision
> **and the new decision is wrong**: it suppressed a timely sepsis alert, or created an alert outside any useful window.

| | Before (flip only) | **After (flip and wrong)** |
|---|---|---|
| Accidental faults flagged or withheld | 8,439 / 8,852 = 95.3% | **6,481 / 6,790 = 95.4%** |
| Deliberately edited inputs | 372 / 864 = 43.1% | **334 / 786 = 42.5%** |

**Attributable coverage.** Counting only flags caused by the fault itself (the same hour was HIGH trust on clean data)
gives 89.4% for accidental faults and 34.5% for deliberate edits. Both numbers are reported.

**Design leakage found and addressed.** Thresholds were always fitted on training data, but the detector *design* was
revised after inspecting test-cohort benchmark results. The full benchmark was therefore re-run on validation patients
who played no part in any integrity design decision (1,543 patients: all septic plus 1,250 random non-septic, as in the
test benchmark; the clean-cost row uses all 4,034 validation patients). *Corrected 30 Sep 2026: this sentence
previously said the benchmark used all 4,034 validation patients.*

| | Test cohort | Validation cohort |
|---|---|---|
| Accidental faults flagged or withheld | 95.4% (6,481 / 6,790) | 94.7% (3,313 / 3,498) |
| Deliberately edited inputs | 42.5% (334 / 786) | 48.0% (212 / 442) |
| Clean patient-hours withheld | 0.46% | 0.48% |

**Scope wording.** Every mention now reads "95.4% of alert-changing *accidental* data faults *in our corruption
benchmark*", with the deliberate-edit result (42.5%) next to it. It never implies coverage of deliberate attacks or of
fault types that were not simulated.

## 3. Clean abstention rate

The originally reported 0.48% came from the benchmark sample, which over-represents septic patients (19% vs 7%). It was
recomputed on **all 8,068 test patients with no corruption**:

* **0.46% of patient-hours withheld** (1,438 / 309,270); 4.7% flagged.
* **12% of patients** have at least one withheld hour, typically a single hour (median 1).
* Causes of clean withholds: 48% physically impossible values or blood-pressure combinations *in the original
  records* (withholding is correct); 52% the coordinated-shift detector firing on real patients (likely false flags).
* `decision == WITHHELD` coincides exactly with `trust == LOW` (verified on every row).

## 4. Bootstrap confidence intervals

The code resamples **patients** (with all their hours), not hours. An independent reimplementation with a different
random stream gave an AUROC interval of 0.839–0.864, matching the shipped 0.840–0.865. For contrast, a naive hour-level
bootstrap gives 0.848–0.857, about 2.7× too narrow. The README now explains this.

## 5. Baselines

SIRS (0.635), qSOFA-partial (0.576), logistic regression (0.751), single LightGBM (0.852), ensemble (0.853) and the final
model (0.852) all use **the same 8,068 test patients, the same hourly rows and the same metric code**
(`experiments2.py`, one `dte` frame). Thresholds for learned models were chosen on validation data. The clinical rules
use fixed standard cut-offs, with no tuning. *Disclosed:* logistic regression was trained on a random 400,000-hour
subsample of the training set.

## 6. Threshold sensitivity: test-set leakage?

The submitted threshold (3.0%) comes from `models/config.json`. `tests/test_isolation.py` re-derives it from the shipped
validation predictions alone. The test-set sweep is reporting only, and its best threshold (2.0%) was **not** adopted.

## 7. Hospital recalibration

The zero-shot external results (AUROC 0.790 / 0.775) use no destination-hospital data. The recalibration experiment
**does use destination-hospital outcome labels** (a labelled 20% sample) to refit the calibrator and threshold only. The
README and Devpost now say this explicitly and label it as separate from zero-shot validation.

## 8. Subgroups

Patient counts, sepsis cases and patient-level prevalence were recomputed independently from `test_predictions.parquet`;
all 11 groups match `subgroups_ci.json`. The README table now includes prevalence and both intervals, generated directly
from the JSON (a hand-typed version was diffed and replaced).

## 9. Dashboard abstention behaviour

`tools/dashboard_abstention_check.py` drives the running dashboard in a real browser:

| Scenario | Expected | Observed |
|---|---|---|
| Clean, trusted inputs | shown | shown |
| °F fault inside window | withheld | withheld ("Withheld", "Not issued", LOW) |
| Edited chart, hour 48 | withheld | withheld |
| Edited chart, hour 57 | shown with warning | shown with warning |

Added to GitHub Actions. *Extended on 30 Sep 2026 (section 14):* the check now reads the Final Decision card, adds the
°F case one hour later (VERIFY INPUTS), the default page, and a click on each Judge Demo button: 10 / 10 pass.

## 10. Clean install and CI

The submitted package was installed into a new virtualenv with the pinned `requirements.txt`; the tests pass and the
dashboard health endpoint responds. The workflow YAML parses, and its steps are install → tests → dashboard health →
browser abstention test. *Not verified:* an actual GitHub Actions run, which requires the repository to be pushed.

## 11. Claims against artifacts

`tools/check_claims.py` recomputes 37 headline numbers directly from `results/*.json` and checks each appears in the
README, Devpost, model card or video script (and, since section 14, that every number in the ML Empowerment documents is
one of the verified values). It also fails if superseded numbers (8,439, 8,852, 95.3%, 372/864, 43.1%) or
removed phrases ("costs a life", "on par with") reappear. Result: 0 problems.

## 12. Video

Sampled every 3 s (46 frames) and reviewed. *Fixed:* a 1.3-second explanation shot that dissolved into a double
exposure; crossfades shortened; the benchmark figure (too small at video size) replaced by an evidence card that
reveals 95.4% → 0.46% → 42.5% as each is narrated, with the definition on screen; the end card now holds 3.5 s after
the narration and states the deliberate-edit result beneath the three numbers. Audio −16.6 dB mean, −1.4 dB peak.

**Clinical opening (added after the audit):** a 7.5 s code-drawn animation (`tools/video/intro.html`,
`render_intro.py`). Its numbers match the real °F case for held-out patient p119917 (temperature 36.9 → 98.4 "°C", risk 0.6% →
3.9%, alert threshold 3.0%, then LOW trust and withheld) and it is labelled on screen as an illustration.

## 13. References

Checked for existence and details; DOIs added where the metadata was confirmed (Kahn 2016 eGEMs; Adams 2022 *Nature
Medicine*).

## 14. Competition upgrade audit (ML Empowerment Build Challenge 3.0, 30 Sep 2026)

The dashboard gained a Judge Demo Mode, a four-card first screen (Sepsis risk · Model confidence · Input trust · Final
decision), a *Why was this flagged?* panel, and tabs for the design comparison and results/limitations. No model,
threshold, calibrator, integrity rule or result file was changed. No test was changed or removed.

| Check | Method | Result |
|---|---|---|
| Numbers in the brief vs artifacts | 14 values (40,336 · 1,552,210 · 2,932 · 8,068 · 0.852 · 0.840–0.865 · 79.4% · 53.9% · 95.4% · 6,481 / 6,790 · 0.46% · 42.5% · 334 / 786 · external AUROC) recomputed from `results/*.json` and a recount of the raw data | all match; no conflicts. Note: the B→A external AUROC is 0.7749, reported as 0.775 (3 dp); "0.78" in older text is double rounding (0.77 at 2 dp) |
| Model outputs unchanged | Risk at each Judge Demo scenario hour pinned to 1e-9 in `tests/test_judge_demo.py` | pass |
| Demo scenarios truthful | Each scenario's decision, trust, alert, confidence and firing check asserted from the shipped pipeline; the fault's effect vs clean data asserted (B creates a false alert, C hides an alert) | pass |
| Test suite | `pytest` (judge mode / with raw data) | 40 passed + 1 skipped / 41 passed |
| Browser | `tools/dashboard_abstention_check.py`: 5 links, the default page, 4 button clicks | 10 / 10 pass |
| Manual UI | category and patient change, hour slider, .psv upload, all five tabs, 390 px phone width (no horizontal scroll), 1280×800 laptop (all four cards above the fold) | no errors |
| Deployment | `requirements.txt` and `packages.txt` unchanged; no new dependencies; relative paths only; no secrets | unchanged from the verified deployment |
| Overclaims | `tools/check_claims.py` fails on unnegated "clinically validated", "saves lives", "guaranteed", "production-ready", "detects all" etc. | 0 problems |
| Deliberate-edit realism (raised by an independent review) | The simulated edit subtracts fixed amounts and can create impossible values (1,177 of 30,705 edited hours have Resp < 4). `tools/masking_realism_check.py` rebuilds the identical benchmark windows: the original edit reproduces 334 / 786 exactly; with edited vitals kept inside normal ranges, coverage is **26.5% (232 / 876)** | disclosed next to the 42.5% in README, app limitations, story; the reported benchmark numbers are unchanged |
| Judge Demo scenario C | At hour 48 the withhold also relied on an impossible Resp value created by the simulation; moved to hour 46, where only the coordinated-shift detector fires and the edited values are physiologically possible (HR 58, Resp 10) | done |
| Stale media | old-layout screenshots and gallery images removed and regenerated (`tools/screenshots.py`, `tools/gallery.py`) | done |

## 15. Distribution-shift awareness (added 30 Sep 2026)

| Check | Method | Result |
|---|---|---|
| Method pre-specified | Top-20 features by mean gain (ICULOS excluded), training 0.5–99.5th percentile ranges, 6-h causal window, q95 / q99 thresholds: fixed before any held-out result was looked at | `src/shift.py` docstring |
| Training-only reference | `models/shift_reference.json` records 28,234 training patients; asserted in `tests/test_shift.py` | pass |
| Thresholds transfer | flagged (MODERATE or HIGH) share, training vs held-out test | 4.8% vs 4.6% |
| Responds to an unseen hospital | reference refitted on one hospital, flagged share same vs other hospital | A-ref 4.3% vs 7.4%; B-ref 4.7% vs 8.5% (modest) |
| Link to model error (exploratory) | test AUROC by state, patient-level bootstrap 95% CI | LOW 0.852 (0.838–0.865); MODERATE 0.857; HIGH 0.749 (0.621–0.862), wide and overlapping |
| No effect on the model | pipeline output identical with and without the shift computation; `src/pipeline.py` never imports it | pass |
| States | unit tests for LOW / MODERATE / HIGH, strict thresholds, missing values, causality, explanations | pass |
| Browser | shift LOW, MODERATE, HIGH links and the optional scenario D button | pass (15 / 15 total) |
| Deployment | no new dependencies, a 3 KB JSON reference, no data or keys needed | pass |

## 16. AI4S experiment suite (added 9 Oct 2026)

No model, threshold, integrity rule or shift reference was changed. New experiments use the shipped models and the
existing split; the comparison models are trained on the same split and never shipped.

| Claim | Source | Check |
|---|---|---|
| XGBoost 0.853, random forest 0.839, CatBoost 0.839 AUROC | `src/exp_models.py` → `results/exp_models.json` | same split / protocol asserted in `tests/test_ai4s_results.py` |
| Benchmark re-creation matches the published 6,481 / 6,790 | `results/exp_robustness.json` → `benchmark_check` | asserted in tests |
| Component ablation 0% / 0% / 0% / 4.0% / 95.4% (+ shift counterfactual 96.4%, 9.3% clean flags) | `ablation` | asserted in tests |
| Impossible values withheld 1,232 / 1,232 | `sweeps.extreme` | asserted in tests |
| Missingness / noise sweeps, simulated cohort shifts, risk-coverage | `sweeps`, `cohort_shift`, `selective` | reported with the stress-test cohort baseline (AUROC 0.811), not the full-test 0.852 |
| Calibration ECE 0.0032 → 0.0028 | `calibration` | modest; stated as such |
| Scenarios E (p110755 h46) and F (p010049 h62) | `app/scenarios.py` | decision, trust, confidence, shift and risk pinned in `tests/test_judge_demo.py` |
| Technical report numbers | `tools/report/build_report.py` reads every number from `results/` | regenerated, not typed |
| Operating points (3% shipped: 79.4% alerted, 32.5 alert episodes / 100 non-septic patient-days) | `src/exp_operating_points.py` → `results/exp_operating_points.json` | shipped row equals the main result; chosen on validation utility (tests) |
| Cross-signal detector: plausible edits 26.5% → 28.2%, not shipped | `src/exp_crosssignal.py` → `results/exp_crosssignal.json` | protocol and ship rule fixed before the test run; decision recomputed in tests; dashboard does not import it |

## Known limitations that remain

* The integrity design was informed by test-cohort results; validation replication mitigates this but is not a fully
  independent third cohort.
* Deliberately edited inputs: 42.5% coverage (334 / 786); 26.5% (232 / 876) when the edits stay physiologically plausible.
* Random missingness and measurement noise are not treated as input faults; the decision changes they cause are mostly silent.
* Simulated corruptions only; real-world fault frequencies are unknown.
* External AUROC drops to 0.790 / 0.775.
