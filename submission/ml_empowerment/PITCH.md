# SepsisShield AI · pitches and statements (ML Empowerment Build Challenge 3.0)

Every number below comes from `results/*.json` and is checked by `tools/check_claims.py`.

## Tagline (one sentence)
Predicts sepsis early, and checks whether the data behind the prediction can be trusted.

## Project subtitle
Trust-aware early warning for sepsis: prediction + input trust.

## 100-character short description
Sepsis early warning that checks if its ICU input data can be trusted, and withholds when it can't.
*(99 characters)*

## Elevator pitch (one sentence)
SepsisShield predicts sepsis early and runs independent input-integrity checks on the clinical data behind each
prediction, warning about or withholding the prediction when those inputs look unreliable.

## 30-second pitch (~75 words)
Sepsis models are usually evaluated on accuracy, but ICU data can go wrong: a thermometer in Fahrenheit, a frozen monitor, an edited
chart. The model can still be confident. SepsisShield runs two paths: one predicts sepsis risk, the other checks
whether the inputs are believable. High trust: show. Reduced: verify. Low: withhold. On 8,068 held-out patients it
reaches AUROC 0.852, and it flagged or withheld 95.4% of wrong decisions caused by simulated accidental faults.

## 60-second pitch (~155 words)
Sepsis early-warning evaluations typically focus on predictive performance, while input reliability receives much less attention. But ICU data are not always right.
A thermometer reports Fahrenheit into a Celsius field, a monitor feed freezes, a chart is edited, and the model
still returns a confident-looking score.

SepsisShield separates two questions. The prediction path asks how likely sepsis is. Independent input-integrity checks ask
whether the inputs are believable. A decision layer then shows the prediction, shows it with a warning, or withholds it
until the data are checked, and explains why.

On 8,068 held-out ICU patients, AUROC is 0.852 and 53.9% of septic patients are alerted at least six hours early. In a
corruption benchmark, 95.4% of alert-changing wrong decisions from four accidental fault types were flagged or
withheld, at a cost of 0.46% of clean hours. Deliberate edits are harder: 42.5%, and 26.5% when the edits stay realistic. I report both.

## Architecture caption
SepsisShield separates prediction from input verification. One path estimates sepsis risk, while independent
input-integrity checks evaluate whether the clinical inputs appear reliable. A decision layer then determines whether
the prediction should be shown, shown with a warning, or withheld. Model disagreement may reduce trust to REDUCED, but LOW trust and prediction withholding are triggered only by evidence from the clinical inputs.

## Target-user statement
SepsisShield is designed, as a research prototype, for three groups: ICU clinicians, who need to know when a sepsis
score rests on unreliable data (it supports clinical judgement and does not replace it); clinical informatics teams,
who maintain the data pipelines that feed such models; and hospital ML / AI monitoring teams, who audit deployed
models and need a per-hour signal of input quality.

## Social-impact statement
Clinical AI is usually evaluated on clean, curated data, but it runs on real hospital data, where unit errors, frozen
feeds and edited values happen. A model that stays confident on broken inputs can create false alarms that erode trust,
or silent all-clears that delay care. SepsisShield shows a practical pattern for safer clinical ML: evaluate input
trust separately from model confidence, and let the system say "check the data first" instead of guessing. It runs on
a standard CPU, uses public data and is fully reproducible, so others can test and build on the idea. Outcomes such as
mortality were not measured, and the project makes no claim about them.

## Known-limitations statement
SepsisShield is a research prototype, not a medical device, and has not been prospectively validated. Deliberately
edited inputs were flagged or withheld in only 42.5% of cases (334 / 786), and in 26.5% when the simulated edits
stay inside normal physiological ranges. Performance drops at a hospital the model
did not train on (AUROC 0.790 / 0.775). The integrity benchmark uses simulated faults, not recorded hospital
incidents. Distribution-shift awareness indicates that an input pattern differs from the training distribution; it does not establish that a prediction is incorrect or clinically unsafe. Frozen feeds are caught in 46.8% of cases, and 26% of non-septic patients receive at least one false alert.

## Judge-demo instructions (paste under "Try it out" or in the story)
1. Open the live demo (or run `pip install -r requirements.txt` then `streamlit run app/app.py`).
2. Click **A · Clean inputs**: trust HIGH → **SHOW PREDICTION**.
3. Click **B · Accidental data fault**: model *Confident*, trust **LOW** → **PREDICTION WITHHELD**. Read *Why was this
   flagged?* and the clean-vs-fault comparison card.
4. Click **C · Edited chart**: the edit hides an alert, and SepsisShield withholds instead of giving a silent all-clear.
   (A caught edit; the benchmark shows most edits are not caught.)
5. Optional: click **D · Unusual patient (distribution shift)**: inputs trusted, but the separate shift panel shows
   HIGH because a white cell count is far outside the training data.
6. Open the **Results & limitations** tab for the evidence and the known weaknesses.
