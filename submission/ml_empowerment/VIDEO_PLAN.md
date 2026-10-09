# SepsisShield AI · demo video plan (ML Empowerment Build Challenge 3.0)

**Target length:** 2:30 · **Format:** 1920×1080, captions in a black band below the picture (as in the current video) or
burned in at the bottom · **Voice:** calm, about 140 words per minute (~350 words total).

All footage is the live dashboard (Judge Demo Mode). Every scenario is a real held-out test patient, and every number
comes from `results/*.json`. Record the browser at 1500×1000 CSS px with the sidebar collapsed.

---

## Opening frame (0:00, hold 2 s under the first line)
Black background, white text, centred:
> **A model can be confident even when the clinical data are wrong.**

Small grey line beneath: *SepsisShield AI · research prototype · de-identified PhysioNet 2019 data*

## Final frame (2:26–2:30)
Off-white background, shield logo, then:
> **SepsisShield AI**
> Trust-aware AI should know not only when to predict, but when not to trust its inputs.
> AUROC 0.852 · 53.9% alerted ≥ 6 h early · 95.4% of accidental-fault wrong decisions flagged or withheld · deliberate edits 42.5% (26.5% if realistic)
> *Research prototype · not a medical device* · github.com/Israt76/SepsisShield_GITHUB_Repo

---

## Script

| Time | Screen | Narration (exact) | Caption | Overlay |
|---|---|---|---|---|
| **0:00–0:15** Problem | Opening frame (2 s), then fade to scenario B's four cards, still, with *Confident* and *LOW* side by side | "Sepsis early-warning models are usually evaluated on how well they predict. But what happens when the data going in are wrong: a thermometer in Fahrenheit, a frozen monitor, an edited chart? The model can still be confident." | A model can be confident even when the data are wrong | Circle *Confident* (card ②) at 0:11 |
| **0:15–0:35** Innovation | Architecture diagram (How it works tab), with a slow zoom from the prediction path to the trust path to the decision | "SepsisShield separates two questions. The prediction path asks: how likely is sepsis? Independent input-integrity checks ask: are these inputs believable? A decision layer then shows the prediction, shows it with a warning, or withholds it until the data are checked." | Prediction path + independent input-integrity checks → show / warn / withhold | Labels "HOW LIKELY IS SEPSIS?" over the blue path and "ARE THE INPUTS BELIEVABLE?" over the orange path |
| **0:35–0:52** Demo A | Click **A · Clean inputs**, then the four cards, then scroll to the risk trajectory and SHAP bars | "Here is the live dashboard. Scenario A: a real held-out patient. Clean inputs, the five models agree, and input trust is high, so the prediction is shown. Its first alert came nine hours before the recorded onset, and the bars show what drove it." | A · clean inputs → trust HIGH → SHOW PREDICTION | Arrow to card ④ *SHOW PREDICTION*; "alert 9 h before onset" tag on the chart |
| **0:52–1:10** Demo B | Click **B · Accidental data fault**. Hold on the cards, then the **clean vs fault** comparison card, then *Why was this flagged?* | "Scenario B: the same patient, but the thermometer now reports Fahrenheit into a Celsius field. The risk jumps above the alert line, a false alarm. Model confidence still says confident. But input trust falls to low, a physiologically implausible temperature, and the prediction is withheld." | B · °F thermometer → models confident, trust LOW → WITHHELD | Split highlight: card ② *Confident* (grey) vs card ③ *LOW* (red). Text overlay: **Confidence ≠ trust** |
| **1:10–1:15** Demo B+1 h | Drag the hour slider from 36 to 37 | "One hour later, trust is reduced: verify inputs." | Trust REDUCED → VERIFY INPUTS | Arrow to the dashed amber card ④ |
| **1:15–1:25** Demo C | Click **C · Edited chart**, then the cards and the **clean vs fault** card (3.5% alert → 2.3% no alert → withheld) | "Scenario C: a second patient's vitals are overwritten to look normal, and the model's alert disappears. The models still agree. SepsisShield sees several vital signs normalising at once and withholds the prediction instead of giving a silent all-clear." | C · edited chart hides an alert → WITHHELD, not a silent all-clear | Highlight "3.5% → 2.3%" in the *What the fault changed* box |
| **1:25–1:55** Evidence | **Results & limitations** tab: the top row, then the trust row. Reveal each tile as it is narrated | "Does it hold up beyond three examples? On 8,068 held-out patients, the model reaches an AUROC of 0.852, and 53.9 percent of septic patients are alerted at least six hours before onset. In a corruption benchmark, 95.4 percent of alert-changing wrong decisions caused by four simulated accidental fault types were flagged or withheld, while only 0.46 percent of clean patient-hours were withheld." | AUROC 0.852 · 53.9% ≥ 6 h early · 95.4% (6,481 / 6,790) · 0.46% clean withheld | Box each tile as named; small note "accidental faults · simulated benchmark" |
| **1:55–2:12** Limitations | Scroll to **Known limitations**: highlight the deliberate-edit and cross-hospital boxes | "The limits matter too. Deliberately edited inputs were caught only 42.5 percent of the time, and about a quarter of the time when the edits stay realistic. At a hospital the model never trained on, AUROC drops to 0.790 and 0.775." | Deliberate edits 42.5% (334 / 786), 26.5% if realistic · unseen hospital AUROC 0.790 / 0.775 | Amber outline on the two boxes |
| **2:12–2:18** Shift signal (brief) | Click **Optional: D · Unusual patient**; hold on the purple *Distribution shift* panel | "A separate, advisory signal also warns when a patient looks unlike the training data. It is a caution, not proof a prediction is wrong." | Distribution shift: advisory only | Outline the shift panel |
| **2:18–2:26** Impact | Back to scenario B's four cards | "SepsisShield is a research prototype, not a medical device. Its idea is simple: trust-aware AI should know not only when to predict, but when not to trust its inputs." | Know when not to trust the inputs | none |
| **2:26–2:30** Close | Final frame | (silence or a soft end tone) | none | none |

Word count of the narration: about 350, which fits 2:30 at a natural pace. If the take runs long, cut "and the bars show
what drove it" (0:35) and "The models still agree" (1:15).

## Rendered version
`tools/video_mle/` renders this plan automatically from the live dashboard: `tts.py` (Piper voice, en-US lessac) →
`capture.py` (app running on :8501) → `render.py`. Output: 1920×1200 (picture plus a 120 px caption band), with every
number taken from the dashboard or the captions in `script.py`.

**Submitted cut (2:31.6):** `splice_opening.py` keeps the first 17.1 s of the original submitted video unchanged (the
ICU / patient-monitor opening with its sound, the slide "Most early-warning models assume their inputs are correct.",
and the "SepsisShield AI · Predict early. Explain clearly. Know when not to trust the model." slide), then cross-fades
into the current demo from the architecture scene onward. The current demo's own 0:00–0:15 "problem" scene is dropped.

## Recording checklist (if you record your own voice-over instead)
* Use the deployed app URL or `streamlit run app/app.py`. The page opens on scenario A.
* Collapse the sidebar (the `«` arrow) before recording, so the Judge Demo buttons and cards fill the frame.
* Scenario B at hour 37 (VERIFY INPUTS) is the link `?pid=p119917&corr=Thermometer%20reports%20°F&start=34&len=8&hour=37`.
* Keep the mouse still while narrating a card. Move only to click.
* Do not add music louder than the voice. Do not add claims that are not in this script.

## What changed from the earlier video (`submission/SepsisShield_demo.mp4`, 2:28)
The earlier video tells the same story with the previous dashboard layout. It stays accurate and can be submitted if
there is no time to re-record. This plan adds the Judge Demo buttons, the Final Decision card, the VERIFY INPUTS state
and the 53.9% early-alert result, and moves the limitations into their own 20-second segment.
