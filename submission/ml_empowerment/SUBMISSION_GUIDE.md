# Submission guide · ML Empowerment Build Challenge 3.0

**Deadline:** Oct 5, 2026 @ 11:45 pm PDT (check the Devpost page for changes) · solo project · students only, ages 13+.

Judging weights: Technical Implementation 30% · Creativity & Innovation 20% · Real-World Impact 20% · Project Design & UX
15% · Presentation & Documentation 15%.

## 1. Devpost fields

| Field | Paste |
|---|---|
| **Project title** | SepsisShield AI: Trust-Aware Early Warning for Sepsis |
| **Tagline / elevator pitch** | Predicts sepsis early, and checks whether the data behind the prediction can be trusted. |
| **About the project** | `STORY.md`, from "## Inspiration" to the end |
| **Problem, solution, key features, technologies, target users** (required by the rules) | All covered in `STORY.md`: problem (Inspiration), solution and key features (What it does), technologies (How we built it), target users (end of What it does) |
| **Built with** | python, pandas, numpy, lightgbm, scikit-learn, plotly, streamlit, pytest, github-actions, physionet |
| **Try it out** | 1) https://sepsisshieldapprepo-diqdknfnksrq5yyf6wkcrd.streamlit.app/ · 2) https://github.com/Israt76/SepsisShield_GITHUB_Repo |
| **Video** | YouTube or Vimeo link (unlisted is fine); see `VIDEO_PLAN.md` |
| **Team details** | Solo: Israt Jahan Aunika (M.S. Data Science, Montclair State University): research, data pipeline, modelling, trust layer, evaluation, dashboard, documentation |
| **Thumbnail** | `submission/gallery/00_devpost_thumbnail_1500x1000.png` |

## 2. Gallery: upload in this order

Judges should see image 1 first. It shows the whole idea in one picture: prediction on one side, a withheld prediction
with *Confident* models on the other.

| # | File (`submission/gallery/`) | Caption |
|---|---|---|
| 1 | `01_hero_1920x1080.png` | Prediction + trust: the models are confident, the inputs are not believable, so the prediction is withheld. |
| 2 | `02_architecture.png` | Two paths from the same ICU data: one predicts sepsis risk; independent input-integrity checks decide whether the inputs can be trusted. Model disagreement can only add a warning, never a withhold. |
| 3 | `03_trust_warning_verify_inputs.png` | Trust-aware warning: after a °F thermometer fault, input trust is REDUCED and the dashboard asks to verify inputs. |
| 4 | `04_edited_chart_withheld.png` | Edited chart on a held-out patient: the edit hides an alert, the models stay confident, and SepsisShield withholds instead of giving a silent all-clear. |
| 5 | `05_trust_benchmark.png` | Corruption benchmark: 95.4% of alert-changing accidental-fault wrong decisions flagged or withheld (6,481 / 6,790); 0.46% of clean hours withheld; deliberate edits weaker at 42.5% (334 / 786), 26.5% when edits stay physiologically plausible. |
| 6 | `06_early_warning.png` | Clean inputs, trust HIGH: the prediction is shown; this patient's first alert came 9 hours before the recorded onset of sepsis. |
| 7 | `07_results_and_limitations.png` | Results and known limitations side by side in the app: AUROC 0.852, 53.9% alerted ≥ 6 h early, cross-hospital AUROC 0.790 / 0.775. |
| 8 | `08_cross_hospital.png` | Zero-shot cross-hospital validation: performance drops at a hospital the model never trained on. |

Optional ninth image: `09_abstention_by_fault.png` (where fault-induced wrong decisions end up, by fault type).

## 3. What you still need to do by hand

1. **Push this version to GitHub** and confirm the **Actions** run is green (tests plus browser test, about 5–8 min).
2. **Deploy or redeploy on Streamlit Community Cloud**: repo `Israt76/SepsisShield_GITHUB_Repo`, branch `main`, main file
   `app/app.py`, Python 3.11 (Advanced settings). If it is already deployed, it redeploys itself when you push.
3. The live URL is already in `README.md`. After pushing, the deployed app updates itself within a few minutes.
4. Open the live URL in a private window: check that the page loads on scenario A and that all three buttons work.
5. Upload the video `SepsisShield_MLE_demo.mp4` (2:31.6: the original ICU opening and first two slides, then the current
   demo; built by `tools/video_mle/`, see `VIDEO_PLAN.md`) to YouTube
   (unlisted is fine), use `submission/gallery/01_hero_1920x1080.png` as the thumbnail, and paste the link into Devpost.
   Optionally add the YouTube link to the README's *Try it* row, which currently points to the Devpost page.
6. Check the challenge rules on Devpost for whether a project started before the submission period, or also entered
   in another hackathon (GIBC V2), is allowed. I could not read the full rules page.
7. Paste the fields, upload the gallery in order, and submit with at least an hour to spare.

## 4. Last checks before pressing Submit
- [ ] Every "95.4%" says *accidental* faults in a *simulated benchmark*, with 42.5% and 26.5% for deliberate edits nearby
- [ ] "Research prototype, not a medical device" appears in the story
- [ ] The live demo link opens in a private window
- [ ] The video link plays in a private window
- [ ] Team details completed
