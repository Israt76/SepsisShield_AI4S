# SepsisShield: request for clinical review (about 60–90 minutes)

SepsisShield is a research prototype that predicts sepsis risk from hourly ICU data (PhysioNet 2019 Challenge). It also
checks whether the clinical inputs behind each prediction look believable. Each prediction is then shown, shown with a
"verify inputs" warning, or withheld.

**Research prototype only. Not a medical device and not intended for clinical decision-making.** No patient data
leave the public dataset, and nothing here is used for care.

I am looking for someone with clinical, nursing, biomedical or physiology expertise to review the clinical logic. If
your review leads to changes, you would be listed as a contributor, with your contribution described accurately.

## What I would like you to check

1. **Physiological limits** (`src/integrity.py`, `PLAUSIBLE`). A value outside these limits is treated as "cannot be
   real", not just "abnormal", and the prediction is withheld. Examples:

   | Measure | Limits |
   |---|---|
   | Heart rate | 20–250 /min |
   | Temperature | 30–43 °C |
   | SBP | 40–280 mmHg |
   | MAP | 20–220 mmHg |
   | DBP | 10–200 mmHg |
   | Respiratory rate | 3–70 /min |
   | SpO₂ | 50–100% |
   | Lactate | 0.1–35 mmol/L |
   | Creatinine | 0.1–25 mg/dL |
   | FiO₂ | 0.21–1.0 |

   Are any too tight, so that they would withhold real ICU values? Are any too loose?

2. **Consistency rules.** Two combinations are treated as impossible: DBP ≥ SBP, and MAP more than 5 mmHg above SBP.
   Two are treated as soft discordance: MAP more than 5 mmHg below DBP, and SpO₂ differing from SaO₂ by more than 25 points. Is that
   split clinically right?

3. **Simulated faults** (`src/benchmark_integrity.py`). Are these realistic, and is anything important missing?
   - °F entered into a °C field
   - SI/conventional unit mix-ups for creatinine, glucose and lactate
   - Monitor artefacts: HR 190–240 together with SBP 25–45
   - Frozen monitor feeds
   - Vitals edited toward normal

4. **The "coordinated normalising shift" rule.** It flags several vitals improving within the same hour. It also fires
   on real patients, probably after effective treatment. Which clinical situations would cause this legitimately?

5. **Wording on the dashboard** (live demo, scenarios A–F). Is anything misleading to a clinician? Two phrases to look
   at: "Physiologically implausible temperature — check units or sensor" and "VERIFY INPUTS".

6. **Alert burden.**
   - At the shipped threshold the system produces about 32.5 new alert episodes per 100 non-septic patient-days.
   - It alerts 79.4% of septic patients in the 12 h before to 3 h after onset.
   - In your setting, which operating point would be acceptable, if any? (`results/figures/17_operating_points.png`)

7. **Limitations section** of the technical report (section 17). Is anything clinically important missing?

## Links

- **Live demo (no login):** https://sepsisshieldapprepo-diqdknfnksrq5yyf6wkcrd.streamlit.app/
- **Code and technical report:** https://github.com/Israt76/SepsisShield_GITHUB_Repo

Comments in any form are welcome: margin notes on the PDF, a short email, or a call.
