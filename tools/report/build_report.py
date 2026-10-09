"""Build the AI4S technical report (docs/technical_report.html -> docs/technical_report.pdf).

Every number in the report is read from the result files in results/ at build time, so the PDF cannot drift away
from the experiments. Run:  python tools/report/build_report.py   (needs playwright + chromium for the PDF step)
"""
import asyncio
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES, FIG, SHOT = ROOT / "results", ROOT / "results" / "figures", ROOT / "results" / "screenshots"
OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)

J = lambda name: json.load(open(RES / name))  # noqa: E731
R, E2, A, V = J("results.json"), J("experiments2.json"), J("abstention.json"), J("abstention_val.json")
IB, MR, SE = J("integrity_benchmark.json"), J("masking_realism.json"), J("shift_evaluation.json")
EM, ER = J("exp_models.json"), J("exp_robustness.json")
SG = J("subgroups_ci.json")
OP = J("exp_operating_points.json")
XJ = J("exp_crosssignal.json")
XT, XV = XJ["test_benchmark"], XJ["validation_benchmark"]
XD = XT["decision"]


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def f3(x):
    return f"{x:.3f}"


def n(x):
    return f"{x:,}"


# ------------------------------------------------------------------ numbers
M = R["main"]["calibrated"]
CI = E2["bootstrap_ci"]["sepsisshield"]
acc, val = A["dfc_accidental"]["all"], V["dfc_accidental"]["all"]
mask = A["dfc_masking"]["all"]
cf = A["clean_cost_full_test"]
base = {b["model"]: b for b in E2["baselines"]}
models = {r["model"].split(" (")[0]: r for r in EM["rows"]}
abl = ER["ablation"]["rows"]
sweep = ER["sweeps"]
cs = ER["cohort_shift"]
sel = ER["selective"]
cal = ER["calibration"]
a2b, b2a = R["cross_A2B"], R["cross_B2A"]
rc = {x["transfer"]: x for x in E2["recalibration"]}
by_fault = {s["corruption"]: s["all"] for s in A["scenarios"]}
FAULT = {"temp_fahrenheit": "Thermometer reports °F into a °C field", "lab_unit_error": "Lab unit mix-up (SI ↔ conventional)",
         "sensor_artifact": "Monitor artefact (HR 190–240, SBP 25–45)", "frozen_feed": "Frozen monitor feed",
         "masking_attack": "Deliberate edit: vitals overwritten to look normal",
         "measurement_noise": "Benign control: ordinary measurement noise"}
covered = acc["withheld"] + acc["flagged"]
val_cov = val["withheld"] + val["flagged"]
mask_cov = mask["withheld"] + mask["flagged"]
pe = MR["plausible_edit"]
pe_cov = pe["withheld"] + pe["flagged"]
sig = sel["benchmark_signal_comparison"]
ref = cs["reference"]
shift_rate = lambda d: d["shift_moderate_pct"] + d["shift_high_pct"]  # noqa: E731
miss40, noise50 = sweep["missingness"][-1], sweep["noise"][-1]
ext = sweep["extreme"]
rcov = sel["clean_risk_coverage_full_test"]
rc70 = [r for r in rcov if r["coverage"] == 0.7][0]
prev = cs["prevalence_x3 (non-septic patients subsampled)"]


def fig(name, caption, width="100%"):
    return (f'<figure><img src="../results/figures/{name}" style="width:{width}"><figcaption>{caption}</figcaption></figure>')


def shot(name, caption, width="100%"):
    return (f'<figure><img src="../results/screenshots/{name}" style="width:{width}"><figcaption>{caption}</figcaption></figure>')


def table(head, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


# ------------------------------------------------------------------ tables
t_models = table(
    ["Model", "AUROC", "AUPRC", "Brier", "ECE", "Utility", "Sens.", "Spec."],
    [[name, f3(r["auroc"]), f3(r["auprc"]), (f"{r['brier']:.4f}" if r.get("brier") is not None else "–"),
      (f"{r['ece']:.4f}" if r.get("ece") is not None else "–"), f3(r["utility"]),
      pct(r["patient_sensitivity"]), pct(r["patient_specificity"])] for name, r in [
        ("SIRS rule (≥2 of 4)", base["SIRS criteria (rule: ≥2 of 4)"]),
        ("qSOFA-partial rule (≥1 of 2)", base["qSOFA-partial (rule: ≥1 of 2)"]),
        ("Logistic regression + isotonic", base["Logistic regression (+ isotonic)"]),
        ("Random forest + isotonic", models["Random forest"]),
        ("CatBoost + isotonic", models["CatBoost"]),
        ("XGBoost + isotonic", models["XGBoost"]),
        ("Single LightGBM (uncalibrated)", base["Single LightGBM (uncalibrated)"]),
        ("<b>SepsisShield: 5× LightGBM + isotonic</b>", {**base["Ensemble + isotonic calibration (SepsisShield model)"],
                                                          "brier": M["brier"]}),
    ]], "num")

_op = {round(r["threshold"], 3): r for r in OP["rows"]}
t_op = table(["Alert threshold", "Septic alerted", "≥ 6 h early", "Non-septic never alerted", "Utility (val / test)",
              "False-alert hours / 100 non-septic pt-days", "Alert episodes / 100 non-septic pt-days"],
             [[("<b>3% (shipped)</b>" if abs(r["threshold"] - OP["shipped_threshold"]) < 1e-9 else f"{r['threshold']*100:g}%"),
               pct(r["patient_sensitivity"]), pct(r["pct_detected_ge6h_early"]), pct(r["patient_specificity"]),
               f"{r['utility_val']:.3f} / {r['utility_test']:.3f}", f"{r['false_alert_hours_per_100_patient_days']:.0f}",
               f"{r['alert_episodes_per_100_nonseptic_patient_days']:.1f}"]
              for r in OP["rows"] if round(r["threshold"], 3) in (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2)], "num")
_xk = [("Accidental faults (4 types)", "accidental"), ("Frozen monitor feed", "frozen_feed"),
       ("Deliberate edit (original)", "masking_attack"), ("Deliberate edit kept plausible", "masking_plausible"),
       ("Benign measurement noise", "measurement_noise")]
def _xc(d, k, w):
    return d[k][w]["coverage"]
t_xsig = table(["Fault", "Validation: before → after", "Test: before → after", "Test: dangerous failures"],
               [[lab, f"{pct(_xc(XV, k, 'before'))} → {pct(_xc(XV, k, 'after'))}",
                 f"<b>{pct(_xc(XT, k, 'before'))} → {pct(_xc(XT, k, 'after'))}</b>",
                 n(XT[k]['before']['n'])] for lab, k in _xk], "num")
t_faults = table(
    ["Simulated fault (10-hour window)", "Dangerous failures", "Withheld", "Warned", "Silent", "Flagged or withheld"],
    [[FAULT[k], n(by_fault[k]["n"]), n(by_fault[k]["withheld"]), n(by_fault[k]["flagged"]), n(by_fault[k]["silent"]),
      pct((by_fault[k]["withheld"] + by_fault[k]["flagged"]) / by_fault[k]["n"])]
     for k in ("temp_fahrenheit", "lab_unit_error", "sensor_artifact", "frozen_feed")] +
    [["<b>All four accidental faults</b>", f"<b>{n(acc['n'])}</b>", n(acc["withheld"]), n(acc["flagged"]),
      n(acc["silent"]), f"<b>{pct(acc['coverage'])}</b>"],
     [FAULT["masking_attack"], n(mask["n"]), n(mask["withheld"]), n(mask["flagged"]), n(mask["silent"]),
      pct(mask["coverage"])],
     ["Deliberate edit kept physiologically plausible (sensitivity check)", n(pe["n"]), n(pe["withheld"]),
      n(pe["flagged"]), n(pe["silent"]), pct(pe["coverage"])],
     [FAULT["measurement_noise"], n(by_fault["measurement_noise"]["n"]), n(by_fault["measurement_noise"]["withheld"]),
      n(by_fault["measurement_noise"]["flagged"]), n(by_fault["measurement_noise"]["silent"]),
      pct(A["dfc_noise_control"]["all"]["coverage"])]], "num")

t_abl = table(
    ["Configuration", "AUROC", "ECE", "Dangerous failures caught", "Clean hours flagged or withheld (total) / withheld only"],
    [[r["config"].split(". ", 1)[1], f3(r["auroc"]), f"{r['ece']:.4f}",
      f"{n(r['dangerous_failures_caught'])} / {n(r['dangerous_failures'])} ({pct(r['coverage'])})",
      f"{pct(r['clean_hours_flagged_pct'])} / {pct(r['clean_hours_withheld_pct'], 2)}"] for r in abl], "num")

t_sweep = table(
    ["Perturbation", "AUROC", "Utility", "Sens.", "Trust LOW / REDUCED hours", "Shift MOD+HIGH hours",
     "Dangerous decision flips (flagged)"],
    [["None (stress-test cohort)", f3(sweep["baseline_clean"]["auroc"]), f3(sweep["baseline_clean"]["utility"]),
      pct(sweep["baseline_clean"]["patient_sensitivity"]),
      f"{pct(sweep['baseline_clean']['trust_low_pct'], 2)} / {pct(sweep['baseline_clean']['trust_reduced_pct'])}",
      pct(shift_rate(sweep["baseline_clean"])), "–"]] +
    [[f"Missingness {int(r['fraction_removed'] * 100)}%", f3(r["auroc"]), f3(r["utility"]), pct(r["patient_sensitivity"]),
      f"{pct(r['trust_low_pct'], 2)} / {pct(r['trust_reduced_pct'])}", pct(shift_rate(r)),
      f"{n(r['decision_flips']['n'])} ({pct(r['decision_flips']['coverage'])})"] for r in sweep["missingness"]] +
    [[f"Noise {r['noise_sd_multiple']} SD", f3(r["auroc"]), f3(r["utility"]), pct(r["patient_sensitivity"]),
      f"{pct(r['trust_low_pct'], 2)} / {pct(r['trust_reduced_pct'])}", pct(shift_rate(r)),
      f"{n(r['decision_flips']['n'])} ({pct(r['decision_flips']['coverage'])})"] for r in sweep["noise"]], "num small")

COH = [("Reference cohort", ref), ("Real subgroup: age ≥ 80", cs["age_ge_80"]), ("Real subgroup: age &lt; 45", cs["age_lt_45"]),
       ("Shifted vitals (HR +20, Resp +6, Temp +1 °C)", cs["shifted_vitals (HR+20, Resp+6, Temp+1 C)"]),
       ("Missing labs (60% of lab values removed)", cs["missing_labs (60% of lab values removed)"]),
       ("Older cohort (Age +25 y, capped at 100)", cs["older_cohort (Age +25 y, capped at 100)"]),
       ("Measurement noise (0.3 SD)", cs["measurement_noise (0.3 SD)"]),
       ("Prevalence ×3 (non-septic patients subsampled)", prev)]
t_cohort = table(["Simulated cohort", "Shift MODERATE + HIGH hours", "vs reference", "AUROC", "ECE"],
                 [[k, pct(shift_rate(d)), f"{shift_rate(d) / shift_rate(ref):.1f}×", f3(d["auroc"]), f"{d['ece']:.4f}"]
                  for k, d in COH], "num")

t_sig = table(["Hours abstained", "Input-trust ranking", "Shift score only", "Ensemble disagreement only", "Random"],
              [[pct(r["abstain_fraction_of_hours"], 0), pct(r["input_trust_policy"]), pct(r["shift_score_only"]),
                pct(r["ensemble_disagreement_only"]), pct(r["random"])] for r in sig["rows"]], "num")

t_sub = table(["Subgroup", "Patients", "Septic", "AUROC (95% CI)", "Patient sensitivity (95% CI)"],
              [[f"{g['attribute'].replace('_', ' ')}: {html.escape(g['group'])}", n(g["n_patients"]), n(g["n_septic"]),
                f"{f3(g['auroc'])} ({f3(g['auroc_ci'][0])}–{f3(g['auroc_ci'][1])})",
                f"{pct(g['patient_sensitivity'])} ({pct(g['sens_ci'][0])}–{pct(g['sens_ci'][1])})"] for g in SG], "num small")

t_feat_abl = table(["Feature set", "Features", "AUROC", "AUPRC", "Utility"],
                   [[a["config"], a["n_features"], f3(a["auroc"]), f3(a["auprc"]), f3(a["utility"])] for a in R["ablation"]], "num")

t_cross = table(["Transfer", "Internal AUROC (source)", "External AUROC", "External utility", "Sensitivity", "Specificity",
                 "Specificity after local recalibration"],
                [["Hospital A → B", f3(a2b["internal_val"]["auroc"]), f3(a2b["auroc"]), f3(a2b["utility"]),
                  pct(a2b["patient_sensitivity"]), pct(a2b["patient_specificity"]),
                  pct(rc["A→B"]["recalibrated"]["patient_specificity"])],
                 ["Hospital B → A", f3(b2a["internal_val"]["auroc"]), f3(b2a["auroc"]), f3(b2a["utility"]),
                  pct(b2a["patient_sensitivity"]), pct(b2a["patient_specificity"]),
                  pct(rc["B→A"]["recalibrated"]["patient_specificity"])]], "num")

t_cmp = table(["Scenario", "Standard ML model", "SepsisShield (what the dashboard shows)"], [
    ["Clean patient", "Predicts", "Predicts; trust HIGH → SHOW PREDICTION (scenarios A, E)"],
    ["Physiologically impossible value", "Predicts confidently", f"Trust LOW → PREDICTION WITHHELD ({n(ext['injected_hours_trust_low'])} / {n(ext['injected_hours'])} injected hours)"],
    ["°F temperature in a °C field", "Predicts confidently (false alert)", "Trust LOW → withheld; models stay confident (scenario B)"],
    ["Vitals edited to look normal", "Confident all-clear", "Coordinated-shift detector → withheld (scenario C); caught far less often overall"],
    ["Models disagree, inputs clean", "Forced prediction", "Trust REDUCED → VERIFY INPUTS, never withheld (scenario F)"],
    ["Patient unlike the training data", "No warning", "Advisory shift indicator MODERATE / HIGH beside the decision (scenario D)"],
    ["Random missing values", "Predicts", "Predicts; missingness is a model feature, not a trust flag (limitation: decision changes mostly not flagged)"],
    ["Explanation", "SHAP only", "SHAP risk drivers + the integrity check that fired + shift reason + decision rule"],
])

CSS = """
@page { size: Letter; margin: 0.8in 0.75in 0.85in 0.75in; }
body { font-family: 'Source Serif 4', 'DejaVu Serif', Georgia, serif; font-size: 9.9pt; line-height: 1.36; color: #111; }
h1 { font-family: 'Inter', 'DejaVu Sans', sans-serif; font-size: 22pt; line-height: 1.15; margin: 0 0 6pt; }
h2 { font-family: 'Inter', 'DejaVu Sans', sans-serif; font-size: 13.5pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ddd; padding-bottom: 3pt; break-after: avoid; }
h3 { font-family: 'Inter', 'DejaVu Sans', sans-serif; font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; hyphens: auto; }
.sub { font-family: 'Inter', 'DejaVu Sans', sans-serif; font-size: 12pt; color: #333; margin-bottom: 10pt; }
.meta { font-family: 'Inter', 'DejaVu Sans', sans-serif; font-size: 9.5pt; color: #444; margin-bottom: 14pt; }
.disclaimer { border: 1.5px solid #b42318; background: #fef3f2; padding: 7pt 10pt; font-family: 'Inter', sans-serif; font-size: 9.6pt; margin: 8pt 0 12pt; }
.abstract { background: #f6f7f9; border-left: 3px solid #2a78d6; padding: 9pt 12pt; margin: 8pt 0 12pt; }
.keybox { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6pt; margin: 8pt 0 12pt; font-family: 'Inter', sans-serif; }
.keybox div { border: 1px solid #ddd; border-radius: 4pt; padding: 6pt; font-size: 8.4pt; color: #333; }
.keybox b { display: block; font-size: 14pt; color: #111; }
figure { margin: 8pt 0 12pt; break-inside: avoid; text-align: center; }
figcaption { font-family: 'Inter', sans-serif; font-size: 8.6pt; color: #444; text-align: left; margin-top: 3pt; }
.tw { break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt; font-family: 'Inter', 'DejaVu Sans', sans-serif; font-size: 8.4pt; break-inside: avoid; }
th { text-align: left; border-bottom: 1.2px solid #333; padding: 3pt 4pt; font-weight: 600; vertical-align: bottom; }
td { border-bottom: 0.6px solid #ddd; padding: 3pt 4pt; vertical-align: top; }
table.num td:not(:first-child) { font-variant-numeric: tabular-nums; }
table.small { font-size: 7.8pt; }
.tcap { font-family: 'Inter', sans-serif; font-size: 8.6pt; color: #444; margin: -6pt 0 12pt; break-before: avoid; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; background: #f2f2f2; padding: 0 2pt; }
ul, ol { margin: 0 0 7pt 16pt; padding: 0; } li { margin-bottom: 2pt; }
.pb { break-before: page; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 10pt; }
.note { font-size: 9.4pt; color: #333; background: #fffaeb; border-left: 3px solid #eda100; padding: 6pt 9pt; margin: 6pt 0 10pt; }
.refs { columns: 2; column-gap: 16pt; } .refs li { font-size: 8pt; margin-bottom: 2pt; break-inside: avoid; }
pre { font-size: 7.6pt; background: #f6f7f9; padding: 6pt; line-height: 1.25; white-space: pre-wrap; }
"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><title>SepsisShield — Technical Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>

<h1>SepsisShield: Trustworthy AI for Early Sepsis Risk Detection Under Clinical Data Failures</h1>
<div class="sub">Technical report · Category: End-to-End System</div>
<div class="meta">Israt Jahan Aunika · M.S. Data Science, Montclair State University · Code: github.com/Israt76/SepsisShield_GITHUB_Repo ·
Live demo: sepsisshieldapprepo-diqdknfnksrq5yyf6wkcrd.streamlit.app</div>
<div class="disclaimer"><b>Research prototype only. Not a medical device and not intended for clinical decision-making.</b>
All results are retrospective, on de-identified PhysioNet 2019 data; data faults are simulated, not recorded hospital incidents;
nothing here has been prospectively validated.</div>

<div class="abstract"><b>Abstract.</b> Sepsis early-warning models are usually judged on how well they rank patients when the input
data are clean. In practice, ICU data can be wrong in ways the model cannot see: a thermometer reporting °F into a °C field, a laboratory
result in the wrong units, a frozen monitor feed, or a chart edited to look normal. A model can remain confident on such inputs. SepsisShield
is an end-to-end system that pairs a calibrated sepsis-risk model (5-seed LightGBM ensemble, 172 causal features, isotonic calibration,
TreeSHAP explanations) with an <i>independent input-integrity layer</i> that grades the clinical inputs of every patient-hour as HIGH,
REDUCED or LOW trust and turns that grade into an action: show the prediction, show it with a "verify inputs" warning, or withhold it.
A separate distribution-shift indicator reports, as advisory context only, whether a patient's recent inputs look unlike the training data.
On a held-out test set of {n(R['main']['n_patients']['test'])} patients the model reaches AUROC {f3(M['auroc'])} (95% CI
{f3(CI['auroc'][0])}–{f3(CI['auroc'][1])}) and alerts {pct(M['pct_detected_ge6h_early'])} of septic patients at least 6 h before onset.
Across four simulated accidental fault types, {pct(acc['coverage'])} ({n(covered)} / {n(acc['n'])}) of the alert decisions that the faults
flipped to a wrong answer were flagged or withheld (validation-cohort replication {pct(val['coverage'])}), while only
{pct(cf['withheld_pct'], 2)} of clean patient-hours were withheld. A component ablation shows that the model, ensembling and calibration
catch none of these failures, ensemble disagreement alone catches {pct(abl[3]['coverage'])}, and adding the input-integrity checks raises
coverage to {pct(abl[4]['coverage'])}. The system is weaker on deliberate edits ({pct(mask['coverage'])}, {n(mask_cov)} / {n(mask['n'])}; {pct(pe['coverage'])} when
edits stay physiologically plausible), on random missingness, and at an unseen hospital (AUROC {f3(a2b['auroc'])} / {f3(b2a['auroc'])}).</div>

<div class="keybox">
<div><b>{f3(M['auroc'])}</b>held-out AUROC (8,068 patients)</div>
<div><b>{pct(acc['coverage'])}</b>accidental-fault failures flagged or withheld</div>
<div><b>{pct(cf['withheld_pct'], 2)}</b>clean patient-hours withheld</div>
<div><b>{pct(mask['coverage'])} / {pct(pe['coverage'])}</b>deliberate edits caught (limitation)</div>
</div>

<h2>1. Introduction</h2>
<p>Sepsis is a dysregulated host response to infection that causes life-threatening organ dysfunction [1]. It is common and deadly: a
global burden study estimated 48.9 million cases and 11 million sepsis-related deaths in 2017 [2], and observational work has linked
each hour of delay in effective antimicrobial therapy for septic shock to lower survival [3]. These facts motivate early-warning systems
that predict sepsis from routinely collected ICU data, and the PhysioNet/Computing in Cardiology Challenge 2019 provided a public benchmark
for exactly that task [4].</p>
<p>Accuracy on clean validation data is necessary but not sufficient. Deployed sepsis models have performed worse than expected outside
the setting in which they were developed [5], and clinical AI in general is vulnerable to changes in the data it receives [6]. A less
discussed failure mode is <i>input corruption at the bedside</i>: unit mix-ups, device artefacts, interface feeds that stop updating, and
manual edits. A tree ensemble given a body temperature of 98.6 in a Celsius field does not know that the number is impossible; it simply
produces a probability, often a confident one.</p>
<p>SepsisShield addresses this by asking a second question next to the usual one. The model asks <i>"what is this patient's sepsis
risk?"</i>; an independent integrity layer asks <i>"can the clinical data behind this prediction be trusted?"</i>. The answer to the second
question decides what the clinician sees. Put briefly: model uncertainty asks whether the models disagree; SepsisShield asks whether
the data themselves deserve to be believed. The contributions of this work are:</p>
<ol>
<li><b>An end-to-end trust-aware pipeline</b> that separates model confidence (agreement between five models) from input trust (evidence
from the raw clinical inputs) and maps them to a three-level action policy, with a rule that only input evidence can withhold a
prediction.</li>
<li><b>A reproducible corruption benchmark</b> with six simulated fault types on held-out patients and a safety metric, <i>dangerous
failure coverage</i>, that counts only alert decisions a fault flipped to a wrong answer.</li>
<li><b>An experiment suite</b> covering model families, calibration, missingness and noise sweeps, selective prediction, a
component-by-component ablation, simulated cohort shifts, cross-hospital transfer and subgroup audits, with every number regenerated from
code.</li>
<li><b>A public, no-login interactive demo</b> with one-click scenarios that show the system's behaviour on real held-out patients,
including the cases where it fails.</li>
</ol>

<h2>2. Clinical motivation</h2>
<p>Three properties of ICU data make input trust a practical concern rather than a theoretical one. First, data arrive from many devices
and interfaces (bedside monitors, point-of-care analysers, central laboratories, manual charting), each with its own units and failure
modes. Second, early-warning features are dominated by trends and recent extremes, so a single corrupted value can shift rolling features
for hours after it occurs. Third, the most dangerous corrupted prediction is not a noisy one but a <i>plausible-looking</i> one: a false
all-clear for a deteriorating patient, or a false alarm that teaches staff to ignore alerts.</p>
<p>The original PhysioNet records already contain such values. Applying SepsisShield's plausibility limits to all
{n(IB['real_data_artifacts']['hours'])} patient-hours flags {n(IB['real_data_artifacts']['hours_implausible'])} hours from
{n(IB['real_data_artifacts']['patients_with_implausible'])} patients with values that cannot be physiologically real (for example FiO₂ of
4000 or a respiratory rate of 1) and {n(IB['real_data_artifacts']['hours_inconsistent'])} hours with internally inconsistent blood pressure.
These are signs that the problem exists in research-grade data; they are not a measured rate of clinical incidents.</p>
<p>The system therefore does not try to correct data or to recommend treatment. Its job is narrower: make the reliability of each
prediction visible, and avoid presenting an untrustworthy prediction as actionable.</p>

<h2>3. Related work</h2>
<p><b>Sepsis prediction.</b> The 2019 PhysioNet challenge [4] introduced a time-dependent utility score that rewards alerts 6–12 hours
before onset and penalises late or false alerts; gradient-boosted trees on engineered features were among the strongest approaches.
External evaluations of deployed proprietary models have reported substantially lower discrimination than developers reported [5], and
implementation studies stress workflow integration as much as accuracy [7].</p>
<p><b>Calibration and uncertainty.</b> Isotonic regression is a standard non-parametric calibration method [8]; calibration error is
commonly summarised by expected calibration error and Brier score [9]. Ensembles provide a simple predictive-uncertainty signal through
member disagreement [10].</p>
<p><b>Selective prediction.</b> Selective classification lets a model abstain on inputs where it is unreliable and studies the
trade-off between coverage and risk [11, 12]. Most work abstains on <i>model</i> uncertainty; SepsisShield abstains on <i>input</i>
evidence, and we show empirically (Section 11) why the two are not interchangeable for data faults.</p>
<p><b>Distribution shift and data validation.</b> Dataset-shift detection has been studied extensively [13], and clinicians have been
urged to treat dataset shift as a patient-safety issue [6]. Production ML systems routinely validate incoming data against schemas and
expected ranges [14]. Robustness benchmarks built from controlled corruptions are well established in computer vision [15]; our
corruption benchmark applies the same idea to ICU time series, with clinically motivated fault types.</p>
<p><b>Explainability.</b> SHAP [16] and its exact tree variant TreeSHAP [17] are widely used to explain tree-ensemble risk scores. We use
TreeSHAP for risk drivers but treat it as one part of a broader explanation that also names the integrity check behind each trust
decision.</p>

<h2>4. Dataset</h2>
<p>We use the public PhysioNet/CinC 2019 Sepsis Challenge training data [4, 18] (CC BY 4.0): 40,336 ICU stays from two US hospital
systems (Hospital A, 20,336 patients; Hospital B, 20,000 patients), 1,552,210 patient-hours and 2,932 patients who met the Sepsis-3 based
challenge definition. Each hour contains up to 8 vital signs, 26 laboratory values and 6 demographic or administrative variables; most
laboratory values are missing at most hours, which is normal for ICU data. The challenge labels every hour from 6 hours before the
clinical onset time onward as positive; we follow the challenge convention and recover onset as the first positive hour plus 6 hours.</p>
<p><b>Splits.</b> Patients (never hours) are split 70/10/20 into training, validation and test sets, stratified by hospital and sepsis
status ({n(R['main']['n_patients']['train'])} / {n(R['main']['n_patients']['val'])} / {n(R['main']['n_patients']['test'])} patients).
Every threshold, calibrator, integrity rule and shift reference was fitted on training or validation patients only. For external
validation we additionally train on one hospital and test on the other. Test-set prevalence is {pct(M['prevalence'], 2)} of hours
({n(M['n_septic'])} septic patients).</p>

<h2>5. System architecture</h2>
{fig("0_architecture.png", "Figure 1. SepsisShield architecture. The prediction path and the trust path run side by side on the same hourly data; a decision layer combines them. The distribution-shift indicator is shown beside the decision and never changes it.")}
<p>The system has two paths and a decision layer:</p>
<ul>
<li><b>Prediction path:</b> raw hourly data → 172 causal features → five LightGBM models → mean probability → isotonic calibration →
risk score and alert (threshold {M['threshold']:.2f}) → TreeSHAP risk drivers. Model confidence is the spread of the five models'
log-odds.</li>
<li><b>Trust path:</b> integrity checks run on the <i>raw</i> measurements, before feature engineering, so they do not depend on the
model. They grade input trust as HIGH, REDUCED or LOW and record which check fired.</li>
<li><b>Decision layer:</b> HIGH → <b>SHOW PREDICTION</b>; REDUCED → <b>VERIFY INPUTS</b> (prediction shown with a warning); LOW →
<b>PREDICTION WITHHELD</b> (the research score is kept for audit but not presented as actionable).</li>
<li><b>Advisory context:</b> a distribution-shift indicator (LOW / MODERATE / HIGH) is computed from the model inputs and displayed
beside the decision. It never changes the risk, the alert, the trust state or the decision.</li>
</ul>
<p>The key design rule, implemented in <code>src/integrity.py::trust_level</code>, is asymmetric: LOW trust, and therefore withholding,
is triggered <i>only</i> by evidence from the clinical inputs (physiologically implausible values, impossible blood-pressure combinations,
or a coordinated "normalising" shift across vitals). Ensemble disagreement can lower trust to REDUCED but can never cause a withhold and
can never raise trust. This keeps the strongest action tied to evidence a clinician can check.</p>

<h2>6. Sepsis prediction model</h2>
<p><b>Features.</b> All 172 features are causal: the value at hour t uses only observations at hours ≤ t, so the model could run in
real time. Feature groups are last observed values, hours since each variable was last measured (informative missingness), rolling
statistics and deltas over 6 and 12 hours, and clinical composites (SIRS score, partial qSOFA, shock index and similar). Missing values
are left as missing; LightGBM handles them natively, and the missingness pattern itself is predictive (Table 6).</p>
<p><b>Model.</b> Five LightGBM [19] models with different random seeds (learning rate 0.02, 31 leaves, minimum 500 samples per leaf,
feature fraction 0.5, bagging 0.8, L2 1.0; early stopping on validation AUROC). Hyper-parameters were chosen on the validation set only.
The ensemble probability is the mean of the five members.</p>
<p><b>Model families compared.</b> To check that the choice of model is reasonable, we trained random forest, XGBoost [20] and CatBoost
[21] on the same split with the same protocol (early stopping, isotonic calibration and utility-optimal threshold all chosen on validation
patients; one evaluation on test). These are comparison models only; the deployed model was not changed or retrained.</p>
<div class="tw">{t_models}<div class="tcap">Table 1. Model families on the held-out test set ({n(R['main']['n_patients']['test'])} patients, {n(cf['hours'])} hours).
Sensitivity = septic patients alerted between 12 h before and 3 h after onset; specificity = non-septic patients never alerted. Random forest
was trained on a 400,000-row sub-sample (as was logistic regression) for memory reasons.</div></div>
<p><b>Finding.</b> Gradient-boosted trees dominate: XGBoost (AUROC {f3(models['XGBoost']['auroc'])}, utility
{f3(models['XGBoost']['utility'])}) lies inside the 95% confidence interval of the SepsisShield ensemble (AUROC {f3(M['auroc'])}, 95% CI
{f3(CI['auroc'][0])}–{f3(CI['auroc'][1])}; utility {f3(M['utility'])}, CI {f3(CI['utility'][0])}–{f3(CI['utility'][1])}). Random forest
and CatBoost reach {f3(models['Random forest']['auroc'])} and {f3(models['CatBoost']['auroc'])}; logistic regression
{f3(base['Logistic regression (+ isotonic)']['auroc'])}; SIRS and partial qSOFA rules {f3(base['SIRS criteria (rule: ≥2 of 4)']['auroc'])}
and {f3(base['qSOFA-partial (rule: ≥1 of 2)']['auroc'])}. We kept the LightGBM ensemble because it was the model already shipped and
evaluated end to end, and because its five members provide the disagreement signal used by the trust layer. <b>The contribution of
SepsisShield is not a better risk model; it is what surrounds the risk model.</b></p>
<p><b>Clinical operating point.</b> At the validation-chosen threshold the system alerts {pct(M['patient_sensitivity'])}
({CI['patient_sensitivity'][0]*100:.1f}–{CI['patient_sensitivity'][1]*100:.1f}%) of septic patients within the useful window, alerts
{pct(M['pct_detected_ge6h_early'])} at least 6 hours before onset (median lead time {M['median_lead_time_h']:.0f} h), and leaves
{pct(M['patient_specificity'])} of non-septic patients without any alert. The false-alert burden is
{M['false_alert_hours_per_100_patient_days']:.0f} alert-hours per 100 non-septic patient-days, which is high and is a known weakness of
hourly sepsis alerting.</p>
<p><b>Operating-point trade-off.</b> The threshold is a deployment choice, not a property of the model. Figure 2 and Table 1b
show the trade-off on the held-out test set. The shipped 3% threshold maximizes utility on validation patients
({_op[0.03]['utility_val']:.3f}); 2% would alert more septic patients ({pct(_op[0.02]['patient_sensitivity'])}) at
{_op[0.02]['alert_episodes_per_100_nonseptic_patient_days']:.0f} new alert episodes per 100 non-septic patient-days instead of
{_op[0.03]['alert_episodes_per_100_nonseptic_patient_days']:.1f}; 5% would cut that to
{_op[0.05]['alert_episodes_per_100_nonseptic_patient_days']:.0f} but alert only {pct(_op[0.05]['patient_sensitivity'])} of septic
patients. Counting alert <i>episodes</i> (an alert that starts after a non-alert hour) is closer to what a clinician experiences
than counting alert-hours. A site would choose its own point with local data; the trust layer works the same at any threshold.</p>
{fig("17_operating_points.png", "Figure 2. Operating points of the shipped model on the held-out test set. The dotted line marks the shipped threshold, chosen on validation utility.", "88%")}
<div class="tw">{t_op}<div class="tcap">Table 1b. Selected operating points (held-out test set; utility also shown on validation, where the threshold was chosen).</div></div>

<h2>7. Data trust layer</h2>
<p>The integrity layer runs five checks on raw hourly measurements, each fitted on training patients only:</p>
<ol>
<li><b>Plausibility:</b> value outside physiologically possible limits (deliberately wide: "cannot be real", not "abnormal"; e.g.
HR 20–250, temperature 30–43 °C, SBP 40–280 mmHg).</li>
<li><b>Consistency:</b> impossible combinations within one blood-pressure reading (DBP ≥ SBP, MAP &gt; SBP + 5); softer discordance
(MAP below DBP, SpO₂ vs SaO₂ disagreement &gt; 25 points) is recorded separately because it also occurs legitimately.</li>
<li><b>Jump:</b> hour-to-hour change larger than the 99.95th percentile seen in clean training data for that vital sign.</li>
<li><b>Coordinated shift:</b> several vitals move toward "normal" together (HR, respiratory rate and temperature down; SBP and MAP up),
scored as a summed, clipped z-score against the patient's own recent baseline. This is a signature of manual overwriting; its threshold
flags 0.25% of clean training hours.</li>
<li><b>Flatline:</b> three core vitals identical for at least 8 consecutive hours (frozen feed or copied-forward values).</li>
</ol>
<p>Because corrupted values propagate into rolling features, any hard flag keeps the patient at REDUCED trust for 6 hours afterwards.
Ensemble disagreement above the 99th percentile of validation hours (log-odds spread &gt; 0.293) also lowers trust to REDUCED.</p>
<p><b>Why there is no 0–100 "trust score".</b> A numeric trust score would invite readers to treat 43/100 as meaningfully different
from 47/100. We have no outcome data that could calibrate such a scale, so we report three evidence-based levels and always name the
check that fired. <b>Why missing values do not lower trust.</b> Most laboratory values are missing at most hours by design, and missingness
is itself predictive; flagging it would withhold most predictions. Section 10 measures what this choice costs.</p>

<h2>8. Calibration</h2>
<p>The ensemble's mean probability is passed through isotonic regression fitted on validation patients. Calibration matters because
the dashboard shows a risk percentage, and because the alert threshold is applied to the calibrated score.</p>
<div class="two">
{fig("1_calibration.png", "Figure 3. Reliability diagram on the held-out test set (15 equal-mass bins), before and after isotonic calibration.")}
<div><p style="margin-top:14pt"><b>Result.</b> The raw ensemble was already well calibrated (ECE {cal['before']['ece']:.4f}, Brier
{cal['before']['brier']:.4f}). Isotonic calibration lowered ECE modestly to {cal['after']['ece']:.4f}, with an essentially unchanged
Brier score ({cal['after']['brier']:.4f}) and AUROC. We report this honestly: on this dataset calibration is a small refinement, not a
major contributor.</p>
<p>Calibration does not survive a change of hospital or prevalence unchanged. In the cross-hospital experiments (Section 14) the
transferred model's ECE rose to {rc['A→B']['as_is']['ece']:.4f} (A→B) and {rc['B→A']['as_is']['ece']:.4f} (B→A); refitting the
calibrator and threshold on 20% of local patients reduced it to {rc['A→B']['recalibrated']['ece']:.4f} and
{rc['B→A']['recalibrated']['ece']:.4f}. Under a simulated three-fold prevalence increase the mean predicted risk
({pct(prev['mean_predicted'], 1)}) fell well below the observed rate ({pct(prev['observed_rate'], 1)}).</p></div>
</div>

<h2>9. Distribution-shift detection</h2>
<p><b>Method.</b> A transparent, model-free indicator. We take the 20 model features with the highest mean LightGBM gain (excluding ICU
length of stay, a time index), record each feature's 0.5th–99.5th percentile range on training patients, compute at each hour the share
of observed features outside their range, and average it over the previous 6 hours. Thresholds are the 95th and 99th percentiles of that
score on training patient-hours: LOW ≤ {SE['thresholds']['moderate']:.4f} &lt; MODERATE ≤ {SE['thresholds']['high']:.4f} &lt; HIGH.</p>
<p><b>Behaviour on real data.</b> On the test set {pct(SE['test_rates']['MODERATE'])} of hours are MODERATE and
{pct(SE['test_rates']['HIGH'])} HIGH, close to the 4% and 1% expected by construction. When the reference is fitted on one hospital and
applied to the other, MODERATE + HIGH hours rise from {pct(SE['cross_hospital']['reference_A']['test_A_same']['MODERATE'] + SE['cross_hospital']['reference_A']['test_A_same']['HIGH'])}
to {pct(SE['cross_hospital']['reference_A']['test_B_other']['MODERATE'] + SE['cross_hospital']['reference_A']['test_B_other']['HIGH'])} (A→B)
and from {pct(SE['cross_hospital']['reference_B']['test_B_same']['MODERATE'] + SE['cross_hospital']['reference_B']['test_B_same']['HIGH'])}
to {pct(SE['cross_hospital']['reference_B']['test_A_other']['MODERATE'] + SE['cross_hospital']['reference_B']['test_A_other']['HIGH'])} (B→A).
AUROC on HIGH-shift hours was {f3(SE['by_state_test']['HIGH']['auroc'])} (95% CI {f3(SE['by_state_test']['HIGH']['auroc_ci95'][0])}–{f3(SE['by_state_test']['HIGH']['auroc_ci95'][1])})
versus {f3(SE['by_state_test']['LOW']['auroc'])} on LOW-shift hours: lower, but with a wide interval.</p>
<p><b>Simulated cohort shifts.</b> We applied five whole-cohort perturbations to the stress-test cohort (Section 10) and compared two
real age subgroups. The indicator reacts clearly to shifted vital signs (2.3× the reference flag rate) and to measurement noise (2.2×),
but barely to missing laboratory values, to an older cohort whose ages remain inside the training range, or to a change in prevalence
(which does not change any input). The prevalence change is nevertheless harmful: hourly AUROC fell to {f3(prev['auroc'])} because the
case mix of negative hours changed, and calibration-in-the-large broke (above). A range-based indicator cannot see such shifts.</p>
{fig("15_cohort_shift.png", "Figure 4. Shift flag rate and AUROC under simulated cohort shifts (stress-test cohort: all septic test patients + 2,500 random non-septic test patients; reference AUROC 0.811 because the cohort is enriched for septic patients).", "82%")}
<div class="tw">{t_cohort}<div class="tcap">Table 2. Simulated cohort shifts. "vs reference" is the ratio of MODERATE + HIGH hours to the unperturbed cohort.</div></div>
<p class="note"><b>Interpretation.</b> Distribution-shift awareness indicates that an input pattern differs from the training
distribution; it does not establish that a prediction is incorrect or clinically unsafe. Section 13 shows that turning it into a
flag would almost double the share of clean hours that are flagged for a small gain, which is why it remains advisory.</p>

<h2>10. Robustness and corruption testing</h2>
<p><b>Cohort and protocol.</b> All corruption experiments use the same stress-test cohort: every septic test patient (586) plus 2,500
randomly chosen non-septic test patients ({n(ER['cohort']['hours'])} hours). Faults are injected into the <i>raw</i> data and the full
pipeline is re-run: features, models, calibration, integrity checks, shift indicator. No model is retrained. For septic patients the
corruption window is placed in the clinically decisive period (12 h before to 3 h after onset), the worst case.</p>
<p><b>Safety metric.</b> A <i>dangerous failure</i> is a patient-hour inside the corruption window where the fault flipped the alert
decision <i>and</i> the new decision is wrong: a timely alert suppressed in a septic patient's useful window, or a new alert outside any
useful window. Dangerous failure coverage is the share of these that are withheld or warned rather than shown silently.</p>
<h3>10.1 Accidental faults and deliberate edits</h3>
<div class="tw">{t_faults}<div class="tcap">Table 3. Corruption benchmark on held-out patients. "Warned" = shown with VERIFY INPUTS. The validation-cohort
replication (1,543 validation patients never used to design the input-integrity checks) gives {pct(val['coverage'])} ({n(val_cov)} / {n(val['n'])}) for the
accidental faults.</div></div>
<p>Unit errors and monitor artefacts are almost always caught because they produce impossible values. Frozen feeds are the weakest
accidental fault ({pct((by_fault['frozen_feed']['withheld'] + by_fault['frozen_feed']['flagged']) / by_fault['frozen_feed']['n'])}): a
copied-forward value is plausible by definition, and the flatline rule needs 8 identical hours. Deliberate edits are the principal
limitation. The simulated edit subtracts fixed amounts, which sometimes pushes values outside physiology and helps detection; when the
edited values are kept inside normal adult ranges, coverage falls from {pct(mask['coverage'])} to {pct(pe['coverage'])}
({n(pe_cov)} / {n(pe['n'])}). The benign noise control is rarely flagged, as intended.</p>
<h3>10.2 Physiologically impossible values</h3>
<p>Injecting HR = 900, temperature = 70 °C or SBP = −20 into 1% of hours ({n(ext['injected_hours'])} hours) produced LOW trust and a
withheld prediction at <b>{n(ext['injected_hours_trust_low'])} of {n(ext['injected_hours'])}</b> injected hours.</p>
<h3>10.3 Missingness and measurement-noise sweeps</h3>
{fig("12_robustness_sweeps.png", "Figure 5. AUROC of the shipped model as observed values are deleted at random (left) or Gaussian noise is added to HR, SBP, temperature, respiratory rate, lactate and creatinine (right). Stress-test cohort; baseline 0.811.", "95%")}
<div class="tw">{t_sweep}<div class="tcap">Table 4. Missingness removes the given share of all observed vital and lab values at random, across the whole stay.
Noise adds N(0, k·SD) to six variables across the whole stay. "Dangerous decision flips" counts alert decisions flipped to a wrong
answer; the percentage is the share flagged or withheld.</div></div>
<p><b>Findings.</b> Discrimination degrades gradually: AUROC falls from {f3(sweep['baseline_clean']['auroc'])} to {f3(miss40['auroc'])} with
40% of observed values deleted and to {f3(noise50['auroc'])} with noise at half a population standard deviation. Patient-level sensitivity
falls from {pct(sweep['baseline_clean']['patient_sensitivity'])} to {pct(miss40['patient_sensitivity'])} under 40% missingness. The trust
layer deliberately does <i>not</i> treat missingness as an input fault, so the alert decisions that missingness changes are mostly
<b>not flagged</b> ({pct(miss40['decision_flips']['coverage'])} at 40%). Heavy noise raises the shift flag rate (to
{pct(shift_rate(noise50))}) and the REDUCED-trust rate, but most noise-induced decision flips are also silent. This is a real limitation:
SepsisShield is designed to catch inputs that are <i>wrong</i>, not inputs that are <i>absent</i> or merely noisy.</p>

<h3>10.4 A pre-registered attempt to catch plausible edits: cross-signal consistency</h3>
<p>The weakest areas above are plausible deliberate edits and frozen feeds: the values are individually normal. We therefore
tested one new trust mechanism under a protocol fixed before the test run (<code>src/exp_crosssignal.py</code>). For each of six
vitals, a LightGBM regressor fitted on training patients predicts the current value from the other current vitals, the last
laboratory values, every vital's own baseline 7–24 hours earlier, and demographics. The hourly score is the mean absolute
standardized residual (each residual scaled by its training-patient MAD and capped at 10); the flag threshold ({XJ['threshold']:.2f}) is the 99th percentile on clean validation hours, and a flag
could only lower trust to REDUCED. The ship criterion, also fixed in advance, was a gain of at least 10 percentage points on
plausible edits with at most 2 points more clean hours flagged.</p>
<div class="tw">{t_xsig}<div class="tcap">Table 4b. Dangerous-failure coverage of the current trust layer before and after adding the
cross-signal flag. Validation = development cohort; test = the single pre-specified evaluation. Clean test hours flagged or
withheld: {pct(XT['clean_full_test']['flagged_or_withheld_before'])} → {pct(XT['clean_full_test']['flagged_or_withheld_after'])}.</div></div>
<p><b>Result.</b> On the held-out test set, plausible-edit coverage moved from {pct(XT['masking_plausible']['before']['coverage'])}
to {pct(XT['masking_plausible']['after']['coverage'])} ({XD['plausible_edit_gain']*100:+.1f} points) while clean flags rose by
{XD['clean_cost_increase']*100:.1f} points. {"This meets the pre-registered criterion, so the detector is shipped as a REDUCED-only warning." if XD['ship'] else "This does not meet the pre-registered criterion, so the detector is <b>not shipped</b>; it is reported as a negative result."}
The reason is instructive: a coordinated edit changes several vitals together, and each vital's prediction is built from the
other (also edited) vitals, so the edited values remain mutually consistent. Catching such edits probably needs signals the
editor does not control — device-captured versus manually entered values and the EHR audit trail — rather than more modelling
of the same vitals.</p>

<h2>11. Selective prediction</h2>
<p><b>Clean data.</b> Abstaining on the hours with the highest ensemble disagreement lowers the Brier score of the retained hours (from
{rcov[0]['brier_uncertainty']:.4f} at full coverage to {rc70['brier_uncertainty']:.4f} at 70% coverage, versus
{rc70['brier_random']:.4f} for random abstention) but slightly <i>lowers</i> AUROC ({f3(rcov[0]['auroc_uncertainty'])} →
{f3(rc70['auroc_uncertainty'])}). Disagreement concentrates on higher-risk hours, so removing them improves squared error while making
the remaining ranking task harder. Model uncertainty is therefore a weak basis for abstention on clean data, which is one reason
SepsisShield never withholds on disagreement alone.</p>
{fig("16_risk_coverage.png", "Figure 6. Risk-coverage curves on all 309,270 clean held-out test hours.", "70%")}
<p><b>Data faults: which abstention signal catches failures?</b> On the {n(sig['hours'])} patient-hours of the four accidental-fault runs
({n(sig['dangerous_failures'])} dangerous failures), we ranked hours by four signals and abstained on the top 5–40%.</p>
<div class="two">
{fig("13_abstention_signals.png", "Figure 7. Share of dangerous failures caught when abstaining on the top-ranked hours by each signal.")}
<div>{t_sig}<div class="tcap">Table 5. Dangerous failures caught at matched abstention rates (accidental-fault benchmark).</div>
<p>Ensemble disagreement catches between a third and two-thirds as many failures as the input-trust ranking: the five models agree with each other
on corrupted inputs. The shift score is competitive at low abstention rates, largely because the °F fault drives temperature far outside
its training range. At its natural operating point the trust policy flags or withholds {pct(sel['trust_policy_operating_point']['hours_flagged_or_withheld_pct'])}
of benchmark hours and catches {pct(acc['coverage'])} of failures.</p></div>
</div>

<h2>12. Explainability</h2>
<p>Every decision on the dashboard is explained in one panel, "Why was this decision made?", which combines four pieces of existing
evidence: (1) the top three TreeSHAP risk drivers for that hour, with direction and value; (2) the integrity check that set the trust level,
in plain language (for example "Physiologically implausible temperature — check units or sensor"); (3) the shift state and the features
outside their training range; and (4) the decision rule that was applied. Global importance (Figure 8) is led by care-process and timing
features (hours since FiO₂ was measured, hours in the ICU, time from hospital to ICU admission), followed by SIRS criteria, ICU type,
renal labs (BUN, creatinine) and temperature. The prominence of process features is a caution: part of the signal reflects how
clinicians monitor sick patients, which can change between hospitals.</p>
<div class="two" style="grid-template-columns: 0.8fr 1.2fr">
{fig("7_shap_global.png", "Figure 8. Global TreeSHAP importance (mean |SHAP|, held-out test hours).")}
{shot("02_prediction_withheld.png", "Figure 9. Dashboard, scenario C: the models are confident, the coordinated-shift detector fires, and the prediction is withheld. The four cards separate risk, model confidence, input trust and decision.")}
</div>
<p>SHAP is not the novelty here. What the panel adds is a <i>trust explanation</i>: a clinician can see that a prediction was withheld
because a temperature of 98.6 is impossible in °C, not because the model was unsure.</p>

<h2>13. Ablation study</h2>
<h3>13.1 Component ablation</h3>
<p>We added components one at a time and measured, on the accidental-fault benchmark, how many dangerous failures each configuration
catches and what it costs on clean data (all hours of the 8,068 test patients).</p>
<div class="tw">{t_abl}<div class="tcap">Table 6a. Component ablation. AUROC and ECE are on clean held-out data; failures are the 6,790 accidental-fault dangerous
failures. Row 6 is a counterfactual and is not deployed.</div></div>
{fig("14_component_ablation.png", "Figure 10. What each component adds. Model-side improvements catch no data-fault failures; the input-integrity checks do almost all of the work. A shift flag would add one percentage point of coverage at the cost of nearly doubling clean flags.", "82%")}
<p><b>Findings.</b> (1) A better or better-calibrated model does nothing for data faults: rows 1–3 catch 0 failures, because they never
look at whether inputs are believable. (2) Ensemble disagreement catches only {pct(abl[3]['coverage'])}. (3) The input-integrity checks
raise coverage to {pct(abl[4]['coverage'])} while flagging or withholding {pct(abl[4]['clean_hours_flagged_pct'])} of clean hours in
total (withholding {pct(abl[4]['clean_hours_withheld_pct'], 2)}). (4) Also flagging on MODERATE/HIGH shift would catch {n(abl[5]['dangerous_failures_caught'] - abl[4]['dangerous_failures_caught'])}
more failures ({pct(abl[5]['coverage'])}) but raise clean flags to {pct(abl[5]['clean_hours_flagged_pct'])}. We judged that trade-off not
worth making, so shift stays advisory.</p>
<h3>13.2 Feature-group ablation</h3>
<div class="tw">{t_feat_abl}<div class="tcap">Table 6b. Incremental feature groups (single LightGBM, same split). Missingness patterns and temporal trends add the most.</div></div>

<h2>14. Cross-hospital validation and subgroups</h2>
<div class="tw">{t_cross}<div class="tcap">Table 7. Train on one hospital, test on the other. Local recalibration refits only the isotonic calibrator and threshold
on 20% of destination patients and evaluates on the other 80%.</div></div>
<p>Discrimination drops at an unseen hospital (AUROC {f3(a2b['auroc'])} and {f3(b2a['auroc'])}, versus {f3(a2b['internal_val']['auroc'])}
and {f3(b2a['internal_val']['auroc'])} internally). The B→A operating point is badly placed (specificity {pct(b2a['patient_specificity'])});
local recalibration helps calibration and specificity but cannot recover discrimination. Any deployment would need local validation.</p>
<div class="tw">{t_sub}<div class="tcap">Table 8. Subgroup audit on the held-out test set with patient-level bootstrap 95% CIs. No subgroup is far below the
overall AUROC, but Hospital A, age 80+ and patients with unknown ICU type are at the low end.</div></div>

<h2>15. Failure-case analysis</h2>
<p>All cases are real held-out test patients, scored live in the public demo; each is pinned by an automated test so the demo cannot
drift from what the models do.</p>
<p><b>Case 1 — false alarm from a unit error (scenario B, p119917).</b> The thermometer reports °F into the °C field for hours 34–41.
Without the fault the risk at hour 36 is 0.9% (no alert); with it, 3.9% — an alert 28 hours before onset, outside the useful window. All
five models agree (confident). The plausibility check fires, trust is LOW, and the false alert is withheld with the message
"Physiologically implausible temperature — check units or sensor". At hour 37 the same patient is at REDUCED trust (VERIFY INPUTS),
because the sticky window keeps the patient flagged while corrupted values remain in the trend features.</p>
<p><b>Case 2 — hidden alert from an edited chart (scenario C, p018345).</b> Vitals are overwritten to look normal for hours 46–57. On
the unedited data the model alerts at hour 46 (risk 3.5%); on the edited data the risk drops to 2.3%, below the alert threshold — a
confident all-clear. The
coordinated-shift detector fires and the prediction is withheld rather than shown as reassuring. This case succeeds, but Table 3 shows
that most edits of this kind are missed.</p>
<p><b>Case 3 — an unusual patient (scenario D, p105030).</b> Inputs are physiologically possible (trust HIGH), but a white-cell count of
100–170 ×10⁹/L is far outside the training range (shift HIGH). The model shows low risk; the patient met the sepsis definition 7 hours
later. The shift indicator gave the only warning. This is one example, not evidence that shift predicts error.</p>
<p><b>Case 4 — models disagree on clean inputs (scenario F, p010049).</b> At hour 62 the inputs pass every integrity check, but the
five models disagree more than on 99% of validation hours. Trust is REDUCED and the alert (risk 23.8%, 10 hours before onset) is shown
with a verification warning. It is never withheld, because there is no input evidence against it.</p>
<p><b>Case 5 — where SepsisShield is silent.</b> Of the {n(acc['n'])} accidental-fault failures, {n(acc['silent'])} were shown silently;
{n(by_fault['frozen_feed']['silent'])} of them came from frozen feeds whose copied-forward values are plausible and change too little for
the flatline rule. Random missingness and plausible deliberate edits produce further silent failures (Tables 3 and 4). These are the
cases a clinician would not be warned about.</p>
{shot("03_fahrenheit_fault.png", "Figure 11. Dashboard, scenario B: the clean-vs-corrupted comparison shows the risk crossing the alert line while model confidence stays unchanged and input trust falls to LOW.", "72%")}

<h2>16. Interactive demonstration</h2>
<p>The Streamlit dashboard is public and needs no login or setup. One-click Judge Demo scenarios load real held-out patients: A (clean
inputs, early warning), B (accidental °F fault, withheld), C (edited chart, withheld), and optional D (unusual patient, shift HIGH), E
(early sepsis warning: risk rises from 1.1% to 13.1% and the alert is shown 8 hours before onset) and F (models disagree → VERIFY INPUTS).
Every screen shows four cards — sepsis risk, model confidence, input trust, final decision — a clean-vs-corrupted comparison, the "Why
was this decision made?" panel, the shift panel, the patient's risk trajectory, and an "Evidence behind this claim" expander that traces
the 95.4% figure to its numerator and denominator. Users can also inject any of five faults into any demo patient at any hour, or upload a
PhysioNet .psv file.</p>
<div class="tw">{t_cmp}<div class="tcap">Table 9. Behaviour of a standard model versus SepsisShield in the scenarios the demo and benchmark cover.</div></div>

<h2>17. Limitations</h2>
<ul>
<li><b>Deliberate manipulation is largely undetected</b>: {pct(mask['coverage'])} ({n(mask_cov)} / {n(mask['n'])}) of alert-changing edits
caught, and {pct(pe['coverage'])} ({n(pe_cov)} / {n(pe['n'])}) when edits stay physiologically plausible. Vitals-only checks cannot
reliably detect careful falsification; a pre-registered cross-signal consistency detector added only
{XD['plausible_edit_gain']*100:.1f} points on held-out data and was not shipped (Section 10.4).</li>
<li><b>Missing and noisy data are not flagged</b>; the decision changes they cause are mostly silent (Section 10.3).</li>
<li><b>Generalisation:</b> AUROC {f3(a2b['auroc'])} / {f3(b2a['auroc'])} at an unseen hospital; prevalence and case-mix changes affect both
discrimination and calibration and are invisible to the shift indicator.</li>
<li><b>All faults are simulated.</b> Their parameters are plausible but are not drawn from recorded hospital incidents; real fault rates
and mixtures are unknown.</li>
<li><b>Clean-data cost:</b> {pct(cf['patients_ever_withheld_pct'])} of test patients have at least one withheld hour (median 1 hour);
about half of clean withholds come from the coordinated-shift detector firing on real patients, likely false flags such as rapid
improvement after treatment.</li>
<li><b>Labels and alert burden:</b> the challenge's label definition is a retrospective approximation of sepsis onset, and the false-alert
rate ({M['false_alert_hours_per_100_patient_days']:.0f} alert-hours per 100 non-septic patient-days) would need workflow design.</li>
<li><b>No prospective or clinician evaluation</b> has been done. Nothing here establishes clinical benefit.</li>
</ul>

<h2>18. Clinical and scientific impact</h2>
<p>The main scientific finding is negative for a common assumption: improving or calibrating a risk model, or adding ensemble
uncertainty, does almost nothing to protect against corrupted inputs (Table 6a). Protection comes from checking the inputs themselves,
independently of the model, and from making the strongest action depend on evidence a human can verify. For accidental faults, this
catches most dangerous failures at a small clean-data cost; for deliberate manipulation and missing data it does not, and we show where.</p>
<p>If developed further and validated prospectively, the pattern could reduce two harms of early-warning systems: false reassurance from
corrupted data and alarm fatigue from impossible values. Because the vitals-only cross-signal detector failed its pre-registered test (Section 10.4), the next research
steps are provenance signals from the EHR audit trail, consistency checks against signals an editor does not control, conformal
bounds on abstention, and local
recalibration with drift monitoring at each new site. A clinical collaborator would be essential for any of these.</p>

<h2>19. Reproducibility</h2>
<p>Code, models and results are public. <code>make test</code> runs the automated test suite (75 tests; works without the raw data
because the demo patients and result files are bundled). <code>./run_all.sh</code> rebuilds every number in this report from the raw
PhysioNet files in about 75 minutes on two CPU cores; <code>tools/report/build_report.py</code> regenerates this PDF directly from the
result files, so the text cannot drift from the experiments. All randomness is seeded; thresholds, calibrators and integrity rules are fitted
on training or validation patients only, and demo patients are held-out test patients (checked by a test).</p>
{table(["File", "What it does / writes"], [['<code>src/train.py</code>', 'model, splits, cross-hospital, feature ablation     results/results.json'], ['<code>src/integrity.py</code>', 'trust-layer thresholds (training patients only)       models/integrity.json'], ['<code>src/abstention.py</code>', 'corruption benchmark, dangerous-failure coverage      results/abstention*.json'], ['<code>src/shift.py</code>', 'distribution-shift reference and evaluation           results/shift_evaluation.json'], ['<code>src/experiments2.py</code>', 'baselines, bootstrap CIs, recalibration               results/experiments2.json'], ['<code>src/exp_models.py</code>', 'random forest, XGBoost, CatBoost (comparison only)    results/exp_models.json'], ['<code>src/exp_robustness.py</code>', 'sweeps, selective prediction, ablation, cohort shift   results/exp_robustness.json'], ['<code>src/exp_operating_points.py</code>', 'threshold / alert-burden trade-off → results/exp_operating_points.json'], ['<code>src/crosssignal.py</code>, <code>src/exp_crosssignal.py</code>', 'cross-signal detector and its pre-registered evaluation (not shipped) → results/exp_crosssignal.json'], ['<code>app/app.py</code>', 'public Streamlit dashboard (Judge Demo scenarios A–F)']], "small")}

<h2>20. Conclusion</h2>
<p>SepsisShield does not only ask "what is this patient's sepsis risk?" It also asks "can this prediction safely be trusted?", and it
answers the second question from the clinical inputs, independently of the model. On held-out data this catches most of the dangerous
failures caused by simulated accidental faults while withholding under half a percent of clean predictions. It is a research prototype
with clear limits — deliberate edits, missing data and new hospitals — and those limits are reported alongside the results.</p>

<h2>References</h2>
<ol class="refs">
<li>Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). <i>JAMA</i>. 2016;315(8):801–810.</li>
<li>Rudd KE, et al. Global, regional, and national sepsis incidence and mortality, 1990–2017. <i>Lancet</i>. 2020;395:200–211.</li>
<li>Kumar A, et al. Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. <i>Crit Care Med</i>. 2006;34(6):1589–1596.</li>
<li>Reyna MA, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. <i>Crit Care Med</i>. 2020;48(2):210–217.</li>
<li>Wong A, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. <i>JAMA Intern Med</i>. 2021;181(8):1065–1070.</li>
<li>Finlayson SG, et al. The clinician and dataset shift in artificial intelligence. <i>N Engl J Med</i>. 2021;385:283–286.</li>
<li>Sendak MP, et al. Real-world integration of a sepsis deep learning technology into routine clinical care: implementation study. <i>JMIR Med Inform</i>. 2020;8(7):e15182.</li>
<li>Zadrozny B, Elkan C. Transforming classifier scores into accurate multiclass probability estimates. <i>KDD</i>. 2002.</li>
<li>Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. <i>ICML</i>. 2017.</li>
<li>Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. <i>NeurIPS</i>. 2017.</li>
<li>El-Yaniv R, Wiener Y. On the foundations of noise-free selective classification. <i>JMLR</i>. 2010;11:1605–1641.</li>
<li>Geifman Y, El-Yaniv R. Selective classification for deep neural networks. <i>NeurIPS</i>. 2017.</li>
<li>Rabanser S, Günnemann S, Lipton ZC. Failing loudly: an empirical study of methods for detecting dataset shift. <i>NeurIPS</i>. 2019.</li>
<li>Breck E, Polyzotis N, Roy S, Whang SE, Zinkevich M. Data validation for machine learning. <i>MLSys</i>. 2019.</li>
<li>Hendrycks D, Dietterich T. Benchmarking neural network robustness to common corruptions and perturbations. <i>ICLR</i>. 2019.</li>
<li>Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. <i>NeurIPS</i>. 2017.</li>
<li>Lundberg SM, et al. From local explanations to global understanding with explainable AI for trees. <i>Nat Mach Intell</i>. 2020;2:56–67.</li>
<li>Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. <i>Circulation</i>. 2000;101(23):e215–e220.</li>
<li>Ke G, et al. LightGBM: a highly efficient gradient boosting decision tree. <i>NeurIPS</i>. 2017.</li>
<li>Chen T, Guestrin C. XGBoost: a scalable tree boosting system. <i>KDD</i>. 2016.</li>
<li>Prokhorenkova L, et al. CatBoost: unbiased boosting with categorical features. <i>NeurIPS</i>. 2018.</li>
</ol>
</body></html>"""



async def pdf(html_path, pdf_path):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.goto(html_path.as_uri(), wait_until="networkidle")
        await pg.pdf(path=str(pdf_path), format="Letter", print_background=True, display_header_footer=True,
                     header_template="<div></div>",
                     footer_template="<div style='width:100%;font-size:8px;color:#666;font-family:sans-serif;"
                                     "padding:0 0.75in;display:flex;justify-content:space-between'>"
                                     "<span>SepsisShield · technical report · research prototype, not a medical device</span>"
                                     "<span><span class='pageNumber'></span> / <span class='totalPages'></span></span></div>",
                     margin={"top": "0.75in", "bottom": "0.8in", "left": "0.75in", "right": "0.75in"})
        await b.close()


if __name__ == "__main__":
    h = OUT / "technical_report.html"
    h.write_text(HTML)
    asyncio.run(pdf(h, OUT / "technical_report.pdf"))
    print("wrote", h, OUT / "technical_report.pdf")
