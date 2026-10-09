"""Check every headline number in README / DEVPOST / MODEL_CARD / video script against the result artifacts."""
import json, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
DOCS = Path(__file__).resolve().parents[1]
J = lambda n: json.load(open(ROOT / "results" / n))
R, E, A, V, IB = J("results.json"), J("experiments2.json"), J("abstention.json"), J("abstention_val.json"), J("integrity_benchmark.json")
SG = {(r["attribute"], r["group"]): r for r in J("subgroups_ci.json")}
m, ci = R["main"]["calibrated"], E["bootstrap_ci"]["sepsisshield"]
acc, mk, cf = A["dfc_accidental"], A["dfc_masking"]["all"], A["clean_cost_full_test"]
S = {s["corruption"]: s["all"] for s in A["scenarios"]}
B = {b["model"]: b for b in E["baselines"]}
RC = {r["transfer"]: r for r in E["recalibration"]}
cov = lambda a: f"{(a['withheld'] + a['flagged']):,} / {a['n']:,}"
pct = lambda a: f"{(a['withheld'] + a['flagged']) / a['n'] * 100:.1f}%"

claims = {
    # prediction
    "test patients": f"{R['main']['n_patients']['test']:,}",
    "AUROC": f"{m['auroc']:.3f}", "utility": f"{m['utility']:.3f}",
    "AUROC CI": f"{ci['auroc'][0]:.3f}–{ci['auroc'][1]:.3f}", "utility CI": f"{ci['utility'][0]:.3f}–{ci['utility'][1]:.3f}",
    "AUPRC CI": f"{ci['auprc'][0]:.3f}–{ci['auprc'][1]:.3f}",
    "sensitivity": f"{m['patient_sensitivity']*100:.1f}%", "sens CI": f"{ci['patient_sensitivity'][0]*100:.1f}–{ci['patient_sensitivity'][1]*100:.1f}",
    "specificity": f"{m['patient_specificity']*100:.1f}%", "spec CI": f"{ci['patient_specificity'][0]*100:.1f}–{ci['patient_specificity'][1]*100:.1f}",
    ">=6h early": f"{m['pct_detected_ge6h_early']*100:.1f}%", "ECE": f"{m['ece']*100:.2f}",
    "external A->B": f"{R['cross_A2B']['auroc']:.3f}", "external B->A": f"{R['cross_B2A']['auroc']:.3f}",
    "baseline SIRS": f"{B['SIRS criteria (rule: ≥2 of 4)']['auroc']:.3f}", "baseline LR": f"{B['Logistic regression (+ isotonic)']['auroc']:.3f}",
    "baseline single": f"{B['Single LightGBM (uncalibrated)']['auroc']:.3f}",
    "recal B->A ECE": f"{RC['B→A']['as_is']['ece']*100:.2f} → {RC['B→A']['recalibrated']['ece']*100:.2f}",
    "recal B->A utility": f"{RC['B→A']['as_is']['utility']:.3f} → {RC['B→A']['recalibrated']['utility']:.3f}",
    # trust path (strict definition)
    "accidental coverage %": f"{acc['all']['coverage']*100:.1f}%", "accidental coverage n": cov(acc["all"]),
    "spurious": cov(acc["spurious"]), "suppressed": cov(acc["suppressed"]),
    "attributable": f"{acc['all']['attributable_coverage']*100:.1f}%",
    "masking %": f"{mk['coverage']*100:.1f}%", "masking n": cov(mk),
    "val accidental": f"{V['dfc_accidental']['all']['coverage']*100:.1f}% ({cov(V['dfc_accidental']['all'])})",
    "val masking": f"{V['dfc_masking']['all']['coverage']*100:.1f}% ({cov(V['dfc_masking']['all'])})",
    "°F": pct(S["temp_fahrenheit"]), "lab": pct(S["lab_unit_error"]), "frozen": pct(S["frozen_feed"]),
    "clean withheld %": f"{cf['withheld_pct']*100:.2f}%", "clean withheld n": f"{cf['withheld_hours']:,} / {cf['hours']:,}",
    "clean patients ever": f"{cf['patients_ever_withheld_pct']*100:.0f}% of patients",
    "real impossible hours": f"{IB['real_data_artifacts']['hours_implausible']:,}",
    # subgroups
    "80+ sens": f"73% (63–83)", "SICU sens": f"73% (65–79)",
}
# sanity: the subgroup strings above must equal the JSON
assert f"{SG[('age_group','80+')]['patient_sensitivity']*100:.0f}% ({SG[('age_group','80+')]['sens_ci'][0]*100:.0f}–{SG[('age_group','80+')]['sens_ci'][1]*100:.0f})" == claims["80+ sens"]
assert f"{SG[('icu','SICU')]['patient_sensitivity']*100:.0f}% ({SG[('icu','SICU')]['sens_ci'][0]*100:.0f}–{SG[('icu','SICU')]['sens_ci'][1]*100:.0f})" == claims["SICU sens"]

norm = lambda t: t.replace("**", "").replace(" / ", "/").replace(" ", "")
docs = {n: norm((DOCS / n).read_text()) for n in ["README.md", "submission/DEVPOST.md", "MODEL_CARD.md", "submission/VIDEO_SCRIPT.md"]}
bad = 0
for k, v in claims.items():
    where = [n.split("/")[-1] for n, t in docs.items() if norm(v) in t]
    bad += not where
    print(("OK  " if where else "MISS"), f"{k:26s} {v:28s} found in: {', '.join(where) or '—'}")
# stale numbers that must no longer appear anywhere
stale = ["8,439", "8,852", "95.3%", "372/864", "43.1%", "costs a life", "on par with"]
for n, t in docs.items():
    for sv in stale:
        if norm(sv) in t:
            bad += 1; print(f"STALE '{sv}' still in {n}")
# ML Empowerment documents: every percentage / AUROC-style number must be one of the verified values
import re
ME = sorted((DOCS / "submission" / "ml_empowerment").glob("*.md"))
ok_nums = {norm(v) for v in claims.values()} | {
    f"{m['patient_sensitivity']*100:.1f}%", f"{m['pct_detected_ge6h_early']*100:.1f}%",
    f"{V['dfc_accidental']['all']['coverage']*100:.1f}%", f"{(1-m['patient_specificity'])*100:.0f}%",
    f"{R['cross_A2B']['auroc']:.3f}", f"{R['cross_B2A']['auroc']:.3f}", "1.2%", "0.5%",   # design-iteration figures (DEVPOST)
    "3.5%", "2.3%",   # Judge Demo scenario C risk (asserted in tests/test_judge_demo.py)
}
SEV = J("shift_evaluation.json")    # distribution-shift awareness evaluation (src/shift.py)
_fl = lambda r: f"{(r['MODERATE'] + r['HIGH']) * 100:.1f}%"
for _r in SEV["cross_hospital"].values():
    ok_nums.update(_fl(v) for v in _r.values())
ok_nums.update({_fl(SEV["train_rates"]), _fl(SEV["test_rates"])})
for _v in SEV["by_state_test"].values():
    ok_nums.update({f"{_v['auroc']:.3f}", f"{_v['auroc_ci95'][0]:.3f}", f"{_v['auroc_ci95'][1]:.3f}"})
MRJ = J("masking_realism.json")    # deliberate-edit sensitivity check (tools/masking_realism_check.py)
pe = MRJ["plausible_edit"]
ok_nums.add(f"{pe['coverage']*100:.1f}%")
assert MRJ["original_edit"]["n"] == mk["n"] and MRJ["original_edit"]["withheld"] + MRJ["original_edit"]["flagged"] == mk["withheld"] + mk["flagged"]
claims_extra = {"plausible-edit coverage": f"{pe['coverage']*100:.1f}% ({pe['withheld']+pe['flagged']} / {pe['n']})"}
for k, v in claims_extra.items():
    where = [n.split("/")[-1] for n, t in docs.items() if norm(v) in t]
    bad += not where
    print(("OK  " if where else "MISS"), f"{k:26s} {v:28s} found in: {', '.join(where) or '—'}")
for f in ME:
    t = f.read_text()
    for tok in re.findall(r"(?<![\w.])\d+\.\d+%?", t):
        if tok.endswith("%") and tok not in ok_nums and norm(tok) not in "".join(ok_nums):
            bad += 1; print(f"UNVERIFIED '{tok}' in {f.name}")
        if not tok.endswith("%") and tok.startswith("0.") and len(tok) == 5 and tok not in "".join(claims.values()) and tok not in ok_nums:
            bad += 1; print(f"UNVERIFIED '{tok}' in {f.name}")
    for k in ["95.4%", "42.5%", "0.852"]:
        if k not in t and f.name in ("STORY.md", "PITCH.md", "VIDEO_PLAN.md"):
            bad += 1; print(f"MISSING '{k}' in {f.name}")
# the deliberate-edit result must never appear without the plausible-edit sensitivity result nearby
AI4S = sorted((DOCS / "submission" / "ai4s").glob("*.md")) + [DOCS / "docs" / "technical_report.html"]
AI4S = [f for f in AI4S if f.exists()]
for f in [DOCS / n for n in ["README.md", "MODEL_CARD.md", "submission/DEVPOST.md", "submission/SUBMISSION_GUIDE.md"]] + ME + AI4S:
    t = f.read_text()
    for mm in re.finditer(r"42\.5", t):
        if "26.5" not in t[max(0, mm.start() - 450):mm.end() + 450]:
            bad += 1; print(f"ALONE '42.5%' without 26.5% nearby in {f.name}, line {t[:mm.start()].count(chr(10)) + 1}")
# overclaims: phrases that must never appear unless negated just before
OVER = ["found real problems", "independently evaluat", "independent trust path", "clinically validated", "clinically deployed", "saves lives", "save lives", "proven to", "guaranteed safe",
        "detects all", "detect all", "prevents all", "prevent all", "production-ready", "universally general",
        "reduces mortality", "improves patient outcomes"]
scan = [DOCS / n for n in ["README.md", "MODEL_CARD.md", "DATA_CARD.md", "AUDIT.md", "submission/DEVPOST.md",
                           "submission/VIDEO_SCRIPT.md", "app/app.py", "app/scenarios.py"]] + ME + AI4S
for f in scan:
    t = f.read_text().lower().replace("\n", " ")
    for ph in OVER:
        for mm in re.finditer(re.escape(ph), t):
            before = t[max(0, mm.start() - 40):mm.start()]
            quoted = before.endswith('"')                     # the phrase itself being listed, e.g. in AUDIT.md
            if not quoted and not re.search(r"\b(not|never|no|n't|nor|without|does not|claim)\b", before):
                bad += 1; print(f"OVERCLAIM '{ph}' in {f.name}: ...{t[max(0, mm.start()-40):mm.end()+20]}...")
print(f"\n{len(claims)} claims checked, plus {len(ME)} competition documents and {len(scan)} files for overclaims; problems: {bad}")
sys.exit(1 if bad else 0)
