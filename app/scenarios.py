"""Judge Demo Mode scenarios and the stress-test corruptions used by the dashboard.

Every scenario is a real held-out test patient (see tests/test_isolation.py) scored by the shipped pipeline.
The expected decisions below are asserted in tests/test_judge_demo.py, so the demo cannot silently drift
away from what the models actually do.
"""
import numpy as np

CORRUPTIONS = ["None", "Thermometer reports °F", "Lab unit mix-up (SI units)", "Monitor artifact",
               "Frozen monitor feed", "Vitals overwritten to look normal"]


def apply_corruption(raw, kind, start, length):
    """Inject one simulated input failure into hours [start, start + length). Unchanged from the original dashboard."""
    raw = raw.copy()
    m = (raw["hour"] >= start) & (raw["hour"] < start + length)
    if kind == "Thermometer reports °F":
        raw.loc[m, "Temp"] = raw.loc[m, "Temp"] * 1.8 + 32
    elif kind == "Lab unit mix-up (SI units)":
        raw.loc[m, "Creatinine"] *= 88.4
        raw.loc[m, "Glucose"] /= 18.0
        raw.loc[m, "Lactate"] *= 9.0
    elif kind == "Monitor artifact":
        rng = np.random.default_rng(0)
        hit = m & (rng.random(len(raw)) < 0.5)
        raw.loc[hit, "HR"] = rng.uniform(190, 240, hit.sum()).round()
        raw.loc[hit, "SBP"] = rng.uniform(25, 45, hit.sum()).round()
    elif kind == "Frozen monitor feed":
        for c in ["HR", "SBP", "MAP", "DBP", "Resp", "O2Sat", "Temp"]:
            base = raw.loc[raw["hour"] <= start, c].ffill()
            v = base.iloc[-1] if len(base) else np.nan
            raw.loc[m, c] = v
    elif kind == "Vitals overwritten to look normal":
        raw.loc[m, "HR"] -= 25; raw.loc[m, "Resp"] -= 8; raw.loc[m, "Temp"] -= 1.2
        raw.loc[m, "SBP"] += 20; raw.loc[m, "MAP"] += 15
        raw.loc[m, "Lactate"] = raw.loc[m, "Lactate"].clip(upper=1.5)
        raw.loc[m, "WBC"] = raw.loc[m, "WBC"].clip(5, 11)
    return raw


# The three one-click scenarios. These are the same cases as the documented shareable links and the
# browser test (tools/dashboard_abstention_check.py); `expect` is what the shipped models produce.
SCENARIOS = {
    "A": dict(
        title="A · Clean inputs",
        blurb="The normal case. Clean inputs, trust HIGH, so the prediction is shown. This patient's first alert came "
              "at hour 55, 9 h before the recorded onset of sepsis.",
        pid="p119917", corr="None", start=None, length=None, hour=58,
        expect=dict(decision="SHOW", trust="HIGH", alert=True, confident=True),
    ),
    "B": dict(
        title="B · Accidental data fault",
        blurb="The thermometer reports °F into a °C field. The models stay confident, trust falls to LOW, "
              "and the false alert (28 h before onset, outside the useful window) is withheld.",
        pid="p119917", corr="Thermometer reports °F", start=34, length=8, hour=36,
        expect=dict(decision="WITHHELD", trust="LOW", alert=True, confident=True,
                    clean_alert=False, flag="flag_implausible"),
    ),
    "C": dict(
        title="C · Edited chart",
        blurb="Vitals overwritten to look normal hide the model's alert. The models stay confident; the manipulation "
              "detector fires, so the prediction is withheld, not a silent all-clear. Such edits are caught far less "
              "often than accidental faults (see limitations).",
        pid="p018345", corr="Vitals overwritten to look normal", start=46, length=12, hour=46,
        expect=dict(decision="WITHHELD", trust="LOW", alert=False, confident=True,
                    clean_alert=True, flag="flag_coordinated_shift"),
    ),
}
DEFAULT_SCENARIO = "A"

# Optional example for the distribution-shift awareness panel (clean data; nothing is injected).
OPTIONAL_SCENARIOS = {
    "D": dict(
        title="D · Unusual patient (distribution shift)",
        blurb="Possible values (trust HIGH), but a white cell count of 100–170 is far outside the training data "
              "(shift HIGH). The model shows low risk; this patient developed sepsis 7 h later. One example, not proof.",
        pid="p105030", corr="None", start=None, length=None, hour=40,
        expect=dict(decision="SHOW", trust="HIGH", alert=False, shift="HIGH"),
    ),
    "E": dict(
        title="E · Early sepsis warning",
        blurb="Clean inputs. Risk jumps from 1.1% to 13.1% at hour 44. At hour 46 the alert is shown with trust HIGH, "
              "8 h before the recorded onset of sepsis.",
        pid="p110755", corr="None", start=None, length=None, hour=46,
        expect=dict(decision="SHOW", trust="HIGH", alert=True, confident=True, shift="LOW"),
    ),
    "F": dict(
        title="F · Uncertain → verify",
        blurb="Clean inputs, but the 5 models disagree more than on 99% of validation hours. Trust is REDUCED, so the "
              "alert is shown with a warning. Model disagreement alone never withholds a prediction.",
        pid="p010049", corr="None", start=None, length=None, hour=62,
        expect=dict(decision="SHOW_WITH_WARNING", trust="REDUCED", alert=True, confident=False, shift="LOW"),
    ),
}
ALL_SCENARIOS = {**SCENARIOS, **OPTIONAL_SCENARIOS}

# Plain-language wording for the three trust levels and the decisions they lead to.
TRUST_TEXT = {
    "HIGH": "Inputs appear reliable.",
    "REDUCED": "Suspicious measurements detected — verify inputs.",
    "LOW": "Prediction withheld until the clinical data are checked.",
}
DECISION_TEXT = {
    "SHOW": "SHOW PREDICTION",
    "SHOW_WITH_WARNING": "VERIFY INPUTS",
    "WITHHELD": "PREDICTION WITHHELD",
}


def query_for(key):
    """Shareable query string for a scenario (same format the dashboard has always accepted)."""
    s = SCENARIOS[key]
    q = {"pid": s["pid"], "hour": str(s["hour"])}
    if s["corr"] != "None":
        q.update(corr=s["corr"], start=str(s["start"]), len=str(s["length"]))
    return q


def fault_effect(clean_row, row):
    """One-sentence description of what an injected fault changed at this hour (clean vs corrupted scoring)."""
    c, r = clean_row["risk"] * 100, row["risk"] * 100
    if row["decision"] == "WITHHELD":
        if row["alert"] and not clean_row["alert"]:
            return (f"Without the fault the risk is {c:.1f}% (no alert). The fault pushes it to {r:.1f}%, a false alert. "
                    "SepsisShield withholds it instead of raising an alarm.")
        if clean_row["alert"] and not row["alert"]:
            return (f"On the unedited data the model raises an alert ({c:.1f}%). The fault drops it to {r:.1f}%, below "
                    "the alert line. Instead of a silent all-clear, SepsisShield withholds the prediction.")
    if row["alert"] != clean_row["alert"] and row["trust"] == "HIGH":
        return (f"The fault changes the alert decision (risk {c:.1f}% without it, {r:.1f}% with it) and the integrity "
                "checks do not catch it at this hour. This is a silent failure: see Known limitations.")
    if row["alert"] != clean_row["alert"]:
        return (f"The fault changes the alert decision (risk {c:.1f}% without it, {r:.1f}% with it). "
                "Trust is not HIGH, so the dashboard asks for the inputs to be verified.")
    return f"Risk without the fault: {c:.1f}% · with the fault: {r:.1f}%. The alert decision is unchanged at this hour."
