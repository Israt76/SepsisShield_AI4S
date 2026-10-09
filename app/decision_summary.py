"""Judge-facing summaries built only from outputs the pipeline already produces (no new scoring or metrics).

* comparison_rows: clean vs corrupted, side by side, for the same patient-hour
* decision_reasons: the evidence behind the final decision (risk drivers, trust reasons, shift warning)
* evidence_95: the provenance of the 95.4% headline result, read from results/abstention*.json
"""
import pandas as pd

DECISION_WORDS = {"SHOW": "SHOW PREDICTION", "SHOW_WITH_WARNING": "VERIFY INPUTS", "WITHHELD": "PREDICTION WITHHELD"}
RULE = {
    "SHOW": "Shown: the inputs passed every integrity check.",
    "SHOW_WITH_WARNING": "Shown with a warning: input trust is REDUCED, so the flagged measurements should be verified.",
    "WITHHELD": "Input trust is LOW. Clinical-input integrity checks triggered this safety action.",
}
RULE_DISAGREEMENT = ("Shown with a warning: the inputs passed the integrity checks, but the 5 models disagree more than "
                     "usual, so trust is REDUCED. Verify the inputs and interpret the risk with care.")
INPUT_FLAGS = ("flag_implausible", "flag_inconsistent", "flag_coordinated_shift", "input_alert", "flag_flatline",
               "flag_discordant", "flag_unit_note")
FAULT_NAMES = {"temp_fahrenheit": "Thermometer reports °F", "lab_unit_error": "Lab unit mix-up",
               "sensor_artifact": "Monitor artifact", "frozen_feed": "Frozen feed"}


def _confidence(r, std_thr):
    return "Confident" if r["ens_std"] <= std_thr else "Uncertain"


def _risk(r):
    return f"{r['risk'] * 100:.1f}%" + (" (research score only)" if r["decision"] == "WITHHELD" else "")


def comparison_rows(clean_row: pd.Series, row: pd.Series, std_thr: float) -> list[tuple[str, str, str, bool]]:
    """(label, clean value, value with the fault, changed?) for the same patient-hour."""
    items = [
        ("Sepsis risk", _risk(clean_row), _risk(row)),
        ("Alert", "Alert" if clean_row["alert"] else "No alert", "Alert" if row["alert"] else "No alert"),
        ("Model confidence", _confidence(clean_row, std_thr), _confidence(row, std_thr)),
        ("Input trust", clean_row["trust"], row["trust"]),
        ("Final decision", DECISION_WORDS[clean_row["decision"]], DECISION_WORDS[row["decision"]]),
    ]
    return [(lab, a, b, a != b) for lab, a, b in items]


def risk_drivers(shap_df: pd.DataFrame, labeler=str, k=3) -> list[str]:
    """Top-k contributions by absolute SHAP value (shap_df as returned by pipeline.explain, already sorted)."""
    out = []
    for f, v, s in shap_df.head(k)[["feature", "value", "shap"]].itertuples(index=False):
        val = "not measured" if pd.isna(v) else f"{v:.4g}"
        out.append(f"{'↑' if s > 0 else '↓'} {labeler(f)} = {val} ({'raises' if s > 0 else 'lowers'} risk)")
    return out


def decision_reasons(row: pd.Series, shap_df: pd.DataFrame, trust_msgs: list[str], shift_state: str,
                     shift_msgs: list[str], labeler=str) -> dict:
    """The evidence behind this hour's decision, grouped for one compact panel."""
    trust = trust_msgs or ["All integrity checks passed: plausible, consistent, no abrupt or coordinated shifts, live feed."]
    if shift_state == "LOW":
        shift = ["LOW: recent inputs look like the training data."]
    else:
        shift = [f"{shift_state}: " + (shift_msgs[0] if shift_msgs else "inputs differ from the training data.")]
    rule = RULE[row["decision"]]
    if row["decision"] == "SHOW_WITH_WARNING" and not any(bool(row.get(f, False)) for f in INPUT_FLAGS):
        rule = RULE_DISAGREEMENT
    return {"rule": rule, "decision": DECISION_WORDS[row["decision"]],
            "risk": risk_drivers(shap_df, labeler), "trust": trust, "shift": shift}


def evidence_95(A: dict, V: dict | None = None) -> dict:
    """Numbers behind the accidental-fault headline, straight from results/abstention.json (and the validation file)."""
    acc = A["dfc_accidental"]["all"]
    by = []
    for s in A["scenarios"]:
        if s["corruption"] in FAULT_NAMES:
            a = s["all"]
            by.append((FAULT_NAMES[s["corruption"]], a["withheld"] + a["flagged"], a["n"]))
    ev = {"covered": acc["withheld"] + acc["flagged"], "n": acc["n"], "withheld": acc["withheld"],
          "flagged": acc["flagged"], "silent": acc["silent"], "coverage": acc["coverage"], "by_fault": by,
          "patients": A["n_patients"]}
    if V:
        va = V["dfc_accidental"]["all"]
        ev["validation"] = (va["withheld"] + va["flagged"], va["n"], va["coverage"])
    return ev
