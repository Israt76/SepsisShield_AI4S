"""Judge-facing summaries: clean-vs-corrupted rows, decision reasons, and the provenance of the 95.4% result.
All values must come from existing pipeline outputs and result files."""
import json
import sys

import pandas as pd
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "app"))
import integrity  # noqa: E402
import pipeline  # noqa: E402
import shift  # noqa: E402
from decision_summary import comparison_rows, decision_reasons, evidence_95, risk_drivers  # noqa: E402
from scenarios import SCENARIOS, apply_corruption  # noqa: E402

STD_THR = pipeline.load()[3]["ens_std_thr"]


def _rows(demo_raw, key):
    s = SCENARIOS[key]
    raw = demo_raw[demo_raw["patient_id"] == s["pid"]].reset_index(drop=True)
    clean, _ = pipeline.score(raw)
    bad, feats = pipeline.score(apply_corruption(raw, s["corr"], s["start"], s["length"]))
    return clean.iloc[s["hour"]], bad.iloc[s["hour"]], feats, s["hour"]


def test_comparison_b_false_alert_withheld_while_models_stay_confident(demo_raw):
    c, b, _, _ = _rows(demo_raw, "B")
    rows = {lab: (x, y, ch) for lab, x, y, ch in comparison_rows(c, b, STD_THR)}
    assert list(rows) == ["Sepsis risk", "Alert", "Model confidence", "Input trust", "Final decision"]
    assert rows["Sepsis risk"][0] == f"{c['risk'] * 100:.1f}%"                       # real pipeline values
    assert rows["Sepsis risk"][1].startswith(f"{b['risk'] * 100:.1f}%") and "research score" in rows["Sepsis risk"][1]
    assert rows["Alert"][:2] == ("No alert", "Alert")
    assert rows["Model confidence"] == ("Confident", "Confident", False)              # confidence does not move...
    assert rows["Input trust"] == ("HIGH", "LOW", True)                               # ...input trust does
    assert rows["Final decision"] == ("SHOW PREDICTION", "PREDICTION WITHHELD", True)


def test_comparison_c_hidden_alert_withheld(demo_raw):
    c, b, _, _ = _rows(demo_raw, "C")
    rows = {lab: (x, y) for lab, x, y, _ in comparison_rows(c, b, STD_THR)}
    assert rows["Alert"] == ("Alert", "No alert")
    assert rows["Final decision"] == ("SHOW PREDICTION", "PREDICTION WITHHELD")


def test_identical_inputs_give_no_changes(demo_raw):
    c, _, _, _ = _rows(demo_raw, "B")
    assert not any(ch for *_, ch in comparison_rows(c, c, STD_THR))


def test_decision_reasons_combine_existing_evidence(demo_raw):
    c, b, feats, h = _rows(demo_raw, "B")
    shap = pipeline.explain(feats.iloc[[h]])
    st_ = shift.assess(feats)["shift_state"].iloc[h]
    why = decision_reasons(b, shap, integrity.explain_flags(b), st_, shift.explain(feats, h))
    assert why["decision"] == "PREDICTION WITHHELD" and "LOW" in why["rule"]
    assert "Physiologically implausible temperature — check units or sensor" in why["trust"]   # the real integrity reason
    assert len(why["risk"]) == 3
    assert why["shift"][0].startswith(st_)


def test_decision_reasons_for_clean_trusted_hour(demo_raw):
    s = SCENARIOS["A"]
    out, feats = pipeline.score(demo_raw[demo_raw["patient_id"] == s["pid"]].reset_index(drop=True))
    r = out.iloc[s["hour"]]
    why = decision_reasons(r, pipeline.explain(feats.iloc[[s["hour"]]]), integrity.explain_flags(r), "LOW", [])
    assert why["decision"] == "SHOW PREDICTION"
    assert why["trust"][0].startswith("All integrity checks passed")
    assert why["shift"] == ["LOW: recent inputs look like the training data."]


def test_risk_driver_arrows_follow_the_shap_sign():
    df = pd.DataFrame({"feature": ["HR", "Temp", "WBC"], "value": [120.0, float("nan"), 4.0], "shap": [0.5, -0.2, 0.1]})
    d = risk_drivers(df)
    assert d[0].startswith("↑ HR = 120") and "raises" in d[0]
    assert d[1].startswith("↓ Temp = not measured") and "lowers" in d[1]


def test_evidence_95_matches_result_files(root):
    A = json.load(open(root / "results" / "abstention.json"))
    V = json.load(open(root / "results" / "abstention_val.json"))
    ev = evidence_95(A, V)
    assert (ev["covered"], ev["n"]) == (6481, 6790)
    assert round(ev["coverage"] * 100, 1) == 95.4
    assert len(ev["by_fault"]) == 4                                                   # the four accidental fault types
    assert sum(c for _, c, _ in ev["by_fault"]) == 6481 and sum(t for *_, t in ev["by_fault"]) == 6790
    assert ev["validation"][:2] == (3313, 3498)
    assert ev["withheld"] + ev["flagged"] + ev["silent"] == ev["n"]


def test_implausible_message_wording():
    assert integrity.implausible_message("Temp") == "Physiologically implausible temperature — check units or sensor"
    assert integrity.implausible_message("MAP DBP") == \
        "Physiologically implausible values: mean arterial pressure, diastolic blood pressure — check units or sensors"
    assert integrity.implausible_message("XYZ").startswith("Physiologically implausible XYZ")
