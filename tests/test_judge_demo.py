"""Judge Demo Mode: the three one-click scenarios must show what the dashboard says they show.

Each scenario is scored with the shipped pipeline. If a model, threshold or integrity rule changes, these tests fail
rather than letting the demo silently tell a different story.
"""
import sys

import numpy as np
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "app"))
import pipeline  # noqa: E402
import scenarios  # noqa: E402
from scenarios import SCENARIOS, apply_corruption, fault_effect, query_for  # noqa: E402

# Risk values produced by the shipped models at each scenario hour (regression guard: model outputs unchanged).
PINNED_RISK = {"A": 0.0714913565777908, "B": 0.039431157078215914, "C": 0.023383998663771506}


def _score(demo_raw, s, corrupt=True):
    raw = demo_raw[demo_raw["patient_id"] == s["pid"]].reset_index(drop=True)
    if corrupt and s["corr"] != "None":
        raw = apply_corruption(raw, s["corr"], s["start"], s["length"])
    out, _ = pipeline.score(raw)
    return out.iloc[s["hour"]]


@pytest.mark.parametrize("key", list(SCENARIOS))
def test_scenario_gives_documented_decision(demo_raw, key):
    s = SCENARIOS[key]
    row = _score(demo_raw, s)
    e = s["expect"]
    assert row["decision"] == e["decision"]
    assert row["trust"] == e["trust"]
    assert bool(row["alert"]) == e["alert"]
    confident = row["ens_std"] <= pipeline.load()[3]["ens_std_thr"]
    assert confident == e["confident"]
    if "flag" in e:
        assert bool(row[e["flag"]])


@pytest.mark.parametrize("key", list(SCENARIOS))
def test_scenario_model_output_unchanged(demo_raw, key):
    row = _score(demo_raw, SCENARIOS[key])
    assert row["risk"] == pytest.approx(PINNED_RISK[key], abs=1e-9)


@pytest.mark.parametrize("key", ["B", "C"])
def test_fault_changes_the_alert_and_is_not_silent(demo_raw, key):
    """B: the fault creates a false alert. C: the edit hides an alert. In both, the prediction is withheld."""
    s = SCENARIOS[key]
    clean, bad = _score(demo_raw, s, corrupt=False), _score(demo_raw, s)
    assert bool(clean["alert"]) == s["expect"]["clean_alert"]
    assert bool(bad["alert"]) != bool(clean["alert"])
    assert clean["trust"] == "HIGH" and bad["decision"] == "WITHHELD"
    assert "withholds" in fault_effect(clean, bad)


def test_scenarios_use_held_out_demo_patients_and_known_corruptions(demo_raw):
    ids = set(demo_raw["patient_id"])
    for s in SCENARIOS.values():
        assert s["pid"] in ids
        assert s["corr"] in scenarios.CORRUPTIONS
        n = (demo_raw["patient_id"] == s["pid"]).sum()
        assert 0 <= s["hour"] < n


def test_scenario_links_match_documented_query_format():
    assert query_for("A") == {"pid": "p119917", "hour": "58"}
    assert query_for("B") == {"pid": "p119917", "hour": "36", "corr": "Thermometer reports °F", "start": "34", "len": "8"}
    assert query_for("C")["corr"] == "Vitals overwritten to look normal"
    readme = (ROOT / "README.md").read_text()
    for key in SCENARIOS:
        q = query_for(key)
        assert f"pid={q['pid']}" in readme and f"hour={q['hour']}" in readme


def test_trust_and_decision_wording_covers_every_state():
    assert set(scenarios.TRUST_TEXT) == {"HIGH", "REDUCED", "LOW"}
    assert set(scenarios.DECISION_TEXT) == {"SHOW", "SHOW_WITH_WARNING", "WITHHELD"}


def test_apply_corruption_only_touches_the_fault_window(demo_raw):
    raw = demo_raw[demo_raw["patient_id"] == "p119917"].reset_index(drop=True)
    bad = apply_corruption(raw, "Thermometer reports °F", 34, 8)
    outside = (raw["hour"] < 34) | (raw["hour"] >= 42)
    num = raw.select_dtypes("number").columns
    assert np.allclose(raw.loc[outside, num].fillna(-999), bad.loc[outside, num].fillna(-999))
    assert (bad.loc[~outside, "Temp"].dropna() > 90).all()


# ---------------------------------------------------------------- optional scenarios E (early warning) and F (uncertain)
PINNED_RISK_OPTIONAL = {"E": 0.12781954887218044, "F": 0.23825503355704697}


@pytest.mark.parametrize("key", ["E", "F"])
def test_optional_scenario_gives_documented_decision(demo_raw, key):
    import shift
    s = scenarios.OPTIONAL_SCENARIOS[key]
    raw = demo_raw[demo_raw["patient_id"] == s["pid"]].reset_index(drop=True)
    out, feats = pipeline.score(raw)
    row, e = out.iloc[s["hour"]], s["expect"]
    assert (row["decision"], row["trust"], bool(row["alert"])) == (e["decision"], e["trust"], e["alert"])
    assert (row["ens_std"] <= pipeline.load()[3]["ens_std_thr"]) == e["confident"]
    assert shift.assess(feats)["shift_state"].iloc[s["hour"]] == e["shift"]
    assert row["risk"] == pytest.approx(PINNED_RISK_OPTIONAL[key], abs=1e-9)


def test_early_warning_scenario_is_before_onset(demo_raw):
    from metrics import onset_hour
    s = scenarios.OPTIONAL_SCENARIOS["E"]
    out, _ = pipeline.score(demo_raw[demo_raw["patient_id"] == s["pid"]].reset_index(drop=True))
    assert onset_hour(out).iloc[0] - s["hour"] == 8                          # "8 h before the recorded onset"
    assert round(out.iloc[43]["risk"] * 100, 1) == 1.1 and round(out.iloc[44]["risk"] * 100, 1) == 13.1


def test_uncertain_scenario_is_disagreement_only_and_never_withheld(demo_raw):
    from decision_summary import INPUT_FLAGS, RULE_DISAGREEMENT, decision_reasons
    s = scenarios.OPTIONAL_SCENARIOS["F"]
    out, feats = pipeline.score(demo_raw[demo_raw["patient_id"] == s["pid"]].reset_index(drop=True))
    row = out.iloc[s["hour"]]
    assert not any(bool(row[f]) for f in INPUT_FLAGS)                         # trust reduced by disagreement alone
    why = decision_reasons(row, pipeline.explain(feats.iloc[[s["hour"]]]), [], "LOW", [])
    assert why["rule"] == RULE_DISAGREEMENT and why["decision"] == "VERIFY INPUTS"
