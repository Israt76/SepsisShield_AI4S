"""Distribution-shift awareness: states, causality, training-only reference, and no effect on the model."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "app"))
import pipeline  # noqa: E402
import shift  # noqa: E402
from scenarios import OPTIONAL_SCENARIOS, SCENARIOS, apply_corruption  # noqa: E402

REF = shift.load()


def frame(n_out, hours=12):
    """Synthetic feature frame: every reference feature at the middle of its training range, except `n_out`
    features pushed above the range at every hour."""
    mid = {c: (lo + hi) / 2 for c, lo, hi in zip(REF["features"], REF["lo"], REF["hi"])}
    df = pd.DataFrame([mid] * hours)
    for c, hi in list(zip(REF["features"], REF["hi"]))[:n_out]:
        df[c] = hi + abs(hi) + 1.0
    return df


def n_for(target):
    """Smallest number of out-of-range features whose fraction lands in the requested state."""
    k = len(REF["features"])
    for n in range(k + 1):
        if shift.state_of([n / k], REF)[0] == target:
            return n
    raise AssertionError(target)


def test_reference_is_fitted_on_training_patients_only(root):
    cfg = json.load(open(root / "models" / "config.json"))
    assert REF["fitted_on"]["cohort"] == "training"
    assert REF["fitted_on"]["patients"] == 28234          # the training split (MODEL_CARD, tests/test_isolation.py)
    assert len(REF["features"]) == shift.TOP_K and set(REF["features"]) <= set(cfg["features"])
    assert "ICULOS" not in REF["features"]
    assert all(lo <= hi for lo, hi in zip(REF["lo"], REF["hi"]))
    assert 0 < REF["thr_moderate"] < REF["thr_high"] < 1


@pytest.mark.parametrize("state", ["LOW", "MODERATE", "HIGH"])
def test_shift_states(state):
    a = shift.assess(frame(n_for(state)))
    assert (a["shift_state"] == state).all()


def test_state_thresholds_are_strict_and_monotone():
    t_m, t_h = REF["thr_moderate"], REF["thr_high"]
    assert list(shift.state_of([0, t_m, t_m + 1e-9, t_h, t_h + 1e-9, 1], REF)) == \
        ["LOW", "LOW", "MODERATE", "MODERATE", "HIGH", "HIGH"]


def test_missing_values_are_not_counted_as_shift():
    df = frame(0)
    df.iloc[:, :10] = np.nan
    assert (shift.assess(df)["shift_state"] == "LOW").all()


def test_shift_is_causal():
    """The score at hour t uses only hours <= t: changing later hours changes nothing earlier."""
    a = frame(0, hours=20)
    b = a.copy()
    b.iloc[12:, :8] = 1e9
    sa, sb = shift.assess(a), shift.assess(b)
    pd.testing.assert_frame_equal(sa.iloc[:12], sb.iloc[:12])
    assert (sb["shift_state"].iloc[12:] != "LOW").any()


def test_explain_names_the_out_of_range_features():
    df = frame(3)
    msgs = shift.explain(df, len(df) - 1)
    assert len(msgs) == 3 and all("above the training range" in m for m in msgs)
    assert shift.explain(frame(0), 5) == []


def test_shift_does_not_change_model_output(demo_raw):
    s = SCENARIOS["A"]
    raw = demo_raw[demo_raw["patient_id"] == s["pid"]].reset_index(drop=True)
    before, feats = pipeline.score(raw)
    shift.assess(feats)
    after, _ = pipeline.score(raw)
    pd.testing.assert_frame_equal(before, after)
    assert "shift" not in (ROOT / "src" / "pipeline.py").read_text()        # the pipeline never consults it


def test_optional_shift_scenario(demo_raw):
    s = OPTIONAL_SCENARIOS["D"]
    raw = demo_raw[demo_raw["patient_id"] == s["pid"]].reset_index(drop=True)
    out, feats = pipeline.score(raw)
    row, a = out.iloc[s["hour"]], shift.assess(feats).iloc[s["hour"]]
    assert a["shift_state"] == s["expect"]["shift"]
    assert row["trust"] == s["expect"]["trust"] and row["decision"] == s["expect"]["decision"]
    assert bool(row["alert"]) == s["expect"]["alert"]
    assert any("White cell count" in m for m in shift.explain(feats, s["hour"], labeler=lambda c: {"WBC": "White cell count"}.get(c, c)))


@pytest.mark.parametrize("key", list(SCENARIOS))
def test_core_scenarios_have_a_defined_shift_state(demo_raw, key):
    s = SCENARIOS[key]
    raw = demo_raw[demo_raw["patient_id"] == s["pid"]].reset_index(drop=True)
    if s["corr"] != "None":
        raw = apply_corruption(raw, s["corr"], s["start"], s["length"])
    _, feats = pipeline.score(raw)
    assert shift.assess(feats)["shift_state"].iloc[s["hour"]] in shift.STATES


def test_evaluation_file_matches_reference(root):
    ev = json.load(open(root / "results" / "shift_evaluation.json"))
    assert ev["features"] == REF["features"]
    assert ev["thresholds"] == {"moderate": REF["thr_moderate"], "high": REF["thr_high"]}
    flagged = ev["train_rates"]["MODERATE"] + ev["train_rates"]["HIGH"]
    assert 0.03 <= flagged <= 0.06                               # ~5% by construction of the q95 threshold
    assert abs(ev["train_rates"]["HIGH"] - 0.01) < 0.005
