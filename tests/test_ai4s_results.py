"""AI4S experiment results: internal consistency with the published benchmark, and the honesty constraints the
report relies on (shift advisory, model comparison on the same split, no retraining of the shipped model)."""
import json

import pytest


@pytest.fixture(scope="module")
def R(root):
    return json.load(open(root / "results" / "exp_robustness.json"))


def test_benchmark_recreation_matches_published_abstention_result(R, root):
    A = json.load(open(root / "results" / "abstention.json"))["dfc_accidental"]["all"]
    chk = R["benchmark_check"]
    assert (chk["caught"], chk["dangerous_failures"]) == (A["withheld"] + A["flagged"], A["n"]) == (6481, 6790)


def test_component_ablation_rows(R):
    rows = {r["config"][:2]: r for r in R["ablation"]["rows"]}
    assert [rows[k]["dangerous_failures_caught"] for k in ("1.", "2.", "3.")] == [0, 0, 0]
    assert rows["5."]["dangerous_failures_caught"] == 6481
    assert round(rows["5."]["clean_hours_withheld_pct"] * 100, 2) == 0.46            # 1,438 / 309,270
    assert "not deployed" in rows["6."]["note"]                                     # shift stays advisory
    assert rows["6."]["clean_hours_flagged_pct"] > rows["5."]["clean_hours_flagged_pct"]


def test_extreme_values_are_always_withheld(R):
    e = R["sweeps"]["extreme"]
    assert e["injected_hours"] > 1000 and e["injected_hours_trust_low"] == e["injected_hours"]


def test_sweeps_are_monotone_enough_and_complete(R):
    S = R["sweeps"]
    assert [r["fraction_removed"] for r in S["missingness"]] == [0.05, 0.1, 0.2, 0.3, 0.4]
    assert [r["noise_sd_multiple"] for r in S["noise"]] == [0.05, 0.1, 0.2, 0.3, 0.5]
    assert S["missingness"][-1]["auroc"] < S["baseline_clean"]["auroc"]


def test_model_comparison_uses_the_deployed_split(root):
    M = json.load(open(root / "results" / "exp_models.json"))
    assert {r["model"].split(" (")[0] for r in M["rows"]} == {"Random forest", "XGBoost", "CatBoost"}
    assert "same 70/10/20 patient split" in M["protocol"]
    cfg = json.load(open(root / "models" / "config.json"))
    assert cfg["threshold"] == pytest.approx(0.03) and len(cfg["features"]) == 172     # shipped model unchanged


def test_crosssignal_decision_follows_the_preregistered_rule(root):
    X = json.load(open(root / "results" / "exp_crosssignal.json"))
    T, d = X["test_benchmark"], X["test_benchmark"]["decision"]
    gain = T["masking_plausible"]["after"]["coverage"] - T["masking_plausible"]["before"]["coverage"]
    cost = T["clean_full_test"]["flagged_or_withheld_after"] - T["clean_full_test"]["flagged_or_withheld_before"]
    assert d["plausible_edit_gain"] == pytest.approx(gain) and d["clean_cost_increase"] == pytest.approx(cost)
    assert d["ship"] == (gain >= 0.10 and cost <= 0.02)
    # "before" must reproduce the published benchmark exactly (same patients, same windows)
    assert (T["accidental"]["before"]["covered"], T["accidental"]["before"]["n"]) == (6481, 6790)
    assert T["masking_plausible"]["before"]["n"] == 876


def test_unshipped_detector_is_not_used_by_the_dashboard(root):
    X = json.load(open(root / "results" / "exp_crosssignal.json"))
    if not X["test_benchmark"]["decision"]["ship"]:
        for f in ("src/pipeline.py", "src/integrity.py", "app/app.py"):
            assert "crosssignal" not in (root / f).read_text()


def test_operating_point_at_shipped_threshold_matches_main_result(root):
    O = json.load(open(root / "results" / "exp_operating_points.json"))
    M = json.load(open(root / "results" / "results.json"))["main"]["calibrated"]
    row = [r for r in O["rows"] if r["threshold"] == O["shipped_threshold"]][0]
    for k in ("patient_sensitivity", "pct_detected_ge6h_early", "patient_specificity", "false_alert_hours_per_100_patient_days"):
        assert row[k] == pytest.approx(M[k])
    assert row["utility_val"] == max(r["utility_val"] for r in O["rows"])           # chosen on validation utility
