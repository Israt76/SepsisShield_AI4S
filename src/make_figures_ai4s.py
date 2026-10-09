"""Figures for the AI4S technical report, drawn only from results/exp_models.json, results/exp_robustness.json and
the existing result files. Style matches src/make_figures.py."""
import json
import numpy as np
import matplotlib.pyplot as plt

from make_figures import RES, FIG, BLUE, ORANGE, AQUA, RED, GRAY, INK, INK2, save

VIOLET, YELLOW = "#4a3aa7", "#eda100"
E2 = json.load(open(RES / "experiments2.json"))
EM = json.load(open(RES / "exp_models.json"))
ER = json.load(open(RES / "exp_robustness.json"))


def model_comparison():
    rows = [b for b in E2["baselines"] if b["model"].startswith(("SIRS", "qSOFA", "Logistic"))] + EM["rows"] + \
           [b for b in E2["baselines"] if b["model"].startswith("Ensemble + isotonic")]
    order = sorted(rows, key=lambda r: r["auroc"])
    names = [r["model"].replace(" (+ isotonic)", "").replace("Ensemble + isotonic calibration (SepsisShield model)",
                                                               "SepsisShield: 5× LightGBM + isotonic")
             .replace(" (rule: ≥2 of 4)", " rule").replace(" (rule: ≥1 of 2)", " rule") for r in order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, key, title in ((axes[0], "auroc", "Hourly AUROC"), (axes[1], "utility", "PhysioNet normalized utility")):
        v = [r[key] for r in order]
        col = [BLUE if "SepsisShield" in n else GRAY for n in names]
        ax.hlines(range(len(v)), min(0, min(v)), v, color=col, lw=2)
        ax.scatter(v, range(len(v)), s=70, color=col, edgecolor="white", linewidth=1.5, zorder=3)
        for i, x in enumerate(v):
            ax.text(x + (0.01 if key == "auroc" else 0.012), i, f"{x:.3f}", va="center", fontsize=9, color=INK2)
        ax.set_title(title)
        ax.set_xlim((0.5, 0.9) if key == "auroc" else (min(0, min(v)) - 0.02, 0.5))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), names)
    fig.suptitle("11 · Model families on the same held-out test patients", x=0.01, ha="left", fontweight="bold")
    save(fig, "11_model_comparison.png")


def robustness():
    S = ER["sweeps"]
    base = S["baseline_clean"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, xk, title, xl in ((axes[0], "missingness", "fraction_removed", "Random missingness", "observed values removed (%)"),
                                   (axes[1], "noise", "noise_sd_multiple", "Measurement noise on 6 variables", "noise SD (× population SD)")):
        xs = [0] + [r[xk] for r in S[key]]
        au = [base["auroc"]] + [r["auroc"] for r in S[key]]
        xs_plot = [x * 100 for x in xs] if key == "missingness" else xs
        ax.plot(xs_plot, au, "-o", color=BLUE, lw=2, ms=7, markeredgecolor="white", markeredgewidth=1.5, label="AUROC")
        for x, a in zip(xs_plot, au):
            ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=INK2)
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel("hourly AUROC")
        ax.set_ylim(min(au) - 0.03, base["auroc"] + 0.02)
    fig.suptitle("12 · Graceful degradation under missingness and noise (shipped model, no retraining)",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "12_robustness_sweeps.png")


def signals():
    B = ER["selective"]["benchmark_signal_comparison"]
    xs = [r["abstain_fraction_of_hours"] * 100 for r in B["rows"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, lab, c, ls in (("input_trust_policy", "Input-trust ranking (SepsisShield)", BLUE, "-"),
                            ("shift_score_only", "Distribution-shift score only", AQUA, "-"),
                            ("ensemble_disagreement_only", "Ensemble disagreement only", ORANGE, "-"),
                            ("random", "Random abstention", GRAY, "--")):
        ax.plot(xs, [r[key] * 100 for r in B["rows"]], ls, marker="o", color=c, lw=2, ms=7,
                markeredgecolor="white", markeredgewidth=1.5, label=lab)
    ax.set_xlabel("hours abstained / flagged (% of benchmark hours)")
    ax.set_ylabel("dangerous failures caught (%)")
    ax.set_ylim(0, 100)
    ax.set_title("13 · Which signal catches data-fault failures?")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "13_abstention_signals.png")


def ablation():
    rows = ER["ablation"]["rows"]
    labs = ["Base model", "+ ensemble", "+ calibration", "+ disagreement\nwarning", "+ input-integrity\nchecks",
            "+ shift flag\n(not deployed)"]
    cov = [r["coverage"] * 100 for r in rows]
    cost = [r["clean_hours_flagged_pct"] * 100 for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, v, title, c, fmt in ((axes[0], cov, "Dangerous failures caught (%)", BLUE, "{:.1f}%"),
                                 (axes[1], cost, "Clean hours flagged or withheld (%)", ORANGE, "{:.1f}%")):
        cols = [c if i < 5 else GRAY for i in range(len(v))]
        ax.barh(y, v, color=cols, height=0.6)
        for yi, x in zip(y, v):
            ax.text(x + 1, yi, fmt.format(x), va="center", fontsize=9, color=INK2)
        ax.set_title(title, fontsize=11); ax.set_xlim(0, 110 if c == BLUE else 12)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labs)
    fig.suptitle("14 · Component ablation on the accidental-fault benchmark (6,790 dangerous failures)",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "14_component_ablation.png")


def cohort_shift():
    C = ER["cohort_shift"]
    keys = [k for k in C if k not in ("reference",)]
    nice = {"age_ge_80": "Real subgroup: age ≥ 80", "age_lt_45": "Real subgroup: age < 45"}
    ref = C["reference"]
    names = ["Reference cohort"] + [nice.get(k, k.split(" (")[0].replace("_", " ")) for k in keys]
    flag = [(ref["shift_moderate_pct"] + ref["shift_high_pct"]) * 100] + \
           [(C[k]["shift_moderate_pct"] + C[k]["shift_high_pct"]) * 100 for k in keys]
    au = [ref["auroc"]] + [C[k]["auroc"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    y = np.arange(len(names))[::-1]
    axes[0].barh(y, flag, color=[GRAY] + [AQUA] * len(keys), height=0.6)
    for yi, x in zip(y, flag):
        axes[0].text(x + 1, yi, f"{x:.1f}%", va="center", fontsize=9, color=INK2)
    axes[0].set_title("Hours with shift MODERATE or HIGH", fontsize=11); axes[0].set_xlim(0, max(flag) * 1.25)
    axes[1].scatter(au, y, s=70, color=[GRAY] + [BLUE] * len(keys), edgecolor="white", linewidth=1.5, zorder=3)
    for yi, x in zip(y, au):
        axes[1].text(x + 0.004, yi, f"{x:.3f}", va="center", fontsize=9, color=INK2)
    axes[1].set_title("Hourly AUROC", fontsize=11); axes[1].set_xlim(min(au) - 0.03, max(au) + 0.04)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names)
    fig.suptitle("15 · Does the shift indicator react to simulated cohort shifts?", x=0.01, ha="left", fontweight="bold")
    save(fig, "15_cohort_shift.png")


def risk_coverage():
    C = ER["selective"]["clean_risk_coverage_full_test"]
    xs = [r["coverage"] * 100 for r in C]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, k, t in ((axes[0], "brier", "Brier score of retained hours (lower is better)"),
                     (axes[1], "auroc", "AUROC of retained hours")):
        ax.plot(xs, [r[f"{k}_uncertainty"] for r in C], "-o", color=BLUE, lw=2, ms=7, markeredgecolor="white",
                markeredgewidth=1.5, label="abstain on highest ensemble disagreement")
        ax.plot(xs, [r[f"{k}_random"] for r in C], "--o", color=GRAY, lw=2, ms=6, markeredgecolor="white",
                markeredgewidth=1.5, label="abstain at random")
        ax.set_title(t, fontsize=11); ax.set_xlabel("coverage (% of clean test hours kept)"); ax.invert_xaxis()
    axes[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("16 · Selective prediction on clean held-out data (all 309,270 test hours)", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "16_risk_coverage.png")


def operating_points():
    O = json.load(open(RES / "exp_operating_points.json"))
    rows, thr = O["rows"], O["shipped_threshold"]
    x = [r["threshold"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    a = axes[0]
    for key, lab, c in (("patient_sensitivity", "septic patients alerted", BLUE),
                        ("pct_detected_ge6h_early", "alerted ≥ 6 h before onset", AQUA),
                        ("patient_specificity", "non-septic never alerted", ORANGE)):
        a.plot(x, [r[key] * 100 for r in rows], "-o", color=c, lw=2, ms=6, markeredgecolor="white", markeredgewidth=1.2, label=lab)
    a.set_ylabel("%"); a.set_title("Detection vs specificity", fontsize=11)
    fig.legend(*a.get_legend_handles_labels(), loc="lower left", ncol=3, fontsize=9, bbox_to_anchor=(0.03, 0.0))
    b = axes[1]
    b.plot(x, [r["utility_val"] for r in rows], "-o", color=VIOLET, lw=2, ms=6, markeredgecolor="white", label="validation (used to choose)")
    b.plot(x, [r["utility_test"] for r in rows], "--o", color=GRAY, lw=2, ms=6, markeredgecolor="white", label="held-out test")
    b.set_title("PhysioNet utility", fontsize=11); b.legend(fontsize=8)
    c = axes[2]
    c.plot(x, [r["alert_episodes_per_100_nonseptic_patient_days"] for r in rows], "-o", color=RED, lw=2, ms=6,
           markeredgecolor="white")
    c.set_title("Alert episodes per 100\nnon-septic patient-days", fontsize=10.5)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3], ["1%", "2%", "3%", "5%", "10%", "20%", "30%"])
        ax.minorticks_off(); ax.axvline(thr, color=INK, lw=1, ls=":"); ax.set_xlabel("alert threshold (calibrated risk)")
    c.annotate("shipped 3%", (thr, c.get_ylim()[1] * 0.9), xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK)
    fig.suptitle("17 · Operating points of the shipped model (held-out test set)", x=0.01, ha="left", fontweight="bold")
    save(fig, "17_operating_points.png")


if __name__ == "__main__":
    for f in (model_comparison, robustness, signals, ablation, cohort_shift, risk_coverage, operating_points):
        f()
        print("ok", f.__name__)
