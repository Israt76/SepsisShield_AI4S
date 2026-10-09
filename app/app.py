"""SepsisShield AI — Streamlit dashboard.

Run:  streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
import pipeline  # noqa: E402
import integrity  # noqa: E402
import shift  # noqa: E402
from decision_summary import INPUT_FLAGS, comparison_rows, decision_reasons, evidence_95  # noqa: E402
from labels import pretty  # noqa: E402
from scenarios import (ALL_SCENARIOS, CORRUPTIONS, DECISION_TEXT, DEFAULT_SCENARIO, OPTIONAL_SCENARIOS,  # noqa: E402
                       SCENARIOS, TRUST_TEXT,
                       apply_corruption, fault_effect)

# palette (validated reference palette from the dataviz method)
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#8a8984"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
WARN_TXT = "#8a5d00"                                  # amber that passes contrast on white, for text
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
TRUST_STYLE = {"HIGH": (GOOD, "✅", TRUST_TEXT["HIGH"]),
               "REDUCED": (WARN, "⚠️", TRUST_TEXT["REDUCED"]),
               "LOW": (CRIT, "⛔", TRUST_TEXT["LOW"])}
DECISION_STYLE = {"SHOW": (GOOD, "✅", "solid"), "SHOW_WITH_WARNING": (WARN, "⚠️", "dashed"),
                  "WITHHELD": (CRIT, "⛔", "double")}

st.set_page_config(page_title="SepsisShield AI", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
.tile {border: 1px solid #e6e5e1; border-radius: 10px; padding: 14px 16px; background: #fcfcfb; height: 100%;}
.tile .lbl {font-size: 0.78rem; color: #52514e; text-transform: uppercase; letter-spacing: .04em;}
.tile .val {font-size: 1.75rem; font-weight: 650; color: #0b0b0b; line-height: 1.2; margin-top: 2px;}
.tile .sub {font-size: 0.84rem; color: #3d3c39; margin-top: 4px;}
.tile .q {font-size: 0.74rem; color: #6b6a66; margin-top: 6px; font-style: italic;}
.dec {border-radius: 10px; padding: 14px 16px; height: 100%; background: #fcfcfb;}
.dec .val {font-size: 1.45rem; font-weight: 750; line-height: 1.2; margin-top: 2px; letter-spacing: .01em;}
.msg {border-left: 4px solid; padding: 8px 12px; margin: 6px 0; background: #fcfcfb; border-radius: 0 6px 6px 0; font-size: .9rem; color:#0b0b0b}
.callout {border: 1px solid #e6e5e1; border-radius: 10px; padding: 12px 16px; background: #f6f8fc; font-size: .92rem; color:#0b0b0b; margin: 8px 0;}
.disclaimer {font-size: .8rem; color: #52514e;}
.grp {font-size: .72rem; font-weight: 700; letter-spacing: .08em; color: #52514e; margin-bottom: -6px;}
.judge {border: 1px solid #cfdcf3; background: #f5f8fe; border-radius: 10px; padding: 8px 14px; margin: 4px 0 8px 0; font-size: .95rem;}
.cmp {border: 1px solid #e6e5e1; border-radius: 10px; padding: 12px 16px; height: 100%; background: #fcfcfb; font-size: .9rem;}
.cmp.ours {border: 2px solid #2a78d6; background: #f5f8fe;}
.cmp h5 {margin: 0 0 6px 0;}
.lim {border-left: 4px solid #8a5d00; padding: 8px 12px; margin: 6px 0; background: #fffaf0; border-radius: 0 6px 6px 0; font-size: .9rem;}
.shift {border-radius: 10px; padding: 12px 16px; margin-top: 12px; background: #faf8ff; display: flex; gap: 22px; align-items: flex-start; flex-wrap: wrap;}
.shift .state {min-width: 190px;}
.shift .val {font-size: 1.35rem; font-weight: 750; color: #4b2fa0; line-height: 1.2; margin-top: 2px;}
.shift .body {flex: 1; min-width: 260px; font-size: .88rem; color: #1f1d24;}
.shift ul {margin: 4px 0 0 0; padding-left: 18px;}
.ss-compare, .why {border: 1px solid #e6e5e1; border-radius: 10px; padding: 10px 14px; background: #fcfcfb; margin-top: 12px; font-size: .88rem;}
.lbl2 {font-size: .76rem; color: #52514e; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 4px;}
.cmpt {width: 100%; border-collapse: collapse;}
.cmpt th {text-align: left; font-size: .78rem; color: #52514e; font-weight: 600; padding: 3px 8px; border-bottom: 1px solid #e6e5e1;}
.cmpt td {padding: 3px 8px; border-bottom: 1px solid #f0efeb;}
.cmpt td.k {color: #52514e; width: 22%;}
.cmpt td.chg {font-weight: 700; background: #fff4e5;}
.why .rule {margin: 2px 0 6px 0;}
.why .cols {display: flex; gap: 16px; flex-wrap: wrap;}
.why .col {flex: 1; min-width: 220px;}
.why .col.trust {flex: 1.35; border-left: 4px solid var(--tc); background: var(--tb); border-radius: 0 8px 8px 0; padding: 6px 10px;}
.why .col.trust .h {font-size: .95rem;}
.why .col.advisory {flex: .85; color: #6b6a66; font-size: .82rem;}
.why .col.advisory .h {font-size: .78rem; font-weight: 600;}
.why .h {font-weight: 700; font-size: .84rem;}
.why ul {margin: 2px 0 0 0; padding-left: 18px;}
.legend {font-size: .8rem; color: #52514e; margin: 2px 0 6px 0;}
.str {border-left: 4px solid #0ca30c; padding: 8px 12px; margin: 6px 0; background: #f4fbf4; border-radius: 0 6px 6px 0; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def demo_patients():
    return pd.read_parquet(ROOT / "app" / "demo_patients.parquet")


@st.cache_data
def demo_index():
    return pd.read_parquet(ROOT / "app" / "demo_index.parquet")


@st.cache_data(show_spinner=False)
def run(raw: pd.DataFrame):
    return pipeline.score(raw)


@st.cache_data
def load_json(name):
    p = ROOT / "results" / name
    return json.load(open(p)) if p.exists() else {}


def tile(label, value, sub="", color=None, q=""):
    style = f"color:{color}" if color else ""
    qh = f"<div class='q'>{q}</div>" if q else ""
    return (f"<div class='tile'><div class='lbl'>{label}</div><div class='val' style='{style}'>{value}</div>"
            f"<div class='sub'>{sub}</div>{qh}</div>")


def base_layout(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=12),
                      hoverlabel=dict(bgcolor="white", font_color=INK), showlegend=False)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


REPO = "https://github.com/Israt76/SepsisShield_GITHUB_Repo/blob/main/"


def evidence_expander():
    """'Evidence behind this claim' for the 95.4% accidental-fault result (numbers read from results/*.json)."""
    A, V, MRx = load_json("abstention.json"), load_json("abstention_val.json"), load_json("masking_realism.json")
    if not A:
        return
    ev = evidence_95(A, V)
    mk_ = A["dfc_masking"]["all"]
    with st.expander("Evidence behind the 95.4% claim"):
        rows = "".join(f"<li>{n}: {c:,} / {t:,} ({c / t * 100:.1f}%)</li>" for n, c, t in ev["by_fault"])
        st.markdown(
            f"**{ev['covered']:,} / {ev['n']:,} = {ev['coverage']*100:.1f}%** of *alert-changing wrong decisions* caused by "
            f"the **four simulated accidental fault types** were flagged (trust REDUCED, {ev['flagged']:,}) or withheld "
            f"(trust LOW, {ev['withheld']:,}); {ev['silent']:,} went unflagged.<ul>{rows}</ul>"
            f"An alert-changing wrong decision is a patient-hour inside a 10-hour fault window where the fault flipped the "
            f"alert to a wrong one: it suppressed a timely sepsis alert or created an unwarranted alert. Benchmark: "
            f"{ev['patients']:,} held-out test patients (all septic plus a random non-septic sample), faults injected "
            f"into the raw data and the full pipeline re-run."
            + (f" Validation-cohort replication: {ev['validation'][0]:,} / {ev['validation'][1]:,} = "
               f"{ev['validation'][2]*100:.1f}%." if "validation" in ev else "")
            + f"<br><b>Scope:</b> simulated accidental faults only. Deliberate edits: {mk_['coverage']*100:.1f}% "
            f"({mk_['withheld']+mk_['flagged']} / {mk_['n']})"
            + (f", and {MRx['plausible_edit']['coverage']*100:.1f}% when the edits stay physiologically plausible" if MRx else "")
            + f".<br><b>Sources:</b> <a href='{REPO}results/abstention.json'>results/abstention.json</a> · "
            f"<a href='{REPO}src/abstention.py'>src/abstention.py</a> · <a href='{REPO}AUDIT.md'>AUDIT.md §2</a> "
            f"(independent recomputation and definition audit) · README, section <i>Does the shield do anything?</i>",
            unsafe_allow_html=True)


# ---------------------------------------------------------------- state
# Shareable links keep working: ?pid=p119917&corr=Thermometer%20reports%20°F&start=34&len=8&hour=36
# Without a link, the dashboard opens on Judge Demo scenario A (the clearest story).
idx = demo_index()
ss = st.session_state
qp = st.query_params
ID2CAT = dict(zip(idx["patient_id"], idx["category"]))


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def set_scenario(key):
    s = ALL_SCENARIOS[key]
    ss.src, ss.pid, ss.cat = "Held-out test patients", s["pid"], ID2CAT[s["pid"]]
    ss.corr, ss.start, ss.len, ss.hour = s["corr"], s["start"], s["length"] or 10, s["hour"]


if "init" not in ss:
    ss.init = True
    if qp.get("pid") in ID2CAT:
        ss.src, ss.pid = "Held-out test patients", qp.get("pid")
        ss.cat = ID2CAT[ss.pid]
        ss.corr = qp.get("corr", "None") if qp.get("corr", "None") in CORRUPTIONS else "None"
        ss.start, ss.len = _int_or_none(qp.get("start")), _int_or_none(qp.get("len")) or 10
        ss.hour = _int_or_none(qp.get("hour"))
    else:
        set_scenario(DEFAULT_SCENARIO)


def _sync(k):
    ss[k] = ss["w_" + k]


def bound(k):
    """Widget kwargs that mirror a persistent state key (survives widgets being hidden)."""
    ss["w_" + k] = ss[k]
    return dict(key="w_" + k, on_change=_sync, args=(k,))


def _on_cat():
    ss.cat = ss.w_cat
    ss.pid = idx.loc[idx["category"] == ss.cat, "patient_id"].iloc[0]
    ss.hour = None


def _on_pid():
    ss.pid = ss.w_pid
    ss.hour = None


def active_scenario():
    for k, s in ALL_SCENARIOS.items():
        same_fault = ss.corr == s["corr"] and (s["corr"] == "None" or (ss.start == s["start"] and ss.len == s["length"]))
        if ss.src == "Held-out test patients" and ss.pid == s["pid"] and same_fault and ss.hour == s["hour"]:
            return k
    return None


# ---------------------------------------------------------------- header
st.markdown("## 🛡️ SepsisShield AI")
st.markdown("**Predicts sepsis risk early, and separately checks whether the clinical data behind that prediction "
            "can be trusted.**  <span class='disclaimer'>Research prototype on de-identified PhysioNet 2019 ICU data · "
            "not a medical device · not for clinical use.</span>", unsafe_allow_html=True)

# ---------------------------------------------------------------- judge demo mode
st.markdown("<div class='judge'>🎬 <b>Judge Demo Mode</b> · one click loads a real held-out test patient, scored "
            "live by the shipped models. Watch the four cards below.</div>", unsafe_allow_html=True)
act = active_scenario()
jcols = st.columns(3)
for col, (k, s) in zip(jcols, SCENARIOS.items()):
    with col:
        st.button(s["title"], key=f"scn_{k}", on_click=set_scenario, args=(k,), width="stretch",
                  type="primary" if act == k else "secondary")
        st.caption(s["blurb"])
for col, (k, s) in zip(st.columns(len(OPTIONAL_SCENARIOS)), OPTIONAL_SCENARIOS.items()):
    with col:
        st.button("Optional: " + s["title"], key=f"scn_{k}", on_click=set_scenario, args=(k,), width="stretch",
                  type="primary" if act == k else "secondary")
        st.caption(s["blurb"])

# ---------------------------------------------------------------- sidebar (manual exploration)
raw = clean_raw = None
c_start = c_len = None
with st.sidebar:
    st.markdown("### Explore manually")
    st.caption("Or use the Judge Demo buttons at the top of the page.")
    st.radio("Source", ["Held-out test patients", "Upload a PhysioNet .psv file"], label_visibility="collapsed",
             **bound("src"))
    if ss.src == "Held-out test patients":
        cats = list(idx["category"].unique())
        ss["w_cat"] = ss.cat
        st.selectbox("Show", cats, key="w_cat", on_change=_on_cat)
        sub = idx[idx["category"] == ss.cat]
        ids = list(sub["patient_id"])
        if ss.pid not in ids:
            ss.pid = ids[0]
        ss["w_pid"] = ss.pid
        st.selectbox("Patient", ids, key="w_pid", on_change=_on_pid,
                     format_func=lambda p: sub.set_index("patient_id").loc[p, "label"])
        raw = demo_patients()
        raw = raw[raw["patient_id"] == ss.pid].reset_index(drop=True)
    else:
        up = st.file_uploader("PSV file (pipe-separated, PhysioNet 2019 format)", type=["psv", "txt", "csv"])
        if up is not None:
            raw = pd.read_csv(up, sep="|")
            raw.insert(0, "patient_id", Path(up.name).stem)
            raw.insert(1, "hour", np.arange(len(raw)))
    if raw is not None and len(raw):
        clean_raw = raw
        st.markdown("### Stress test")
        st.caption("Inject a realistic input failure and watch the integrity layer respond.")
        st.selectbox("Corruption", CORRUPTIONS, **bound("corr"))
        n = len(raw)
        if ss.corr != "None":
            d_start = min(max(0, n - 20), max(0, n - 2))
            ss.start = d_start if ss.start is None else min(max(0, ss.start), max(0, n - 2))
            ss.len = min(max(2, ss.len or 10), 24)
            c_start = st.slider("Starts at hour", 0, max(0, n - 2), **bound("start"))
            c_len = st.slider("Duration (hours)", 2, 24, **bound("len"))
            raw = apply_corruption(raw, ss.corr, c_start, c_len)

# ---------------------------------------------------------------- scoring + first screen
row = scored = feats = None
if raw is None or not len(raw):
    st.info("Choose a patient or upload a file in the sidebar, or click a Judge Demo scenario.")
else:
    with st.spinner("Scoring..."):
        scored, feats = run(raw)
    n = len(scored)
    ss.hour = n - 1 if ss.hour is None else min(max(0, ss.hour), n - 1)
    hour = st.slider("Hour in ICU (the model only sees data up to this hour)", 0, n - 1, **bound("hour"))
    row = scored.iloc[hour]
    cfg = pipeline.load()[0]
    thr = cfg["threshold"]
    std_thr = pipeline.load()[3]["ens_std_thr"]
    onset = None
    if "SepsisLabel" in scored and scored["SepsisLabel"].max() == 1:
        onset = int(scored.loc[scored["SepsisLabel"] == 1, "hour"].min() + 6)
    # keep the URL shareable
    if ss.src == "Held-out test patients":
        q = {"pid": ss.pid, "hour": str(hour)}
        if ss.corr != "None":
            q.update(corr=ss.corr, start=str(c_start), len=str(c_len))
        if dict(qp) != q:
            qp.clear()
            qp.update(q)

    tcol, ticon, ttext = TRUST_STYLE[row["trust"]]
    risk_pct = row["risk"] * 100
    decision = row["decision"]
    withheld = decision == "WITHHELD"
    confident = row["ens_std"] <= std_thr
    pctx = f"Patient {row['patient_id']} · hour {hour} of {n - 1}"
    if onset is not None:
        pctx += f" · sepsis onset recorded at hour {onset} (retrospective label)"
    if c_start is not None:
        pctx += f" · simulated fault: {ss.corr}, hours {c_start}–{c_start + c_len - 1}"
    st.caption(pctx)
    st.markdown("<div class='legend'><b>① Sepsis risk</b> = estimated clinical risk · <b>② Model confidence</b> = do "
                "the five models agree? · <b>③ Input trust</b> = do the current measurements look reliable? · "
                "<b>⑤ Distribution shift</b> (below) = does this input pattern differ from the training data?</div>",
                unsafe_allow_html=True)

    sh = shift.assess(feats)
    sstate, sscore = sh["shift_state"].iloc[hour], sh["shift_score"].iloc[hour]
    sref = shift.load()

    # trust REDUCED by model disagreement alone (no input flag): say so instead of "suspicious measurements"
    disagree_only = row["trust"] == "REDUCED" and not any(bool(row.get(f, False)) for f in INPUT_FLAGS)
    if disagree_only:
        ttext = "Inputs passed the checks, but the 5 models disagree — verify inputs."
    c1, c2, c3, c4 = st.columns(4)
    if withheld:
        c1.markdown(tile("① Sepsis risk", "Withheld",
                         f"Alert: <b>Not issued</b>. Research score {risk_pct:.1f}% is kept for audit only, not for use.",
                         CRIT, "Calibrated risk that sepsis onset is near · average hour: 1.8%"), unsafe_allow_html=True)
    else:
        alert_txt = f"🔔 <b>ALERT</b>: above the {thr*100:.1f}% alert line" if row["alert"] else \
            f"No alert: below the {thr*100:.1f}% alert line"
        c1.markdown(tile("① Sepsis risk", f"{risk_pct:.1f}%", alert_txt, CRIT if row["alert"] else None,
                         "Calibrated risk that sepsis onset is near · average hour: 1.8%"), unsafe_allow_html=True)
    c2.markdown(tile("② Model confidence", "Confident" if confident else "Uncertain",
                     ("The 5 models agree" if confident else "The 5 models disagree more than usual") +
                     f" (spread ±{row['ens_std_prob']*100:.1f} pp)", None if confident else CRIT,
                     "Do the models agree? This says nothing about whether the data are correct."),
                 unsafe_allow_html=True)
    c3.markdown(tile("③ Input trust", f"{ticon} {row['trust']}", ttext,
                     tcol if row["trust"] != "REDUCED" else WARN_TXT,
                     "Are the inputs believable? Withholding is decided from the inputs alone."), unsafe_allow_html=True)
    dcol, dicon, dborder = DECISION_STYLE[decision]
    dsub = {"SHOW": "Inputs passed every check. Read the risk as shown." if sstate == "LOW" else
                    "Inputs passed every check. Note the distribution-shift signal below.",
            "SHOW_WITH_WARNING": ("Prediction shown with a warning. The models disagree: verify the inputs and read the risk "
                                  "with care." if disagree_only else
                                  "Prediction shown with a warning. Check the flagged measurements first."),
            "WITHHELD": "No alert is issued until the clinical data are checked."}[decision]
    c4.markdown(f"<div class='dec ss-decision ss-decision-{decision}' style='border:3px {dborder} {dcol}'>"
                f"<div class='lbl' style='font-size:.78rem;color:#52514e;text-transform:uppercase;letter-spacing:.04em'>"
                f"④ Final decision</div><div class='val' style='color:{WARN_TXT if decision == 'SHOW_WITH_WARNING' else dcol}'>"
                f"{dicon} {DECISION_TEXT[decision]}</div><div class='sub' style='font-size:.84rem;color:#3d3c39;"
                f"margin-top:4px'>{dsub}</div></div>", unsafe_allow_html=True)

    # clean vs corrupted, side by side (shown whenever a fault is injected, including faults that go uncaught)
    if c_start is not None and clean_raw is not None:
        clean_scored, _ = run(clean_raw)
        crow = clean_scored.iloc[hour]
        if bool(row["alert"]) == bool(crow["alert"]):
            change = "same alert decision"
        elif row["alert"]:
            change = "the fault creates an alert"
        else:
            change = "the fault hides the alert"
        trs = "".join(f"<tr><td class='k'>{lab}</td><td>{a}</td><td class='{'chg' if ch else ''}'>{b}</td></tr>"
                      for lab, a, b, ch in comparison_rows(crow, row, std_thr))
        st.markdown(
            "<div class='ss-compare'><div class='lbl2'>🔍 Clean vs corrupted · same patient, same hour "
            f"(<i>{change}</i>)</div><table class='cmpt'><tr><th></th><th>Clean data</th>"
            f"<th>With the simulated fault ({ss.corr})</th></tr>{trs}</table>"
            f"<div class='disclaimer' style='margin-top:4px'>{fault_effect(crow, row)} Changed values are "
            "highlighted. All values are live outputs of the shipped pipeline.</div></div>",
            unsafe_allow_html=True)

    # why was this decision made? (existing evidence only: SHAP drivers, integrity reasons, shift warning)
    tmsgs = integrity.explain_flags(row)
    if not confident:
        tmsgs.append("The 5 models disagree more than on 99% of validation hours (unfamiliar pattern)")
    why = decision_reasons(row, pipeline.explain(feats.iloc[[hour]]), tmsgs, sstate,
                           shift.explain(feats, hour, labeler=pretty) if sstate != "LOW" else [], labeler=pretty)
    li = lambda xs: "".join(f"<li>{x}</li>" for x in xs)
    st.markdown(
        f"<div class='why ss-why'><div class='lbl2'>Why was this decision made?</div>"
        f"<div class='rule'><b>{why['decision']}</b> · {why['rule']}</div><div class='cols'>"
        f"<div class='col'><div class='h'>① Top risk drivers (SHAP)</div><ul>{li(why['risk'])}</ul></div>"
        f"<div class='col trust' style='--tc:{tcol};--tb:{ {'HIGH': '#f4fbf4', 'REDUCED': '#fffaf0', 'LOW': '#fdf2f2'}[row['trust']] }'>"
        f"<div class='h' style='color:{WARN_TXT if row['trust'] == 'REDUCED' else tcol}'>"
        f"③ Input trust: {ticon} {row['trust']}</div><ul>{li(why['trust'])}</ul></div>"
        f"<div class='col advisory'><div class='h' style='color:#6b5aa8'>⑤ Distribution shift · advisory only</div>"
        f"<ul>{li(why['shift'])}</ul></div></div></div>", unsafe_allow_html=True)
    if row["trust"] != "HIGH" and confident:
        st.markdown("<div class='callout'>⚡ <b>Model confidence is not input trust.</b> All five models agree with "
                    "each other, so a confidence-only safety check would pass this prediction. The inputs "
                    + ("themselves fail the integrity checks" if row["trust"] == "LOW" else
                       "themselves were flagged by the integrity checks") +
                    ", and SepsisShield reacts to that.</div>", unsafe_allow_html=True)

    # distribution-shift awareness (separate signal; never changes risk, alert, trust or decision)
    sicon, sborder = {"LOW": ("○", "1px solid #cfc6ea"), "MODERATE": ("◇", "2px dashed #7a5bd6"),
                      "HIGH": ("◆", "3px solid #5b3fb5")}[sstate]
    trig = shift.explain(feats, hour, labeler=pretty) if sstate != "LOW" else []
    tlist = "".join(f"<li>{t}</li>" for t in trig)
    note = ("<br><i>Input trust is also reduced, so this unusual pattern may come from a data fault rather than an "
            "unusual patient. Check input trust first.</i>" if sstate != "LOW" and row["trust"] != "HIGH" else "")
    st.markdown(
        f"<div class='shift ss-shift ss-shift-{sstate}' style='border:{sborder}'>"
        f"<div class='state'><div class='lbl' style='font-size:.74rem;color:#52514e;text-transform:uppercase;"
        f"letter-spacing:.04em'>⑤ Distribution shift awareness</div><div class='val'>{sicon} {sstate} SHIFT</div>"
        f"<div style='font-size:.76rem;color:#6b6a66'>{sscore*100:.0f}% of key inputs outside the training range "
        f"(last {sref['window']} h)</div></div>"
        f"<div class='body'>{shift.TEXT[sstate]}{('<ul>' + tlist + '</ul>') if tlist else ''}{note}"
        f"<div style='font-size:.76rem;color:#6b6a66;margin-top:6px'>Awareness signal only. It does not change the "
        f"risk, the alert or the decision, and it does not show that this prediction is wrong or unsafe.</div></div></div>",
        unsafe_allow_html=True)

st.write("")
tab_mon, tab_diff, tab_res, tab_val, tab_how = st.tabs(
    ["Patient detail", "Why SepsisShield is different", "Results & limitations", "Validation evidence", "How it works"])

# ---------------------------------------------------------------- patient detail tab
with tab_mon:
    if row is None:
        st.info("Choose a patient first.")
    else:
        # risk trajectory
        seen = scored.iloc[: hour + 1]
        fig = go.Figure()
        for lvl, col in (("REDUCED", WARN), ("LOW", CRIT)):
            mask = (seen["trust"] == lvl).to_numpy()
            for h in seen["hour"][mask]:
                fig.add_vrect(x0=h - 0.5, x1=h + 0.5, fillcolor=col, opacity=0.18, line_width=0, layer="below")
        if c_start is not None:
            fig.add_vrect(x0=c_start - 0.5, x1=c_start + c_len - 0.5, line=dict(color=GRAY, dash="dot", width=1),
                          fillcolor="rgba(0,0,0,0)", annotation_text="injected corruption",
                          annotation_position="bottom left", annotation_font_color=INK2)
        fig.add_hline(y=thr * 100, line=dict(color=GRAY, dash="dash", width=1),
                      annotation_text="alert threshold", annotation_position="top right",
                      annotation_font_color=INK2)
        if onset is not None:
            fig.add_vline(x=onset, line=dict(color=INK2, dash="dot", width=1.5),
                          annotation_text="sepsis onset (retrospective)", annotation_position="top left",
                          annotation_font_color=INK2)
        fig.add_trace(go.Scatter(x=seen["hour"], y=seen["risk"] * 100, mode="lines", line=dict(color=BLUE, width=2),
                                 customdata=np.stack([seen["trust"], seen["ens_std_prob"] * 100], axis=1),
                                 hovertemplate="Hour %{x}<br>Risk %{y:.1f}%<br>Trust %{customdata[0]}"
                                               "<br>Model spread ±%{customdata[1]:.2f} pp<extra></extra>"))
        fig.add_trace(go.Scatter(x=[hour], y=[risk_pct], mode="markers",
                                 marker=dict(size=10, color=BLUE, line=dict(color="white", width=2)), hoverinfo="skip"))
        fig.update_xaxes(title="Hours since ICU admission", range=[-0.5, max(n - 0.5, 10)])
        ymax = max(scored["risk"].max() * 100 * 1.15, thr * 100 * 2, 5)
        fig.update_yaxes(title="Sepsis risk (%)", range=[0, ymax])
        fig = base_layout(fig, 330)
        fig.update_layout(title=dict(text="Risk trajectory · amber = verify inputs · red = prediction withheld (low input trust)",
                                     font=dict(size=13, color=INK2), x=0))
        st.plotly_chart(fig, width="stretch")

        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("##### Why this risk? Top drivers at this hour")
            ex = pipeline.explain(feats.iloc[[hour]])
            ex = ex.iloc[::-1]
            names = [f"{pretty(f)} = {v:.4g}" if pd.notna(v) else f"{pretty(f)} = not measured"
                     for f, v in zip(ex["feature"], ex["value"])]
            fx = go.Figure(go.Bar(x=ex["shap"], y=names, orientation="h",
                                  marker=dict(color=[RED if s > 0 else BLUE for s in ex["shap"]], cornerradius=4),
                                  hovertemplate="%{y}<br>contribution %{x:+.3f} log-odds<extra></extra>"))
            fx.update_xaxes(title="← lowers risk      contribution (log-odds)      raises risk →")
            st.plotly_chart(base_layout(fx, 330), width="stretch")
            st.caption("SHAP values averaged across the 5-model ensemble. Red raises risk, blue lowers it.")
        with right:
            st.markdown("##### Input integrity at this hour")
            msgs = integrity.explain_flags(row)
            if row["ens_std"] > std_thr:
                msgs.append("The 5 models disagree more than on 99% of validation hours (unfamiliar pattern)")
            if not msgs:
                st.markdown(f"<div class='msg' style='border-color:{GOOD}'>✅ All checks passed: plausible values, "
                            "consistent readings, no abrupt or coordinated shifts, live feed.</div>", unsafe_allow_html=True)
            for mtxt in msgs:
                st.markdown(f"<div class='msg' style='border-color:{tcol}'>{ticon} {mtxt}</div>", unsafe_allow_html=True)
            st.markdown("##### Over this stay so far")
            counts = {
                "Physiologically implausible values": int(seen["flag_implausible"].sum()),
                "Inconsistent readings": int(seen["flag_inconsistent"].sum()),
                "Abrupt jumps": int(seen["flag_jump"].sum()),
                "Coordinated 'normalising' shifts": int(seen["flag_coordinated_shift"].sum()),
                "Frozen-feed hours": int(seen["flag_flatline"].sum()),
            }
            st.dataframe(pd.DataFrame({"Check": counts.keys(), "Hours flagged": counts.values()}),
                         hide_index=True, width="stretch")
            tl = seen["trust"].value_counts()
            st.caption(f"Trust over {len(seen)} hours: HIGH {tl.get('HIGH', 0)} · REDUCED {tl.get('REDUCED', 0)} · LOW {tl.get('LOW', 0)}")

        st.markdown("##### Vital signs and key labs (as received)")
        panels = [("HR", "Heart rate (bpm)"), ("Temp", "Temperature (°C)"), ("MAP", "Mean arterial pressure"),
                  ("Resp", "Respiratory rate"), ("O2Sat", "O₂ saturation (%)"), ("Lactate", "Lactate (mmol/L)")]
        vf = make_subplots(rows=2, cols=3, subplot_titles=[p[1] for p in panels], horizontal_spacing=0.06,
                           vertical_spacing=0.18)
        rs = raw.iloc[: hour + 1]
        for i, (c, _) in enumerate(panels):
            r_, c_ = i // 3 + 1, i % 3 + 1
            d = rs[["hour", c]].dropna()
            vf.add_trace(go.Scatter(x=d["hour"], y=d[c], mode="lines+markers" if c == "Lactate" else "lines",
                                    line=dict(color=BLUE, width=2), marker=dict(size=7),
                                    hovertemplate=f"Hour %{{x}}<br>{c} %{{y}}<extra></extra>"), row=r_, col=c_)
            if onset is not None:
                vf.add_vline(x=onset, line=dict(color=INK2, dash="dot", width=1), row=r_, col=c_)
        vf.update_annotations(font=dict(size=12, color=INK2))
        st.plotly_chart(base_layout(vf, 420), width="stretch")

# ---------------------------------------------------------------- why different tab
with tab_diff:
    st.markdown("#### Why model confidence is not enough")
    st.markdown(
        "- **Model confidence** (here: how closely the five models agree) measures how *consistent the model is with itself*.\n"
        "- It does **not** show that the clinical data are correct. A thermometer reporting °F, a frozen monitor or an "
        "edited chart can still produce a confident prediction.\n"
        "- SepsisShield therefore treats **model confidence** and **input trust** as separate signals and shows both.")
    st.markdown("<div class='callout'>Real example from this dashboard: in Judge Demo scenarios <b>B</b> and <b>C</b>, "
                "model confidence stays <b>Confident</b> while input trust drops to <b>LOW</b>. B is a thermometer "
                "reporting °F, which causes a false alert. C is an edited chart, which hides an alert. In both, "
                "SepsisShield withholds the prediction.</div>", unsafe_allow_html=True)
    st.markdown("#### Three system designs, compared")
    st.caption("A conceptual comparison of design patterns, not a claim about any specific commercial or academic system.")
    k1, k2, k3 = st.columns(3)
    k1.markdown("<div class='cmp'><h5>Standard sepsis predictor</h5>• Predicts sepsis risk<br>• No independent "
                "input-trust evaluation<br>• May still return a confident prediction when inputs are corrupted</div>",
                unsafe_allow_html=True)
    k2.markdown("<div class='cmp'><h5>Confidence-only safety</h5>• Shows model uncertainty / confidence<br>• Confidence "
                "can stay high when corrupted inputs look plausible<br>• Does not independently decide whether inputs are "
                "trustworthy</div>", unsafe_allow_html=True)
    k3.markdown("<div class='cmp ours'><h5>🛡️ SepsisShield</h5>• Predicts sepsis risk<br>• Runs independent input-integrity checks on the "
                "clinical inputs<br>• Shows confidence and trust separately<br>• Can warn about or withhold a prediction</div>",
                unsafe_allow_html=True)
    st.markdown("#### The real-world problem")
    st.markdown(
        "Clinical measurements are not always right. A temperature can be charted in the wrong unit, a lab value in SI "
        "instead of conventional units, a monitor feed can freeze and copy the same values forward, a value can be typed "
        "incorrectly, and chart data can be edited. An ML model receiving such inputs can still return a "
        "plausible-looking number. SepsisShield adds an **input-reliability layer** that checks the data *before* "
        "a prediction is trusted.")
    st.markdown("#### Example workflow")
    w1, w2 = st.columns([1.2, 1])
    with w1:
        st.markdown(
            "1. ICU measurements arrive (vitals, labs, demographics), hour by hour.\n"
            "2. The **prediction path** estimates sepsis risk from data up to the current hour.\n"
            "3. Independent **input-integrity checks** test whether those inputs are believable.\n"
            "4. A measurement fault is detected (e.g. a physiologically implausible temperature).\n"
            "5. The models may still be confident.\n"
            "6. Input trust drops to REDUCED or LOW.\n"
            "7. The decision layer shows the prediction with a warning, or withholds it.\n"
            "8. The user is asked to verify the flagged measurements. The explanation names which check fired.")
    with w2:
        st.markdown("**Intended users (research setting)**\n"
                    "- ICU clinicians, as decision *support* that does not replace clinical judgement\n"
                    "- Clinical informatics teams checking data pipelines\n"
                    "- Hospital ML / AI monitoring teams auditing model inputs")
        st.caption("SepsisShield is a research prototype. It has not been prospectively validated and does not "
                   "replace clinical judgement.")

# ---------------------------------------------------------------- results & limitations tab
with tab_res:
    R, A, E = load_json("results.json"), load_json("abstention.json"), load_json("experiments2.json")
    MR = load_json("masking_realism.json")
    if R and A:
        m = R["main"]["calibrated"]
        ci = E.get("bootstrap_ci", {}).get("sepsisshield", {})
        acc, mk = A["dfc_accidental"]["all"], A["dfc_masking"]["all"]
        cf = A["clean_cost_full_test"]
        ff = next(s for s in A["scenarios"] if s["corruption"] == "frozen_feed")["all"]
        st.markdown(f"#### Strongest results · held-out test set ({R['main']['n_patients']['test']:,} patients never used "
                    "for training or tuning)")
        r1 = st.columns(4)
        r1[0].markdown(tile("AUROC", f"{m['auroc']:.3f}",
                            f"95% CI {ci['auroc'][0]:.3f}–{ci['auroc'][1]:.3f}" if "auroc" in ci else "",
                            q="How well risk ranks septic above non-septic hours"), unsafe_allow_html=True)
        r1[1].markdown(tile("Sepsis patients alerted", f"{m['patient_sensitivity']*100:.1f}%",
                            f"{round(m['patient_sensitivity']*m['n_septic'])} / {m['n_septic']} septic patients"),
                       unsafe_allow_html=True)
        r1[2].markdown(tile("Alerted ≥ 6 h before onset", f"{m['pct_detected_ge6h_early']*100:.1f}%",
                            f"{round(m['pct_detected_ge6h_early']*m['n_septic'])} / {m['n_septic']} septic patients"),
                       unsafe_allow_html=True)
        r1[3].markdown(tile("Cross-hospital AUROC", f"{R['cross_A2B']['auroc']:.3f} / {R['cross_B2A']['auroc']:.3f}",
                            "train on one hospital, test on the other (see limitations)"), unsafe_allow_html=True)
        r2 = st.columns(3)
        r2[0].markdown(tile("Accidental faults: wrong decisions caught", f"{acc['coverage']*100:.1f}%",
                            f"{acc['withheld']+acc['flagged']:,} / {acc['n']:,} alert-changing wrong decisions flagged "
                            f"(warned) or withheld · 4 simulated accidental fault types", GOOD),
                       unsafe_allow_html=True)
        r2[1].markdown(tile("Clean patient-hours withheld", f"{cf['withheld_pct']*100:.2f}%",
                            f"{cf['withheld_hours']:,} / {cf['hours']:,} hours · all {cf['patients']:,} test patients, "
                            "no corruption"), unsafe_allow_html=True)
        r2[2].markdown(tile("Deliberately edited inputs caught", f"{mk['coverage']*100:.1f}%",
                            f"{mk['withheld']+mk['flagged']} / {mk['n']} · substantially weaker (main limitation)"
                            + (f"; {MR['plausible_edit']['coverage']*100:.1f}% when edits stay physiologically plausible" if MR else ""),
                            WARN_TXT), unsafe_allow_html=True)
        st.caption("Corruption benchmark: faults were injected into held-out test patients. Counted: patient-hours where "
                   "the fault flipped the alert decision to a wrong one, and whether trust was REDUCED (flagged) or LOW "
                   "(withheld) at that hour. Validation-cohort replication: 94.7% accidental, 48.0% deliberate.")
        evidence_expander()

        s_col, l_col = st.columns(2)
        with s_col:
            st.markdown("#### Strengths")
            for t in [f"<b>Early warning:</b> {m['pct_detected_ge6h_early']*100:.1f}% of septic patients alerted at least "
                      f"6 h before onset (median lead {m['median_lead_time_h']:.0f} h among those alerted; the evaluation window "
                      "caps lead time at 12 h).",
                      f"<b>Trust layer works on accidental faults:</b> {acc['coverage']*100:.1f}% of alert-changing wrong "
                      f"decisions flagged or withheld, at a cost of {cf['withheld_pct']*100:.2f}% of clean hours withheld.",
                      "<b>Leakage-tested:</b> patient-level splits, causal features, and threshold and calibration chosen on "
                      "validation only. All of this is enforced by unit tests.",
                      "<b>Explained:</b> every hour has SHAP drivers and a plain-language reason for any trust flag."]:
                st.markdown(f"<div class='str'>{t}</div>", unsafe_allow_html=True)
        with l_col:
            st.markdown("#### Known limitations")
            for t in [f"<b>Deliberately edited inputs:</b> only {mk['coverage']*100:.1f}% ({mk['withheld']+mk['flagged']} / "
                      f"{mk['n']}) caught in the benchmark"
                      + (f", and only {MR['plausible_edit']['coverage']*100:.1f}% ({MR['plausible_edit']['withheld']+MR['plausible_edit']['flagged']} / "
                         f"{MR['plausible_edit']['n']}) when the simulated edits are kept inside normal physiological ranges "
                         "(sensitivity check)" if MR else "") +
                      ". Edited values that stay plausible are hard to detect from the data alone. <i>Future work:</i> "
                      "audit-trail and cross-source checks (e.g. device vs charted values).",
                      f"<b>Cross-hospital shift:</b> AUROC drops from 0.852 to {R['cross_A2B']['auroc']:.3f} / "
                      f"{R['cross_B2A']['auroc']:.3f} at an unseen hospital. <i>Future work:</i> local recalibration "
                      "(tested retrospectively in Validation evidence) and multi-site training.",
                      "<b>Simulated faults:</b> the integrity benchmark injects simulated corruptions, not recorded "
                      "hospital data-quality incidents. <i>Future work:</i> evaluation on logged real-world data errors.",
                      f"<b>Frozen feeds and alert burden:</b> frozen feeds are caught in only {(ff['withheld']+ff['flagged'])/ff['n']*100:.1f}% "
                      f"of cases, because they are only recognisable after 8 identical hours. "
                      f"{(1-m['patient_specificity'])*100:.0f}% of non-septic patients receive at least one false alert.",
                      "<b>Distribution-shift awareness:</b> it indicates that an input pattern differs from the training "
                      "distribution; it does not establish that a prediction is incorrect or clinically unsafe. It is "
                      "patient-level, uses 20 model inputs, and its link to model error is exploratory (below).",
                      "<b>Clinical status:</b> research prototype only. Not a medical device and not prospectively validated. "
                      "<i>Future work:</i> silent prospective evaluation with clinical partners."]:
                st.markdown(f"<div class='lim'>{t}</div>", unsafe_allow_html=True)
        SE = load_json("shift_evaluation.json")
        if SE:
            st.markdown("#### Distribution-shift awareness · exploratory evidence (held-out test set)")
            flag = lambda r: (r["MODERATE"] + r["HIGH"]) * 100
            ca, cb = SE["cross_hospital"]["reference_A"], SE["cross_hospital"]["reference_B"]
            st.markdown(
                f"*Does it notice a different hospital?* Refitting the reference on one hospital's training patients only: "
                f"hours flagged MODERATE or HIGH are **{flag(ca['test_A_same']):.1f}%** at the same hospital vs "
                f"**{flag(ca['test_B_other']):.1f}%** at the other (reference A), and **{flag(cb['test_B_same']):.1f}%** vs "
                f"**{flag(cb['test_A_other']):.1f}%** (reference B). A consistent but modest increase.")
            bs = SE["by_state_test"]
            st.dataframe(pd.DataFrame([{"Shift state": k, "Patient-hours": v["hours"], "Patients": v["patients"],
                                        "AUROC": v["auroc"], "AUROC 95% CI": f"{v['auroc_ci95'][0]:.3f}–{v['auroc_ci95'][1]:.3f}",
                                        "Observed sepsis-label rate": v["observed_rate"], "Mean predicted risk": v["mean_predicted"]}
                                       for k, v in bs.items()]).round(3), hide_index=True, width="stretch")
            st.caption("Shipped model, reference fitted on all training patients. AUROC is lower in HIGH-shift hours, but "
                       "the interval is wide (few patients) and overlaps; treat this as suggestive, not as validation. "
                       "The shift signal is never used to change a prediction.")
    else:
        st.info("Result files not found.")

# ---------------------------------------------------------------- validation tab
with tab_val:
    res_p, ib_p = ROOT / "results" / "results.json", ROOT / "results" / "integrity_benchmark.json"
    if not res_p.exists():
        st.info("Run the training pipeline to populate validation results.")
    else:
        R = json.load(open(res_p))
        IB = json.load(open(ib_p)) if ib_p.exists() else None
        m = R["main"]["calibrated"]
        st.markdown("#### Held-out test set · "
                    f"{R['main']['n_patients']['test']:,} patients never seen in training or tuning")
        E = json.load(open(ROOT / "results" / "experiments2.json")) if (ROOT / "results" / "experiments2.json").exists() else {}
        A = json.load(open(ROOT / "results" / "abstention.json")) if (ROOT / "results" / "abstention.json").exists() else {}
        ci = E.get("bootstrap_ci", {}).get("sepsisshield", {})

        def fmt_ci(k, pct=False):
            if k not in ci:
                return ""
            lo, hi = ci[k]
            return f"95% CI {lo*100:.1f}–{hi*100:.1f}%" if pct else f"95% CI {lo:.3f}–{hi:.3f}"
        st.markdown("<div class='grp'>PREDICTION PATH</div>", unsafe_allow_html=True)
        cs = st.columns(4)
        cs[0].markdown(tile("AUROC", f"{m['auroc']:.3f}", fmt_ci("auroc")), unsafe_allow_html=True)
        cs[1].markdown(tile("Challenge utility", f"{m['utility']:.3f}", fmt_ci("utility")), unsafe_allow_html=True)
        cs[2].markdown(tile("Sepsis patients alerted", f"{m['patient_sensitivity']*100:.1f}%",
                            fmt_ci("patient_sensitivity", True)), unsafe_allow_html=True)
        cs[3].markdown(tile("External AUROC", f"{R['cross_A2B']['auroc']:.3f} / {R['cross_B2A']['auroc']:.3f}",
                            "train A → test B / train B → test A"), unsafe_allow_html=True)
        if A:
            acc, mk, cc = A["dfc_accidental"]["all"], A["dfc_masking"]["all"], A["clean_cost"]
            st.markdown("<div class='grp' style='margin-top:14px'>TRUST PATH</div>", unsafe_allow_html=True)
            ts = st.columns(4)
            ts[0].markdown(tile("Alert-changing accidental faults flagged or withheld", f"{acc['coverage']*100:.1f}%",
                                f"{acc['withheld']+acc['flagged']:,} / {acc['n']:,} patient-hours in the corruption benchmark"),
                           unsafe_allow_html=True)
            ts[1].markdown(tile("Withheld outright", f"{acc['withheld_share']*100:.0f}%",
                                f"{acc['withheld']:,} of {acc['n']:,} (LOW trust → prediction withheld)"), unsafe_allow_html=True)
            MRv = load_json("masking_realism.json")
            ts[2].markdown(tile("Deliberately edited inputs", f"{mk['coverage']*100:.1f}%",
                                f"{mk['withheld']+mk['flagged']} / {mk['n']} · substantially weaker — principal limitation"
                                + (f"; {MRv['plausible_edit']['coverage']*100:.1f}% when edits stay physiologically plausible" if MRv else "")),
                           unsafe_allow_html=True)
            cf = A.get("clean_cost_full_test", cc)
            ts[3].markdown(tile("Clean predictions withheld", f"{cf['withheld_pct']*100:.2f}%",
                                f"of patient-hours, all {cf.get('patients', 0):,} test patients · "
                                f"{cf.get('patients_ever_withheld_pct', 0)*100:.0f}% of patients ever (median 1 h)"),
                           unsafe_allow_html=True)
            evidence_expander()
        st.write("")
        figdir = ROOT / "results" / "figures"
        figs = sorted(figdir.glob("*.png"), key=lambda p: int(p.name.split("_")[0]))
        st.image(str(figs[0]), width="stretch")
        for i in range(1, len(figs), 2):
            cols = st.columns(2)
            for c, f in zip(cols, figs[i:i + 2]):
                c.image(str(f), width="stretch")
        if E:
            st.markdown("#### Baselines (same test set)")
            st.dataframe(pd.DataFrame(E["baselines"]).drop(columns=["threshold"], errors="ignore").round(3),
                         hide_index=True, width="stretch")
            st.markdown("#### Local recalibration at the unseen hospital (retrospective)")
            st.dataframe(pd.DataFrame([{"Transfer": r["transfer"], "ECE as-is (pp)": r["as_is"]["ece"] * 100,
                                        "ECE recalibrated (pp)": r["recalibrated"]["ece"] * 100,
                                        "Utility as-is": r["as_is"]["utility"], "Utility recalibrated": r["recalibrated"]["utility"]}
                                       for r in E["recalibration"]]).round(3), hide_index=True, width="stretch")
        if A:
            st.markdown("#### Trust-aware abstention by scenario (patient-hours where the fault flipped the alert decision to a wrong one)")
            NAMES = {"temp_fahrenheit": "Thermometer reports °F", "lab_unit_error": "Lab unit mix-up",
                     "sensor_artifact": "Monitor artifact", "frozen_feed": "Frozen feed",
                     "masking_attack": "Deliberately edited vitals", "measurement_noise": "Ordinary noise (control)"}
            st.dataframe(pd.DataFrame([{"Scenario": NAMES.get(s["corruption"], s["corruption"]),
                                        "Alert-changing wrong decisions": s["all"]["n"],
                                        "Withheld": s["all"]["withheld"], "Flagged": s["all"]["flagged"],
                                        "Silent": s["all"]["silent"]} for s in A["scenarios"]]),
                         hide_index=True, width="stretch")
        st.markdown("#### Cross-hospital generalization")
        rows = []
        for k, name in (("cross_A2B", "Train Hospital A → test Hospital B"), ("cross_B2A", "Train Hospital B → test Hospital A")):
            if k in R:
                x = R[k]
                rows.append({"Setting": name, "Internal AUROC": x["internal_val"]["auroc"], "External AUROC": x["auroc"],
                             "Internal utility": x["internal_val"]["utility"], "External utility": x["utility"]})
        if rows:
            st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        if "ablation" in R:
            st.markdown("#### Ablation: what each feature group adds")
            st.dataframe(pd.DataFrame(R["ablation"]).round(3), hide_index=True, width="stretch")
        if "subgroups" in R:
            st.markdown("#### Subgroup performance")
            st.dataframe(pd.DataFrame(R["subgroups"]).round(3), hide_index=True, width="stretch")
        if IB:
            st.markdown("#### Integrity-layer corruption benchmark (earlier, hour-level view)")
            st.caption("Hour-level AUROC and detection rates on the 3,086-patient benchmark sample, not the full test set. "
                       "AUROC can rise under a fault that pushes risk up, so it is not the right safety metric; the "
                       "alert-changing analysis above is the primary result.")
            st.dataframe(pd.DataFrame(IB["corruptions"]).round(3), hide_index=True, width="stretch")

# ---------------------------------------------------------------- how tab
with tab_how:
    st.markdown("#### Architecture")
    st.markdown("SepsisShield separates prediction from input verification. One path estimates sepsis risk, while "
                "independent input-integrity checks evaluate whether the clinical inputs appear reliable. A decision "
                "layer then decides whether the prediction is shown, shown with a warning, or withheld. "
                "Model disagreement may reduce trust to REDUCED, but LOW trust and prediction withholding are triggered only by evidence from the clinical inputs.")
    st.image(str(ROOT / "results" / "figures" / "0_architecture.png"), width="stretch")
    st.caption("Exact rule: LOW trust (withhold) comes only from input evidence: physiologically implausible values, inconsistent "
               "blood-pressure readings, or a coordinated 'normalising' shift. REDUCED (warn) comes from other input "
               "flags (a recent problem, a frozen feed, device disagreement, a unit note), or from unusual disagreement "
               "between the five models.")
    st.markdown("""
#### Pipeline
1. **Raw hourly data**: 8 vital signs, 26 labs and 6 demographic/context fields (PhysioNet 2019 format).
2. **Input-integrity layer** (runs on raw inputs, independent of the model): physiological plausibility,
   internal consistency, abrupt jumps, coordinated "normalising" shifts and frozen feeds.
3. **Causal feature engineering**: 172 features using only data up to the current hour, covering last values,
   informative missingness, 6 h / 12 h trends, lab trajectories and SIRS/qSOFA/SOFA-style composites.
4. **5-model LightGBM ensemble** → **isotonic calibration** → risk %, with ensemble spread as model uncertainty.
5. **Alert threshold** chosen on the validation set to maximise the official challenge utility.
6. **Trust level** = input checks + model disagreement → HIGH / REDUCED / LOW.
7. **SHAP explanations** for every hour.

#### What the trust levels mean
- ✅ **HIGH → SHOW PREDICTION**: inputs passed every check; the risk score can be read as calibrated.
- ⚠️ **REDUCED → VERIFY INPUTS**: a recent input problem, a frozen feed, or unusual model disagreement. The prediction
  is shown with a warning.
- ⛔ **LOW → PREDICTION WITHHELD**: physiologically implausible or contradictory values, or a pattern consistent with manipulation. The
  prediction is **withheld** ("requires data verification"). The research score is kept for audit but not presented
  as actionable.

#### Terms
- **AUROC**: how well the risk score ranks septic above non-septic patient-hours (0.5 = chance, 1.0 = perfect).
- **Calibrated**: a risk of 5% means that about 5 in 100 such hours precede sepsis.
- **SHAP**: how much each measurement pushed this hour's risk up or down.
- **Ensemble spread**: how much the five models disagree (log-odds scale).
- **Distribution shift**: whether this patient's recent inputs differ from the training data. It is computed as the
  share of the 20 most important model inputs (by LightGBM gain, excluding the ICU-hour index) that fall outside the
  0.5th–99.5th percentile range seen in training patients, averaged over the last 6 hours. MODERATE means more unusual
  than 95% of training patient-hours, HIGH more unusual than 99%. It is an awareness signal only and never changes the
  risk, alert, trust or decision.

#### Limits
Retrospective data from two US hospital systems; sepsis labels follow the challenge's Sepsis-3-based definition.
Not prospectively validated, not a medical device, and not a substitute for clinical judgement.
""")
