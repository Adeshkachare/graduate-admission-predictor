import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Graduate Admission Predictor", page_icon="🎓", layout="wide")

# --------------------------------------------------------------------------
# Styling: editorial look (ivory / ink / clay, serif headlines, hairlines),
# subtle reveal animations that play once per session.
# --------------------------------------------------------------------------
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Inter:wght@400;500;600&display=swap');
:root{--paper:#faf9f5;--ink:#141413;--clay:#d97757;--oat:#e8e6dc;--soft:rgba(20,20,19,.62);--card:#fffefb;--olive:#788c5d;color-scheme:light}
@property --p{syntax:"<number>";inherits:false;initial-value:0}
.stApp{background:var(--paper)}
.stApp,.stApp button,.stApp input{font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
.stApp [data-testid="stMarkdownContainer"] p,.stApp [data-testid="stWidgetLabel"] p,.stApp [data-testid="stWidgetLabel"] label,.stApp [data-testid="stToggle"] p,.stApp [data-testid="stCaptionContainer"] p{color:var(--ink)}
[data-testid="stHeader"]{background:transparent}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{max-width:1100px;padding:1.2rem 1.5rem 3rem}
.stApp [data-testid="stSlider"],.stApp [data-testid="stToggle"],.stApp [data-baseweb="tab-highlight"]{filter:hue-rotate(123deg) saturate(.72)}
.stApp [data-baseweb="tab-list"]{gap:1.4rem}
.stApp [data-baseweb="tab-border"]{background:var(--oat)}
.stApp button[data-baseweb="tab"] p{font-weight:500;color:var(--ink)}
.serif,.stApp .serif{font-family:Newsreader,Georgia,"Times New Roman",serif;font-weight:400;letter-spacing:-.01em}
.nav{display:flex;justify-content:space-between;align-items:center;padding:.3rem .2rem 1rem;border-bottom:1px solid var(--oat);margin-bottom:1.2rem}
.stApp .brand{font-family:Newsreader,Georgia,serif;font-size:1.3rem;color:var(--ink)}
.brand i{font-style:normal;color:var(--clay);margin:0 .3rem}
.stApp .pill{padding:.35rem .9rem;border-radius:999px;border:1px solid var(--oat);font-size:.8rem;color:var(--soft)}
.hero{position:relative;overflow:hidden;background:var(--ink);border-radius:26px;padding:3.2rem 2.6rem 2.8rem}
.hero::after{content:"";position:absolute;right:-90px;top:-100px;width:340px;height:340px;border-radius:50%;background:radial-gradient(circle,rgba(217,119,87,.5),transparent 68%)}
.stApp .hero .eyebrow{position:relative;z-index:1;font-size:.76rem;letter-spacing:.16em;font-weight:600;color:var(--clay)}
.stApp .hero .h1{position:relative;z-index:1;font-size:clamp(2.1rem,5.2vw,3.5rem);line-height:1.07;color:var(--paper);margin:.7rem 0 1rem;max-width:12em}
.stApp .hero p{position:relative;z-index:1;color:rgba(250,249,245,.74);max-width:35rem;font-size:1.05rem;line-height:1.55;margin:0}
.pillars{display:grid;grid-template-columns:repeat(3,1fr);gap:2rem;margin:2rem 0 .2rem}
.stApp .pillar b{display:block;font-family:Newsreader,Georgia,serif;font-weight:400;font-size:1.55rem;color:var(--ink);border-top:1px solid var(--ink);padding-top:.7rem;margin-bottom:.35rem}
.stApp .pillar span{color:var(--soft);font-size:.93rem;line-height:1.5}
.sec-h{display:flex;align-items:baseline;gap:.9rem;margin:2.4rem 0 1rem}
.stApp .sec-h .n{font-size:.8rem;font-weight:600;letter-spacing:.1em;color:var(--clay)}
.stApp .sec-h .t{font-size:clamp(1.5rem,3.4vw,2rem);color:var(--ink)}
.stApp .card{height:100%;background:var(--card);border:1px solid var(--oat);border-radius:20px;padding:1.4rem 1.5rem;color:var(--ink);transition:border-color .25s,transform .25s,box-shadow .25s}
.card:hover{border-color:var(--clay);transform:translateY(-2px);box-shadow:0 14px 30px rgba(20,20,19,.06)}
.stApp .ct{font-size:.76rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--soft);margin-bottom:1rem}
.center{text-align:center}
.gauge-wrap{position:relative;width:min(100%,210px);aspect-ratio:1;margin:.2rem auto .9rem}
.ring{position:absolute;inset:0;border-radius:50%;background:conic-gradient(var(--c) calc(var(--p)*1%),rgba(20,20,19,.08) 0);-webkit-mask:radial-gradient(farthest-side,transparent 74%,#000 75%);mask:radial-gradient(farthest-side,transparent 74%,#000 75%)}
.gauge-label{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.stApp .gauge-label b{font-family:Newsreader,Georgia,serif;font-weight:400;font-size:3.1rem;line-height:1;color:var(--ink)}
.stApp .gauge-label span{font-size:.8rem;color:var(--soft);margin-top:.3rem}
.stApp .badge{display:inline-block;padding:.32rem .95rem;border-radius:999px;color:#fff;font-weight:600;font-size:.85rem}
.stApp .headline{font-family:Newsreader,Georgia,serif;font-weight:400;font-size:clamp(1.4rem,3vw,1.85rem);line-height:1.2;color:var(--ink);margin-bottom:.6rem}
.stApp .card p{color:var(--ink);line-height:1.55;margin:.35rem 0}
.stApp .card p.muted{color:var(--soft)}
.cmp{margin-bottom:1rem}
.stApp .cmp-h{display:flex;justify-content:space-between;gap:.5rem;font-size:.9rem;margin-bottom:.4rem;color:var(--ink)}
.stApp .cmp-h em{font-style:normal;color:var(--soft)}
.track{position:relative;height:8px;border-radius:99px;background:rgba(20,20,19,.08)}
.fill{height:100%;border-radius:99px}
.mark{position:absolute;top:-5px;width:2px;height:18px;background:var(--ink);transform:translateX(-50%)}
.stApp .legend{font-size:.78rem;color:var(--soft);margin-top:.3rem}
.stApp .sec{font-size:.76rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--soft);margin:.9rem 0 .4rem}
.stApp .chip{display:inline-block;margin:.2rem .35rem .2rem 0;padding:.3rem .8rem;border-radius:999px;font-size:.85rem;border:1px solid;color:var(--ink)}
.chip.good{background:rgba(120,140,93,.14);border-color:rgba(120,140,93,.5)}
.chip.gap{background:rgba(217,119,87,.12);border-color:rgba(217,119,87,.5)}
.rows{border-top:1px solid var(--ink)}
.row{display:grid;grid-template-columns:3.4rem 1fr auto;gap:1rem;align-items:start;padding:1.2rem .4rem;border-bottom:1px solid var(--oat);transition:background .25s,padding-left .25s}
.row:hover{background:rgba(217,119,87,.07);padding-left:.9rem}
.stApp .row .no{font-family:Newsreader,Georgia,serif;font-size:1.7rem;line-height:1.1;color:var(--clay)}
.stApp .row .t{font-size:1.08rem;font-weight:600;color:var(--ink)}
.stApp .row p{margin:.3rem 0 0;color:var(--soft);font-size:.92rem;line-height:1.5}
.stApp .row .g{font-family:Newsreader,Georgia,serif;font-size:1.55rem;color:var(--olive);white-space:nowrap}
.proj{margin-top:1.4rem;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:1rem;background:var(--ink);border-radius:22px;padding:1.5rem 1.8rem}
.stApp .proj span{color:rgba(250,249,245,.74)}
.stApp .proj .big{font-family:Newsreader,Georgia,serif;font-size:2.3rem;color:var(--clay)}
.stApp .tbl{width:100%;border-collapse:collapse;font-size:.93rem}
.stApp .tbl th{text-align:left;font-weight:500;color:var(--soft);border-bottom:1px solid var(--ink);padding:.65rem .5rem}
.stApp .tbl td{color:var(--ink);padding:.8rem .5rem;border-bottom:1px solid var(--oat)}
.tbl tr.best td{background:rgba(217,119,87,.08)}
.stApp .foot{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--oat);font-size:.8rem;color:var(--soft)}
@keyframes up{from{opacity:0;transform:translateY(18px)}}
@keyframes grow{from{transform:scaleX(0)}}
@keyframes sweep{from{--p:0}}
.an{animation:up .8s cubic-bezier(.2,.7,.2,1) backwards;animation-delay:var(--d,0s)}
.an .fill{transform-origin:left;animation:grow 1s cubic-bezier(.2,.7,.2,1) backwards;animation-delay:calc(var(--d,0s) + .25s)}
.an .ring{animation:sweep 1.2s cubic-bezier(.2,.7,.2,1) backwards;animation-delay:.25s}
@media (max-width:640px){.hero{padding:2rem 1.4rem}.pillars{grid-template-columns:1fr;gap:1.2rem}.row{grid-template-columns:2.4rem 1fr}.row .g{grid-column:2}.pill{display:none}}
@media (prefers-reduced-motion:reduce){.an,.an *,.an *::after{animation:none!important}}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

# Play entrance animations only on the first render of a session (not on every slider move)
intro = not st.session_state.get("seen", False)
st.session_state["seen"] = True


def an(delay=0.0):
    return f' an" style="--d:{delay}s' if intro else ""


# --------------------------------------------------------------------------
# Model + data
# --------------------------------------------------------------------------
FEATURES = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
TARGET = "Chance of Admit"
FALLBACK_TOP = {"GRE Score": 330, "TOEFL Score": 113, "University Rating": 4.0,
                "SOP": 4.3, "LOR": 4.2, "CGPA": 9.3, "Research": 0.9}
TIERS = [(0.85, "Excellent", "#5d7a3f"), (0.75, "Strong", "#788c5d"),
         (0.60, "Competitive", "#6a9bcc"), (0.40, "Reach", "#d97757"), (0.0, "Long shot", "#b5503c")]
SUMMARY = {
    "Excellent": "Your profile sits at the top of the range. Focus on polishing the application itself rather than your scores.",
    "Strong": "You are a strong candidate. Small gains in the areas below can make you a safer bet.",
    "Competitive": "You are in the running, but not a safe bet at this university tier. Targeted improvements could move you into the strong range.",
    "Reach": "Admission is possible but uncertain. Improve several areas and balance your list with safer universities.",
    "Long shot": "Chances are low at this tier right now. Prioritise the improvements below and widen your university list.",
}


@st.cache_resource
def load_model():
    bundle = joblib.load("model.pkl")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return bundle["model"], bundle["features"], metrics


@st.cache_data
def load_df():
    p = Path("Admission_Predict.csv")
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d.columns = d.columns.str.strip()
    return d


model, features, metrics = load_model()
df = load_df()
TOP = df[df[TARGET] >= 0.8][FEATURES].mean().to_dict() if df is not None else FALLBACK_TOP


def predict(row):
    return float(np.clip(model.predict(pd.DataFrame([row])[features])[0], 0, 1))


def tier_of(p):
    for thr, name, color in TIERS:
        if p >= thr:
            return name, color


def build_levers(r):
    L = []
    if r["CGPA"] < 10:
        n = round(min(10, r["CGPA"] + 0.3), 2)
        L.append((f"Raise CGPA to {n:.2f}", {"CGPA": n},
                  "Strong final-semester grades and an upward trend count. Spotlight your best coursework and projects."))
    if r["GRE Score"] < 340:
        n = int(min(340, r["GRE Score"] + 8))
        L.append((f"Raise GRE to {n}", {"GRE Score": n},
                  f"A retake after focused prep on your weaker section is the quickest gain. Top applicants in the data average {TOP['GRE Score']:.0f}."))
    if r["TOEFL Score"] < 120:
        n = int(min(120, r["TOEFL Score"] + 4))
        L.append((f"Raise TOEFL to {n}", {"TOEFL Score": n},
                  f"Practise speaking and writing, then retake. Top applicants in the data average {TOP['TOEFL Score']:.0f}."))
    if r["SOP"] < 5:
        n = min(5, r["SOP"] + 1)
        L.append((f"Strengthen your SOP to {n:g}", {"SOP": n},
                  "Lead with a specific motivation, tie your projects to the program, and have a mentor review each draft."))
    if r["LOR"] < 5:
        n = min(5, r["LOR"] + 1)
        L.append((f"Strengthen your LORs to {n:g}", {"LOR": n},
                  "Ask professors or supervisors who know your work closely, and share your résumé and goals so letters are specific."))
    if r["Research"] == 0:
        L.append(("Add research experience", {"Research": 1},
                  "Join a faculty project, present or publish a paper, or do a research internship. Even one solid project helps."))
    if r["University Rating"] > 1:
        n = r["University Rating"] - 1
        L.append((f"Add safer universities (rating {n})", {"University Rating": n},
                  "Balance reach schools with target and safe options so you secure an offer."))
    return L


def gauge_html(p, color):
    return (f'<div class="gauge-wrap"><div class="ring" style="--p:{p*100:.1f};--c:{color}"></div>'
            f'<div class="gauge-label"><b>{p:.0%}</b><span>estimated chance</span></div></div>')


def bar_html(label, you, top, lo, hi, fmt):
    pu = float(np.clip((you - lo) / (hi - lo), 0, 1) * 100)
    pt = float(np.clip((top - lo) / (hi - lo), 0, 1) * 100)
    color = "#788c5d" if you >= top - 0.02 * (hi - lo) else "#d97757"
    return (f'<div class="cmp"><div class="cmp-h"><span>{label}</span><span>{fmt.format(you)} <em>vs {fmt.format(top)}</em></span></div>'
            f'<div class="track"><div class="fill" style="width:{pu:.0f}%;background:{color}"></div>'
            f'<div class="mark" style="left:{pt:.0f}%"></div></div></div>')


def section(n, title, delay=0.0):
    st.markdown(f'<div class="sec-h{an(delay)}"><span class="n">{n}</span><span class="t serif">{title}</span></div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.markdown('<div class="nav"><span class="brand">Admission<i>/</i>Predictor</span>'
            '<span class="pill">Machine learning · scikit-learn</span></div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="hero{an(0)}"><div class="eyebrow">GRADUATE ADMISSIONS</div>'
    '<div class="h1 serif">Know your odds.<br>Know what to improve.</div>'
    '<p>Enter your academic profile. Get an estimated chance of admission, an honest read on where you stand, '
    'and a ranked plan showing how much each change could help.</p></div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="pillars{an(0.15)}">'
    '<div class="pillar"><b>Predict.</b><span>A model trained on historical applicant records estimates your chance of admission.</span></div>'
    '<div class="pillar"><b>Evaluate.</b><span>See exactly how each score compares with the profile of top applicants.</span></div>'
    '<div class="pillar"><b>Improve.</b><span>Get a ranked list of changes, each with its estimated gain.</span></div></div>',
    unsafe_allow_html=True)

tab_predict, tab_model = st.tabs(["Predict", "About the model"])

with tab_predict:
    section("01", "Your profile", 0.2)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        gre = st.slider("GRE Score", 260, 340, 316)
        toefl = st.slider("TOEFL Score", 0, 120, 107)
    with c2:
        cgpa = st.slider("CGPA (out of 10)", 5.0, 10.0, 8.6, 0.01)
        uni = st.select_slider("University Rating", options=[1, 2, 3, 4, 5], value=3)
    with c3:
        sop = st.select_slider("SOP Strength", options=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], value=3.5)
        lor = st.select_slider("LOR Strength", options=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], value=3.5)
    research = st.toggle("I have research experience", value=True)

    row = {"GRE Score": gre, "TOEFL Score": toefl, "University Rating": uni,
           "SOP": sop, "LOR": lor, "CGPA": cgpa, "Research": int(research)}
    p = predict(row)
    tier, color = tier_of(p)

    levers = []
    for title, upd, tip in build_levers(row):
        gain = predict({**row, **upd}) - p
        if gain >= 0.004:
            levers.append((gain, title, upd, tip))
    levers.sort(key=lambda x: -x[0])

    specs = [("GRE Score", "GRE", 260, 340, "{:.0f}"), ("TOEFL Score", "TOEFL", 80, 120, "{:.0f}"),
             ("CGPA", "CGPA", 6, 10, "{:.2f}"), ("SOP", "SOP", 1, 5, "{:.1f}"), ("LOR", "LOR", 1, 5, "{:.1f}")]
    strengths, gaps = [], []
    for key, label, lo, hi, fmt in specs:
        if row[key] >= TOP[key] - 0.02 * (hi - lo):
            strengths.append(f"{label} {fmt.format(row[key])}")
        else:
            gaps.append(f"{label} {fmt.format(row[key])} vs {fmt.format(TOP[key])}")
    if row["Research"]:
        strengths.append("Research experience")
    else:
        gaps.append(f"No research (about {TOP['Research']:.0%} of top applicants have it)")

    section("02", "Your estimate", 0.3)
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        st.markdown(f'<div class="card center{an(0.35)}">{gauge_html(p, color)}'
                    f'<span class="badge" style="background:{color}">{tier}</span></div>', unsafe_allow_html=True)
    with right:
        higher = [t for t in sorted(TIERS) if t[0] > p]
        nxt = ""
        if higher:
            thr, name, _ = higher[0]
            nxt = f'<p class="muted">You are <b>{(thr - p) * 100:.1f} points</b> away from the <b>{name}</b> range.</p>'
        top_line = ""
        if levers:
            g, t, _, _ = levers[0]
            top_line = f'<p>Biggest lever: <b>{t}</b>, about <b>+{g * 100:.1f} pts</b>.</p>'
        st.markdown(f'<div class="card{an(0.45)}"><div class="headline">{tier} profile at {p:.1%}</div>'
                    f'<p>{SUMMARY[tier]}</p>{nxt}{top_line}</div>', unsafe_allow_html=True)

    section("03", "How you compare", 0.5)
    b1, b2 = st.columns(2, gap="large")
    with b1:
        bars = "".join(bar_html(l, row[k], TOP[k], lo, hi, f) for k, l, lo, hi, f in specs)
        st.markdown(f'<div class="card{an(0.55)}"><div class="ct">Profile vs top applicants</div>{bars}'
                    '<div class="legend">Bar = you &nbsp;|&nbsp; black tick = average of applicants with 80%+ chance</div></div>',
                    unsafe_allow_html=True)
    with b2:
        good = "".join(f'<span class="chip good">✓ {s}</span>' for s in strengths) or '<p class="muted">None yet</p>'
        gap = "".join(f'<span class="chip gap">▲ {g}</span>' for g in gaps) or '<p class="muted">No gaps. You match or beat top applicants everywhere.</p>'
        st.markdown(f'<div class="card{an(0.65)}"><div class="ct">Evaluation</div>'
                    f'<div class="sec" style="margin-top:0">Strengths</div>{good}'
                    f'<div class="sec">Needs improvement</div>{gap}</div>', unsafe_allow_html=True)

    section("04", "Priority improvements", 0.7)
    if levers:
        rows = "".join(
            f'<div class="row{an(0.75 + i * 0.1)}"><span class="no">{i:02d}</span>'
            f'<div><div class="t">{t}</div><p>{tip}</p></div><span class="g">+{g * 100:.1f} pts</span></div>'
            for i, (g, t, _, tip) in enumerate(levers[:4], 1))
        st.markdown(f'<div class="rows">{rows}</div>', unsafe_allow_html=True)

        combined = dict(row)
        for _, _, upd, _ in levers[:3]:
            combined.update(upd)
        p2 = predict(combined)
        if p2 > p:
            t2, _ = tier_of(p2)
            st.markdown(
                f'<div class="proj{an(1.1)}"><span>Projected chance if you act on the top {min(3, len(levers))}</span>'
                f'<span class="big">{p:.0%} → {p2:.0%}</span><span>{t2} range</span></div>', unsafe_allow_html=True)
    else:
        st.success("No single change moves your estimate meaningfully. Your profile is already at the top of the range.")

    st.markdown('<div class="foot">Estimates come from patterns in historical data, not causal guarantees. '
                'Gains are the model\'s what-if predictions for each change on its own.</div>', unsafe_allow_html=True)

with tab_model:
    section("01", "Model comparison")
    best = metrics["best"]
    trs = "".join(
        f'<tr class="{"best" if k == best else ""}"><td>{k}{" ●" if k == best else ""}</td>'
        f'<td>{v["cv_r2"]:.3f}</td><td>{v["test_r2"]:.3f}</td><td>{v["test_mae"]:.3f}</td></tr>'
        for k, v in metrics["results"].items())
    st.markdown(
        f'<div class="card"><div class="ct">Deployed model: {best} (selected by 5-fold cross-validated R²)</div>'
        f'<table class="tbl"><tr><th>Model</th><th>CV R²</th><th>Test R²</th><th>Test MAE</th></tr>{trs}</table></div>',
        unsafe_allow_html=True)

    imp = None
    if df is not None:
        @st.cache_resource
        def get_importance():
            r = permutation_importance(model, df[features], df[TARGET], n_repeats=10, random_state=42)
            return pd.Series(r.importances_mean, index=features)
        imp = get_importance()
    elif hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=features)
    if imp is not None:
        section("02", "What drives the prediction")
        imp = imp.clip(lower=0).sort_values(ascending=False)
        mx = float(imp.max()) or 1.0
        bars = "".join(
            f'<div class="cmp"><div class="cmp-h"><span>{k}</span><span>{v / mx:.0%}</span></div>'
            f'<div class="track"><div class="fill" style="width:{v / mx * 100:.0f}%;background:#d97757"></div></div></div>'
            for k, v in imp.items())
        st.markdown(f'<div class="card"><div class="ct">Permutation importance (relative)</div>{bars}'
                    '<div class="legend">How much accuracy drops when a feature is shuffled.</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="foot">Trained on the public Graduate Admissions dataset (500 records). '
                'For learning and demonstration only, not real admissions advice.</div>', unsafe_allow_html=True)