"""
AppendiCheck Kids — Paediatric Appendicitis Screening Tool
A single-file Streamlit app with an editorial "portfolio" aesthetic (cream/light theme).

Run:
    pip install streamlit pandas numpy joblib scikit-learn
    streamlit run appendicheck_kids.py

If a trained model `appendicitis_model.pkl` is present in the same folder it is
used for prediction; otherwise a transparent clinical heuristic is used so the
app still runs standalone.
"""

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="AppendiCheck Kids",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# Design tokens + global style (forces light/cream theme)
# =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --cream: #F5F2ED;
        --cream-deep: #EFEAE0;
        --card: #FFFFFF;
        --ink: #0B0B0B;
        --ink-soft: #2A2A28;
        --muted: #777777;
        --line: #0B0B0B;
        --line-soft: #D8D3C9;
        --brick: #C25C4E;
        --win: #EBEBEB;
    }

    /* Force the cream/light theme regardless of Streamlit's saved theme preference */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"], section[data-testid="stMain"],
    [data-testid="stHeader"], [data-testid="stBottomBlock"],
    .block-container, [data-testid="stVerticalBlock"] {
        background: var(--cream) !important;
        color: var(--ink) !important;
    }
    [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    h1, h2, h3 { font-family: 'Playfair Display', serif; color: var(--ink) !important; }
    .stMarkdown p { color: var(--ink); }

    .wrap { max-width: 1000px; margin: 0 auto; padding: 0 1rem; }

    /* ---------- Window bar ---------- */
    .winbar { display:flex; align-items:center; gap:.75rem; height:44px; padding:0 1rem;
              background: var(--win); border-bottom:1px solid var(--ink); }
    .winbar .dots { display:flex; gap:.5rem; }
    .winbar .dot { width:12px; height:12px; border-radius:50%; }
    .winbar .url { flex:1; display:flex; justify-content:center; }
    .winbar .url span { height:24px; padding:0 1rem; min-width:55%; background: rgba(255,255,255,.6);
                        border:1px solid rgba(11,11,11,.15); display:flex; align-items:center; justify-content:center;
                        font-family:'IBM Plex Mono',monospace; font-size:.7rem; color: rgba(11,11,11,.6); }

    /* ---------- Hero ---------- */
    .hero { border:1px solid var(--ink); background: var(--card); overflow:hidden; }
    .hero-grid { display:flex; min-height:340px; }
    .hero-visual { flex:0 0 36%; background:#000000; border-right:1px solid var(--ink);
                   display:flex; align-items:stretch; justify-content:center; min-height:300px; }
    .hero-content { flex:1; padding:2.4rem 2.6rem; display:flex; flex-direction:column; justify-content:center; background: var(--card); }
    .eyebrow { font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.18em;
               text-transform:uppercase; color:var(--muted); margin-bottom:1rem; }
    .hero h1 { font-size:3.4rem; font-weight:800; line-height:1.02; letter-spacing:.01em;
               text-transform:uppercase; margin:0 0 1rem 0; }
    .hero p { font-size:.98rem; max-width:520px; color:var(--muted); margin:0 0 1.4rem 0; }
    .hero-credit { display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:.6rem;
                   border-top:1px solid var(--line-soft); padding-top:1rem; margin-top:.4rem; }
    .hero-credit .who { font-size:.92rem; font-weight:500; color: var(--ink); }
    .hero-credit .who span { display:block; color:var(--muted); font-size:.8rem; font-weight:400; }
    .hero-credit .stats { font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.04em;
                          color:var(--muted); text-align:right; }
    .hero-credit .stats b { color:var(--ink); font-weight:600; }

    /* ---------- Disclaimer ---------- */
    .disclaimer { background:var(--card); border:1px solid var(--line-soft); border-left:3px solid var(--ink);
                  padding:.9rem 1.2rem; margin:1.2rem 0; font-size:.88rem; color:var(--muted); }
    .disclaimer b { color:var(--ink); }

    /* ---------- Section nav (thumbnail cards) ---------- */
    div[data-testid="stRadio"] > label { display:none; }
    div[data-testid="stRadio"] > div[role="radiogroup"] { gap:10px; flex-wrap:wrap; margin-bottom:1rem; }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background:var(--card); border:1px solid var(--line-soft); padding:.85rem 1.3rem;
        display:flex; align-items:center; gap:.5rem; cursor:pointer; transition:all .15s ease; }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display:none; }
    div[data-testid="stRadio"] label[data-baseweb="radio"] p {
        margin:0; font-weight:600; font-size:.8rem; color:var(--muted); white-space:nowrap;
        text-transform:uppercase; letter-spacing:.04em; }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background:var(--cream-deep); border:1px solid var(--ink); }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p { color:var(--ink); }

    /* ---------- Section panels ---------- */
    .st-key-sec_patient, .st-key-sec_symptoms, .st-key-sec_blood, .st-key-sec_ultrasound {
        background:var(--card) !important; border:1px solid var(--ink) !important; border-radius:0 !important;
        padding:1.8rem 2rem 1.2rem 2rem !important; margin-bottom:1.1rem !important; }
    .section-title { font-family:'Playfair Display',serif; font-weight:700; font-size:1.3rem; margin-bottom:.15rem; color: var(--ink); }
    .section-sub { font-size:.85rem; color:var(--muted); margin-bottom:1.2rem; }
    .subsection-divider { height:1px; background:var(--line-soft); margin:1.4rem 0 1.2rem 0; }

    label, .stSlider label, .stSelectbox label {
        color:var(--ink) !important; font-size:.83rem !important; font-weight:500 !important; }
    .stSlider [data-baseweb="slider"] > div > div { background:var(--ink) !important; }
    .stSlider [role="slider"] { background-color:var(--ink) !important; border:3px solid var(--cream) !important; }
    [data-testid="stSelectbox"] > div > div {
        background-color:var(--card) !important; border-color:var(--ink) !important; border-radius:0 !important; }
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSelectbox"] > div > div * { color:var(--ink) !important; }
    [data-testid="stSelectbox"] svg { fill:var(--ink) !important; }
    div[data-baseweb="popover"] div[data-baseweb="menu"] { background-color:var(--card) !important; border:1px solid var(--ink) !important; }
    div[data-baseweb="popover"] li { background-color:var(--card) !important; color:var(--ink) !important; }
    div[data-baseweb="popover"] li:hover { background-color:var(--cream-deep) !important; }

    /* ---------- Button ---------- */
    div.stButton > button {
        width:100%; height:3.6rem; border-radius:0; background:transparent; color:var(--ink);
        font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:.9rem;
        border:1px solid var(--ink); transition:all .15s ease; }
    div.stButton > button:hover { background:var(--ink); color:var(--cream); border-color:var(--ink); }

    /* ---------- Result ---------- */
    .result-banner { padding:1.8rem 2rem; background:var(--card); border:1px solid var(--ink); border-left:4px solid var(--ink); margin-bottom:1rem; }
    .result-high { border-left-color:var(--brick); }
    .result-banner .kicker { font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.12em;
                             text-transform:uppercase; color:var(--muted); margin-bottom:.5rem; }
    .result-banner h2 { font-size:2rem; font-weight:800; margin:0 0 .6rem 0; text-transform:uppercase; color: var(--ink); }
    .result-high h2 { color:var(--brick); }
    .result-banner p { margin:0; color:var(--muted); max-width:640px; }
    .risk-badge { display:inline-block; padding:.3rem .7rem; font-size:.7rem; font-weight:700;
                  text-transform:uppercase; letter-spacing:.05em; margin-left:.6rem; }
    .risk-low { background:var(--ink); color:var(--cream); }
    .risk-mod, .risk-high { background:var(--brick); color:var(--cream); }

    .gauge-wrap { display:flex; align-items:center; gap:2.5rem; flex-wrap:wrap; }
    .gauge-outer { border-radius:50%; width:160px; height:160px; display:flex; align-items:center; justify-content:center; }
    .gauge-inner { border-radius:50%; width:118px; height:118px; background:var(--card); border:1px solid var(--line-soft);
                   display:flex; flex-direction:column; align-items:center; justify-content:center; }
    .gauge-pct { font-family:'IBM Plex Mono',monospace; font-size:1.8rem; font-weight:600; }
    .gauge-label { font-size:.62rem; color:var(--muted); letter-spacing:.05em; text-transform:uppercase; margin-top:.2rem; }
    .prob-row { display:flex; align-items:center; gap:1rem; padding:.5rem 0; }
    .prob-row .lbl { width:160px; font-size:.85rem; font-weight:600; color: var(--ink); }
    .prob-row .pct { width:56px; text-align:right; font-family:'IBM Plex Mono',monospace; font-size:.82rem; color:var(--muted); }
    .prob-row .bar-bg { flex:1; height:8px; background:var(--cream-deep); overflow:hidden; }
    .prob-row .bar-fill { height:8px; }

    .factors { display:grid; grid-template-columns:1fr 1fr; gap:.75rem; }
    .factor { display:flex; gap:.6rem; border:1px solid var(--line-soft); padding:.7rem; background: var(--card); }
    .factor .dot { width:8px; height:8px; border-radius:50%; margin-top:.35rem; flex:0 0 auto; }
    .factor .f-up { background:var(--brick); }
    .factor .f-down { background:rgba(11,11,11,.4); }
    .factor .f-neu { background:var(--muted); }
    .factor p { margin:0; font-size:.85rem; color: var(--ink); }
    .factor .eff { font-family:'IBM Plex Mono',monospace; font-size:.62rem; text-transform:uppercase; color:var(--muted); }

    .footnote { color:var(--muted); font-size:.78rem; font-family:'IBM Plex Mono',monospace; margin-top:1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Model loader (with heuristic fallback)
# =========================
@st.cache_resource
def load_model():
    path = "appendicitis_model.pkl"
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


MODEL = load_model()


def heuristic_predict(d):
    """Transparent clinical heuristic used when no trained model is available."""
    s = 0.0
    s += (d["Alvarado_Score"] / 10.0) * 0.30
    s += (d["Paedriatic_Appendicitis_Score"] / 10.0) * 0.25
    if d["Lower_Right_Abd_Pain"] == "yes":
        s += 0.06
    if d["Migratory_Pain"] == "yes":
        s += 0.06
    if d["Nausea"] == "yes":
        s += 0.03
    if d["Loss_of_Appetite"] == "yes":
        s += 0.03
    if d["Peritonitis"] == "yes":
        s += 0.08
    if d["WBC_Count"] > 11:
        s += min(0.08, (d["WBC_Count"] - 11) / 50.0)
    if d["CRP"] > 10:
        s += min(0.08, (d["CRP"] - 10) / 1100.0)
    if d["Neutrophil_Percentage"] > 75:
        s += 0.05
    if d["Appendix_on_US"] == "yes" and d["Appendix_Diameter"] > 6:
        s += min(0.10, (d["Appendix_Diameter"] - 6) / 40.0)
    if d["Free_Fluids"] == "yes":
        s += 0.06
    prob = max(0.02, min(0.98, s))
    pred = "appendicitis" if prob >= 0.5 else "no appendicitis"
    return pred, prob


def build_key_factors(d):
    factors = []
    if d["Alvarado_Score"] >= 7 or d["Paedriatic_Appendicitis_Score"] >= 7:
        factors.append(("Clinical scores (Alvarado/PAS) strongly positive", "increases risk"))
    elif d["Alvarado_Score"] < 5 and d["Paedriatic_Appendicitis_Score"] < 5:
        factors.append(("Clinical scores below threshold", "decreases risk"))
    if d["Peritonitis"] == "yes":
        factors.append(("Peritonitis / rebound guarding present", "increases risk"))
    if d["Migratory_Pain"] == "yes":
        factors.append(("Migratory pain pattern", "increases risk"))
    if d["WBC_Count"] > 15 or d["CRP"] > 50:
        factors.append(("Elevated inflammatory markers (WBC/CRP)", "increases risk"))
    if d["Appendix_on_US"] == "yes" and d["Appendix_Diameter"] > 6:
        factors.append(("Enlarged appendix on ultrasound", "increases risk"))
    if d["Free_Fluids"] == "yes":
        factors.append(("Free fluid on ultrasound", "increases risk"))
    if not factors:
        factors.append(("No strong positive findings", "decreases risk"))
    return factors[:4]


# =========================
# Hero
# =========================
st.markdown(
    """
    <div class="hero">
      <div class="winbar">
        <div class="dots">
          <span class="dot" style="background:#FF5F56"></span>
          <span class="dot" style="background:#FFBD2E"></span>
          <span class="dot" style="background:#27C93F"></span>
        </div>
        <div class="url"><span>appendicheck.kids — paediatric screening tool</span></div>
      </div>
      <div class="hero-grid">
        <div class="hero-visual">
          <svg viewBox="0 0 220 200" style="width:92%;height:100%">
            <text x="14" y="14" font-family="IBM Plex Mono" font-size="8" fill="#6A6A66" letter-spacing="1">MACRO F1-SCORE</text>
            <rect x="14" y="150" width="34" height="18" fill="#3A3A38" />
            <text x="14" y="145" font-family="IBM Plex Mono" font-size="9" fill="#6A6A66">0.37</text>
            <text x="14" y="182" font-family="IBM Plex Mono" font-size="7.5" fill="#6A6A66">BASELINE</text>
            <rect x="80" y="103" width="34" height="65" fill="#5A5A56" />
            <text x="80" y="98" font-family="IBM Plex Mono" font-size="9" fill="#9A9A96">0.91</text>
            <text x="72" y="182" font-family="IBM Plex Mono" font-size="7.5" fill="#9A9A96">LOG. REG</text>
            <rect x="146" y="92" width="34" height="76" fill="#F5F2ED" />
            <text x="146" y="87" font-family="IBM Plex Mono" font-size="9" fill="#F5F2ED">0.95</text>
            <text x="140" y="182" font-family="IBM Plex Mono" font-size="7.5" fill="#F5F2ED">RANDOM FOREST</text>
            <line x1="10" y1="168" x2="210" y2="168" stroke="#3A3A38" stroke-width="1" />
          </svg>
        </div>
        <div class="hero-content">
          <div class="eyebrow">Paediatric Screening Tool</div>
          <h1>AppendiCheck<br>Kids</h1>
          <p>Scores appendicitis likelihood in children from clinical scores, symptoms,
          blood work and ultrasound findings — a statistical aid to support a clinical
          decision, never to replace one.</p>
          <div class="hero-credit">
            <div class="who">AppendiCheck Kids<span>Screening support tool</span></div>
            <div class="stats">Model <b>Random Forest</b> &nbsp;·&nbsp; Accuracy <b>94.87%</b> &nbsp;·&nbsp; Macro F1 <b>94.67%</b></div>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer"><b>Disclaimer —</b> This app is an educational screening '
    'support tool. It does not replace professional medical diagnosis. Always consult a '
    'qualified clinician.</div>',
    unsafe_allow_html=True,
)

# =========================
# Inputs
# =========================
st.markdown("## Patient Assessment")
st.markdown(
    "<p style='font-size:.85rem;color:#777777;margin-top:-.5rem;margin-bottom:1rem;'>"
    "Enter the child's clinical data across the four sections, then run the screening.</p>",
    unsafe_allow_html=True,
)

SECTION_OPTIONS = ["👤 Patient Info", "🩻 Symptoms", "🧪 Blood Tests", "🔍 Ultrasound"]
active_section = st.radio("Section", SECTION_OPTIONS, horizontal=True, label_visibility="collapsed", key="active_section")

section_keys = ["sec_patient", "sec_symptoms", "sec_blood", "sec_ultrasound"]
active_key = dict(zip(SECTION_OPTIONS, section_keys))[active_section]
hide_css = "\n".join(f".st-key-{k} {{ display: none; }}" for k in section_keys if k != active_key)
st.markdown(f"<style>{hide_css}</style>", unsafe_allow_html=True)

with st.container(key="sec_patient"):
    st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Basic patient information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age (years)", 0.0, 18.0, 10.0, 0.1, help="How old the child is, in years.")
    with c2:
        sex = st.selectbox("Sex", ["female", "male"], help="Whether the child is female or male.")
    with c3:
        bmi = st.slider("BMI (kg/m²)", 5.0, 40.0, 18.0, 0.1, help="Body Mass Index from height and weight.")

    st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical Scores</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Composite scores from standard assessment tools</div>', unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    with c4:
        alvarado_score = st.slider("Alvarado Score", 0.0, 10.0, 5.0, 0.5, help="A 0–10 score from a doctor's exam and blood test. Higher = more likely appendicitis.")
    with c5:
        paediatric_score = st.slider("Paediatric Appendicitis Score", 0.0, 10.0, 5.0, 0.5, help="Like the Alvarado Score, but made for children.")

with st.container(key="sec_symptoms"):
    st.markdown('<div class="section-title">Reported Symptoms</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Select yes/no for each presenting symptom</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        lower_right_pain = st.selectbox("Lower Right Abdominal Pain", ["no", "yes"], help="Pain on the lower right side of the belly.")
    with c2:
        migratory_pain = st.selectbox("Migratory Pain", ["no", "yes"], help="Pain that started near the belly button and moved to the lower right side.")
    with c3:
        nausea = st.selectbox("Nausea", ["no", "yes"], help="Feeling like you might vomit.")
    c4, c5 = st.columns(2)
    with c4:
        loss_of_appetite = st.selectbox("Loss of Appetite", ["no", "yes"], help="Not wanting to eat, even favourite foods.")
    with c5:
        peritonitis = st.selectbox("Peritonitis", ["no", "yes"], help="Belly hurts more when pressure is released quickly.")

with st.container(key="sec_blood"):
    st.markdown('<div class="section-title">Blood Test Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Numbers from a blood test done at a clinic or hospital</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        wbc_count = st.slider("WBC Count (×10⁹/L)", 0.0, 40.0, 10.0, 0.1, help="White blood cells fight infection. Higher may mean infection.")
    with c2:
        crp = st.slider("CRP (mg/L)", 0.0, 300.0, 10.0, 0.1, help="Rises with swelling or infection. Higher = more inflammation.")
    with c3:
        neutrophil_percentage = st.slider("Neutrophil Percentage (%)", 0.0, 100.0, 60.0, 0.5, help="A white blood cell type that fights bacteria. Higher often means active infection.")

with st.container(key="sec_ultrasound"):
    st.markdown('<div class="section-title">Ultrasound Findings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">From an ultrasound scan done by a doctor or radiographer</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        appendix_on_us = st.selectbox("Appendix Seen on Ultrasound", ["no", "yes"], help="Whether the appendix could be clearly seen on the scan.")
    with c2:
        appendix_diameter = st.slider("Appendix Diameter (mm)", 0.0, 20.0, 6.0, 0.1, help="How wide the appendix measures. Over 6–7mm can suggest appendicitis.")
    with c3:
        free_fluids = st.selectbox("Free Fluids", ["no", "yes"], help="Extra fluid seen in the belly — can be a sign of inflammation.")

# =========================
# Prediction
# =========================
st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
predict_clicked = st.button("Predict Appendicitis Risk")

if predict_clicked:
    data = {
        "Age": age, "Sex": sex, "BMI": bmi,
        "Alvarado_Score": alvarado_score,
        "Paedriatic_Appendicitis_Score": paediatric_score,
        "Lower_Right_Abd_Pain": lower_right_pain,
        "Migratory_Pain": migratory_pain,
        "Nausea": nausea,
        "Loss_of_Appetite": loss_of_appetite,
        "Peritonitis": peritonitis,
        "WBC_Count": wbc_count,
        "CRP": crp,
        "Neutrophil_Percentage": neutrophil_percentage,
        "Appendix_on_US": appendix_on_us,
        "Appendix_Diameter": appendix_diameter,
        "Free_Fluids": free_fluids,
    }

    if MODEL is not None:
        df_input = pd.DataFrame([data])
        df_input = pd.get_dummies(df_input)
        df_input = df_input.reindex(columns=MODEL.feature_names_in_, fill_value=0)
        prediction = MODEL.predict(df_input)[0]
        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(df_input)[0]
            classes = list(getattr(MODEL, "classes_", None) or MODEL.named_steps["model"].classes_)
            appx_idx = classes.index("appendicitis") if "appendicitis" in classes else 1
            appx_prob = float(probs[appx_idx])
        else:
            appx_prob = 0.9 if prediction == "appendicitis" else 0.1
        key_factors = build_key_factors(data)
    else:
        prediction, appx_prob = heuristic_predict(data)
        key_factors = build_key_factors(data)

    st.markdown("## Prediction Result")
    is_high = prediction == "appendicitis"
    risk_level = "high" if appx_prob >= 0.7 else ("moderate" if appx_prob >= 0.4 else "low")
    badge_cls = {"low": "risk-low", "moderate": "risk-mod", "high": "risk-high"}[risk_level]
    badge_label = {"low": "Low risk", "moderate": "Moderate risk", "high": "High risk"}[risk_level]

    if is_high:
        st.markdown(
            f"""
            <div class="result-banner result-high">
              <div class="kicker">Screening Result</div>
              <h2>Possible appendicitis <span class="risk-badge {badge_cls}">{badge_label}</span></h2>
              <p>Based on what was entered, this tool suggests appendicitis is possible. This is not a
              diagnosis — please take the child to see a doctor or go to a clinic or hospital as soon
              as you can, so a real doctor can check them properly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-banner">
              <div class="kicker">Screening Result</div>
              <h2>No appendicitis detected <span class="risk-badge {badge_cls}">{badge_label}</span></h2>
              <p>Based on what was entered, this tool does not think appendicitis is likely right now.
              This is not a diagnosis — if the child is still in pain or you're worried, it's always
              okay to see a doctor anyway.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    gauge_color = "#C25C4E" if is_high else "#0B0B0B"
    gauge_pct = appx_prob * 100
    no_pct = (1 - appx_prob) * 100

    factors_html = ""
    for factor, effect in key_factors:
        dot_cls = {"increases risk": "f-up", "decreases risk": "f-down", "neutral": "f-neu"}[effect]
        factors_html += f"""
        <div class="factor">
          <span class="dot {dot_cls}"></span>
          <div><p>{factor}</p><span class="eff">{effect}</span></div>
        </div>"""

    st.markdown("### Prediction Probability")
    st.markdown(
        f"""
        <div style="background:#fff;border:1px solid #0B0B0B;padding:1.8rem 2rem;">
          <div class="gauge-wrap">
            <div class="gauge-outer" style="background: conic-gradient({gauge_color} {gauge_pct*3.6:.1f}deg, #EFEAE0 0deg);">
              <div class="gauge-inner">
                <div class="gauge-pct" style="color:{gauge_color};">{gauge_pct:.1f}%</div>
                <div class="gauge-label">Appendicitis risk</div>
              </div>
            </div>
            <div style="flex:1;min-width:260px;">
              <div class="prob-row">
                <span class="lbl">Appendicitis</span>
                <span class="pct">{gauge_pct:.1f}%</span>
                <div class="bar-bg"><div class="bar-fill" style="width:{gauge_pct:.1f}%;background:{gauge_color};"></div></div>
              </div>
              <div class="prob-row">
                <span class="lbl">No appendicitis</span>
                <span class="pct">{no_pct:.1f}%</span>
                <div class="bar-bg"><div class="bar-fill" style="width:{no_pct:.1f}%;background:#0B0B0B;"></div></div>
              </div>
            </div>
          </div>
          <div style="height:1px;background:#D8D3C9;margin:1.6rem 0;"></div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#777777;margin-bottom:1rem;">Key Influencing Factors</div>
          <div class="factors">{factors_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footnote">AppendiCheck Kids uses a trained Random Forest machine learning model '
    'when available, otherwise a transparent clinical heuristic. The prediction is based on '
    'selected patient information, symptoms, blood test results and ultrasound-related features.</div>',
    unsafe_allow_html=True,
)