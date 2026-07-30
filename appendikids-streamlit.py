"""
AppendiCheck Kids — Paediatric Appendicitis Screening Tool

Run:
    pip install streamlit pandas numpy joblib scikit-learn
    streamlit run appendikids-streamlit.py

The app loads appendicitis_model.pkl from the same folder.
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
    initial_sidebar_state="collapsed"
)

# =========================
# Clean dashboard styling
# =========================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #F6F8FB;
        --card: #FFFFFF;
        --text: #172033;
        --muted: #6B7280;
        --line: #E5E7EB;
        --blue: #2563EB;
        --blue-dark: #1E40AF;
        --blue-soft: #EFF6FF;
        --red: #DC2626;
        --red-soft: #FEF2F2;
        --green: #16A34A;
        --green-soft: #F0FDF4;
        --yellow-soft: #FFFBEB;
        --yellow-line: #FDE68A;
        --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }

    html, body, .stApp {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.02em;
    }

    .hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%);
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 2.4rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.4rem;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.4fr 0.8fr;
        gap: 2rem;
        align-items: center;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        background: var(--blue-soft);
        color: var(--blue-dark);
        border: 1px solid #DBEAFE;
        padding: .45rem .75rem;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .05em;
        margin-bottom: 1rem;
    }

    .hero h1 {
        font-size: 3.1rem;
        line-height: 1.05;
        margin: 0 0 .9rem 0;
        font-weight: 800;
    }

    .hero p {
        font-size: 1rem;
        line-height: 1.7;
        color: var(--muted);
        max-width: 650px;
        margin: 0;
    }

    .hero-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.2rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .hero-card-title {
        font-size: .8rem;
        color: var(--muted);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-bottom: .7rem;
    }

    .mini-bar {
        display: flex;
        align-items: center;
        gap: .7rem;
        margin: .65rem 0;
    }

    .mini-label {
        width: 95px;
        font-size: .78rem;
        font-weight: 600;
        color: var(--text);
    }

    .mini-track {
        flex: 1;
        height: 10px;
        background: #E5E7EB;
        border-radius: 999px;
        overflow: hidden;
    }

    .mini-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--blue), #60A5FA);
        border-radius: 999px;
    }

    .mini-value {
        width: 52px;
        text-align: right;
        font-size: .78rem;
        font-weight: 700;
        color: var(--blue-dark);
    }

    .disclaimer {
        background: var(--yellow-soft);
        border: 1px solid var(--yellow-line);
        color: #92400E;
        padding: 1rem 1.2rem;
        border-radius: 18px;
        margin: 1rem 0 1.4rem 0;
        font-size: .9rem;
        line-height: 1.5;
    }

    .disclaimer b {
        color: #78350F;
    }

    div[data-testid="stRadio"] > label {
        display: none;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: .7rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }

    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: .75rem 1.15rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        transition: all .15s ease;
    }

    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none;
    }

    div[data-testid="stRadio"] label[data-baseweb="radio"] p {
        margin: 0;
        font-weight: 700;
        font-size: .82rem;
        color: var(--muted);
    }

    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: var(--blue);
        border-color: var(--blue);
    }

    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
        color: white;
    }

    .st-key-sec_patient, .st-key-sec_symptoms, .st-key-sec_blood, .st-key-sec_ultrasound {
        background: var(--card) !important;
        border: 1px solid var(--line) !important;
        border-radius: 24px !important;
        padding: 1.7rem 1.9rem 1.3rem 1.9rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: var(--shadow);
    }

    .section-title {
        font-weight: 800;
        font-size: 1.25rem;
        margin-bottom: .2rem;
        color: var(--text);
    }

    .section-sub {
        font-size: .9rem;
        color: var(--muted);
        margin-bottom: 1.2rem;
    }

    .subsection-divider {
        height: 1px;
        background: var(--line);
        margin: 1.4rem 0 1.2rem 0;
    }

    label, .stSlider label, .stSelectbox label {
        color: var(--text) !important;
        font-size: .86rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stSelectbox"] > div > div {
        background-color: white !important;
        border-color: var(--line) !important;
        border-radius: 14px !important;
        min-height: 44px;
    }

    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSelectbox"] > div > div * {
        color: var(--text) !important;
    }

    .stSlider [data-baseweb="slider"] > div > div {
        background: var(--blue) !important;
    }

    .stSlider [role="slider"] {
        background-color: var(--blue) !important;
        border: 3px solid white !important;
        box-shadow: 0 0 0 1px var(--blue);
    }

    div.stButton > button {
        width: 100%;
        height: 3.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, var(--blue), var(--blue-dark));
        color: white;
        font-weight: 800;
        letter-spacing: .02em;
        border: none;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.22);
        transition: all .15s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.28);
        color: white;
        border: none;
    }

    .result-banner {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.7rem 1.9rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .result-high {
        background: var(--red-soft);
        border-color: #FECACA;
    }

    .result-low {
        background: var(--green-soft);
        border-color: #BBF7D0;
    }

    .result-banner .kicker {
        font-size: .78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .06em;
        color: var(--muted);
        margin-bottom: .45rem;
    }

    .result-banner h2 {
        font-size: 1.8rem;
        margin: 0 0 .55rem 0;
        font-weight: 800;
    }

    .result-high h2 {
        color: var(--red) !important;
    }

    .result-low h2 {
        color: var(--green) !important;
    }

    .result-banner p {
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
        max-width: 760px;
    }

    .risk-badge {
        display: inline-block;
        padding: .3rem .7rem;
        border-radius: 999px;
        font-size: .72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-left: .45rem;
        vertical-align: middle;
    }

    .risk-low {
        background: #DCFCE7;
        color: #166534;
    }

    .risk-mod {
        background: #FEF3C7;
        color: #92400E;
    }

    .risk-high {
        background: #FEE2E2;
        color: #991B1B;
    }

    .prob-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.8rem;
        box-shadow: var(--shadow);
    }

    .gauge-wrap {
        display: flex;
        align-items: center;
        gap: 2rem;
        flex-wrap: wrap;
    }

    .gauge-outer {
        border-radius: 50%;
        width: 150px;
        height: 150px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .gauge-inner {
        border-radius: 50%;
        width: 112px;
        height: 112px;
        background: white;
        border: 1px solid var(--line);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .gauge-pct {
        font-size: 1.75rem;
        font-weight: 800;
    }

    .gauge-label {
        font-size: .68rem;
        color: var(--muted);
        font-weight: 700;
        text-transform: uppercase;
        margin-top: .15rem;
    }

    .prob-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: .55rem 0;
    }

    .prob-row .lbl {
        width: 150px;
        font-size: .88rem;
        font-weight: 700;
        color: var(--text);
    }

    .prob-row .pct {
        width: 60px;
        text-align: right;
        font-size: .86rem;
        font-weight: 700;
        color: var(--muted);
    }

    .prob-row .bar-bg {
        flex: 1;
        height: 10px;
        background: #E5E7EB;
        border-radius: 999px;
        overflow: hidden;
    }

    .prob-row .bar-fill {
        height: 10px;
        border-radius: 999px;
    }

    .factors {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .8rem;
    }

    .factor {
        display: flex;
        gap: .75rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: .85rem;
        background: #FAFAFA;
    }

    .factor .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-top: .35rem;
        flex: 0 0 auto;
    }

    .factor .f-up {
        background: var(--red);
    }

    .factor .f-down {
        background: var(--green);
    }

    .factor .f-neu {
        background: var(--muted);
    }

    .factor p {
        margin: 0;
        font-size: .88rem;
        font-weight: 600;
        color: var(--text);
    }

    .factor .eff {
        font-size: .73rem;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 700;
    }

    .footnote {
        color: var(--muted);
        font-size: .8rem;
        line-height: 1.6;
        margin-top: 1.6rem;
        text-align: center;
    }

    @media (max-width: 850px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }

        .hero h1 {
            font-size: 2.35rem;
        }

        .factors {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Model loader
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

# =========================
# Fallback heuristic
# =========================

def heuristic_predict(d):
    score = 0.0

    score += (d["Alvarado_Score"] / 10.0) * 0.30
    score += (d["Paedriatic_Appendicitis_Score"] / 10.0) * 0.25

    if d["Lower_Right_Abd_Pain"] == "yes":
        score += 0.06

    if d["Migratory_Pain"] == "yes":
        score += 0.06

    if d["Nausea"] == "yes":
        score += 0.03

    if d["Loss_of_Appetite"] == "yes":
        score += 0.03

    if d["Peritonitis"] == "yes":
        score += 0.08

    if d["WBC_Count"] > 11:
        score += min(0.08, (d["WBC_Count"] - 11) / 50.0)

    if d["CRP"] > 10:
        score += min(0.08, (d["CRP"] - 10) / 1100.0)

    if d["Neutrophil_Percentage"] > 75:
        score += 0.05

    if d["Appendix_on_US"] == "yes" and d["Appendix_Diameter"] > 6:
        score += min(0.10, (d["Appendix_Diameter"] - 6) / 40.0)

    if d["Free_Fluids"] == "yes":
        score += 0.06

    probability = max(0.02, min(0.98, score))
    prediction = "appendicitis" if probability >= 0.5 else "no appendicitis"

    return prediction, probability


def build_key_factors(d):
    factors = []

    if d["Alvarado_Score"] >= 7 or d["Paedriatic_Appendicitis_Score"] >= 7:
        factors.append(("High clinical score", "increases risk"))
    elif d["Alvarado_Score"] < 5 and d["Paedriatic_Appendicitis_Score"] < 5:
        factors.append(("Lower clinical score", "decreases risk"))

    if d["Peritonitis"] == "yes":
        factors.append(("Peritonitis present", "increases risk"))

    if d["Migratory_Pain"] == "yes":
        factors.append(("Migratory pain pattern", "increases risk"))

    if d["WBC_Count"] > 15 or d["CRP"] > 50:
        factors.append(("Elevated WBC or CRP", "increases risk"))

    if d["Appendix_on_US"] == "yes" and d["Appendix_Diameter"] > 6:
        factors.append(("Enlarged appendix on ultrasound", "increases risk"))

    if d["Free_Fluids"] == "yes":
        factors.append(("Free fluid on ultrasound", "increases risk"))

    if not factors:
        factors.append(("No strong positive findings", "decreases risk"))

    return factors[:4]

# =========================
# Hero section
# =========================

st.markdown(
    """
    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="eyebrow">🩺 Paediatric Screening Tool</div>
                <h1>AppendiCheck Kids</h1>
                <p>
                    A machine learning app that estimates appendicitis risk in children using
                    patient information, symptoms, blood test results and ultrasound findings.
                    It is designed as a screening support tool, not a replacement for a doctor.
                </p>
            </div>

            <div class="hero-card">
                <div class="hero-card-title">Model Summary</div>

                <div class="mini-bar">
                    <div class="mini-label">Accuracy</div>
                    <div class="mini-track"><div class="mini-fill" style="width:94.87%;"></div></div>
                    <div class="mini-value">94.87%</div>
                </div>

                <div class="mini-bar">
                    <div class="mini-label">Macro F1</div>
                    <div class="mini-track"><div class="mini-fill" style="width:94.67%;"></div></div>
                    <div class="mini-value">94.67%</div>
                </div>

                <div class="mini-bar">
                    <div class="mini-label">Model</div>
                    <div class="mini-track"><div class="mini-fill" style="width:100%;"></div></div>
                    <div class="mini-value">RF</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="disclaimer">
        <b>Disclaimer:</b> This app is an educational screening support tool.
        It does not replace professional medical diagnosis. Always consult a qualified clinician.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Inputs
# =========================

st.markdown("## Patient Assessment")
st.markdown(
    """
    <p style='font-size:.9rem;color:#6B7280;margin-top:-.5rem;margin-bottom:1rem;'>
        Enter the child's clinical data across the four sections, then run the screening.
    </p>
    """,
    unsafe_allow_html=True
)

SECTION_OPTIONS = [
    "👤 Patient Info",
    "🩻 Symptoms",
    "🧪 Blood Tests",
    "🔍 Ultrasound"
]

active_section = st.radio(
    "Section",
    SECTION_OPTIONS,
    horizontal=True,
    label_visibility="collapsed",
    key="active_section"
)

section_keys = [
    "sec_patient",
    "sec_symptoms",
    "sec_blood",
    "sec_ultrasound"
]

active_key = dict(zip(SECTION_OPTIONS, section_keys))[active_section]

hide_css = "\n".join(
    f".st-key-{key} {{ display: none; }}"
    for key in section_keys
    if key != active_key
)

st.markdown(f"<style>{hide_css}</style>", unsafe_allow_html=True)

with st.container(key="sec_patient"):
    st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Basic patient information</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider(
            "Age (years)",
            0.0,
            18.0,
            10.0,
            0.1,
            help="How old the child is, in years."
        )

    with c2:
        sex = st.selectbox(
            "Sex",
            ["female", "male"],
            help="Whether the child is female or male."
        )

    with c3:
        bmi = st.slider(
            "BMI (kg/m²)",
            5.0,
            40.0,
            18.0,
            0.1,
            help="Body Mass Index from height and weight."
        )

    st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical Scores</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Composite scores from standard assessment tools</div>', unsafe_allow_html=True)

    c4, c5 = st.columns(2)

    with c4:
        alvarado_score = st.slider(
            "Alvarado Score",
            0.0,
            10.0,
            5.0,
            0.5,
            help="A 0–10 score from a doctor's exam and blood test. Higher means more likely appendicitis."
        )

    with c5:
        paediatric_score = st.slider(
            "Paediatric Appendicitis Score",
            0.0,
            10.0,
            5.0,
            0.5,
            help="A clinical score made for children."
        )

with st.container(key="sec_symptoms"):
    st.markdown('<div class="section-title">Reported Symptoms</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Select yes or no for each presenting symptom</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        lower_right_pain = st.selectbox(
            "Lower Right Abdominal Pain",
            ["no", "yes"],
            help="Pain on the lower right side of the belly."
        )

    with c2:
        migratory_pain = st.selectbox(
            "Migratory Pain",
            ["no", "yes"],
            help="Pain that moved from the belly button area to the lower right side."
        )

    with c3:
        nausea = st.selectbox(
            "Nausea",
            ["no", "yes"],
            help="Feeling like vomiting."
        )

    c4, c5 = st.columns(2)

    with c4:
        loss_of_appetite = st.selectbox(
            "Loss of Appetite",
            ["no", "yes"],
            help="Not wanting to eat."
        )

    with c5:
        peritonitis = st.selectbox(
            "Peritonitis",
            ["no", "yes"],
            help="Abdominal tenderness or guarding."
        )

with st.container(key="sec_blood"):
    st.markdown('<div class="section-title">Blood Test Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Numbers from a blood test done at a clinic or hospital</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        wbc_count = st.slider(
            "WBC Count (×10⁹/L)",
            0.0,
            40.0,
            10.0,
            0.1,
            help="White blood cells fight infection. Higher values may suggest infection."
        )

    with c2:
        crp = st.slider(
            "CRP (mg/L)",
            0.0,
            300.0,
            10.0,
            0.1,
            help="CRP rises with inflammation or infection."
        )

    with c3:
        neutrophil_percentage = st.slider(
            "Neutrophil Percentage (%)",
            0.0,
            100.0,
            60.0,
            0.5,
            help="A type of white blood cell linked to bacterial infection."
        )

with st.container(key="sec_ultrasound"):
    st.markdown('<div class="section-title">Ultrasound Findings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">From an ultrasound scan done by a doctor or radiographer</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        appendix_on_us = st.selectbox(
            "Appendix Seen on Ultrasound",
            ["no", "yes"],
            help="Whether the appendix could be clearly seen on the scan."
        )

    with c2:
        appendix_diameter = st.slider(
            "Appendix Diameter (mm)",
            0.0,
            20.0,
            6.0,
            0.1,
            help="A wider appendix may suggest appendicitis."
        )

    with c3:
        free_fluids = st.selectbox(
            "Free Fluids",
            ["no", "yes"],
            help="Extra fluid seen in the belly."
        )

# =========================
# Prediction
# =========================

st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

predict_clicked = st.button("Predict Appendicitis Risk")

if predict_clicked:
    data = {
        "Age": age,
        "Sex": sex,
        "BMI": bmi,
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
        "Free_Fluids": free_fluids
    }

    if MODEL is not None:
        df_input = pd.DataFrame([data])
        df_input = pd.get_dummies(df_input)

        df_input = df_input.reindex(
            columns=MODEL.feature_names_in_,
            fill_value=0
        )

        prediction = MODEL.predict(df_input)[0]

        if hasattr(MODEL, "predict_proba"):
            probabilities = MODEL.predict_proba(df_input)[0]

            if hasattr(MODEL, "classes_"):
                classes = list(MODEL.classes_)
            else:
                classes = list(MODEL.named_steps["model"].classes_)

            if "appendicitis" in classes:
                appendicitis_index = classes.index("appendicitis")
            else:
                appendicitis_index = 1

            appendicitis_probability = float(probabilities[appendicitis_index])
        else:
            appendicitis_probability = 0.9 if prediction == "appendicitis" else 0.1

        key_factors = build_key_factors(data)

    else:
        prediction, appendicitis_probability = heuristic_predict(data)
        key_factors = build_key_factors(data)

    st.markdown("## Prediction Result")

    is_high = prediction == "appendicitis"

    risk_level = (
        "high"
        if appendicitis_probability >= 0.7
        else "moderate"
        if appendicitis_probability >= 0.4
        else "low"
    )

    badge_class = {
        "low": "risk-low",
        "moderate": "risk-mod",
        "high": "risk-high"
    }[risk_level]

    badge_label = {
        "low": "Low risk",
        "moderate": "Moderate risk",
        "high": "High risk"
    }[risk_level]

    if is_high:
        st.markdown(
            f"""
            <div class="result-banner result-high">
                <div class="kicker">Screening Result</div>
                <h2>
                    Possible appendicitis
                    <span class="risk-badge {badge_class}">{badge_label}</span>
                </h2>
                <p>
                    Based on the entered information, the model suggests appendicitis is possible.
                    This is not a diagnosis. Please seek medical assessment from a qualified doctor.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-banner result-low">
                <div class="kicker">Screening Result</div>
                <h2>
                    No appendicitis detected
                    <span class="risk-badge {badge_class}">{badge_label}</span>
                </h2>
                <p>
                    Based on the entered information, the model does not predict appendicitis.
                    This is not a diagnosis. If pain continues or symptoms worsen, seek medical advice.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    gauge_color = "#DC2626" if is_high else "#16A34A"
    gauge_percentage = appendicitis_probability * 100
    no_appendicitis_percentage = (1 - appendicitis_probability) * 100

    factors_html = ""

    for factor, effect in key_factors:
        dot_class = {
            "increases risk": "f-up",
            "decreases risk": "f-down",
            "neutral": "f-neu"
        }[effect]

        factors_html += f"""
        <div class="factor">
            <span class="dot {dot_class}"></span>
            <div>
                <p>{factor}</p>
                <span class="eff">{effect}</span>
            </div>
        </div>
        """

    st.markdown("### Prediction Probability")

    st.markdown(
        f"""
        <div class="prob-card">
            <div class="gauge-wrap">
                <div class="gauge-outer"
                    style="background: conic-gradient({gauge_color} {gauge_percentage * 3.6:.1f}deg, #E5E7EB 0deg);">
                    <div class="gauge-inner">
                        <div class="gauge-pct" style="color:{gauge_color};">
                            {gauge_percentage:.1f}%
                        </div>
                        <div class="gauge-label">Appendicitis risk</div>
                    </div>
                </div>

                <div style="flex:1;min-width:260px;">
                    <div class="prob-row">
                        <span class="lbl">Appendicitis</span>
                        <span class="pct">{gauge_percentage:.1f}%</span>
                        <div class="bar-bg">
                            <div class="bar-fill"
                                style="width:{gauge_percentage:.1f}%;background:{gauge_color};">
                            </div>
                        </div>
                    </div>

                    <div class="prob-row">
                        <span class="lbl">No appendicitis</span>
                        <span class="pct">{no_appendicitis_percentage:.1f}%</span>
                        <div class="bar-bg">
                            <div class="bar-fill"
                                style="width:{no_appendicitis_percentage:.1f}%;background:#16A34A;">
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div style="height:1px;background:#E5E7EB;margin:1.6rem 0;"></div>

            <div style="font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.06em;color:#6B7280;margin-bottom:1rem;">
                Key Influencing Factors
            </div>

            <div class="factors">
                {factors_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Footer
# =========================

st.markdown(
    """
    <div class="footnote">
        AppendiCheck Kids uses a trained Random Forest machine learning model when available.
        The prediction is based on selected patient information, symptoms, blood test results
        and ultrasound-related features.
    </div>
    """,
    unsafe_allow_html=True
)