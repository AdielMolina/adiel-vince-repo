import joblib
import streamlit as st
import pandas as pd
import numpy as np

# =========================
# Load trained model
# =========================

model = joblib.load("appendicitis_model.pkl")

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
# Design tokens + global style
# =========================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,600;0,700;1,500;1,600&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --cream: #FBF1E4;
        --cream-deep: #F3E6D3;
        --ink: #14213D;
        --ink-soft: #4A5674;
        --coral: #FF5A44;
        --coral-tint: #FFE4DE;
        --sky: #3E8FD1;
        --sky-tint: #DCEBFA;
        --amber: #FFD23F;
        --amber-tint: #FFF3D2;
        --mint: #34C77B;
        --mint-tint: #DBF5E7;
        --red: #E23B3B;
        --red-tint: #FBDEDE;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: var(--cream);
        color: var(--ink);
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* ---------- Hero ---------- */
    .hero {
        padding: 2.6rem 2.6rem 2.2rem 2.6rem;
        border-radius: 26px;
        background: var(--coral);
        color: white;
        box-shadow: 0 16px 40px rgba(255, 90, 68, 0.25);
        margin-bottom: 1.2rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-style: italic;
        font-size: 3rem;
        line-height: 1.05;
        margin: 0 0 0.6rem 0;
        color: white;
    }
    .hero p {
        font-size: 1.05rem;
        max-width: 600px;
        opacity: 0.95;
        margin: 0 0 1.2rem 0;
    }
    .pill-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
    .pill {
        background: white;
        color: var(--ink);
        border-radius: 999px;
        padding: 0.55rem 1.1rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* ---------- Section blocks ---------- */
    .section-block {
        border-radius: 22px;
        padding: 1.5rem 1.7rem 0.7rem 1.7rem;
        margin-bottom: 1.1rem;
        border: 1px solid rgba(20,33,61,0.08);
    }
    .block-cream { background: var(--cream-deep); }
    .block-sky { background: var(--sky-tint); }
    .block-amber { background: var(--amber-tint); }
    .block-mint { background: var(--mint-tint); }

    .section-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.35rem;
        color: var(--ink);
        margin-bottom: 0.15rem;
    }
    .section-sub {
        font-size: 0.88rem;
        color: var(--ink-soft);
        margin-bottom: 1.1rem;
    }

    h2 { font-family: 'Fraunces', serif; font-style: italic; color: var(--ink); }

    /* ---------- Streamlit widget overrides ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: none; }
    .stTabs [data-baseweb="tab"] {
        background: white;
        color: var(--ink-soft);
        font-weight: 600;
        border-radius: 999px;
        padding: 0.55rem 1.2rem;
        border: 1px solid rgba(20,33,61,0.1);
    }
    .stTabs [aria-selected="true"] {
        color: white !important;
        background: var(--ink) !important;
        border: none !important;
    }

    label, .stSlider label, .stSelectbox label { color: var(--ink-soft) !important; font-size: 0.85rem !important; font-weight: 500 !important; }

    .stSlider [data-baseweb="slider"] > div > div { background: var(--coral) !important; }
    .stSlider [role="slider"] { background-color: var(--coral) !important; border: 3px solid var(--cream) !important; }

    div[data-baseweb="select"] > div {
        background-color: white !important;
        border-color: rgba(20,33,61,0.15) !important;
        color: var(--ink) !important;
        border-radius: 12px !important;
    }

    div.stButton > button {
        width: 100%;
        height: 3.3rem;
        border-radius: 999px;
        background: var(--ink);
        color: var(--cream);
        font-weight: 700;
        font-family: 'Fraunces', serif;
        font-style: italic;
        border: none;
        font-size: 1.15rem;
        box-shadow: 0 10px 26px rgba(20, 33, 61, 0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 30px rgba(20, 33, 61, 0.35);
        color: var(--amber);
    }

    /* ---------- Result banner ---------- */
    .result-banner {
        padding: 2rem 2.2rem;
        border-radius: 26px;
        margin-bottom: 1rem;
    }
    .result-high { background: var(--coral); color: white; }
    .result-low { background: var(--mint); color: white; }
    .result-banner .kicker {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.85;
        margin-bottom: 0.3rem;
    }
    .result-banner h2 {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-style: normal;
        font-size: 2.4rem;
        color: white;
        margin: 0 0 0.5rem 0;
        line-height: 1.05;
    }
    .result-banner p { margin: 0; opacity: 0.95; max-width: 640px; }

    /* Gauge */
    .gauge-wrap { display: flex; align-items: center; gap: 2rem; flex-wrap: wrap; }
    .gauge-outer { border-radius: 50%; width: 168px; height: 168px; display: flex; align-items: center; justify-content: center; }
    .gauge-inner {
        border-radius: 50%; width: 128px; height: 128px; background: var(--cream);
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        border: 1px solid rgba(20,33,61,0.1);
    }
    .gauge-pct { font-family: 'IBM Plex Mono', monospace; font-size: 1.85rem; font-weight: 600; color: var(--ink); }
    .gauge-label { font-size: 0.68rem; color: var(--ink-soft); letter-spacing: 0.05em; text-transform: uppercase; margin-top: 0.15rem; }

    .prob-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; }
    .prob-table td { padding: 0.55rem 0.2rem; }
    .prob-table .bar-bg { background: rgba(20,33,61,0.08); border-radius: 6px; height: 9px; width: 100%; overflow: hidden; }
    .prob-table .bar-fill { height: 9px; border-radius: 6px; }

    .footnote { color: var(--ink-soft); font-size: 0.82rem; font-family: 'IBM Plex Mono', monospace; }
    hr { border-color: rgba(20,33,61,0.12) !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Header
# =========================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">🩺 Paediatric Screening Tool</div>
        <h1>Check the signs,<br>skip the guesswork.</h1>
        <p>AppendiCheck Kids estimates appendicitis risk in children from clinical scores,
        symptoms, blood work and ultrasound findings — built to support a clinical decision,
        never replace one.</p>
        <div class="pill-row">
            <div class="pill">Random Forest</div>
            <div class="pill">94.87% Accuracy</div>
            <div class="pill">94.67% Macro F1</div>
            <div class="pill">Screening only</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "Disclaimer: This app is an educational screening support tool. "
    "It does not replace professional medical diagnosis."
)

# =========================
# Inputs
# =========================

st.markdown("## Patient Assessment")

tab1, tab2, tab3, tab4 = st.tabs([
    "👤  Patient Info",
    "🩻  Symptoms",
    "🧪  Blood Tests",
    "🔍  Ultrasound"
])

with tab1:
    st.markdown('<div class="section-block block-cream">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age (years)", 0.0, 18.0, 10.0, 0.1)
    with col2:
        sex = st.selectbox("Sex", ["female", "male"])
    with col3:
        bmi = st.slider("BMI (kg/m²)", 5.0, 40.0, 18.0, 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-block block-cream">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical Scores</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Composite scores from standard assessment tools</div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        alvarado_score = st.slider("Alvarado Score", 0.0, 10.0, 5.0, 0.5)
    with col5:
        paediatric_score = st.slider("Paediatric Appendicitis Score", 0.0, 10.0, 5.0, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-block block-sky">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Reported Symptoms</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Select yes/no for each presenting symptom</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        lower_right_pain = st.selectbox("Lower Right Abdominal Pain", ["no", "yes"])
    with col2:
        migratory_pain = st.selectbox("Migratory Pain", ["no", "yes"])
    with col3:
        nausea = st.selectbox("Nausea", ["no", "yes"])
    col4, col5 = st.columns(2)
    with col4:
        loss_of_appetite = st.selectbox("Loss of Appetite", ["no", "yes"])
    with col5:
        peritonitis = st.selectbox("Peritonitis", ["no", "yes"])
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-block block-amber">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Blood Test Results</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        wbc_count = st.slider("WBC Count (×10⁹/L)", 0.0, 40.0, 10.0, 0.1)
    with col2:
        crp = st.slider("CRP (mg/L)", 0.0, 300.0, 10.0, 0.1)
    with col3:
        neutrophil_percentage = st.slider("Neutrophil Percentage (%)", 0.0, 100.0, 60.0, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-block block-mint">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Ultrasound Findings</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        appendix_on_us = st.selectbox("Appendix Seen on Ultrasound", ["no", "yes"])
    with col2:
        appendix_diameter = st.slider("Appendix Diameter (mm)", 0.0, 20.0, 6.0, 0.1)
    with col3:
        free_fluids = st.selectbox("Free Fluids", ["no", "yes"])
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Prediction
# =========================

st.markdown("---")

predict_clicked = st.button("Predict Appendicitis Risk")

if predict_clicked:

    df_input = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "BMI": [bmi],
        "Alvarado_Score": [alvarado_score],
        "Paedriatic_Appendicitis_Score": [paediatric_score],
        "Lower_Right_Abd_Pain": [lower_right_pain],
        "Migratory_Pain": [migratory_pain],
        "Nausea": [nausea],
        "Loss_of_Appetite": [loss_of_appetite],
        "Peritonitis": [peritonitis],
        "WBC_Count": [wbc_count],
        "CRP": [crp],
        "Neutrophil_Percentage": [neutrophil_percentage],
        "Appendix_on_US": [appendix_on_us],
        "Appendix_Diameter": [appendix_diameter],
        "Free_Fluids": [free_fluids]
    })

    # One-hot encode user input
    df_input = pd.get_dummies(df_input)

    # Match the training columns stored in the saved pipeline
    df_input = df_input.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Predict
    prediction = model.predict(df_input)[0]

    st.markdown("## Prediction Result")

    is_high_risk = (prediction == "appendicitis")

    if is_high_risk:
        st.markdown(
            """
            <div class="result-banner result-high">
                <div class="kicker">Screening Result</div>
                <h2>Possible appendicitis</h2>
                <p>The model predicts that this case may show signs of appendicitis. Further medical
                assessment is recommended.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="result-banner result-low">
                <div class="kicker">Screening Result</div>
                <h2>No appendicitis detected</h2>
                <p>The model predicts that this case may not show signs of appendicitis.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Prediction probability
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(df_input)[0]

        try:
            classes = model.classes_
        except Exception:
            classes = model.named_steps["model"].classes_

        classes = list(classes)

        gauge_color = "#E23B3B" if is_high_risk else "#34C77B"
        gauge_pct = 0.0
        if "appendicitis" in classes:
            gauge_pct = float(probabilities[classes.index("appendicitis")]) * 100

        rows_html = ""
        for cls, prob in zip(classes, probabilities):
            pct = float(prob) * 100
            bar_color = "#E23B3B" if str(cls).lower() == "appendicitis" else "#34C77B"
            rows_html += f"""
            <tr>
                <td style="width:38%; color:#14213D; font-weight:600;">{cls}</td>
                <td style="width:12%; text-align:right; color:#4A5674;">{pct:.2f}%</td>
                <td style="width:50%;">
                    <div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%; background:{bar_color};"></div></div>
                </td>
            </tr>
            """

        st.markdown("### Prediction Probability")
        st.markdown(
            f"""
            <div class="section-block block-cream">
                <div class="gauge-wrap">
                    <div class="gauge-outer" style="background: conic-gradient({gauge_color} {gauge_pct * 3.6:.1f}deg, rgba(20,33,61,0.08) 0deg);">
                        <div class="gauge-inner">
                            <div class="gauge-pct" style="color:{gauge_color};">{gauge_pct:.1f}%</div>
                            <div class="gauge-label">Appendicitis risk</div>
                        </div>
                    </div>
                    <table class="prob-table" style="flex:1; min-width:260px;">
                        {rows_html}
                    </table>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

st.markdown(
    '<div class="footnote">AppendiCheck Kids uses a trained Random Forest machine learning model. '
    'The prediction is based on selected patient information, symptoms, blood test results and '
    'ultrasound-related features.</div>',
    unsafe_allow_html=True
)