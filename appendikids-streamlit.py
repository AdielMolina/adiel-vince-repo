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
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --bg-deep: #0B1220;
        --bg-panel: #131C2E;
        --bg-panel-alt: #182238;
        --border: #24304A;
        --text-primary: #E7ECF5;
        --text-muted: #8E9BB3;
        --accent: #3ED6C4;
        --accent-dim: #23443F;
        --amber: #F2B84B;
        --danger: #FB7185;
        --danger-dim: #3A2230;
        --success: #34D399;
        --success-dim: #16332B;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #0E1830 0%, var(--bg-deep) 45%) fixed;
        color: var(--text-primary);
    }

    #MainMenu, footer, header { visibility: hidden; }

    h1, h2, h3, .hero h1 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }

    /* ---------- Hero ---------- */
    .hero {
        padding: 2.2rem 2.4rem 1.6rem 2.4rem;
        border-radius: 20px;
        background: linear-gradient(155deg, #101B33 0%, #0D1526 100%);
        border: 1px solid var(--border);
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
        margin-bottom: 0.4rem;
        position: relative;
        overflow: hidden;
    }

    .hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }

    .hero h1 {
        font-size: 2.3rem;
        margin: 0 0 0.35rem 0;
        color: var(--text-primary);
    }

    .hero p {
        font-size: 1rem;
        color: var(--text-muted);
        max-width: 640px;
        margin: 0;
    }

    /* Heartbeat divider — signature element */
    .pulse-line { width: 100%; height: 34px; margin-top: 1.2rem; opacity: 0.85; }
    .pulse-line path {
        fill: none;
        stroke: var(--accent);
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
        filter: drop-shadow(0 0 4px rgba(62, 214, 196, 0.55));
    }

    /* ---------- Metric strip ---------- */
    .metric-card {
        padding: 1.1rem 1rem;
        border-radius: 14px;
        background: var(--bg-panel);
        border: 1px solid var(--border);
        text-align: left;
    }
    .metric-card .label {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .metric-card .value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--accent);
        margin: 0;
    }

    /* ---------- Section cards ---------- */
    .section-card {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        margin-bottom: 1.1rem;
    }
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-sub {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }

    /* ---------- Streamlit widget overrides ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-muted);
        font-weight: 500;
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        background: var(--bg-panel-alt) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    label, .stSlider label, .stSelectbox label { color: var(--text-muted) !important; font-size: 0.85rem !important; }

    .stSlider [data-baseweb="slider"] > div > div { background: var(--accent) !important; }
    .stSlider [role="slider"] { background-color: var(--accent) !important; border: 3px solid #0B1220 !important; }

    div[data-baseweb="select"] > div {
        background-color: var(--bg-panel-alt) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
    }

    div.stButton > button {
        width: 100%;
        height: 3.1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--accent) 0%, #21A896 100%);
        color: #06231F;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        border: none;
        font-size: 1.02rem;
        box-shadow: 0 8px 24px rgba(62, 214, 196, 0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 28px rgba(62, 214, 196, 0.4);
        color: #06231F;
    }

    /* ---------- Result panels ---------- */
    .result-banner {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        display: flex;
        align-items: center;
        gap: 1rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .result-high { background: var(--danger-dim); border: 1px solid var(--danger); color: #FFD9DF; }
    .result-low { background: var(--success-dim); border: 1px solid var(--success); color: #D3FBEA; }

    .result-note { color: var(--text-muted); font-size: 0.92rem; margin-top: -0.4rem; margin-bottom: 1.2rem; }

    /* Gauge */
    .gauge-wrap { display: flex; align-items: center; gap: 2rem; flex-wrap: wrap; }
    .gauge-outer {
        border-radius: 50%;
        width: 168px; height: 168px;
        display: flex; align-items: center; justify-content: center;
    }
    .gauge-inner {
        border-radius: 50%;
        width: 128px; height: 128px;
        background: var(--bg-panel);
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        border: 1px solid var(--border);
    }
    .gauge-pct { font-family: 'IBM Plex Mono', monospace; font-size: 1.9rem; font-weight: 600; }
    .gauge-label { font-size: 0.68rem; color: var(--text-muted); letter-spacing: 0.06em; text-transform: uppercase; margin-top: 0.15rem; }

    .prob-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; }
    .prob-table td { padding: 0.5rem 0.2rem; border-bottom: 1px solid var(--border); }
    .prob-table .bar-bg { background: var(--bg-panel-alt); border-radius: 6px; height: 8px; width: 100%; overflow: hidden; }
    .prob-table .bar-fill { height: 8px; border-radius: 6px; }

    .footnote { color: var(--text-muted); font-size: 0.82rem; }
    hr { border-color: var(--border) !important; }
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
        <div class="hero-eyebrow">Paediatric Screening Support · v2</div>
        <h1>🩺 AppendiCheck Kids</h1>
        <p>A machine learning screening support tool that estimates appendicitis risk in
        children from clinical scores, symptoms, blood work and ultrasound findings.</p>
        <svg class="pulse-line" viewBox="0 0 600 34" preserveAspectRatio="none">
            <path d="M0 17 H220 L235 3 L250 31 L265 10 L280 24 L295 17 H600" />
        </svg>
    </div>
    """,
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)
metrics = [
    (m1, "Final Model", "Random Forest"),
    (m2, "Accuracy", "94.87%"),
    (m3, "Macro F1", "94.67%"),
    (m4, "Purpose", "Screening only"),
]
for col, label, value in metrics:
    with col:
        st.markdown(
            f"""<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>""",
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
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age (years)", 0.0, 18.0, 10.0, 0.1)
    with col2:
        sex = st.selectbox("Sex", ["female", "male"])
    with col3:
        bmi = st.slider("BMI (kg/m²)", 5.0, 40.0, 18.0, 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical Scores</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Composite scores from standard assessment tools</div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        alvarado_score = st.slider("Alvarado Score", 0.0, 10.0, 5.0, 0.5)
    with col5:
        paediatric_score = st.slider("Paediatric Appendicitis Score", 0.0, 10.0, 5.0, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
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
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
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
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
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
            """<div class="result-banner result-high">⚠️ Predicted Result: Possible Appendicitis</div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="result-note">The model predicts that this case may show signs of appendicitis. '
            'Further medical assessment is recommended.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div class="result-banner result-low">✅ Predicted Result: No Appendicitis</div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="result-note">The model predicts that this case may not show signs of appendicitis.</div>',
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

        gauge_color = "#FB7185" if is_high_risk else "#34D399"
        gauge_pct = 0.0
        if "appendicitis" in classes:
            gauge_pct = float(probabilities[classes.index("appendicitis")]) * 100

        rows_html = ""
        for cls, prob in zip(classes, probabilities):
            pct = float(prob) * 100
            bar_color = "#FB7185" if str(cls).lower() == "appendicitis" else "#34D399"
            rows_html += f"""
            <tr>
                <td style="width:38%; color:#E7ECF5;">{cls}</td>
                <td style="width:12%; text-align:right; color:#8E9BB3;">{pct:.2f}%</td>
                <td style="width:50%;">
                    <div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%; background:{bar_color};"></div></div>
                </td>
            </tr>
            """

        st.markdown("### Prediction Probability")
        st.markdown(
            f"""
            <div class="section-card">
                <div class="gauge-wrap">
                    <div class="gauge-outer" style="background: conic-gradient({gauge_color} {gauge_pct * 3.6:.1f}deg, #182238 0deg);">
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