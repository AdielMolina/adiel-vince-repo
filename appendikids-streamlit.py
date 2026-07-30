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
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --bg-deep: #0B0B0B;
        --bg-panel: #161616;
        --bg-panel-alt: #1F1F1F;
        --border: #3A3A38;
        --border-soft: #2A2A28;
        --ink: #F2F0EC;
        --ink-muted: #9A9A96;
        --brick: #C25C4E;
        --brick-tint: #241A18;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: var(--bg-deep);
        color: var(--ink);
    }

    #MainMenu, footer, header { visibility: hidden; }
    h1, h2, h3 { font-family: 'Playfair Display', serif; color: var(--ink); }

    /* ---------- Hero (split layout, like a portfolio cover) ---------- */
    .hero {
        border: 1px solid var(--border);
        margin-bottom: 1.2rem;
        overflow: hidden;
    }
    .hero-grid { display: flex; min-height: 300px; }
    .hero-visual {
        flex: 0 0 34%;
        background: #0E0E0E;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        border-right: 1px solid var(--border);
    }
    .hero-visual svg { width: 92%; height: 82%; }
    .hero-content {
        flex: 1;
        padding: 2.2rem 2.4rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--ink-muted);
        margin-bottom: 0.9rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        text-transform: uppercase;
        line-height: 1.08;
        margin: 0 0 0.9rem 0;
    }
    .hero p {
        font-size: 0.98rem;
        max-width: 520px;
        color: var(--ink-muted);
        margin: 0 0 1.3rem 0;
        font-family: 'Inter', sans-serif;
    }
    .hero-credit {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-top: 1px solid var(--border-soft);
        padding-top: 0.9rem;
        margin-top: 0.3rem;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    .hero-credit .who { font-size: 0.92rem; color: var(--ink); }
    .hero-credit .who span { display: block; color: var(--ink-muted); font-size: 0.8rem; }
    .hero-credit .stats {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        color: var(--ink-muted);
        text-align: right;
    }
    .hero-credit .stats b { color: var(--ink); }

    /* ---------- Disclaimer ---------- */
    .disclaimer-block {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-left: 2px solid var(--ink-muted);
        padding: 0.85rem 1.2rem;
        color: var(--ink-muted);
        font-size: 0.88rem;
        margin: 1.1rem 0 1.5rem 0;
    }
    .disclaimer-block b { color: var(--ink); }

    /* ---------- Section titles ---------- */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 1.25rem;
        color: var(--ink);
        margin-bottom: 0.15rem;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .section-sub { font-size: 0.85rem; color: var(--ink-muted); margin-bottom: 1rem; }
    .subsection-divider { height: 1px; background: var(--border-soft); margin: 1.3rem 0 1.1rem 0; }

    /* Every section container gets the SAME consistent panel treatment. */
    .st-key-sec_patient, .st-key-sec_symptoms, .st-key-sec_blood, .st-key-sec_ultrasound {
        background: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
        padding: 1.6rem 1.8rem 1.1rem 1.8rem !important;
        margin-bottom: 1.1rem !important;
    }
    .section-panel {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 1.6rem 1.8rem 0.7rem 1.8rem;
    }

    label, .stSlider label, .stSelectbox label {
        color: var(--ink-muted) !important; font-size: 0.83rem !important; font-weight: 500 !important;
    }

    .stSlider [data-baseweb="slider"] > div > div { background: var(--ink) !important; }
    .stSlider [role="slider"] { background-color: var(--ink) !important; border: 3px solid var(--bg-deep) !important; }

    [data-testid="stSelectbox"] > div > div {
        background-color: var(--bg-panel-alt) !important;
        border-color: var(--border) !important;
        border-radius: 2px !important;
    }
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSelectbox"] > div > div * { color: var(--ink) !important; }
    /* Dropdown chevron icon: force a visible glyph instead of a blank swatch */
    [data-testid="stSelectbox"] svg {
        fill: var(--ink) !important;
        background: transparent !important;
        opacity: 1 !important;
    }
    [data-testid="stSelectbox"] div:has(> svg) { background: transparent !important; }
    [data-testid="stSelectbox"] [data-baseweb="icon"] { background: transparent !important; }
    div[data-baseweb="popover"] div[data-baseweb="menu"] { background-color: var(--bg-panel-alt) !important; border: 1px solid var(--border) !important; }
    div[data-baseweb="popover"] li { background-color: var(--bg-panel-alt) !important; color: var(--ink) !important; }
    div[data-baseweb="popover"] li:hover { background-color: var(--bg-panel) !important; }

    /* ---------- Expander (used for plain-language glossary) ---------- */
    [data-testid="stExpander"] {
        background: var(--bg-panel-alt);
        border: 1px solid var(--border);
        border-radius: 2px;
        margin-bottom: 1rem;
    }
    [data-testid="stExpander"] summary {
        color: var(--ink) !important;
        font-weight: 600;
        font-size: 0.92rem;
        display: flex !important;
        align-items: center;
        min-height: 2.4rem;
    }
    [data-testid="stExpander"] summary svg { fill: var(--ink-muted) !important; }
    [data-testid="stExpander"] summary:hover { color: var(--brick) !important; }
    [data-testid="stExpander"] p { color: var(--ink); font-size: 0.92rem; margin-bottom: 0.5rem; }
    [data-testid="stExpander"] b { color: var(--ink); }
    .glossary-term { margin-bottom: 0.7rem; }
    .glossary-term b {
        display: block; color: var(--ink); font-family: 'Playfair Display', serif; font-weight: 700;
    }

    /* ---------- Section nav — thumbnail cards, like the portfolio strip ---------- */
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 10px; flex-wrap: wrap; margin-bottom: 0.2rem;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 0.7rem 1.3rem;
        display: flex; align-items: center; gap: 0.5rem;
        cursor: pointer;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display: none; }
    div[data-testid="stRadio"] label[data-baseweb="radio"] p {
        margin: 0; font-weight: 600; font-size: 0.88rem; color: var(--ink-muted); white-space: nowrap;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: var(--bg-panel-alt);
        border: 1px solid var(--ink);
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p { color: var(--ink); }

    /* ---------- Button — ghost outline, inverts on hover ---------- */
    div.stButton > button {
        width: 100%;
        height: 3.2rem;
        border-radius: 2px;
        background: transparent;
        color: var(--ink);
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.9rem;
        border: 1px solid var(--ink);
        transition: background 0.15s ease, color 0.15s ease;
    }
    div.stButton > button:hover {
        background: var(--ink);
        color: var(--bg-deep);
        border-color: var(--ink);
    }

    /* ---------- Result banner ---------- */
    .result-banner {
        padding: 1.7rem 2rem;
        border-radius: 2px;
        margin-bottom: 1rem;
        background: var(--bg-panel);
        border: 1px solid var(--border);
    }
    .result-high { border-left: 3px solid var(--brick); }
    .result-low { border-left: 3px solid var(--ink); }
    .result-banner .kicker {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em;
        text-transform: uppercase; color: var(--ink-muted); margin-bottom: 0.4rem;
    }
    .result-banner h2 {
        font-size: 1.8rem; font-weight: 700; margin: 0 0 0.5rem 0; text-transform: uppercase;
        letter-spacing: 0.01em;
    }
    .result-high h2 { color: var(--brick); }
    .result-low h2 { color: var(--ink); }
    .result-banner p { margin: 0; color: var(--ink-muted); max-width: 640px; font-family: 'Inter', sans-serif; }

    /* Gauge */
    .gauge-wrap { display: flex; align-items: center; gap: 2rem; flex-wrap: wrap; }
    .gauge-outer { border-radius: 50%; width: 164px; height: 164px; display: flex; align-items: center; justify-content: center; }
    .gauge-inner {
        border-radius: 50%; width: 126px; height: 126px; background: var(--bg-deep);
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        border: 1px solid var(--border);
    }
    .gauge-pct { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; }
    .gauge-label { font-size: 0.65rem; color: var(--ink-muted); letter-spacing: 0.05em; text-transform: uppercase; margin-top: 0.15rem; }

    .prob-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem; }
    .prob-table td { padding: 0.55rem 0.2rem; }
    .prob-table .bar-bg { background: var(--bg-panel-alt); border-radius: 2px; height: 8px; width: 100%; overflow: hidden; }
    .prob-table .bar-fill { height: 8px; border-radius: 2px; }

    .footnote { color: var(--ink-muted); font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; }
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
        <div class="hero-grid">
            <div class="hero-visual">
                <svg viewBox="0 0 220 200">
                    <!-- Bars: Baseline / Logistic Regression / Random Forest (tuned) macro F1 -->
                    <text x="14" y="14" font-family="IBM Plex Mono" font-size="8" fill="#6A6A66" letter-spacing="1">MACRO F1-SCORE</text>

                    <rect x="14" y="150" width="34" height="18" fill="#2A2A28" />
                    <text x="14" y="145" font-family="IBM Plex Mono" font-size="9" fill="#6A6A66">0.37</text>
                    <text x="14" y="182" font-family="IBM Plex Mono" font-size="7.5" fill="#6A6A66">BASELINE</text>

                    <rect x="80" y="103" width="34" height="65" fill="#4A4A46" />
                    <text x="80" y="98" font-family="IBM Plex Mono" font-size="9" fill="#9A9A96">0.91</text>
                    <text x="72" y="182" font-family="IBM Plex Mono" font-size="7.5" fill="#9A9A96">LOG. REG</text>

                    <rect x="146" y="92" width="34" height="76" fill="#F2F0EC" />
                    <text x="146" y="87" font-family="IBM Plex Mono" font-size="9" fill="#F2F0EC">0.95</text>
                    <text x="140" y="182" font-family="IBM Plex Mono" font-size="7.5" fill="#F2F0EC">RANDOM FOREST</text>

                    <line x1="10" y1="168" x2="210" y2="168" stroke="#3A3A38" stroke-width="1" />
                </svg>
            </div>
            <div class="hero-content">
                <div class="hero-eyebrow">Paediatric Screening Tool</div>
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
    unsafe_allow_html=True

)

st.markdown(
    '<div class="disclaimer-block"><b>Disclaimer —</b> This app is an educational screening '
    'support tool. It does not replace professional medical diagnosis.</div>',
    unsafe_allow_html=True
)


def glossary(title, terms):
    """Render a plain-language glossary as a collapsible dropdown.
    terms: list of (word, simple_explanation) tuples."""
    with st.expander(f"❓ {title}"):
        for word, explanation in terms:
            st.markdown(
                f'<div class="glossary-term"><b>{word}</b>{explanation}</div>',
                unsafe_allow_html=True
            )

# =========================
# Inputs
# =========================

st.markdown("## Patient Assessment")

SECTION_OPTIONS = ["👤 Patient Info", "🩻 Symptoms", "🧪 Blood Tests", "🔍 Ultrasound"]
active_section = st.radio(
    "Section", SECTION_OPTIONS, horizontal=True, label_visibility="collapsed", key="active_section"
)

# Dynamically hide non-active section panels, keeping all widgets mounted so
# values persist across selections (mirrors how st.tabs keeps every tab alive).
section_keys = ["sec_patient", "sec_symptoms", "sec_blood", "sec_ultrasound"]
active_key = dict(zip(SECTION_OPTIONS, section_keys))[active_section]
hide_css = "\n".join(f".st-key-{k} {{ display: none; }}" for k in section_keys if k != active_key)
st.markdown(f"<style>{hide_css}</style>", unsafe_allow_html=True)

with st.container(key="sec_patient"):
    st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age (years)", 0.0, 18.0, 10.0, 0.1,
                         help="How old the child is, in years.")
    with col2:
        sex = st.selectbox("Sex", ["female", "male"],
                            help="Whether the child is female or male.")
    with col3:
        bmi = st.slider("BMI (kg/m²)", 5.0, 40.0, 18.0, 0.1,
                         help="Body Mass Index — a number worked out from height and weight "
                              "that shows if a child's weight is in a healthy range for their height. "
                              "A doctor, nurse, or clinic scale can measure this.")

    st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical Scores</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Composite scores from standard assessment tools</div>', unsafe_allow_html=True)

    glossary("What do these scores mean?", [
        ("Alvarado Score",
         " — A number from 0 to 10 that doctors work out by checking a child's symptoms, "
         "examining their belly, and looking at a blood test. It adds points for things like pain, "
         "fever, and high white blood cell count. The higher the number, the more likely it is "
         "appendicitis. This score is usually given to you by a doctor or nurse after they examine "
         "the child — you don't need to calculate it yourself."),
        ("Paediatric Appendicitis Score",
         " — Very similar to the Alvarado Score above, but designed specifically for children. "
         "It's also a number from 0 to 10 given by a doctor after an exam, where higher means "
         "more likely to be appendicitis."),
    ])

    col4, col5 = st.columns(2)
    with col4:
        alvarado_score = st.slider("Alvarado Score", 0.0, 10.0, 5.0, 0.5,
                                    help="A 0–10 score from a doctor's exam and blood test. "
                                         "Higher = more likely to be appendicitis.")
    with col5:
        paediatric_score = st.slider("Paediatric Appendicitis Score", 0.0, 10.0, 5.0, 0.5,
                                      help="Like the Alvarado Score, but made for children. "
                                           "Higher = more likely to be appendicitis.")

with st.container(key="sec_symptoms"):
    st.markdown('<div class="section-title">Reported Symptoms</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Select yes/no for each presenting symptom</div>', unsafe_allow_html=True)

    glossary("What do these words mean?", [
        ("Lower Right Abdominal Pain",
         " — Pain on the lower right side of the belly, roughly between the hip bone and the "
         "belly button. This is where the appendix usually sits."),
        ("Migratory Pain",
         " — Pain that started near the belly button and then moved to the lower right side over "
         "several hours. This moving pattern is a classic early sign of appendicitis."),
        ("Nausea",
         " — Feeling sick to the stomach, like you might vomit, even if you haven't actually "
         "thrown up."),
        ("Loss of Appetite",
         " — Not wanting to eat, even foods the child usually likes."),
        ("Peritonitis",
         " — A sign that the lining inside the belly is irritated or infected. Doctors often check "
         "for this by gently pressing on the belly and then releasing quickly — if it hurts more "
         "when released than when pressed, that can be a sign of peritonitis."),
    ])

    col1, col2, col3 = st.columns(3)
    with col1:
        lower_right_pain = st.selectbox("Lower Right Abdominal Pain", ["no", "yes"],
                                         help="Pain on the lower right side of the belly.")
    with col2:
        migratory_pain = st.selectbox("Migratory Pain", ["no", "yes"],
                                       help="Pain that started near the belly button and moved "
                                            "to the lower right side.")
    with col3:
        nausea = st.selectbox("Nausea", ["no", "yes"],
                               help="Feeling like you might vomit, even without actually vomiting.")
    col4, col5 = st.columns(2)
    with col4:
        loss_of_appetite = st.selectbox("Loss of Appetite", ["no", "yes"],
                                         help="Not wanting to eat, even favourite foods.")
    with col5:
        peritonitis = st.selectbox("Peritonitis", ["no", "yes"],
                                    help="Belly hurts more when pressure is released quickly than "
                                         "when it's pressed — usually checked by a doctor.")

with st.container(key="sec_blood"):
    st.markdown('<div class="section-title">Blood Test Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">These numbers come from a blood test done at a clinic or hospital</div>', unsafe_allow_html=True)

    glossary("What do these blood test words mean?", [
        ("WBC Count (White Blood Cell Count)",
         " — White blood cells are the body's defence against infection. This number counts how "
         "many are in the blood. When the body is fighting an infection like appendicitis, this "
         "number often goes up."),
        ("CRP (C-Reactive Protein)",
         " — A substance made by the body when there's swelling or infection somewhere inside. "
         "The higher this number, the more inflammation is likely happening in the body."),
        ("Neutrophil Percentage",
         " — Neutrophils are a type of white blood cell that respond first and fastest to an "
         "infection, especially one caused by bacteria. A higher percentage of neutrophils often "
         "means the body is actively fighting a bacterial infection."),
    ])

    col1, col2, col3 = st.columns(3)
    with col1:
        wbc_count = st.slider("WBC Count (×10⁹/L)", 0.0, 40.0, 10.0, 0.1,
                               help="White blood cells fight infection. Higher usually means the "
                                    "body may be fighting an infection.")
    with col2:
        crp = st.slider("CRP (mg/L)", 0.0, 300.0, 10.0, 0.1,
                         help="Rises when there's swelling or infection in the body. "
                              "Higher = more inflammation.")
    with col3:
        neutrophil_percentage = st.slider("Neutrophil Percentage (%)", 0.0, 100.0, 60.0, 0.5,
                                           help="A type of white blood cell that fights bacteria. "
                                                "Higher often means an active infection.")

with st.container(key="sec_ultrasound"):
    st.markdown('<div class="section-title">Ultrasound Findings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">These come from an ultrasound scan done by a doctor or radiographer</div>', unsafe_allow_html=True)

    glossary("What do these ultrasound words mean?", [
        ("Ultrasound",
         " — A scan that uses sound waves to take a picture of the inside of the belly. "
         "It doesn't hurt and doesn't use radiation. A trained person (a doctor or radiographer) "
         "does the scan and looks at the pictures."),
        ("Appendix Seen on Ultrasound",
         " — Whether the person doing the scan was able to clearly see the appendix in the pictures."),
        ("Appendix Diameter",
         " — How wide the appendix measures on the scan, in millimetres (mm). A wider appendix "
         "(usually above about 6–7 mm) can be a sign of appendicitis."),
        ("Free Fluid",
         " — Extra fluid seen around the organs in the belly on the scan. This can be a sign of "
         "inflammation, or in serious cases, that the appendix has burst."),
    ])

    col1, col2, col3 = st.columns(3)
    with col1:
        appendix_on_us = st.selectbox("Appendix Seen on Ultrasound", ["no", "yes"],
                                       help="Whether the appendix could be clearly seen on the scan.")
    with col2:
        appendix_diameter = st.slider("Appendix Diameter (mm)", 0.0, 20.0, 6.0, 0.1,
                                       help="How wide the appendix measures on the scan. Wider "
                                            "(usually over 6–7mm) can suggest appendicitis.")
    with col3:
        free_fluids = st.selectbox("Free Fluids", ["no", "yes"],
                                    help="Extra fluid seen in the belly on the scan — can be a "
                                         "sign of inflammation.")

# =========================
# Prediction
# =========================

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

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
                <p>Based on what was entered, this computer program thinks appendicitis is possible.
                This is not a diagnosis — please take the child to see a doctor or go to a clinic
                or hospital as soon as you can, so a real doctor can check them properly.</p>
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
                <p>Based on what was entered, this computer program does not think appendicitis is
                likely right now. This is not a diagnosis — if the child is still in pain or you're
                worried, it's always okay to see a doctor anyway.</p>
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

        gauge_color = "#C25C4E" if is_high_risk else "#F2F0EC"
        gauge_pct = 0.0
        if "appendicitis" in classes:
            gauge_pct = float(probabilities[classes.index("appendicitis")]) * 100

        rows_html = ""
        for cls, prob in zip(classes, probabilities):
            pct = float(prob) * 100
            bar_color = "#C25C4E" if str(cls).lower() == "appendicitis" else "#F2F0EC"
            rows_html += f"""
            <tr>
                <td style="width:38%; color:#F2F0EC; font-weight:600;">{cls}</td>
                <td style="width:12%; text-align:right; color:#9A9A96;">{pct:.2f}%</td>
                <td style="width:50%;">
                    <div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%; background:{bar_color};"></div></div>
                </td>
            </tr>
            """

        st.markdown("### Prediction Probability")
        st.markdown(
            f"""
            <div class="section-panel">
                <div class="gauge-wrap">
                    <div class="gauge-outer" style="background: conic-gradient({gauge_color} {gauge_pct * 3.6:.1f}deg, #1F1F1F 0deg);">
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

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footnote">AppendiCheck Kids uses a trained Random Forest machine learning model. '
    'The prediction is based on selected patient information, symptoms, blood test results and '
    'ultrasound-related features.</div>',
    unsafe_allow_html=True
)