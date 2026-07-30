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
# Load trained model
# =========================

try:
    model = joblib.load("appendicitis_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please make sure appendicitis_model.pkl is in the same folder as this app.")
    st.stop()
except Exception as e:
    st.error("The model could not be loaded. Please check the model file and requirements.txt.")
    st.stop()

# =========================
# Design tokens + global style
# =========================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --bg-deep: #0A0E1A;
        --bg-panel: #131A2C;
        --bg-panel-alt: #1B2338;
        --border: #2A3350;
        --text-primary: #E7ECF7;
        --text-muted: #8892AD;
        --blue: #5B8DEF;
        --blue-deep: #3D6FD1;
        --purple: #8B7CF6;
        --purple-deep: #6C5CE0;
        --success: #43B876;
        --success-tint: #16261F;
        --danger: #E5645A;
        --danger-tint: #2A1A1C;
        --amber: #D9A441;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 10% -10%, #141C36 0%, var(--bg-deep) 45%) fixed;
        color: var(--text-primary);
    }

    #MainMenu, footer, header { visibility: hidden; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--text-primary); }

    /* ---------- Hero ---------- */
    .hero {
        padding: 2.4rem 2.6rem 2rem 2.6rem;
        border-radius: 20px;
        background:
            radial-gradient(circle at 90% 0%, rgba(139,124,246,0.16) 0%, transparent 55%),
            linear-gradient(155deg, #131B32 0%, #0E1424 100%);
        border: 1px solid var(--border);
        box-shadow: 0 16px 40px rgba(0,0,0,0.35);
        margin-bottom: 1.2rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: var(--bg-panel-alt);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--blue);
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.12;
        margin: 0 0 0.6rem 0;
    }
    .hero p {
        font-size: 1rem;
        max-width: 620px;
        color: var(--text-muted);
        margin: 0 0 1.2rem 0;
    }
    .stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1.1rem; }
    .stat-chip {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: var(--text-primary);
    }
    .stat-chip span { color: var(--blue); font-weight: 600; }
    .trend-svg { width: 100%; max-width: 420px; height: 36px; opacity: 0.9; }
    .trend-svg path.line { fill: none; stroke: var(--purple); stroke-width: 2; stroke-linecap: round; }
    .trend-svg circle { fill: var(--blue); }

    /* ---------- Disclaimer ---------- */
    .disclaimer-block {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-left: 3px solid var(--amber);
        border-radius: 10px;
        padding: 0.85rem 1.2rem;
        color: var(--text-muted);
        font-size: 0.88rem;
        margin: 1.1rem 0 1.5rem 0;
    }
    .disclaimer-block b { color: var(--text-primary); }

    /* ---------- Section titles ---------- */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        color: var(--text-primary);
        margin-bottom: 0.15rem;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .section-sub { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem; }
    .subsection-divider { height: 1px; background: var(--border); margin: 1.3rem 0 1.1rem 0; }

    /* Every section container gets the SAME consistent panel treatment.
       Applied directly to the container's key-class (not a split div) so the
       background truly wraps every widget inside it. */
    .st-key-sec_patient, .st-key-sec_symptoms, .st-key-sec_blood, .st-key-sec_ultrasound {
        background: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1.5rem 1.7rem 1rem 1.7rem !important;
        margin-bottom: 1.1rem !important;
    }
    .section-panel {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem 1.7rem 0.6rem 1.7rem;
    }

    label, .stSlider label, .stSelectbox label {
        color: var(--text-muted) !important; font-size: 0.83rem !important; font-weight: 500 !important;
    }

    .stSlider [data-baseweb="slider"] > div > div { background: var(--blue) !important; }
    .stSlider [role="slider"] { background-color: var(--purple) !important; border: 3px solid var(--bg-deep) !important; }

    [data-testid="stSelectbox"] > div > div {
        background-color: var(--bg-panel-alt) !important;
        border-color: var(--border) !important;
        border-radius: 10px !important;
    }
    [data-testid="stSelectbox"] > div > div * { color: var(--text-primary) !important; fill: var(--text-primary) !important; }
    div[data-baseweb="popover"] div[data-baseweb="menu"] { background-color: var(--bg-panel-alt) !important; border: 1px solid var(--border) !important; }
    div[data-baseweb="popover"] li { background-color: var(--bg-panel-alt) !important; color: var(--text-primary) !important; }
    div[data-baseweb="popover"] li:hover { background-color: var(--bg-panel) !important; }

    /* ---------- Expander (used for plain-language glossary) ---------- */
    [data-testid="stExpander"] {
        background: var(--bg-panel-alt);
        border: 1px solid var(--border);
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    [data-testid="stExpander"] summary {
        color: var(--blue) !important;
        font-weight: 600;
        font-size: 0.92rem;
    }
    [data-testid="stExpander"] summary:hover { color: var(--purple) !important; }
    [data-testid="stExpander"] p { color: var(--text-primary); font-size: 0.92rem; margin-bottom: 0.5rem; }
    [data-testid="stExpander"] b { color: var(--blue); }
    .glossary-term { margin-bottom: 0.7rem; }
    .glossary-term b { display: block; color: var(--purple); font-family: 'Space Grotesk', sans-serif; }

    /* ---------- Custom pill selector (replaces st.tabs) ---------- */
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 8px; flex-wrap: wrap; margin-bottom: 0.2rem;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.55rem 1.2rem;
        display: flex; align-items: center; gap: 0.5rem;
        cursor: pointer;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display: none; }
    div[data-testid="stRadio"] label[data-baseweb="radio"] p {
        margin: 0; font-weight: 600; font-size: 0.92rem; color: var(--text-muted); white-space: nowrap;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(135deg, var(--blue-deep), var(--purple-deep));
        border-color: var(--purple);
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p { color: white; }

    /* ---------- Button ---------- */
    div.stButton > button {
        width: 100%;
        height: 3.2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--blue) 0%, var(--purple-deep) 100%);
        color: white;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.01em;
        border: none;
        font-size: 1.02rem;
        box-shadow: 0 10px 26px rgba(91, 141, 239, 0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 30px rgba(139, 124, 246, 0.35);
        color: white;
    }

    /* ---------- Result banner ---------- */
    .result-banner {
        padding: 1.6rem 1.9rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        background: var(--bg-panel);
        border: 1px solid var(--border);
    }
    .result-high { border-left: 4px solid var(--danger); }
    .result-low { border-left: 4px solid var(--success); }
    .result-banner .kicker {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.35rem;
    }
    .result-banner h2 { font-size: 1.7rem; font-weight: 700; margin: 0 0 0.4rem 0; }
    .result-high h2 { color: var(--danger); }
    .result-low h2 { color: var(--success); }
    .result-banner p { margin: 0; color: var(--text-muted); max-width: 640px; }

    /* Gauge */
    .gauge-wrap { display: flex; align-items: center; gap: 2rem; flex-wrap: wrap; }
    .gauge-outer { border-radius: 50%; width: 164px; height: 164px; display: flex; align-items: center; justify-content: center; }
    .gauge-inner {
        border-radius: 50%; width: 126px; height: 126px; background: var(--bg-deep);
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        border: 1px solid var(--border);
    }
    .gauge-pct { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; }
    .gauge-label { font-size: 0.65rem; color: var(--text-muted); letter-spacing: 0.05em; text-transform: uppercase; margin-top: 0.15rem; }

    .prob-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem; }
    .prob-table td { padding: 0.55rem 0.2rem; }
    .prob-table .bar-bg { background: var(--bg-panel-alt); border-radius: 6px; height: 8px; width: 100%; overflow: hidden; }
    .prob-table .bar-fill { height: 8px; border-radius: 6px; }

    .footnote { color: var(--text-muted); font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; }
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
        <div class="hero-badge">◆ Paediatric Screening Model</div>
        <h1>Appendicitis risk, quantified.</h1>
        <p>AppendiCheck Kids scores appendicitis likelihood in children from clinical scores,
        symptoms, blood work and ultrasound findings — a statistical aid to support a clinical
        decision, never to replace one.</p>
        <div class="stat-row">
            <div class="stat-chip">Model <span>Random Forest</span></div>
            <div class="stat-chip">Accuracy <span>94.87%</span></div>
            <div class="stat-chip">Macro F1 <span>94.67%</span></div>
            <div class="stat-chip">Use case <span>Screening only</span></div>
        </div>
        <svg class="trend-svg" viewBox="0 0 420 36" preserveAspectRatio="none">
            <path class="line" d="M0 28 L60 22 L120 26 L180 12 L240 18 L300 6 L360 14 L420 4" />
            <circle cx="420" cy="4" r="3.5" />
        </svg>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="disclaimer-block"><b>Disclaimer —</b> This app is an educational screening '
    'support tool. It does not replace professional medical diagnosis.</div>',
    unsafe_allow_html=True
)

st.info(
    "How to use: Fill in the patient information, symptoms, blood test results and ultrasound findings. "
    "Then click 'Predict Appendicitis Risk' to view the screening result and probability."
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
warnings = []

if appendix_on_us == "no" and appendix_diameter > 6:
    warnings.append(
        "Appendix diameter is high, but appendix is marked as not seen on ultrasound. Please check the ultrasound input."
    )

if alvarado_score <= 3 and paediatric_score >= 8:
    warnings.append(
        "Alvarado Score is low but Paediatric Appendicitis Score is high. Please check if both scores were entered correctly."
    )

if wbc_count <= 5 and crp >= 100:
    warnings.append(
        "CRP is very high while WBC count is low. This may be possible, but please verify the blood test values."
    )

for warning in warnings:
    st.warning(warning)
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

        gauge_color = "#E5645A" if is_high_risk else "#43B876"
        gauge_pct = 0.0
        if "appendicitis" in classes:
            gauge_pct = float(probabilities[classes.index("appendicitis")]) * 100

        summary_text = f"""
            AppendiCheck Kids Result Summary

            Prediction: {prediction}
            Appendicitis probability: {gauge_pct:.2f}%

            Patient inputs:
            Age: {age}
            Sex: {sex}
            BMI: {bmi}
            Alvarado Score: {alvarado_score}
            Paediatric Appendicitis Score: {paediatric_score}
            Lower Right Abdominal Pain: {lower_right_pain}
            Migratory Pain: {migratory_pain}
            Nausea: {nausea}
            Loss of Appetite: {loss_of_appetite}
            Peritonitis: {peritonitis}
            WBC Count: {wbc_count}
            CRP: {crp}
            Neutrophil Percentage: {neutrophil_percentage}
            Appendix Seen on Ultrasound: {appendix_on_us}
            Appendix Diameter: {appendix_diameter}
            Free Fluids: {free_fluids}

            Disclaimer:
            This is an educational screening support tool and does not replace professional medical diagnosis.
            """

        st.download_button(
                label="Download Result Summary",
                data=summary_text,
                file_name="appendicheck_result_summary.txt",
                mime="text/plain"
            )
        
        rows_html = ""
        for cls, prob in zip(classes, probabilities):
            pct = float(prob) * 100
            bar_color = "#E5645A" if str(cls).lower() == "appendicitis" else "#43B876"
            rows_html += f"""
            <tr>
                <td style="width:38%; color:#E7ECF7; font-weight:600;">{cls}</td>
                <td style="width:12%; text-align:right; color:#8892AD;">{pct:.2f}%</td>
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
                    <div class="gauge-outer" style="background: conic-gradient({gauge_color} {gauge_pct * 3.6:.1f}deg, #1B2338 0deg);">
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
        with st.expander("What does this result mean?"):
            st.write(
                "A higher appendicitis probability means the model found patterns similar to appendicitis cases "
                "in the training data. A lower probability means the inputs look more similar to no-appendicitis cases."
            )
            st.write(
                "This result should be used only as screening support. A doctor should make the final diagnosis."
            )
st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footnote">AppendiCheck Kids uses a trained Random Forest machine learning model. '
    'The prediction is based on selected patient information, symptoms, blood test results and '
    'ultrasound-related features.</div>',
    unsafe_allow_html=True
)