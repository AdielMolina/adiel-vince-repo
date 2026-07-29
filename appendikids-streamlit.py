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
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 45%, #fff7f2 100%);
        color: #1f2937;
    }

    .hero {
        padding: 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        box-shadow: 0 12px 35px rgba(37, 99, 235, 0.25);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.95;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 18px;
        background: white;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .metric-card h3 {
        margin-bottom: 0.4rem;
        color: #1e3a8a;
        font-size: 1rem;
    }

    .metric-card p {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
    }

    .result-high {
        padding: 1.5rem;
        border-radius: 20px;
        background: #fee2e2;
        border: 2px solid #ef4444;
        color: #7f1d1d;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
    }

    .result-low {
        padding: 1.5rem;
        border-radius: 20px;
        background: #dcfce7;
        border: 2px solid #22c55e;
        color: #14532d;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
    }

    div.stButton > button {
        width: 100%;
        height: 3.2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        font-weight: 700;
        border: none;
        font-size: 1.05rem;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        color: white;
        border: none;
    }
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
        <h1>🩺 AppendiCheck Kids</h1>
        <p>A machine learning screening support tool for paediatric appendicitis risk prediction.</p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <h3>Final Model</h3>
            <p>Random Forest</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <h3>Accuracy</h3>
            <p>94.87%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <h3>Macro F1</h3>
            <p>94.67%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="metric-card">
            <h3>Purpose</h3>
            <p>Screening</p>
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

st.markdown("## Patient Assessment Form")

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Patient Info",
    "🩻 Symptoms",
    "🧪 Blood Tests",
    "🔍 Ultrasound"
])

with tab1:
    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 0.0, 18.0, 10.0, 0.1)

    with col2:
        sex = st.selectbox("Sex", ["female", "male"])

    with col3:
        bmi = st.slider("BMI", 5.0, 40.0, 18.0, 0.1)

    st.subheader("Clinical Scores")

    col4, col5 = st.columns(2)

    with col4:
        alvarado_score = st.slider("Alvarado Score", 0.0, 10.0, 5.0, 0.5)

    with col5:
        paediatric_score = st.slider("Paediatric Appendicitis Score", 0.0, 10.0, 5.0, 0.5)

with tab2:
    st.subheader("Symptoms")

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

with tab3:
    st.subheader("Blood Test Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        wbc_count = st.slider("WBC Count", 0.0, 40.0, 10.0, 0.1)

    with col2:
        crp = st.slider("CRP", 0.0, 300.0, 10.0, 0.1)

    with col3:
        neutrophil_percentage = st.slider("Neutrophil Percentage", 0.0, 100.0, 60.0, 0.5)

with tab4:
    st.subheader("Ultrasound Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        appendix_on_us = st.selectbox("Appendix Seen on Ultrasound", ["no", "yes"])

    with col2:
        appendix_diameter = st.slider("Appendix Diameter", 0.0, 20.0, 6.0, 0.1)

    with col3:
        free_fluids = st.selectbox("Free Fluids", ["no", "yes"])

# =========================
# Prediction
# =========================

st.markdown("---")

if st.button("Predict Appendicitis Risk"):

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

    if prediction == "appendicitis":
        st.markdown(
            """
            <div class="result-high">
                ⚠️ Predicted Result: Possible Appendicitis
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write(
            "The model predicts that this case may show signs of appendicitis. "
            "Further medical assessment is recommended."
        )

    else:
        st.markdown(
            """
            <div class="result-low">
                ✅ Predicted Result: No Appendicitis
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write(
            "The model predicts that this case may not show signs of appendicitis."
        )

    # Prediction probability
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(df_input)[0]

        try:
            classes = model.classes_
        except:
            classes = model.named_steps["model"].classes_

        probability_df = pd.DataFrame({
            "Diagnosis": classes,
            "Probability": probabilities
        })

        st.markdown("### Prediction Probability")
        st.dataframe(probability_df, use_container_width=True)

        if "appendicitis" in list(classes):
            appendicitis_index = list(classes).index("appendicitis")
            appendicitis_probability = probabilities[appendicitis_index]

            st.progress(float(appendicitis_probability))
            st.write(f"Appendicitis probability: **{appendicitis_probability * 100:.2f}%**")

st.markdown("---")

st.caption(
    "AppendiCheck Kids uses a trained Random Forest machine learning model. "
    "The prediction is based on selected patient information, symptoms, blood test results and ultrasound-related features."
)