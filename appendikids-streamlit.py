import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Load trained model and saved feature information
model = joblib.load("appendicitis_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")
default_values = joblib.load("model/default_values.pkl")

# Streamlit app title
st.title("AppendiCheck Kids")
st.subheader("Appendicitis Screening Support Tool")

st.write(
    "This app predicts whether a child may have appendicitis based on clinical, "
    "blood test, and ultrasound-related information."
)

st.warning(
    "Disclaimer: This is an educational screening support tool. "
    "It does not replace professional medical diagnosis."
)

# User inputs
st.header("Patient Information")

age = st.slider("Age", min_value=0.0, max_value=18.0, value=10.0, step=0.1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", min_value=5.0, max_value=40.0, value=18.0, step=0.1)

st.header("Clinical Scores")

alvarado_score = st.slider("Alvarado Score", min_value=0.0, max_value=10.0, value=5.0, step=0.5)

paediatric_score = st.slider(
    "Paediatric Appendicitis Score",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.5
)

st.header("Symptoms")

lower_right_pain = st.selectbox("Lower Right Abdominal Pain", ["yes", "no"])
migratory_pain = st.selectbox("Migratory Pain", ["yes", "no"])
nausea = st.selectbox("Nausea", ["yes", "no"])
loss_of_appetite = st.selectbox("Loss of Appetite", ["yes", "no"])
peritonitis = st.selectbox("Peritonitis", ["yes", "no"])

st.header("Blood Test Results")

wbc_count = st.slider("WBC Count", min_value=0.0, max_value=40.0, value=10.0, step=0.1)
crp = st.slider("CRP", min_value=0.0, max_value=300.0, value=10.0, step=0.1)
neutrophil_percentage = st.slider(
    "Neutrophil Percentage",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=0.5
)

st.header("Ultrasound Information")

appendix_on_us = st.selectbox("Appendix Seen on Ultrasound", ["yes", "no"])
appendix_diameter = st.slider(
    "Appendix Diameter",
    min_value=0.0,
    max_value=20.0,
    value=6.0,
    step=0.1
)
free_fluids = st.selectbox("Free Fluids", ["yes", "no"])

# Predict button
if st.button("Predict Appendicitis Risk"):

    # Start with default values for all original model input columns
    input_data = default_values.copy()

    # Update only the values that users enter in the app
    if "Age" in input_data:
        input_data["Age"] = age

    if "Sex" in input_data:
        input_data["Sex"] = sex

    if "BMI" in input_data:
        input_data["BMI"] = bmi

    if "Alvarado_Score" in input_data:
        input_data["Alvarado_Score"] = alvarado_score

    if "Paedriatic_Appendicitis_Score" in input_data:
        input_data["Paedriatic_Appendicitis_Score"] = paediatric_score

    if "Lower_Right_Abd_Pain" in input_data:
        input_data["Lower_Right_Abd_Pain"] = lower_right_pain

    if "Migratory_Pain" in input_data:
        input_data["Migratory_Pain"] = migratory_pain

    if "Nausea" in input_data:
        input_data["Nausea"] = nausea

    if "Loss_of_Appetite" in input_data:
        input_data["Loss_of_Appetite"] = loss_of_appetite

    if "Peritonitis" in input_data:
        input_data["Peritonitis"] = peritonitis

    if "WBC_Count" in input_data:
        input_data["WBC_Count"] = wbc_count

    if "CRP" in input_data:
        input_data["CRP"] = crp

    if "Neutrophil_Percentage" in input_data:
        input_data["Neutrophil_Percentage"] = neutrophil_percentage

    if "Appendix_on_US" in input_data:
        input_data["Appendix_on_US"] = appendix_on_us

    if "Appendix_Diameter" in input_data:
        input_data["Appendix_Diameter"] = appendix_diameter

    if "Free_Fluids" in input_data:
        input_data["Free_Fluids"] = free_fluids

    # Convert input into dataframe
    df_input = pd.DataFrame([input_data])

    # One-hot encode like notebook
    df_input = pd.get_dummies(df_input, drop_first=True)

    # Match training columns
    df_input = df_input.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(df_input)[0]

    st.subheader("Prediction Result")

    if prediction == "appendicitis":
        st.error("Predicted Result: Appendicitis")
        st.write("The model predicts that this case may show signs of appendicitis.")
    else:
        st.success("Predicted Result: No Appendicitis")
        st.write("The model predicts that this case may not show signs of appendicitis.")

    # Show probability if the model supports it
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df_input)[0]
        classes = model.classes_

        probability_df = pd.DataFrame({
            "Diagnosis": classes,
            "Probability": probabilities
        })

        st.write("Prediction Probability")
        st.dataframe(probability_df)