# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("src/logistic_model.pkl")
scaler = joblib.load("src/scaler.pkl")

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter the passenger details below to predict survival:")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=6, value=0)
fare = st.number_input("Passenger Fare", min_value=0.0, max_value=600.0, value=32.0)
sex = st.radio("Sex", ["Male", "Female"])
embarked = st.radio("Port of Embarkation", ["S", "C", "Q"])

# Prepare input for prediction
def preprocess_input():
    sex_male = 1 if sex == "Male" else 0
    embarked_Q = 1 if embarked == "Q" else 0
    embarked_S = 1 if embarked == "S" else 0

    input_data = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S]])
    input_scaled = scaler.transform(input_data)
    return input_scaled

if st.button("Predict Survival"):
    features = preprocess_input()
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"🎉 The passenger **would survive** with {proba*100:.2f}% probability.")
    else:
        st.error(f"💀 The passenger **would not survive**. Survival probability: {proba*100:.2f}%.")

