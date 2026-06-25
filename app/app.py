import streamlit as st
import numpy as np
import joblib

# Load model
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "logistic_model.pkl")

model = joblib.load(model_path)

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below:")

age = st.number_input("Age")
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.number_input("Chest Pain Type")
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
restecg = st.number_input("Rest ECG")
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak")

input_data = np.array([[age, sex, cp, trestbps, chol,
                        fbs, restecg, thalach, exang, oldpeak]])

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")