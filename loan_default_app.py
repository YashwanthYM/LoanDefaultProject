import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json

# Load saved files
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
means = pickle.load(open("means.pkl", "rb"))
threshold = pickle.load(open("threshold.pkl", "rb"))
metrics = json.load(open("metrics.json", "r"))

st.title("Loan Default Risk Prediction System")

st.write("Enter applicant details below to predict default risk.")

# Input fields
age = st.number_input("Age", 18, 80, 30)
income = st.number_input("Annual Income", 0, 10000000, 50000)
loan_amt = st.number_input("Loan Amount", 0, 10000000, 20000)
credit_score = st.number_input("Credit Score", 300, 900, 650)
loan_term = st.number_input("Loan Term (months)", 1, 360, 60)
dti = st.number_input("DTI Ratio (%)", 0.0, 100.0, 30.0)

emp_type = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
mortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
cosigner = st.selectbox("Has Co-signer?", ["Yes", "No"])

# Prepare dataframe
input_df = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "LoanAmount": [loan_amt],
    "CreditScore": [credit_score],
    "LoanTerm": [loan_term],
    "DTIRatio": [dti],
    "EmploymentType": [emp_type],
    "MaritalStatus": [marital],
    "HasMortgage": [mortgage],
    "HasCoSigner": [cosigner]
})

# Feature engineering
input_df["RepaymentRatio"] = input_df["LoanAmount"] / (input_df["Income"] + 1)

# Scale numeric features
num_features = means.index.tolist()
input_df[num_features] = scaler.transform(input_df[num_features])

# Encode categorical
for col in label_encoders:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# Predict
if st.button("Predict"):
    prob = model.predict_proba(input_df[feature_columns])[0][1]
    pred = "High Risk" if prob >= threshold else "Low Risk"

    color = "red" if pred == "High Risk" else "green"
    st.markdown(f"### Prediction: <span style='color:{color}'>{pred}</span>", unsafe_allow_html=True)
    st.write(f"**Default Probability:** {prob:.4f}")

    st.subheader("Model Performance")
    st.json(metrics)
