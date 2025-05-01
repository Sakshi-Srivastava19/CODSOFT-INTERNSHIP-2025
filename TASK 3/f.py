import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load model
model = pickle.load(open("fraud_model.pkl", "rb"))

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")

st.markdown("Enter transaction details below to check if it's fraudulent.")

# Input fields for 28 PCA features
v_features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0, format="%.5f")
    v_features.append(val)

# Amount and Time inputs
amount = st.number_input("Transaction Amount", value=0.0)
time = st.number_input("Time (seconds since first transaction)", value=0.0)

# Normalize Amount and Time
scaler = StandardScaler()
amount_norm = scaler.fit_transform([[amount]])[0][0]
time_norm = scaler.fit_transform([[time]])[0][0]

# Combine all inputs
features = np.array(v_features + [amount_norm, time_norm]).reshape(1, -1)

# Predict on button click
if st.button("Check for Fraud"):
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]
    if prediction[0] == 1:
        st.error(f"⚠️ Fraud Detected! (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Confidence: {1 - prob:.2f})")
