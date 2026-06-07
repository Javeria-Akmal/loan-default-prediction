import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="wide")

# Path automatically detect hoga
BASE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(BASE, '..', 'Backend')

@st.cache_resource
def load_models():
    best_lr  = pickle.load(open(os.path.join(BACKEND, 'best_lr.pkl'),  'rb'))
    best_rf  = pickle.load(open(os.path.join(BACKEND, 'best_rf.pkl'),  'rb'))
    best_gb  = pickle.load(open(os.path.join(BACKEND, 'best_gb.pkl'),  'rb'))
    best_xgb = pickle.load(open(os.path.join(BACKEND, 'best_xgb.pkl'), 'rb'))
    scaler     = pickle.load(open(os.path.join(BACKEND, 'scaler.pkl'),     'rb'))
    normalizer = pickle.load(open(os.path.join(BACKEND, 'normalizer.pkl'), 'rb'))

    ann = Sequential([
        Dense(64, activation='relu', input_shape=(16,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann.load_weights(os.path.join(BACKEND, 'ann_weights.weights.h5'))

    dnn = Sequential([
        Dense(128, activation='relu', input_shape=(16,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    dnn.load_weights(os.path.join(BACKEND, 'dnn_weights.weights.h5'))

    cnn = Sequential([
        tf.keras.layers.Conv1D(32, 2, activation='relu', input_shape=(16, 1)),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    cnn.load_weights(os.path.join(BACKEND, 'cnn_weights.weights.h5'))

    lstm = Sequential([
        tf.keras.layers.LSTM(64, input_shape=(16, 1)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    lstm.load_weights(os.path.join(BACKEND, 'lstm_weights.weights.h5'))

    return best_lr, best_rf, best_gb, best_xgb, scaler, normalizer, ann, dnn, cnn, lstm

best_lr, best_rf, best_gb, best_xgb, scaler, normalizer, ann, dnn, cnn, lstm = load_models()

st.title("🏦 Loan Default Prediction")
st.markdown("Fill in the loan details and select a model to predict default risk.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    age              = st.number_input("Age",              min_value=18,    max_value=69,    value=35)
    income           = st.number_input("Income",           min_value=15000, max_value=150000, value=60000)
    loan_amount      = st.number_input("Loan Amount",      min_value=5000,  max_value=250000, value=50000)
    credit_score     = st.number_input("Credit Score",     min_value=300,   max_value=849,   value=600)
    months_employed  = st.number_input("Months Employed",  min_value=0,     max_value=119,   value=24)
    num_credit_lines = st.number_input("Num Credit Lines", min_value=1,     max_value=4,     value=2)

with col2:
    interest_rate = st.slider("Interest Rate (%)", 2.0, 25.0, 10.0)
    loan_term     = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    dti_ratio     = st.slider("DTI Ratio", 0.1, 0.9, 0.5)
    education     = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment    = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])

with col3:
    marital        = st.selectbox("Marital Status",  ["Single", "Married", "Divorced"])
    has_mortgage   = st.selectbox("Has Mortgage",    ["Yes", "No"])
    has_dependents = st.selectbox("Has Dependents",  ["Yes", "No"])
    loan_purpose   = st.selectbox("Loan Purpose",    ["Home", "Business", "Education", "Auto", "Other"])
    has_cosigner   = st.selectbox("Has Co-Signer",   ["Yes", "No"])
    model_type     = st.selectbox("Select Model", [
        "Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost",
        "ANN", "DNN", "CNN", "LSTM"
    ])

st.divider()

edu_map  = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
emp_map  = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
mar_map  = {"Divorced": 0, "Married": 1, "Single": 2}
yn_map   = {"No": 0, "Yes": 1}
purp_map = {"Auto": 0, "Business": 1, "Education": 2, "Home": 3, "Other": 4}

if st.button("🔍 Predict", use_container_width=True):
    raw = np.array([[
        age, income, loan_amount, credit_score,
        months_employed, num_credit_lines, interest_rate,
        loan_term, dti_ratio,
        edu_map[education], emp_map[employment], mar_map[marital],
        yn_map[has_mortgage], yn_map[has_dependents],
        purp_map[loan_purpose], yn_map[has_cosigner]
    ]])

    scaled     = scaler.transform(raw)
    normalized = normalizer.transform(scaled)

    if model_type == "Logistic Regression":
        prob = best_lr.predict_proba(scaled)[0][1]
    elif model_type == "Random Forest":
        prob = best_rf.predict_proba(scaled)[0][1]
    elif model_type == "Gradient Boosting":
        prob = best_gb.predict_proba(scaled)[0][1]
    elif model_type == "XGBoost":
        prob = best_xgb.predict_proba(scaled)[0][1]
    elif model_type == "ANN":
        prob = float(ann.predict(normalized)[0][0])
    elif model_type == "DNN":
        prob = float(dnn.predict(normalized)[0][0])
    elif model_type == "CNN":
        prob = float(cnn.predict(normalized.reshape(1, normalized.shape[1], 1))[0][0])
    elif model_type == "LSTM":
        prob = float(lstm.predict(normalized.reshape(1, normalized.shape[1], 1))[0][0])

    prediction = int(prob > 0.5)

    st.subheader("Prediction Result")
    r1, r2 = st.columns(2)
    with r1:
        if prediction == 1:
            st.error("⚠️ HIGH RISK — Loan Default Likely")
        else:
            st.success("✅ LOW RISK — Loan Default Unlikely")
    with r2:
        st.metric("Default Probability", f"{prob*100:.1f}%")
        st.progress(float(prob))