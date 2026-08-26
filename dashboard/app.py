import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="SME Risk V4.0 Pure ML", layout="wide")

st.title("SME Payment Risk Analytics - V4.0 Pure ML (No Leakage)")

# Load model
model_path = Path(__file__).parent.parent / "models" / "model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.success(f"Data: 48839 | Model: {model.classes_} | Features: {list(model.feature_names_in_)} | Accuracy: ~75%")

# Sidebar - No isOpen now
st.sidebar.header("Live Risk Scoring")
total_open_amount = st.sidebar.number_input("Invoice Amount (total_open_amount)", value=17559.64, min_value=0.0)
posting_month = st.sidebar.slider("Posting Month", 1, 12, 6)
delay_days = st.sidebar.slider("Delay Days (0 = on time)", 0, 60, 0)
buisness_year = st.sidebar.selectbox("Business Year", [2020, 2021, 2022, 2023, 2024], index=3)
is_q_end = st.sidebar.checkbox("Is Quarter End? (Mar/Jun/Sep/Dec)")

if st.sidebar.button("Predict Risk"):
    input_dict = {
        'total_open_amount': total_open_amount,
        'posting_month': posting_month,
        'delay_days': delay_days,
        'buisness_year': buisness_year,
        'is_q_end': int(is_q_end)
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[list(model.feature_names_in_)]

    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    if proba < 0.4:
        risk = "LOW RISK"
        color = "green"
    elif proba < 0.7:
        risk = "MEDIUM RISK"
        color = "orange"
    else:
        risk = "HIGH RISK"
        color = "red"

    col1, col2 = st.columns(2)
    col1.metric("Default Probability", f"{proba*100:.2f}%")
    col2.metric("Risk Level", risk)

    st.markdown(f"<h2 style='color:{color}'>Final: {risk} ({proba*100:.0f}%)</h2>", unsafe_allow_html=True)

    st.write("Input:", input_dict)
    st.info("Test: Amount 5k -> LOW, 50k -> MEDIUM, 1Lakh+ with delay 30 -> HIGH")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("V4.0 - Pure ML without isOpen leakage")