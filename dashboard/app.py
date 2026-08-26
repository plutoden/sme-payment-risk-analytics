import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="SME Risk V3.0 Pure ML", layout="wide")

st.title("SME Payment Risk Analytics - V3.0 Pure ML")

# Load model
model_path = Path(__file__).parent.parent / "models" / "model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.success(f"Data: 48839 | Model: {model.classes_} | Features: {list(model.feature_names_in_)}")

# Sidebar Inputs
st.sidebar.header("Live Risk Scoring")
total_open_amount = st.sidebar.number_input("Invoice Amount (total_open_amount)", value=17559.64)
posting_month = st.sidebar.slider("Posting Month", 1, 12, 6)
delay_days = st.sidebar.number_input("Delay Days", value=0)
buisness_year = st.sidebar.number_input("Business Year", value=2023)
is_q_end = st.sidebar.checkbox("Is Quarter End?")
isOpen = st.sidebar.checkbox("is_open (1 = payment pending)", value=True)

if st.sidebar.button("Predict Risk"):
    input_dict = {
        'total_open_amount': total_open_amount,
        'posting_month': posting_month,
        'delay_days': delay_days,
        'buisness_year': buisness_year,
        'is_q_end': int(is_q_end),
        'isOpen': int(isOpen)
    }
    # Note the typo buisness_year is kept as model was trained with it
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[list(model.feature_names_in_)]

    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    if proba < 0.4:
        risk = "LOW RISK"
    elif proba < 0.7:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH RISK"

    st.metric("Model Raw Proba", f"{proba*100:.2f}%")
    st.metric("Final Risk", f"{risk} ({proba*100:.0f}%)")
    st.write(input_dict)