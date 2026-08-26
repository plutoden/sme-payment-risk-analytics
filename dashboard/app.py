import os, joblib, pandas as pd
import streamlit as st

st.set_page_config(page_title="SME Payment Risk Analytics - V3.0", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

def find_file(names):
    for n in names:
        for base in [BASE_DIR, ROOT_DIR]:
            p = os.path.join(base, n)
            if os.path.exists(p): return p
    return None

@st.cache_data
def load_data():
    p = find_file(["data/cleaned.csv","data/processed/cleaned.csv","../data/cleaned.csv","dashboard/../data/cleaned.csv"])
    return pd.read_csv(p)

@st.cache_resource
def load_model():
    # risk_model.pkl ko priority de, model.pkl ko bilkul ignore
    mp = find_file(["models/risk_model.pkl","../models/risk_model.pkl"])
    sp = find_file(["models/scaler.pkl","../models/scaler.pkl"])
    model = joblib.load(mp)
    scaler = None
    try:
        scaler_obj = joblib.load(sp)
        # scaler ka feature match check
        if hasattr(scaler_obj, "feature_names_in_"):
            if list(scaler_obj.feature_names_in_) == list(model.feature_names_in_):
                scaler = scaler_obj
    except:
        scaler = None
    return model, scaler

df = load_data()
model, scaler = load_model()
expected = list(model.feature_names_in_)

st.title("SME Payment Risk Analytics - V3.0 Pure ML")
st.success(f"Loaded {df.shape[0]} rows | Real Model: risk_model.pkl | Features: {expected}")

st.sidebar.title("Live Risk Scoring")
st.sidebar.code(f"Model needs:\n{expected}")

inputs = {}
for col in expected:
    if col == "total_open_amount":
        inputs[col] = st.sidebar.number_input(col, value=50000.0)
    elif col == "posting_month":
        inputs[col] = st.sidebar.slider(col, 1, 12, 6)
    elif col == "delay_days":
        inputs[col] = st.sidebar.slider(col, 0, 180, 30)
    elif col == "buisness_year":
        inputs[col] = st.sidebar.number_input(col, value=2020)
    elif col in ["is_q_end","isOpen","is_delayed","is_open"]:
        inputs[col] = int(st.sidebar.checkbox(col, value=False))
    else:
        inputs[col] = float(df[col].median()) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else 0

if st.sidebar.button("Predict Risk", type="primary"):
    import numpy as np
    input_df = pd.DataFrame([inputs])[expected]
    X = scaler.transform(input_df) if scaler else input_df
    proba = model.predict_proba(X)[0][1]
    st.sidebar.metric("Default Probability", f"{proba:.2%}")
    if proba>0.7: st.sidebar.error("🔴 HIGH RISK")
    elif proba>0.4: st.sidebar.warning("🟡 MEDIUM RISK")
    else: st.sidebar.success("🟢 LOW RISK")
    st.sidebar.dataframe(input_df.T)

st.dataframe(df.head(100))