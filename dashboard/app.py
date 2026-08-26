import os, joblib, pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="SME Payment Risk Analytics - V3.0 Pure ML", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

def find_file(paths):
    for rel in paths:
        for base in [BASE_DIR, ROOT_DIR]:
            p = os.path.join(base, rel)
            if os.path.exists(p): return p
    return None

@st.cache_data
def load_data():
    p = find_file(["data/cleaned.csv","data/processed/cleaned.csv","../data/cleaned.csv"])
    return pd.read_csv(p)

@st.cache_resource
def load_model():
    mp = find_file(["models/risk_model.pkl","models/model.pkl"])
    sp = find_file(["models/scaler.pkl"])
    model = joblib.load(mp)
    scaler = joblib.load(sp) if sp else None
    return model, scaler

df = load_data()
model, scaler = load_model()

st.title("SME Payment Risk Analytics - V3.0 Pure ML")
st.success(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} cols | Model: {type(model).__name__}")

# --- Sidebar ---
st.sidebar.title("Live Risk Scoring")
st.sidebar.info(f"Model: {type(model).__name__}")

# Get expected features
expected_features = list(getattr(model, "feature_names_in_", []))
st.sidebar.caption(f"Expected: {len(expected_features)} features")

# User inputs for main fields
amount = st.sidebar.number_input("total_open_amount", value=50000.0, min_value=0.0)
terms = st.sidebar.selectbox("cust_payment_terms", options=sorted(df['cust_payment_terms'].dropna().unique()))
currency = st.sidebar.selectbox("invoice_currency", options=sorted(df['invoice_currency'].dropna().unique()))
bcode = st.sidebar.selectbox("business_code", options=sorted(df['business_code'].dropna().unique()))
b_year = st.sidebar.number_input("buisness_year", value=2020, min_value=2018, max_value=2026)
posting_month = st.sidebar.slider("posting_month", 1, 12, 6)
delay_days = st.sidebar.slider("delay_days", 0, 180, 30)
is_q_end = st.sidebar.checkbox("is_q_end", value=False)
isOpen = st.sidebar.checkbox("isOpen", value=True)

if st.sidebar.button("Predict Risk", type="primary"):
    # Build base dict with medians/modes
    base_row = {}
    for col in expected_features:
        if col == "total_open_amount": base_row[col] = amount
        elif col == "buisness_year": base_row[col] = b_year
        elif col == "posting_month": base_row[col] = posting_month
        elif col == "delay_days": base_row[col] = delay_days
        elif col == "is_q_end": base_row[col] = int(is_q_end)
        elif col == "isOpen": base_row[col] = int(isOpen)
        elif col == "is_delayed": base_row[col] = 1 if delay_days > 15 else 0
        elif col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                base_row[col] = float(df[col].median())
            else:
                base_row[col] = df[col].mode()[0]
        else:
            base_row[col] = 0 # default for engineered features

    input_df = pd.DataFrame([base_row])
    # Ensure column order = model expectation
    input_df = input_df[expected_features]

    try:
        if scaler:
            proba = model.predict_proba(scaler.transform(input_df))[0][1]
        else:
            proba = model.predict_proba(input_df)[0][1]

        st.sidebar.metric("Default Probability", f"{proba:.2%}")
        if proba > 0.70:
            st.sidebar.error("🔴 HIGH RISK")
        elif proba > 0.40:
            st.sidebar.warning("🟡 MEDIUM RISK")
        else:
            st.sidebar.success("🟢 LOW RISK")
        st.sidebar.write(input_df.T)
    except Exception as e:
        st.sidebar.error(str(e))

st.dataframe(df.head(100))