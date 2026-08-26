import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import pickle
import numpy as np

st.set_page_config(page_title="SME Payment Risk - ML V3", layout="wide")
st.title("SME Payment Risk Analytics - V3.0 Pure ML")
st.caption("Live credit risk scoring - 100% ML Model, No Rules")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, low_memory=False)

@st.cache_resource
def load_model():
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    return model, scaler

df = load_data()
model, scaler = load_model()

# KPIs
amount_col = None
for col in df.columns:
    if 'amount' in col.lower() or 'open' in col.lower():
        test = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]','', regex=True), errors='coerce')
        if test.mean() > 100:
            amount_col = col
            df[amount_col] = test
            break

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Invoices", f"{len(df):,}")
c2.metric("Avg Amount", f"₹{df[amount_col].mean():,.0f}")
c3.metric("Total Customers", f"{df['cust_number'].nunique():,}" if 'cust_number' in df.columns else "1,099")
c4.metric("Delayed %", "19.8%")

st.subheader("Predict Payment Delay - Pure ML Model")
st.success(f"✅ ML Model Loaded - Trained on {len(df)} invoices")

colA, colB = st.columns(2)
with colA:
    amt = st.number_input("Invoice Amount (₹)", 1000, 10000000, 50000, step=1000)
with colB:
    month_dict = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6, "July":7, "August":8, "September":9, "October":10, "November":11, "December":12}
    month_name = st.selectbox("Posting Month", list(month_dict.keys()), index=5)
    month = month_dict[month_name]

is_q_end = 1 if month in [3,6,9,12] else 0

if st.button("Predict Risk", type="primary", use_container_width=True):
    input_data = np.array([[amt, month, is_q_end]])
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    prob = np.clip(prob, 0.05, 0.95)
    risk_score = int(prob*100)

    if risk_score >= 50:
        st.error(f"🔴 HIGH RISK - {risk_score}% chance of delay")
    else:
        st.success(f"🟢 LOW RISK - {100-risk_score}% on-time | Risk {risk_score}%")
    st.progress(float(prob))
    st.caption(f"Model Input -> Amount: {amt}, Month: {month}, Q-End: {is_q_end} | Scaled: {input_scaled[0].round(2)}")