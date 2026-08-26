import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="SME Payment Risk Analytics - V3.0 Pure ML", layout="wide")

# --- PATH RESOLVER (Cloud + Local dono pe kaam karega) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

def find_file(relative_paths):
    for rel in relative_paths:
        p1 = os.path.join(BASE_DIR, rel)
        p2 = os.path.join(ROOT_DIR, rel)
        p3 = os.path.abspath(rel)
        for p in [p1, p2, p3]:
            if os.path.exists(p):
                return p
    return None

@st.cache_data
def load_data():
    csv_path = find_file([
        "data/cleaned.csv",
        "data/processed/cleaned.csv",
        "../data/cleaned.csv",
        "../data/processed/cleaned.csv",
        "sme-payment-risk-analytics/data/cleaned.csv"
    ])
    if not csv_path:
        st.error("cleaned.csv not found. Searched in data/ and data/processed/")
        st.write("ROOT_DIR:", ROOT_DIR)
        st.write("BASE_DIR:", BASE_DIR)
        st.write("ROOT contents:", os.listdir(ROOT_DIR))
        if os.path.exists(os.path.join(ROOT_DIR, "data")):
            st.write("data contents:", os.listdir(os.path.join(ROOT_DIR, "data")))
        st.stop()
    return pd.read_csv(csv_path)

@st.cache_resource
def load_model():
    model_path = find_file([
        "models/risk_model.pkl",
        "models/model.pkl",
        "../models/risk_model.pkl",
        "../models/model.pkl"
    ])
    scaler_path = find_file([
        "models/scaler.pkl",
        "../models/scaler.pkl"
    ])
    model = joblib.load(model_path) if model_path else None
    scaler = joblib.load(scaler_path) if scaler_path else None
    return model, scaler

# --- LOAD ---
df = load_data()
model, scaler = load_model()

# --- UI ---
st.title("SME Payment Risk Analytics - V3.0 Pure ML")
st.subheader("Live credit risk scoring - 100% ML Model, No Rules + SQL Production Ready")
st.success(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} cols")

st.dataframe(df.head(100))

# Example scoring - adjust columns as per your model
if model is not None:
    st.sidebar.header("Live Risk Scoring")
    # yaha apne features daal de
    st.sidebar.write("Model loaded:", type(model).__name__)