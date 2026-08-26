import os, joblib, pandas as pd
import streamlit as st

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
    return joblib.load(mp), joblib.load(sp) if sp else None

df = load_data()
model, scaler = load_model()
expected = list(getattr(model, "feature_names_in_", []))

st.title("SME Payment Risk Analytics - V3.0 Pure ML")
st.success(f"Data loaded: {df.shape[0]} rows | Model expects {len(expected)} features")

st.sidebar.title("Live Risk Scoring")
# --- Correct inputs exactly as model expects ---
inputs = {}
for col in expected:
    if col == "total_open_amount":
        inputs[col] = st.sidebar.number_input(col, value=50000.0)
    elif col == "buisness_year":
        inputs[col] = st.sidebar.number_input(col, value=2020)
    elif col == "posting_month":
        inputs[col] = st.sidebar.slider(col, 1, 12, 6)
    elif col == "delay_days":
        inputs[col] = st.sidebar.slider(col, 0, 180, 30)
    elif col in ["is_q_end", "isOpen", "is_open", "is_delayed", "isOpen"]:
        inputs[col] = int(st.sidebar.checkbox(col, value=True if col=="isOpen" else False))
    elif col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            inputs[col] = float(df[col].median())
        else:
            inputs[col] = 0
    else:
        inputs[col] = 0

if st.sidebar.button("Predict Risk", type="primary"):
    input_df = pd.DataFrame([inputs])
    input_df = input_df[expected] # exact order

    try:
        X = scaler.transform(input_df) if scaler else input_df
        proba = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else float(model.predict(X)[0])
        st.sidebar.metric("Default Probability", f"{proba:.2%}")
        if proba > 0.7: st.sidebar.error("🔴 HIGH RISK")
        elif proba > 0.4: st.sidebar.warning("🟡 MEDIUM RISK")
        else: st.sidebar.success("🟢 LOW RISK")
        st.sidebar.dataframe(input_df.T)
    except Exception as e:
        st.sidebar.error(e)
        st.write("Expected:", expected)

st.dataframe(df.head(100))