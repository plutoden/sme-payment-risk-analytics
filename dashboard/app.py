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
    p = find_file(["data/cleaned.csv","data/processed/cleaned.csv","../data/cleaned.csv"])
    return pd.read_csv(p)

@st.cache_resource
def load_model():
    mp = find_file(["models/risk_model.pkl","../models/risk_model.pkl","models/model.pkl"])
    return joblib.load(mp)

df = load_data()
model = load_model()
expected = list(model.feature_names_in_)

st.title("SME Payment Risk Analytics - V3.0 Pure ML")
st.success(f"Data: {df.shape[0]} rows | Model: {type(model).__name__} | Features: {expected}")

# --- SIDEBAR - Fix: amount, month inputs dikhao ---
st.sidebar.title("Live Risk Scoring")

amount = st.sidebar.number_input("amount (total_open_amount)", value=50000.0, min_value=0.0)
month = st.sidebar.slider("month (posting_month)", 1, 12, 6)
is_open = st.sidebar.checkbox("is_open", value=True)

if st.sidebar.button("Predict Risk", type="primary"):
    # Exact order me DataFrame banao
    data = {}
    for col in expected:
        if col == "amount": data[col] = amount
        elif col == "month": data[col] = month
        elif col == "is_open": data[col] = int(is_open)
        else: data[col] = 0

    input_df = pd.DataFrame([data])[expected]

    try:
        proba_arr = model.predict_proba(input_df)[0]
        # Fix IndexError: agar 1 hi class hai to proba_arr len 1 hoga
        proba = proba_arr[1] if len(proba_arr) > 1 else proba_arr[0]
        pred = model.predict(input_df)[0]

        st.sidebar.metric("Default Probability", f"{proba:.2%}")
        st.sidebar.metric("Prediction", str(pred))

        if proba > 0.7: st.sidebar.error("🔴 HIGH RISK")
        elif proba > 0.4: st.sidebar.warning("🟡 MEDIUM RISK")
        else: st.sidebar.success("🟢 LOW RISK")

        st.sidebar.write("Input:")
        st.sidebar.dataframe(input_df)
        st.sidebar.write("proba array:", proba_arr)
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

st.dataframe(df.head(100))
st.write("Model classes:", getattr(model, "classes_", "unknown"))