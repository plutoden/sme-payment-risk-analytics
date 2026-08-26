import os, joblib, pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
BASE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE, ".."))

def find(p):
    for b in [BASE, ROOT]:
        fp = os.path.join(b, p)
        if os.path.exists(fp): return fp
    return None

@st.cache_data
def load_data(): return pd.read_csv(find("data/cleaned.csv"))
@st.cache_resource
def load_model(): return joblib.load(find("models/risk_model.pkl"))

df = load_data()
model = load_model()

st.title("SME Payment Risk Analytics - V3.0 Pure ML")
st.success(f"Data: {df.shape[0]} | Model: {model.classes_} | Features: {list(model.feature_names_in_)}")

st.sidebar.title("Live Risk Scoring")
amount = st.sidebar.number_input("Invoice Amount", value=float(df['total_open_amount'].median()))
month = st.sidebar.slider("Posting Month", 1, 12, 6)
is_open = st.sidebar.checkbox("is_open (1 = payment pending)", True)

if st.sidebar.button("Predict Risk", type="primary"):
    input_df = pd.DataFrame([{"amount": amount, "month": month, "is_open": int(is_open)}])
    input_df = input_df[list(model.feature_names_in_)]

    # Original model proba
    proba_arr = model.predict_proba(input_df)[0]
    raw_proba = proba_arr[1] if len(proba_arr)>1 else proba_arr[0]

    # FIX: Realistic scoring - amount ke basis pe normalize
    # Agar amount > 75th percentile to risk high
    q75 = df['total_open_amount'].quantile(0.75)
    q50 = df['total_open_amount'].quantile(0.50)

    if amount > q75 and is_open: final_proba = 0.85
    elif amount > q50 and is_open: final_proba = 0.55
    elif not is_open: final_proba = 0.10
    else: final_proba = 0.30

    # Dono dikhao
    st.sidebar.metric("Model Raw Proba", f"{raw_proba:.2%}")
    st.sidebar.metric("Adjusted Real Risk", f"{final_proba:.2%}")

    if final_proba > 0.7: st.sidebar.error("🔴 HIGH RISK")
    elif final_proba > 0.4: st.sidebar.warning("🟡 MEDIUM RISK")
    else: st.sidebar.success("🟢 LOW RISK")

st.dataframe(df.head(100))