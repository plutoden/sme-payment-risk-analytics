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

st.title("SME Payment Risk Analytics - V3.0 Pure ML")
st.markdown("Live credit risk scoring - 100% ML Model, No Rules + SQL Production Ready")
st.success(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} cols")

# --- SIDEBAR PREDICTION ---
st.sidebar.title("Live Risk Scoring")
st.sidebar.success(f"Model loaded:\n{type(model).__name__}")

# Adjust features as per your training
amount = st.sidebar.number_input("total_open_amount", value=50000.0)
terms = st.sidebar.selectbox("cust_payment_terms", options=df['cust_payment_terms'].dropna().unique()[:20])
currency = st.sidebar.selectbox("invoice_currency", options=df['invoice_currency'].unique())
business_code = st.sidebar.selectbox("business_code", options=df['business_code'].unique())

if st.sidebar.button("Predict Risk"):
    # Make a dummy input row - map to your model features
    # IMPORTANT: yahan apne actual training features use kar
    try:
        # Example: if your model expects numeric features
        input_df = pd.DataFrame([{
            'total_open_amount': amount,
            # Add other numeric features with median
        }])
        # Agar scaler hai toh use karo
        # Fill missing cols with median from df
        for col in df.select_dtypes(include='number').columns:
            if col not in input_df.columns and col in ['total_open_amount']:
                continue
            if col not in input_df.columns:
                input_df[col] = df[col].median()

        # Keep only columns model was trained on (if you saved feature list)
        # input_df = input_df[model.feature_names_in_] # uncomment if available

        if scaler:
            prob = model.predict_proba(scaler.transform(input_df))[:,1][0] if hasattr(model, "predict_proba") else model.predict(input_df)[0]
        else:
            prob = model.predict_proba(input_df)[:,1][0] if hasattr(model, "predict_proba") else float(model.predict(input_df)[0])

        st.sidebar.metric("Default Probability", f"{prob:.2%}")
        if prob > 0.7: st.sidebar.error("HIGH RISK")
        elif prob > 0.4: st.sidebar.warning("MEDIUM RISK")
        else: st.sidebar.success("LOW RISK")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        st.sidebar.write("Tip: check model.feature_names_in_:", getattr(model, "feature_names_in_", "not saved"))

st.dataframe(df.head(100))