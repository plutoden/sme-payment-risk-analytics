import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="SME Payment Risk", layout="wide")
st.title("SME Payment Risk Analytics")
st.caption("Live credit risk scoring for SME invoices")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned.csv"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    return df

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

# --- SAFE KPIs ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Invoices", f"{len(df):,}")

# Find numeric amount column
amount_col = None
for col in ['amount','invoice_amount','total_amount','Total open amount','InvoiceAmount']:
    if col in df.columns:
        amount_col = col
        break
if not amount_col:
    # try second column if numeric
    amount_col = df.select_dtypes(include='number').columns[0] if len(df.select_dtypes(include='number').columns)>0 else None

if amount_col:
    df[amount_col] = pd.to_numeric(df[amount_col].astype(str).str.replace(r'[^0-9.]','', regex=True), errors='coerce')
    avg_amt = df[amount_col].mean()
    c2.metric("Avg Amount", f"₹{avg_amt:,.0f}" if pd.notna(avg_amt) else "N/A")
else:
    c2.metric("Avg Amount", "N/A")

# Customer count
cust_col = 'cust_number' if 'cust_number' in df.columns else df.columns[0]
c3.metric("Total Customers", f"{df[cust_col].nunique():,}")

delay_col = next((c for c in ['is_delayed','isOpen','is_open','delay_flag'] if c in df.columns), None)
if delay_col:
    try:
        df[delay_col] = pd.to_numeric(df[delay_col], errors='coerce')
        c4.metric("Delayed %", f"{df[delay_col].mean()*100:.1f}%")
    except:
        c4.metric("Delayed %", "N/A")
else:
    c4.metric("Status", "Live")

tab1, tab2 = st.tabs(["Risk Predictor", "Data Analytics"])

with tab1:
    st.subheader("Predict Payment Delay")
    amt = st.number_input("Invoice Amount", 1000, 10000000, 50000)
    month = st.slider("Posting Month", 1, 12, 6)
    is_open = st.selectbox("Is Open Invoice?", [0, 1])
    if st.button("Predict Risk", type="primary"):
        try:
            pred = model.predict([[amt, month, is_open]])[0]
            proba = model.predict_proba([[amt, month, is_open]])[0].max()
            if pred == 1:
                st.error(f"HIGH RISK - Delay Prob: {proba*100:.1f}%")
            else:
                st.success(f"LOW RISK - On-time Prob: {proba*100:.1f}%")
        except Exception as e:
            st.error(f"Model error: {e}. Retrain with 3 features [amount, month, is_open]")

with tab2:
    st.subheader("Insights")
    st.write("Columns in data:", list(df.columns))
    st.dataframe(df.head(100), use_container_width=True)
    if delay_col and cust_col in df.columns:
        try:
            top = df[df[delay_col]==1].groupby(cust_col).size().reset_index(name='delayed_count').sort_values('delayed_count', ascending=False).head(10)
            if not top.empty:
                fig = px.bar(top, x=cust_col, y='delayed_count', title="Top Risky Customers")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart skipped: {e}")