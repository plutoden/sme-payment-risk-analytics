import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="SME Payment Risk", layout="wide")
st.title("SME Payment Risk Analytics")
st.markdown("Live credit risk scoring for SME invoices")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned.csv"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

model = load_model()
df = load_data()

# Auto-detect column names
delay_col = None
for c in ['is_delayed', 'isOpen', 'is_open', 'delay_flag', 'risk_flag']:
    if c in df.columns:
        delay_col = c
        break

cust_col = 'cust_number' if 'cust_number' in df.columns else df.columns[0]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Invoices", f"{len(df):,}")
col2.metric("Total Customers", f"{df[cust_col].nunique():,}" if cust_col in df.columns else "N/A")
if delay_col:
    delay_rate = df[delay_col].mean()*100 if df[delay_col].dtype!= 'O' else 0
    col3.metric("Delayed %", f"{delay_rate:.1f}%")
col4.metric("Avg Amount", f"₹{df['amount'].mean():,.0f}" if 'amount' in df.columns else f"₹{df.iloc[:,1].mean():,.0f}")

tab1, tab2 = st.tabs(["Risk Predictor", "Data Analytics"])

with tab1:
    st.subheader("Predict Payment Delay")
    c1, c2, c3 = st.columns(3)
    amt = c1.number_input("Invoice Amount", 1000, 10000000, 50000)
    month = c2.slider("Posting Month", 1, 12, 6)
    is_open = c3.selectbox("Is Open Invoice?", [0, 1])
    if st.button("Predict Risk", type="primary"):
        pred = model.predict([[amt, month, is_open]])[0]
        proba = model.predict_proba([[amt, month, is_open]])[0].max()
        if pred == 1:
            st.error(f"HIGH RISK - Delay Probability: {proba*100:.1f}%")
        else:
            st.success(f"LOW RISK - On-time Probability: {proba*100:.1f}%")

with tab2:
    st.subheader("Risk Insights")
    if delay_col:
        top_customers = df[df[delay_col]==1].groupby(cust_col).size().reset_index(name='delayed_count')
        top_customers = top_customers.sort_values(by='delayed_count', ascending=False).head(5)
        if not top_customers.empty:
            st.write("**Top 5 Risky Customers**")
            st.dataframe(top_customers, use_container_width=True)
            fig = px.bar(top_customers, x=cust_col, y='delayed_count', color='delayed_count', title="Delayed Invoices by Customer")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No delayed records found in {delay_col}. Showing distribution instead.")
            fig = px.histogram(df, x=cust_col, title="Invoices per Customer")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df.head(100), use_container_width=True)
        st.write("Columns found:", list(df.columns))

    st.write("---")
    st.dataframe(df.head(), use_container_width=True)