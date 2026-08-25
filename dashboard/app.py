import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="SME Payment Risk", layout="wide")
st.title("SME Payment Risk Analytics")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"
DATA_PATH_1 = BASE_DIR / "data" / "cleaned_invoices.csv"
DATA_PATH_2 = BASE_DIR / "data" / "sme_payments.csv"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

model = load_model()

@st.cache_data
def load_data():
    if DATA_PATH_1.exists():
        return pd.read_csv(DATA_PATH_1)
    elif DATA_PATH_2.exists():
        return pd.read_csv(DATA_PATH_2)
    else:
        # try relative path for streamlit cloud
        return pd.read_csv("data/cleaned_invoices.csv")

df = load_data()

tab1, tab2 = st.tabs(["Risk Predictor", "Data Analytics"])

with tab1:
    st.subheader("Predict Payment Delay")
    amt = st.number_input("Invoice Amount", 1000, 1000000, 50000)
    month = st.slider("Posting Month", 1, 12, 6)
    is_open = st.selectbox("Is Open Invoice?", [0, 1])

    if st.button("Predict Risk"):
        pred = model.predict([[amt, month, is_open]])[0]
        if pred == 1:
            st.error("HIGH RISK - Payment likely to be delayed")
        else:
            st.success("LOW RISK - Payment likely on time")

with tab2:
    st.subheader("Top 5 Risky Customers")
    if 'cust_number' in df.columns and 'is_delayed' in df.columns:
        top_customers = df[df['is_delayed']==1].groupby('cust_number').size().reset_index(name='delayed_count')
        top_customers = top_customers.sort_values(by='delayed_count', ascending=False).head(5)
        st.dataframe(top_customers)
        st.bar_chart(top_customers.set_index('cust_number'))
    else:
        st.warning("Required columns not found. Showing data overview.")
        st.dataframe(df.head())
        st.write(df.describe())