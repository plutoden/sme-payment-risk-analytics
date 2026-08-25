import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from sqlalchemy import create_engine, URL

st.set_page_config(page_title="SME Payment Risk", layout="wide")
st.title("💰 SME Payment Risk Analytics")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"

with open(MODEL_PATH,'rb') as f:
    model = pickle.load(f)

engine = create_engine(URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password="Mypassword",
    host="localhost",
    port=5432,
    database="sme_payment_risk"
))

tab1, tab2 = st.tabs(["Risk Predictor", "SQL Analytics"])

with tab1:
    amt = st.number_input("Invoice Amount", 1000, 1000000, 50000)
    month = st.slider("Posting Month", 1, 12, 6)
    is_open = st.selectbox("Is Open Invoice?", [0,1])
    if st.button("Predict Risk"):
        pred = model.predict([[amt, month, is_open]])[0]
        if pred==1:
            st.error("🔴 HIGH RISK - Delay hoga!")
        else:
            st.success("🟢 LOW RISK - Time pe payment")

with tab2:
    st.subheader("Top 5 Risky Customers (SQL se)")
    try:
        df_sql = pd.read_sql("SELECT cust_number, COUNT(*) as delayed FROM invoices WHERE is_delayed=1 GROUP BY cust_number ORDER BY delayed DESC LIMIT 5", engine)
        if df_sql.empty:
            st.warning("Table empty hai. 01 notebook wala to_sql wala cell dobara run karo.")
        else:
            st.dataframe(df_sql)
            st.bar_chart(df_sql.set_index('cust_number'))
    except Exception as e:
        st.error(f"SQL Error: {e}")