import streamlit as st
import pandas as pd
import pickle
import sqlite3
import os

st.set_page_config(page_title="SME Payment Risk Analytics", layout="wide")
st.title("SME Payment Risk Analytics - V3.0 Pure ML")
st.caption("Live credit risk scoring - 100% ML Model, No Rules + SQL Production Ready")

# Load data - Fixed path for Streamlit Cloud
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "cleaned.csv")
    return pd.read_csv(csv_path)

df = load_data()

# Load model - Fixed path
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "model.pkl")
    scaler_path = os.path.join(base_dir, "..", "models", "scaler.pkl")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_model()

# Metrics
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Invoices", f"{len(df):,}")
c2.metric("Avg Amount", f"₹{df['total_open_amount'].mean():,.0f}")
c3.metric("Total Customers", f"{df['cust_number'].nunique():,}")
c4.metric("Delayed %", f"{df['is_open'].mean()*100:.1f}%")

tab1, tab2, tab3 = st.tabs(["Risk Predictor (ML)", "Data Analytics", "SQL Scoring (For Interview)"])

with tab1:
    st.subheader("Predict Payment Delay - Pure ML Model")
    if model:
        st.success(f"✅ ML Model Loaded - Trained on {len(df)} Invoices")
    else:
        st.error("Model not found")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Invoice Amount (₹)", value=50000, min_value=1000)
    with col2:
        month = st.selectbox("Posting Month", ["January","February","March","April","May","June","July","August","September","October","November","December"])
        month_map = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
        month_num = month_map[month]

    if st.button("Predict Risk", use_container_width=True, type="primary"):
        if model:
            import numpy as np
            is_q_end = 1 if month_num in [3,6,9,12] else 0
            features = np.array([[amount, month_num, is_q_end]])
            try:
                prob = model.predict_proba(scaler.transform(features))[0][1]
            except:
                prob = model.predict_proba(features)[0][1]
            risk_pct = prob*100
            if risk_pct > 70:
                st.error(f"🔴 HIGH RISK: {risk_pct:.1f}%")
            elif risk_pct > 40:
                st.warning(f"🟡 MEDIUM RISK: {risk_pct:.1f}%")
            else:
                st.success(f"🟢 LOW RISK: {risk_pct:.1f}%")

with tab2:
    st.subheader("Data Analytics")
    st.bar_chart(df.groupby('posting_month')['is_open'].mean())
    st.dataframe(df.head(100), use_container_width=True)

with tab3:
    st.subheader("SQL Based Risk Scoring - Production Approach")
    st.caption("Same logic jo Bank ke core system me chalega")
    conn = sqlite3.connect(':memory:')
    df.to_sql('invoices', conn, index=False, if_exists='replace')
    sql_query = """
    SELECT
        cust_number,
        total_open_amount,
        CASE
            WHEN total_open_amount > 100000 THEN 85
            WHEN total_open_amount > 75000 THEN 75
            WHEN total_open_amount > 50000 THEN 55
            ELSE 15
        END as risk_score,
        CASE
            WHEN total_open_amount > 100000 THEN 'HIGH'
            WHEN total_open_amount > 50000 THEN 'MEDIUM'
            ELSE 'LOW'
        END as risk_category
    FROM invoices
    ORDER BY risk_score DESC
    LIMIT 100
    """
    st.code(sql_query, language='sql')
    sql_df = pd.read_sql_query(sql_query, conn)
    st.dataframe(sql_df, use_container_width=True)
    st.info("Interview Point: Python model for data science, SQL view for production deployment in banking system")