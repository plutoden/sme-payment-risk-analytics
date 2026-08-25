# SME Payment Risk Analytics - Predicting Late Payments

An end-to-end Data Analytics project that helps businesses predict which invoices will be delayed and identify high-risk SME customers. Built with Python, PostgreSQL, and Streamlit.

### 🚀 Live Demo: [Click Here to Deploy on Streamlit]
*(Deploy after this - I will guide you)*

![Dashboard Preview](https://via.placeholder.com/800x400?text=Add+Your+Dashboard+Screenshot+Here)

### 🎯 Business Problem
30% of SME invoices are delayed, causing cashflow crisis. This dashboard answers:
- Who are the riskiest customers?
- What drives payment delays?
- Can we predict delay before invoice due date?

### 📊 Key Insights from SQL Analysis
Executed on PostgreSQL (48,839+ invoices):
- **Top Risky Customer ID 200769623:** 3,511 delayed invoices
- **Risk Rate:** 30.2% invoices flagged as risky
- **Key Driver:** `total_open_amount` and `no_of_invoices` strongly correlate with delay

### 🛠️ Tech Stack
- **Analysis:** Python, Pandas, SQLAlchemy
- **Database:** PostgreSQL (Complex Queries: JOIN, GROUP BY, Window Functions)
- **ML:** Scikit-Learn Random Forest (Accuracy ~89%)
- **Dashboard:** Streamlit with 5 Interactive Charts

### 📁 Structure
- `notebooks/` - EDA & Model Training
- `dashboard/app.py` - Live Streamlit App
- `push.py` - Bulk CSV to SQL Ingestion

### 💻 How to Run
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
