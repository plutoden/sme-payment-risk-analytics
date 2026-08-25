# SME Payment Risk Analytics and Delay Prediction

An end-to-end data analytics project to identify high-risk SME customers and predict payment delays. The project analyzes 48,839 invoices using Python, PostgreSQL, and Machine Learning with an interactive Streamlit dashboard.

Live Dashboard: (Will be added after Streamlit deployment)

## Business Problem
Small and Medium Enterprises face severe cash flow issues due to delayed payments. This project addresses:
- Which customers consistently delay payments?
- What factors drive payment delays?
- Can we predict risky invoices before the due date?

## Key Findings from SQL Analytics
Analysis performed on PostgreSQL:

- Top Risky Customer: Customer ID 200769623 with 3511 delayed invoices
- Overall Risk Rate: 30.2 percent of invoices flagged as delayed
- Primary Risk Drivers: total_open_amount and no_of_invoices

SQL concepts used: GROUP BY, Aggregations, JOINs, Subqueries, Filtering.

## Tech Stack
- Programming: Python 3.11, Pandas, NumPy
- Database: PostgreSQL, SQLAlchemy
- Machine Learning: Scikit-learn, Random Forest Classifier
- Visualization: Matplotlib, Seaborn
- Dashboard: Streamlit

## Project Structure
- data/: Raw and cleaned invoice data
- notebooks/: 01_data_understanding.ipynb, 02_model_training.ipynb
- dashboard/app.py: Main Streamlit application
- models/: Trained model file (payment_risk_model.pkl)
- push.py: Script for bulk ingestion from CSV to PostgreSQL
- requirements.txt: Project dependencies

## Model Performance
- Model: Random Forest Classifier
- Accuracy: 89 percent
- Features: total_open_amount, no_of_invoices, isOpen, customer history
- Target Variable: is_delayed

## How to Run Locally
pip install -r requirements.txt
streamlit run dashboard/app.py

## Author
Suraj - Aspiring Data Analyst
Skills: Python, SQL, Machine Learning, Data Visualization, Streamlit

LinkedIn: [Add your LinkedIn URL]
