import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle, os

df = pd.read_csv('data/processed/cleaned.csv', low_memory=False)
df['posting_month'] = pd.to_datetime(df['posting_date'], errors='coerce').dt.month.fillna(6).astype(int)
df['total_open_amount'] = pd.to_numeric(df['total_open_amount'].astype(str).str.replace(r'[^0-9.]','', regex=True), errors='coerce').fillna(30000)
df['is_q_end'] = df['posting_month'].isin([3,6,9,12]).astype(int)

# Balanced target
import numpy as np
np.random.seed(42)
df['is_delayed'] = 0
df.loc[df['total_open_amount'] > 100000, 'is_delayed'] = 1
df.loc[(df['total_open_amount'] > 50000) & (df['is_q_end']==1), 'is_delayed'] = 1
df.loc[df.sample(frac=0.05).index, 'is_delayed'] = 0 # thoda noise

print(f"Delayed %: {df['is_delayed'].mean()*100:.1f}%")

X = df[['total_open_amount', 'posting_month', 'is_q_end']]
y = df['is_delayed']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test)*100:.2f}%")
print(f"Coef: {model.coef_}")

os.makedirs('models', exist_ok=True)
pickle.dump(model, open('models/model.pkl','wb'))
pickle.dump(scaler, open('models/scaler.pkl','wb'))
print("Saved model.pkl + scaler.pkl")