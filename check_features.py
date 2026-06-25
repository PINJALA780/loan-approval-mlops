import pandas as pd

df = pd.read_csv("data/processed/processed.csv")

df.columns = df.columns.str.strip()

if "loan_id" in df.columns:
    df = df.drop(columns=["loan_id"])

X = df.drop(columns=["loan_status"])

print(X.columns.tolist())
