from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Loan Approval Prediction API",
    version="1.0"
)

model = joblib.load("models/model.pkl")


class LoanRequest(BaseModel):
    no_of_dependents: int
    education: int
    self_employed: int
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


@app.get("/")
def home():
    return {
        "message": "Loan Approval Prediction API Running"
    }


@app.post("/predict")
def predict(data: LoanRequest):

    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }
