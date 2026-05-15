import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(
    title="Credit Scoring API",
    description="API de scoring crédit — Credit Decision",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def probability_to_score(p, base_score=600, PDO=50, base_odds=1):
    if p <= 0 or p >= 1:
        return base_score
    odds = (1 - p) / p
    return round(base_score + PDO * np.log(odds / base_odds), 2)

class ClientData(BaseModel):
    person_income: float
    loan_percent_income: float
    loan_int_rate: float
    loan_amnt: float
    person_emp_length: float
    person_age: float
    cb_person_default_on_file_Y: float
    person_home_ownership_RENT: float
    person_home_ownership_OWN: float
    loan_grade_D: float
    loan_grade_E: float
    loan_grade_B: float
    loan_grade_F: float
    loan_grade_G: float
    loan_intent_VENTURE: float

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/predict")
def predict(client: ClientData):
    columns = [
        'person_income', 'loan_percent_income', 'loan_int_rate', 'loan_amnt',
        'person_emp_length', 'person_age', 'cb_person_default_on_file_Y',
        'person_home_ownership_RENT', 'person_home_ownership_OWN', 'loan_grade_D',
        'loan_grade_E', 'loan_grade_B', 'loan_grade_F', 'loan_grade_G',
        'loan_intent_VENTURE'
    ]
    X = pd.DataFrame([[
        client.person_income, client.loan_percent_income, client.loan_int_rate,
        client.loan_amnt, client.person_emp_length, client.person_age,
        client.cb_person_default_on_file_Y, client.person_home_ownership_RENT,
        client.person_home_ownership_OWN, client.loan_grade_D, client.loan_grade_E,
        client.loan_grade_B, client.loan_grade_F, client.loan_grade_G,
        client.loan_intent_VENTURE
    ]], columns=columns)

    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    score = probability_to_score(prob)
    decision = "Accepté ✅" if prob < 0.5 else "Refusé ❌"

    return {
        "probabilite_defaut": round(float(prob), 4),
        "credit_score": round(float(score), 2),
        "decision": decision
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)