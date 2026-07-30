import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session
import hashlib


from database import engine, get_db, Base
from models import User, Analysis, Log

# ── INIT DB ──────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── LOAD MODEL ───────────────────────────────────────────
model  = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")
from fastapi.staticfiles import StaticFiles
app = FastAPI(
    title="Credit Scoring API",
    description="API de scoring crédit avec PostgreSQL",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# ── HELPERS ───────────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def probability_to_score(p, base_score=600, PDO=50, base_odds=1):
    if p <= 0 or p >= 1:
        return base_score
    odds = (1 - p) / p
    return round(base_score + PDO * np.log(odds / base_odds), 2)

def add_log(db: Session, user_id, method, endpoint, status_code, detail=None, ip=None):
    log = Log(
        user_id=user_id,
        method=method,
        endpoint=endpoint,
        status_code=status_code,
        detail=detail,
        ip_address=ip
    )
    db.add(log)
    db.commit()

# ── SCHEMAS ───────────────────────────────────────────────
class ClientData(BaseModel):
    person_income:              float
    loan_percent_income:        float
    loan_int_rate:              float
    loan_amnt:                  float
    person_emp_length:          float
    person_age:                 float
    cb_person_default_on_file_Y: float
    person_home_ownership_RENT: float
    person_home_ownership_OWN:  float
    loan_grade_D:               float
    loan_grade_E:               float
    loan_grade_B:               float
    loan_grade_F:               float
    loan_grade_G:               float
    loan_intent_VENTURE:        float
    username:                   Optional[str] = None

class UserCreate(BaseModel):
    username: str
    password: str
    role:     Optional[str] = "analyst"

class LoginData(BaseModel):
    username: str
    password: str

class AnalysisOut(BaseModel):
    id:               int
    credit_score:     float
    probabilite_defaut: float
    decision:         str
    risk_level:       str
    person_income:    float
    loan_amnt:        float
    loan_grade:       str
    created_at:       datetime
    username:         Optional[str]

    class Config:
        from_attributes = True

class LogOut(BaseModel):
    id:         int
    method:     str
    endpoint:   str
    status_code: int
    detail:     Optional[str]
    created_at: datetime
    username:   Optional[str]

    class Config:
        from_attributes = True

# ── ROUTES ────────────────────────────────────────────────

@app.get("/")
def home():
    return FileResponse("index.html")

# ── AUTH ──────────────────────────────────────────────────
@app.post("/auth/register")
def register(data: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = User(
        username=data.username,
        password=hash_password(data.password),
        role=data.role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User created", "id": user.id, "username": user.username}

@app.post("/auth/login")
def login(data: LoginData, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user or user.password != hash_password(data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    user.last_login = datetime.utcnow()
    db.commit()

    add_log(db, user.id, "POST", "/auth/login", 200,
            detail=f"Login: {user.username}", ip=request.client.host)

    return {
        "id":       user.id,
        "username": user.username,
        "role":     user.role,
        "message":  "Login successful"
    }

# ── PREDICT ───────────────────────────────────────────────
@app.post("/predict")
def predict(client: ClientData, request: Request, db: Session = Depends(get_db)):
    columns = [
        'person_income','loan_percent_income','loan_int_rate','loan_amnt',
        'person_emp_length','person_age','cb_person_default_on_file_Y',
        'person_home_ownership_RENT','person_home_ownership_OWN',
        'loan_grade_D','loan_grade_E','loan_grade_B','loan_grade_F',
        'loan_grade_G','loan_intent_VENTURE'
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
    prob      = model.predict_proba(X_scaled)[0][1]
    score     = probability_to_score(prob)
    decision  = "Accepté" if prob < 0.5 else "Refusé"
    risk      = "Faible" if prob < 0.2 else ("Modéré" if prob < 0.5 else "Élevé")

    # Determine grade from inputs
    grade = "A/C"
    if client.loan_grade_B: grade = "B"
    elif client.loan_grade_D: grade = "D"
    elif client.loan_grade_E: grade = "E"
    elif client.loan_grade_F: grade = "F"
    elif client.loan_grade_G: grade = "G"

    housing = "RENT" if client.person_home_ownership_RENT else ("OWN" if client.person_home_ownership_OWN else "MORTGAGE")

    # Save to DB
    user_id = None
    if client.username:
        u = db.query(User).filter(User.username == client.username).first()
        if u:
            user_id = u.id

    analysis = Analysis(
        user_id=user_id,
        person_income=client.person_income,
        person_age=int(client.person_age),
        person_emp_length=client.person_emp_length,
        loan_amnt=client.loan_amnt,
        loan_int_rate=client.loan_int_rate,
        loan_percent_income=client.loan_percent_income,
        loan_grade=grade,
        person_home_ownership=housing,
        cb_person_default_on_file=bool(client.cb_person_default_on_file_Y),
        loan_intent_venture=bool(client.loan_intent_VENTURE),
        credit_score=round(float(score), 2),
        probabilite_defaut=round(float(prob), 4),
        decision=decision,
        risk_level=risk
    )
    db.add(analysis)
    db.commit()

    add_log(db, user_id, "POST", "/predict", 200,
            detail=f"Score={score:.0f} Prob={prob:.2%}", ip=request.client.host)

    return {
        "probabilite_defaut": round(float(prob), 4),
        "credit_score":       round(float(score), 2),
        "decision":           decision,
        "risk_level":         risk,
        "analysis_id":        analysis.id
    }

# ── HISTORY ───────────────────────────────────────────────
@app.get("/history", response_model=List[AnalysisOut])
def get_history(
    limit:    int = 100,
    username: Optional[str] = None,
    db: Session = Depends(get_db)
):
    q = db.query(Analysis)
    if username:
        user = db.query(User).filter(User.username == username).first()
        if user:
            q = q.filter(Analysis.user_id == user.id)
    analyses = q.order_by(Analysis.created_at.desc()).limit(limit).all()

    result = []
    for a in analyses:
        result.append(AnalysisOut(
            id=a.id,
            credit_score=a.credit_score,
            probabilite_defaut=a.probabilite_defaut,
            decision=a.decision,
            risk_level=a.risk_level,
            person_income=a.person_income,
            loan_amnt=a.loan_amnt,
            loan_grade=a.loan_grade,
            created_at=a.created_at,
            username=a.user.username if a.user else None
        ))
    return result

@app.delete("/history/{analysis_id}")
def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    db.delete(analysis)
    db.commit()
    return {"message": "Deleted"}

# ── STATS ─────────────────────────────────────────────────
@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    analyses = db.query(Analysis).all()
    if not analyses:
        return {"total": 0}

    scores = [a.credit_score for a in analyses]
    return {
        "total":          len(analyses),
        "avg_score":      round(sum(scores) / len(scores), 2),
        "approved":       sum(1 for a in analyses if a.decision == "Accepté"),
        "refused":        sum(1 for a in analyses if a.decision == "Refusé"),
        "risk_low":       sum(1 for a in analyses if a.risk_level == "Faible"),
        "risk_medium":    sum(1 for a in analyses if a.risk_level == "Modéré"),
        "risk_high":      sum(1 for a in analyses if a.risk_level == "Élevé"),
        "min_score":      min(scores),
        "max_score":      max(scores),
    }

# ── USERS ─────────────────────────────────────────────────
@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"id": u.id, "username": u.username, "role": u.role,
             "is_active": u.is_active, "created_at": u.created_at,
             "last_login": u.last_login} for u in users]

@app.patch("/users/{user_id}/toggle")
def toggle_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = not user.is_active
    db.commit()
    return {"message": "Updated", "is_active": user.is_active}

# ── LOGS ──────────────────────────────────────────────────
@app.get("/logs", response_model=List[LogOut])
def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    logs = db.query(Log).order_by(Log.created_at.desc()).limit(limit).all()
    return [LogOut(
        id=l.id, method=l.method, endpoint=l.endpoint,
        status_code=l.status_code, detail=l.detail,
        created_at=l.created_at,
        username=l.user.username if l.user else None
    ) for l in logs]

@app.delete("/logs")
def clear_logs(db: Session = Depends(get_db)):
    db.query(Log).delete()
    db.commit()
    return {"message": "Logs cleared"}

# ── SEED DEFAULT USERS ────────────────────────────────────
@app.post("/seed")
def seed_users(db: Session = Depends(get_db)):
    defaults = [
        {"username": "admin",   "password": "admin123",   "role": "admin"},
        {"username": "manager", "password": "manager123", "role": "manager"},
        {"username": "analyst", "password": "analyst123", "role": "analyst"},
    ]
    created = []
    for d in defaults:
        exists = db.query(User).filter(User.username == d["username"]).first()
        if not exists:
            u = User(username=d["username"], password=hash_password(d["password"]), role=d["role"])
            db.add(u)
            created.append(d["username"])
    db.commit()
    return {"message": "Seeded", "created": created}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8015)
