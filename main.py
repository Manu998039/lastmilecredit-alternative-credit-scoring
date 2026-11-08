"""LastMileCredit backend - FastAPI.
Run locally:  uvicorn main:app --reload
"""
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DB_PATH = os.getenv("DB_PATH", "microlend.db")
USE_SHAP = os.getenv("USE_SHAP", "0") == "1"   # off by default: saves memory on free tier
WORKING_DAYS = 26

bundle = joblib.load("model.joblib")
model, FEATURES = bundle["model"], bundle["features"]
BIZ_TYPES, PURPOSES = bundle["biz_types"], bundle["purposes"]

explainer = None
if USE_SHAP:
    import shap
    explainer = shap.TreeExplainer(model)

LABELS = {
    "business_age_years": "Business vintage",
    "daily_upi_sales": "Daily UPI sales",
    "utility_bill_score": "Utility bill history",
    "supplier_repay_score": "Supplier repayment",
    "loan_to_revenue_ratio": "Loan-to-revenue ratio",
    "biz_type_encoded": "Business type",
    "city_tier": "City tier",
    "purpose_encoded": "Loan purpose",
}

# Illustrative lender network (8 lenders). Thresholds/ceilings are sample values.
LENDERS = [
    {"name": "Bandhan Bank",          "type": "Bank",            "min_score": 70, "max_loan": 500_000},
    {"name": "Ujjivan Small Finance Bank", "type": "Small finance bank", "min_score": 65, "max_loan": 300_000},
    {"name": "Lendingkart",           "type": "Fintech NBFC",    "min_score": 60, "max_loan": 1_000_000},
    {"name": "Aye Finance",           "type": "Fintech NBFC",    "min_score": 55, "max_loan": 500_000},
    {"name": "Indifi",                "type": "Fintech NBFC",    "min_score": 55, "max_loan": 750_000},
    {"name": "Arohan MFI",            "type": "Microfinance",    "min_score": 45, "max_loan": 100_000},
    {"name": "Mudra Shishu / Kishore","type": "Govt scheme",     "min_score": 0,  "max_loan": 500_000},
    {"name": "PM SVANidhi",           "type": "Govt scheme",     "min_score": 0,  "max_loan": 50_000},
]

app = FastAPI(title="LastMileCredit API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- database ----------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(db()) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            business_type TEXT, city TEXT, business_age_years REAL, daily_upi_sales REAL,
            utility_bill_score INTEGER, supplier_repay_score INTEGER,
            loan_amount_requested REAL, loan_purpose TEXT,
            credit_score INTEGER, verdict TEXT, risk_band TEXT, approved_amount REAL,
            result_json TEXT, created_at TIMESTAMP)""")
        conn.commit()


init_db()


# ---------- schemas ----------
class ScoreRequest(BaseModel):
    business_type: str
    city_tier: Literal[1, 2]
    business_age_years: float = Field(ge=0, le=60)
    daily_upi_sales: float = Field(gt=0)
    utility_bill_score: int = Field(ge=1, le=4)
    supplier_repay_score: int = Field(ge=1, le=4)
    loan_amount: float = Field(gt=0)
    loan_purpose: str


# ---------- scoring logic ----------
def verdict_for(score: int):
    if score >= 70:
        return "APPROVED", "LOW"
    if score >= 45:
        return "CONDITIONAL", "MEDIUM"
    return "DECLINED", "HIGH"


def approved_amount(score: int, monthly_revenue: float, requested: float) -> float:
    multiple = 3.5 if score >= 80 else 2.5 if score >= 70 else 1.5 if score >= 60 else 0.8 if score >= 45 else 0
    return float(round(min(requested, multiple * monthly_revenue), -3))   # never more than asked


def factors_for(X: pd.DataFrame, req: ScoreRequest):
    if explainer is not None:
        sv = explainer.shap_values(X)[0]
        ranked = sorted(zip(FEATURES, sv), key=lambda t: abs(t[1]), reverse=True)[:5]
        return [{"label": LABELS[f], "impact": "positive" if v > 0.05 else "negative" if v < -0.05 else "neutral"}
                for f, v in ranked]
    # lightweight rule-based stand-in when SHAP is disabled
    ratio = X["loan_to_revenue_ratio"].iloc[0]
    items = [
        ("Utility bill history", req.utility_bill_score >= 3, req.utility_bill_score <= 2),
        ("Supplier repayment", req.supplier_repay_score >= 3, req.supplier_repay_score <= 2),
        ("Business vintage", req.business_age_years >= 3, req.business_age_years < 1),
        ("Loan-to-revenue ratio", ratio <= 0.8, ratio > 1.2),
        ("Daily UPI sales", req.daily_upi_sales >= 3000, req.daily_upi_sales < 1000),
    ]
    return [{"label": l, "impact": "positive" if good else "negative" if bad else "neutral"} for l, good, bad in items]


def tips_for(req: ScoreRequest, ratio: float):
    tips = []
    if req.utility_bill_score <= 2:
        tips.append("Pay electricity and water bills on or before the due date for the next 6 months.")
    if req.supplier_repay_score <= 2:
        tips.append("Clear pending supplier dues on time and keep a written ledger.")
    if ratio > 1.0:
        tips.append("Request a smaller loan - ideally under one month of your UPI revenue.")
    if req.business_age_years < 1:
        tips.append("Build a longer UPI transaction history before applying for a larger loan.")
    return tips or ["Keep up your on-time payment record."]


def match_lenders(score: int, amount: float):
    return [l for l in LENDERS if score >= l["min_score"] and amount <= l["max_loan"]]


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/score")
def score(req: ScoreRequest):
    if req.business_type not in BIZ_TYPES:
        raise HTTPException(422, f"business_type must be one of {BIZ_TYPES}")
    if req.loan_purpose not in PURPOSES:
        raise HTTPException(422, f"loan_purpose must be one of {PURPOSES}")

    monthly = req.daily_upi_sales * WORKING_DAYS
    ratio = req.loan_amount / monthly
    row = {
        "business_age_years": req.business_age_years,
        "daily_upi_sales": req.daily_upi_sales,
        "utility_bill_score": req.utility_bill_score,
        "supplier_repay_score": req.supplier_repay_score,
        "loan_to_revenue_ratio": ratio,
        "biz_type_encoded": BIZ_TYPES.index(req.business_type),
        "city_tier": req.city_tier,
        "purpose_encoded": PURPOSES.index(req.loan_purpose),
    }
    X = pd.DataFrame([row])[FEATURES]            # exact training column order

    prob = float(model.predict_proba(X)[0, 1])
    credit_score = int(round(prob * 100))
    verdict, band = verdict_for(credit_score)
    approved = approved_amount(credit_score, monthly, req.loan_amount)
    match_amount = approved if approved > 0 else min(req.loan_amount, 50_000)

    result = {
        "credit_score": credit_score,
        "repayment_probability": round(prob, 4),
        "verdict": verdict,
        "risk_band": band,
        "approved_amount": approved,
        "monthly_revenue": monthly,
        "loan_to_revenue_ratio": round(ratio, 2),
        "factors": factors_for(X, req),
        "tips": tips_for(req, ratio),
        "lenders": match_lenders(credit_score, match_amount),
    }

    with closing(db()) as conn:
        cur = conn.execute(
            """INSERT INTO applications (business_type, city, business_age_years, daily_upi_sales,
               utility_bill_score, supplier_repay_score, loan_amount_requested, loan_purpose,
               credit_score, verdict, risk_band, approved_amount, result_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (req.business_type, f"Tier {req.city_tier}", req.business_age_years, req.daily_upi_sales,
             req.utility_bill_score, req.supplier_repay_score, req.loan_amount, req.loan_purpose,
             credit_score, verdict, band, approved, json.dumps(result),
             datetime.now(timezone.utc).isoformat()))
        conn.commit()
        result["application_id"] = cur.lastrowid
    return result


@app.get("/api/applications")
def list_applications(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    with closing(db()) as conn:
        rows = conn.execute("SELECT id, business_type, city, loan_amount_requested, credit_score, verdict, "
                            "risk_band, approved_amount, created_at FROM applications "
                            "ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
    return {"total": total, "limit": limit, "offset": offset, "items": [dict(r) for r in rows]}


@app.get("/api/applications/{app_id}")
def get_application(app_id: int):
    with closing(db()) as conn:
        row = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Application not found")
    data = dict(row)
    data["result"] = json.loads(data.pop("result_json"))
    return data


@app.get("/api/stats")
def stats():
    with closing(db()) as conn:
        total, avg = conn.execute("SELECT COUNT(*), AVG(credit_score) FROM applications").fetchone()
        approved = conn.execute("SELECT COUNT(*) FROM applications WHERE verdict='APPROVED'").fetchone()[0]
        by_verdict = conn.execute("SELECT verdict, COUNT(*) c FROM applications GROUP BY verdict").fetchall()
    return {"total_applications": total,
            "approval_rate": round(approved / total * 100, 1) if total else 0,
            "average_credit_score": round(avg, 1) if avg else 0,
            "by_verdict": {r["verdict"]: r["c"] for r in by_verdict}}
