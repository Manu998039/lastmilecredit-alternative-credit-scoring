"""Generate the synthetic MSME dataset (msme_loans.csv) and train the XGBoost model.
Run:  python train_model.py
Outputs: msme_loans.csv, model.joblib
"""
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

N = 15_000
SEED = 42
BASE_RATE = 0.712          # ~71.2% repay on time (report, section 2.6)
WORKING_DAYS = 26          # monthly revenue = daily UPI sales x 26

BIZ_TYPES = ["Kirana / General Store", "Tea Stall / Chai Shop", "Tailor",
             "Mobile Repair Shop", "Vegetable Vendor", "Salon / Barber",
             "Street Food Stall", "Tiffin Service"]
PURPOSES = ["Stock / Inventory", "Equipment", "Working Capital", "Shop Renovation"]

FEATURES = ["business_age_years", "daily_upi_sales", "utility_bill_score",
            "supplier_repay_score", "loan_to_revenue_ratio",
            "biz_type_encoded", "city_tier", "purpose_encoded"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def generate_dataset(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.exponential(4.0, n), 0.3, 40).round(1)                    # exponential
    sales = np.clip(rng.lognormal(np.log(2500), 0.8, n), 300, 50_000).round(0)  # log-normal
    util = rng.choice([1, 2, 3, 4], n, p=[0.12, 0.25, 0.38, 0.25])              # ordinal 1-4
    supplier = rng.choice([1, 2, 3, 4], n, p=[0.10, 0.25, 0.40, 0.25])
    city_tier = rng.choice([1, 2], n, p=[0.4, 0.6])
    biz = rng.integers(0, len(BIZ_TYPES), n)
    purpose = rng.integers(0, len(PURPOSES), n)

    monthly = sales * WORKING_DAYS
    loan = np.clip(monthly * rng.uniform(0.3, 3.0, n), 10_000, 400_000).round(-3)
    ratio = (loan / monthly).round(3)

    # Weighted linear combination -> sigmoid, with Gaussian noise (report, section 3.4)
    z = (0.70 * (util - 2.5) + 0.70 * (supplier - 2.5)
         + 0.30 * np.log1p(age) + 0.15 * np.log(sales / 2500)
         + 0.10 * (city_tier == 1) - 1.50 * ratio
         + rng.normal(0, 0.45, n))
    b0 = brentq(lambda b: sigmoid(z + b).mean() - BASE_RATE, -10, 10)  # hit the target base rate
    repaid = (rng.random(n) < sigmoid(z + b0)).astype(int)

    return pd.DataFrame({
        "business_type": [BIZ_TYPES[i] for i in biz],
        "city_tier": city_tier,
        "business_age_years": age,
        "daily_upi_sales": sales,
        "utility_bill_score": util,
        "supplier_repay_score": supplier,
        "loan_amount": loan,
        "loan_purpose": [PURPOSES[i] for i in purpose],
        "loan_to_revenue_ratio": ratio,
        "biz_type_encoded": biz,
        "purpose_encoded": purpose,
        "repaid_on_time": repaid,
    })


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("msme_loans.csv", index=False)
    print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols | repaid_on_time = {df.repaid_on_time.mean():.2%}")

    X, y = df[FEATURES], df["repaid_on_time"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.85, colsample_bytree=0.85,
                          eval_metric="logloss", random_state=SEED)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    print(f"AUC-ROC : {roc_auc_score(y_te, proba):.3f}")
    print(f"Accuracy: {accuracy_score(y_te, (proba >= 0.5).astype(int)):.3f}")

    joblib.dump({"model": model, "features": FEATURES,
                 "biz_types": BIZ_TYPES, "purposes": PURPOSES}, "model.joblib")
    print("Saved model.joblib")
