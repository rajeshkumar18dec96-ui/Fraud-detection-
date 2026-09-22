"""
Trains a toy but genuinely-fitted fraud classifier on a synthetic transaction
dataset. The dataset is generated with a hand-designed fraud probability
function (see `true_fraud_probability`) so the labels reflect realistic,
interpretable risk factors — not random noise. The trained model then has to
recover that structure purely from features + labels, same as a real fraud
model would.

Features (mirror the "Available data" / "Required prediction" rows of the
decision architecture):
  amount                  transaction amount (USD)
  hour_of_day             0-23, local time of transaction
  merchant_risk_score     0-1, known merchant risk (chargeback history etc.)
  customer_tenure_days    how long the account has existed
  distance_from_home_km   distance between transaction and home address
  device_is_new           1 if this device has never been used on the account
  velocity_24h            number of transactions by this customer in last 24h
  shared_device_count     graph-derived: how many OTHER accounts share this
                           device/IP fingerprint (ring-fraud signal)
  ip_risk_score           0-1, IP reputation (proxy/VPN/blacklist signals)

Label:
  is_fraud (0/1)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib

RNG = np.random.default_rng(42)
N = 20000

FEATURES = [
    "amount",
    "hour_of_day",
    "merchant_risk_score",
    "customer_tenure_days",
    "distance_from_home_km",
    "device_is_new",
    "velocity_24h",
    "shared_device_count",
    "ip_risk_score",
]


def generate_dataset(n=N):
    amount = np.round(RNG.gamma(shape=2.0, scale=60.0, size=n), 2)  # skewed, mostly small
    hour_of_day = RNG.integers(0, 24, size=n)
    merchant_risk_score = np.clip(RNG.beta(2, 6, size=n), 0, 1)
    customer_tenure_days = RNG.integers(1, 3650, size=n)
    distance_from_home_km = np.round(np.abs(RNG.normal(loc=5, scale=40, size=n)), 1)
    device_is_new = RNG.binomial(1, 0.12, size=n)
    velocity_24h = RNG.poisson(1.5, size=n)
    shared_device_count = RNG.poisson(0.3, size=n)  # most devices used by 0 other accounts
    ip_risk_score = np.clip(RNG.beta(2, 8, size=n), 0, 1)

    df = pd.DataFrame({
        "amount": amount,
        "hour_of_day": hour_of_day,
        "merchant_risk_score": merchant_risk_score,
        "customer_tenure_days": customer_tenure_days,
        "distance_from_home_km": distance_from_home_km,
        "device_is_new": device_is_new,
        "velocity_24h": velocity_24h,
        "shared_device_count": shared_device_count,
        "ip_risk_score": ip_risk_score,
    })

    p = true_fraud_probability(df)
    df["is_fraud"] = RNG.binomial(1, p)
    return df


def true_fraud_probability(df):
    """Hand-designed risk function — this is the 'ground truth' pattern the
    model has to learn. Combines the kind of signals a real fraud team would
    use, each contributing on a logistic (log-odds) scale."""
    z = (
        -6.0
        + 0.010 * (df["amount"] - 60)
        + 1.6 * df["merchant_risk_score"]
        - 0.0006 * df["customer_tenure_days"]
        + 0.02 * df["distance_from_home_km"]
        + 1.3 * df["device_is_new"]
        + 0.35 * df["velocity_24h"]
        + 1.1 * df["shared_device_count"]
        + 2.4 * df["ip_risk_score"]
        + np.where((df["hour_of_day"] >= 1) & (df["hour_of_day"] <= 4), 0.5, 0.0)
    )
    return 1 / (1 + np.exp(-z))


def main():
    df = generate_dataset()
    print(f"Generated {len(df)} rows, fraud rate: {df['is_fraud'].mean():.3%}")

    X = df[FEATURES]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"Test AUC: {auc:.4f}")
    print(classification_report(y_test, (proba > 0.5).astype(int)))

    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:24s} {imp:.4f}")

    joblib.dump({"model": model, "features": FEATURES}, "fraud_model.joblib")
    print("\nSaved model to fraud_model.joblib")


if __name__ == "__main__":
    main()
