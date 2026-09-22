"""
Fraud detection microservice for the n8n workflow.

Two endpoints, matching the "Available data" and "AI/ML opportunity" steps
of the decision architecture:

  POST /enrich   takes raw transaction identifiers, returns enriched
                  customer/device/network features (synthetic but
                  deterministic — same input always returns same output,
                  so repeated calls behave consistently)

  POST /score    takes the full enriched feature set, returns a fraud
                  probability plus the top contributing factors (reason
                  codes) as a stand-in for the graph-augmented hybrid model
                  described in the assignment

Run locally:    uvicorn app:app --reload --port 8000
Docs:           http://localhost:8000/docs
"""

import hashlib
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Fraud Detection Service",
    description="Toy classifier + enrichment stub backing the n8n fraud workflow.",
    version="1.0.0",
)

_artifact = joblib.load("fraud_model.joblib")
MODEL = _artifact["model"]
FEATURES = _artifact["features"]


def _seeded_random(*parts: str) -> np.random.Generator:
    """Deterministic RNG seeded from the input identifiers, so the same
    transaction/customer/device always enriches to the same values."""
    key = "|".join(parts).encode()
    seed = int(hashlib.sha256(key).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# /enrich
# ---------------------------------------------------------------------------

class EnrichRequest(BaseModel):
    transaction_id: str
    customer_id: str
    device_id: str
    ip_address: str
    merchant_category: Optional[str] = "general"


class EnrichResponse(BaseModel):
    transaction_id: str
    customer_tenure_days: int
    distance_from_home_km: float
    device_is_new: int
    velocity_24h: int
    shared_device_count: int
    ip_risk_score: float
    merchant_risk_score: float


@app.post("/enrich", response_model=EnrichResponse)
def enrich(req: EnrichRequest):
    rng_customer = _seeded_random("customer", req.customer_id)
    rng_device = _seeded_random("device", req.device_id, req.customer_id)
    rng_ip = _seeded_random("ip", req.ip_address)
    rng_merchant = _seeded_random("merchant", req.merchant_category)

    return EnrichResponse(
        transaction_id=req.transaction_id,
        customer_tenure_days=int(rng_customer.integers(1, 3650)),
        distance_from_home_km=round(float(abs(rng_device.normal(5, 40))), 1),
        device_is_new=int(rng_device.binomial(1, 0.12)),
        velocity_24h=int(rng_customer.poisson(1.5)),
        shared_device_count=int(rng_device.poisson(0.3)),
        ip_risk_score=round(float(np.clip(rng_ip.beta(2, 8), 0, 1)), 3),
        merchant_risk_score=round(float(np.clip(rng_merchant.beta(2, 6), 0, 1)), 3),
    )


# ---------------------------------------------------------------------------
# /score
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    transaction_id: str
    amount: float = Field(..., gt=0)
    hour_of_day: int = Field(..., ge=0, le=23)
    merchant_risk_score: float = Field(..., ge=0, le=1)
    customer_tenure_days: int = Field(..., ge=0)
    distance_from_home_km: float = Field(..., ge=0)
    device_is_new: int = Field(..., ge=0, le=1)
    velocity_24h: int = Field(..., ge=0)
    shared_device_count: int = Field(..., ge=0)
    ip_risk_score: float = Field(..., ge=0, le=1)


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    decision_hint: str
    top_factors: list[str]


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    row = {f: getattr(req, f) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)

    try:
        proba = float(MODEL.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scoring failed: {e}")

    if proba > 0.7:
        hint = "auto_block"
    elif proba >= 0.3:
        hint = "step_up_auth"
    else:
        hint = "auto_approve"

    # Reason codes: features weighted by (value * global importance), so the
    # explanation reflects both what the model learned AND this transaction's
    # actual values — a simplified stand-in for SHAP-style explainability.
    importances = dict(zip(FEATURES, MODEL.feature_importances_))
    contributions = {
        f: importances[f] * _normalize(f, row[f]) for f in FEATURES
    }
    top = sorted(contributions.items(), key=lambda x: -x[1])[:3]
    top_factors = [name for name, _ in top]

    return ScoreResponse(
        transaction_id=req.transaction_id,
        fraud_score=round(proba, 4),
        decision_hint=hint,
        top_factors=top_factors,
    )


def _normalize(feature: str, value: float) -> float:
    """Rough 0-1 normalization per feature, just for ranking reason codes."""
    ranges = {
        "amount": 500.0,
        "hour_of_day": 23.0,
        "merchant_risk_score": 1.0,
        "customer_tenure_days": 3650.0,
        "distance_from_home_km": 200.0,
        "device_is_new": 1.0,
        "velocity_24h": 10.0,
        "shared_device_count": 5.0,
        "ip_risk_score": 1.0,
    }
    inverted = {"customer_tenure_days"}  # higher tenure = lower risk contribution
    v = min(max(value, 0), ranges[feature]) / ranges[feature]
    return (1 - v) if feature in inverted else v


@app.get("/health")
def health():
    return {"status": "ok"}
