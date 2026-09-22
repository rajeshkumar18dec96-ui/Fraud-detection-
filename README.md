# Fraud detection service

A real (if toy) fraud classifier trained on a synthetic dataset with
realistic feature relationships, served via FastAPI. This backs the
`Enrich Customer & Device Data` and `Fraud Scoring (Classifier + Graph)`
nodes in `fraud_detection_workflow.json`.

## What's included

| File | Purpose |
|---|---|
| `train_model.py` | Generates 20,000 synthetic transactions with a hand-designed fraud pattern, trains a `GradientBoostingClassifier`, saves `fraud_model.joblib` |
| `fraud_model.joblib` | The trained model (already built — you don't need to retrain unless you want to) |
| `app.py` | FastAPI service exposing `/enrich`, `/score`, `/health` |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build for deployment |

Model performance on held-out test data: **AUC 0.86**, 5.5% base fraud rate
(realistic order of magnitude for a synthetic set). Feature importances
came out sensible without being hand-tuned: transaction amount, account
tenure, distance from home, and the graph-derived `shared_device_count`
dominate — exactly the signals a real fraud model leans on.

## API

**POST /enrich** — simulates the "available data" enrichment step (deterministic: same IDs always return the same synthetic profile)
```json
{
  "transaction_id": "txn_001",
  "customer_id": "cust_123",
  "device_id": "dev_abc",
  "ip_address": "203.0.113.5",
  "merchant_category": "electronics"
}
```

**POST /score** — the classifier + graph-feature hybrid scoring step
```json
{
  "transaction_id": "txn_001",
  "amount": 1200.0,
  "hour_of_day": 3,
  "merchant_risk_score": 0.18,
  "customer_tenure_days": 1243,
  "distance_from_home_km": 32.9,
  "device_is_new": 0,
  "velocity_24h": 1,
  "shared_device_count": 0,
  "ip_risk_score": 0.25
}
```
Returns `{"fraud_score": 0.50, "decision_hint": "step_up_auth", "top_factors": [...]}`.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# Interactive docs at http://localhost:8000/docs
```

Quick test:
```bash
curl -X POST http://localhost:8000/enrich -H "Content-Type: application/json" \
  -d '{"transaction_id":"t1","customer_id":"c1","device_id":"d1","ip_address":"1.2.3.4","merchant_category":"electronics"}'
```

## Deploy for free — Render.com (recommended, easiest)

1. Push this folder to a new GitHub repo.
2. Go to [render.com](https://render.com) → New → Web Service → connect the repo.
3. Render will detect the `Dockerfile` automatically. Leave build/start commands blank (the Dockerfile handles them).
4. Instance type: **Free**. Click Create Web Service.
5. Wait for the build (~2–3 min). Render gives you a public URL like `https://fraud-detection-abc123.onrender.com`.
6. Test it: `curl https://fraud-detection-abc123.onrender.com/health`

Note: Render's free tier spins down after 15 minutes of inactivity, so the first request after idling takes ~30–50 seconds to wake up. Fine for a demo/assignment; not for production.

## Deploy for free — Hugging Face Spaces (alternative)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), SDK: **Docker**.
2. Upload `app.py`, `requirements.txt`, `Dockerfile`, `fraud_model.joblib` to the Space (or `git push` if you clone the Space repo).
3. Hugging Face builds and serves it automatically at `https://<your-username>-<space-name>.hf.space`.
4. Note: Spaces expects the container to listen on port 7860 by default — either add `EXPOSE 7860` and change the `CMD` port in the Dockerfile, or set the Space's "App port" setting to 8000 to match this Dockerfile as-is.

## Wire it into the n8n workflow

In `fraud_detection_workflow.json`, two HTTP Request nodes have their URL set to the placeholder `FRAUD_SERVICE_BASE_URL/enrich` and `FRAUD_SERVICE_BASE_URL/score`. After deploying, open the workflow in n8n and replace `FRAUD_SERVICE_BASE_URL` in both nodes with your actual deployed URL (e.g. `https://fraud-detection-abc123.onrender.com`).

## Retraining on a real dataset

`train_model.py` is self-contained and easy to swap: if you'd rather train on a real dataset (e.g. the Kaggle "Credit Card Fraud Detection" dataset), replace `generate_dataset()` with a `pd.read_csv(...)` call, keep the same `FEATURES` list (or update it to match your columns), and the rest of the training/saving code works unchanged.
