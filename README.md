# Bank Fraud Detection using Machine Learning

Video Link:https://drive.google.com/file/d/1SrIhFh_yuT3NiFRKTzWyMxARSlVpOwoA/view?usp=sharing
An end-to-end fraud detection system: synthetic transaction data generation, EDA,
an XGBoost model tuned with Optuna and explained with SHAP, a FastAPI backend, and
a minimal HTML/CSS/JS frontend that calls it.

---

## 1. Problem Statement

Banks need to flag fraudulent transactions in real time while minimizing
disruption to legitimate customers. This is a classic **imbalanced binary
classification** problem: fraud is rare (real-world rates are typically
well under 1%), so a naive model that predicts "not fraud" for everything
would already be >98% "accurate" while catching zero fraud. The real
objective is a cost-sensitive one:

- **Catch as much fraud as possible** (high recall on the fraud class)
- **Without overwhelming fraud analysts / blocking too many legitimate
  customers** (low false-positive rate)

This project targets **fraud recall > 80% at a false-positive rate < 5%**.

> **Note on the dataset:** No Kaggle dataset was provided, so a synthetic
> transaction dataset was generated (see [Approach](#2-approach) below).
> This was a deliberate design choice for this project, documented in
> `scripts/generate_data.py`; swapping in a real dataset (e.g. the IEEE-CIS
> or Kaggle "Credit Card Fraud" datasets) would only require changing
> `data/transactions.csv` and the column names in `src/data_prep.py`.

---

## 2. Approach

### 2.1 Data (`scripts/generate_data.py`)
Generates 50,000 synthetic transactions with a **probabilistic** (not
hard-coded if/else) fraud label:
- Features: `amount`, `timestamp`, `transaction_type`, `merchant_category`,
  `country`, `device_risk_score`, `ip_risk_score`
- Each transaction's fraud probability is computed as a **logistic function**
  of weighted risk signals (large amount, risky transaction types like wire
  transfers, risky merchant categories like gambling/cash-advance, high-risk
  countries, late-night activity, high device/IP risk scores) plus random
  noise, then a Bernoulli label is sampled from that probability. This
  produces a realistic, imperfectly-separable classification problem rather
  than a trivially deterministic one.
- Final fraud rate: **~1.6%** of transactions.

### 2.2 EDA (`scripts/eda.py`)
Class imbalance check, amount distribution by class, and a feature
correlation heatmap. Plots saved to `outputs/`. Key finding: linear
correlations with `is_fraud` are individually weak (as expected for a
logistic-combination signal), but fraud *rates* vary sharply across
categories -- e.g. `gambling`/`cash_advance` merchants and `WIRE_TRANSFER`
transactions have several times the base fraud rate -- which is exactly the
kind of non-linear, interaction-heavy signal tree ensembles like XGBoost
are good at exploiting.

### 2.3 Modeling (`src/data_prep.py`, `src/model.py`, `scripts/train.py`)
1. **Feature engineering**: `hour`, `day_of_week`, `is_night` derived from
   `timestamp`; categoricals one-hot encoded.
2. **Stratified train/test split** (80/20) so the rare fraud class is
   represented proportionally in both sets.
3. **Baseline model**: Logistic Regression trained on SMOTE-resampled data.
4. **Main model**: XGBoost, tuned with **Optuna** (30 trials, 3-fold
   cross-validation). Both the SMOTE oversampling ratio and the XGBoost
   hyperparameters (`max_depth`, `learning_rate`, `subsample`,
   `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`,
   `reg_lambda`, `n_estimators`) are tuned jointly. The tuning objective is
   **recall at a fixed 5% false-positive rate** -- i.e. we optimize
   directly for the metric the business cares about, not a threshold-agnostic
   proxy like ROC-AUC.
   - *Why tune the SMOTE ratio instead of using its 1:1 default?* With only
     ~1.6% fraud, default SMOTE generates ~60x synthetic minority samples.
     In this dataset's ~30-dimensional, mostly one-hot feature space, that
     much synthetic oversampling created noisy near-duplicate points that
     blurred the true decision boundary and *hurt* the low-FPR operating
     point. Tuning the ratio (found optimum: ~16% of majority class size)
     fixed this.
5. **Cost-sensitive threshold optimization**: rather than the default 0.5
   cutoff, we scan thresholds and pick the one that satisfies
   recall > 80% and FPR < 5% (see `optimize_threshold` in `src/model.py`).
6. **SHAP explainability**: `shap.TreeExplainer` on the final XGBoost model
   gives both global feature importance (`outputs/shap_global_importance.png`)
   and a per-transaction local explanation (also exposed live via the API).

### 2.4 Backend (`app/main.py`)
FastAPI service that loads the trained model **once at startup**, validates
input with Pydantic, and serves `/predict` and `/health`. CORS is enabled
so the static frontend can call it from a different origin/port.

### 2.5 Frontend (`frontend/`)
Plain HTML/CSS/JS (no frameworks, no build step). A form posts to
`/predict` and renders the fraud probability, a red/green flag, and the
top SHAP-driving features for that specific transaction.

---

## 3. Model Performance Achieved

Evaluated on a held-out test set (10,000 transactions, 164 fraud, never
seen during training or hyperparameter tuning):

| Model | Threshold | Recall | Precision | False Positive Rate | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline (Logistic Regression + SMOTE) | 0.5 (default) | 91.5% | 17.2% | 7.4% | 0.976 | 0.622 |
| **XGBoost (tuned) -- default threshold** | 0.5 (default) | 37.2% | 67.8% | 0.3% | 0.974 | 0.579 |
| **XGBoost (tuned) -- optimized threshold** | **0.125** | **82.9%** | **35.5%** | **2.5%** | 0.974 | 0.579 |

**Targets: recall > 80%, FPR < 5% -- both met** (82.9% recall at 2.5% FPR).

Confusion matrix at the optimized threshold (0.125):

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 9,589 | 247 |
| **Actual Fraud** | 28 | 136 |

This illustrates why the default 0.5 threshold is the wrong choice for a
rare-event problem: at 0.5, XGBoost is "confident" and precise (67.8%
precision) but only catches 37% of fraud. Lowering the threshold to 0.125
trades some precision for a large recall gain -- exactly the trade-off a
fraud team would choose, since missing fraud is typically far costlier
than a false alarm that a human reviewer clears in seconds.

Plots: `outputs/class_imbalance.png`, `outputs/amount_distribution.png`,
`outputs/correlation_heatmap.png`, `outputs/confusion_matrix.png`,
`outputs/roc_curve.png`, `outputs/shap_global_importance.png`.

**Top global SHAP features**: `ip_risk_score`, `device_risk_score`,
`transaction_type_WIRE_TRANSFER`, `is_night`, `transaction_type_ONLINE` --
consistent with the fraud-generating logic and with the EDA findings.

---

## 4. Project Structure

```
bank-fraud-detection/
├── data/
│   └── transactions.csv          # generated synthetic dataset
├── scripts/
│   ├── generate_data.py          # Step 1: synthetic data generation
│   ├── eda.py                    # Step 2: exploratory data analysis
│   └── train.py                  # Step 3: full training pipeline
├── src/
│   ├── data_prep.py              # feature engineering + preprocessing (shared)
│   └── model.py                  # training / tuning / eval / SHAP (shared)
├── app/
│   └── main.py                   # Step 4: FastAPI backend
├── frontend/
│   ├── index.html                # Step 5: minimal frontend
│   ├── style.css
│   └── script.js
├── models/                       # saved model artifacts (generated by train.py)
│   ├── preprocessor.joblib
│   ├── xgb_pipeline.joblib
│   └── metadata.json             # threshold, feature names, metrics, SHAP summary
├── outputs/                      # EDA + evaluation plots (generated)
├── requirements.txt
└── README.md
```

---

## 5. How to Run Locally

### 5.1 Setup
```bash
cd bank-fraud-detection
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```
> Tested with **Python 3.11**. (On Windows, avoid Python 3.13+/3.14 for now --
> some pinned packages here don't yet ship prebuilt wheels for the newest
> Python versions.)

### 5.2 Train the model
```bash
python scripts/generate_data.py   # -> data/transactions.csv
python scripts/eda.py             # -> outputs/*.png (optional, informational)
python scripts/train.py           # -> models/*.joblib, models/metadata.json
```
`train.py` takes a few minutes (Optuna runs 30 hyperparameter-search trials).
It prints baseline metrics, best hyperparameters, final test-set metrics at
both the default and optimized thresholds, and the SHAP summary.

### 5.3 Serve the backend
```bash
uvicorn app.main:app --reload --port 8000
```
- `GET  http://127.0.0.1:8000/health` -- health check
- `POST http://127.0.0.1:8000/predict` -- fraud prediction
- Interactive API docs: `http://127.0.0.1:8000/docs`

### 5.4 Run the frontend
The FastAPI app already serves the frontend at `http://127.0.0.1:8000/`
(mounted via `StaticFiles`), so **no extra step is needed** -- just open
that URL in a browser once the backend is running.

Alternatively, serve it standalone on its own port (useful if you want the
frontend and backend on genuinely different origins to exercise CORS):
```bash
cd frontend
python -m http.server 5500
# open http://127.0.0.1:5500 and set "API base URL" in the form to
# http://127.0.0.1:8000
```

---

## 6. API Usage Examples

### Health check
```bash
curl http://127.0.0.1:8000/health
```
```json
{"status":"ok","model_loaded":true,"threshold":0.125,"trained_at":"2026-09-08 22:53:03"}
```

### Predict -- high-risk transaction
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 4500.00,
    "timestamp": "2025-07-15T02:30:00",
    "transaction_type": "WIRE_TRANSFER",
    "merchant_category": "cash_advance",
    "country": "NG",
    "device_risk_score": 0.82,
    "ip_risk_score": 0.77
  }'
```
```json
{
  "fraud_probability": 0.999333,
  "fraud_flag": true,
  "threshold_used": 0.125,
  "top_contributing_features": [
    {"feature": "ip_risk_score", "value": 0.77, "shap_value": 2.96, "direction": "increases_fraud_risk"},
    {"feature": "device_risk_score", "value": 0.82, "shap_value": 2.49, "direction": "increases_fraud_risk"},
    {"feature": "merchant_category_cash_advance", "value": 1.0, "shap_value": 1.22, "direction": "increases_fraud_risk"},
    {"feature": "transaction_type_WIRE_TRANSFER", "value": 1.0, "shap_value": 0.97, "direction": "increases_fraud_risk"},
    {"feature": "country_NG", "value": 1.0, "shap_value": 0.82, "direction": "increases_fraud_risk"}
  ]
}
```

### Predict -- low-risk transaction
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 42.50,
    "timestamp": "2025-07-15T14:20:00",
    "transaction_type": "POS",
    "merchant_category": "grocery",
    "country": "US",
    "device_risk_score": 0.05,
    "ip_risk_score": 0.03
  }'
```
```json
{"fraud_probability": 0.000015, "fraud_flag": false, "threshold_used": 0.125, "top_contributing_features": [...]}
```

Both examples above were run against the live server as part of building
this project -- see the console output in the development session for the
exact responses.

---

## 7. What a Real Deployment Would Add

This project is a local, self-contained demo. A production deployment on
AWS would conceptually add:

- **Compute**: containerize the FastAPI app (Docker) and run it on
  **AWS ECS/Fargate** or **EC2** behind an Application Load Balancer for
  steady traffic; or **AWS Lambda** (via API Gateway + a container-image
  Lambda) for spiky/low-volume traffic, since inference here is a single
  fast XGBoost + SHAP call per request.
- **Storage**: model artifacts (`preprocessor.joblib`, `xgb_pipeline.joblib`,
  `metadata.json`) would live in **S3**, loaded at container/Lambda cold
  start instead of from local disk, with versioned model artifacts for
  safe rollback.
- **Data pipeline**: replace the synthetic generator with a real ingestion
  pipeline (e.g. transactions streamed via **Kinesis**, batch-scored or
  retrained via **Step Functions** / **SageMaker Pipelines** on a schedule).
- **Monitoring**: track live fraud-recall/FPR against the offline
  evaluation (data drift and model-performance drift), log predictions to
  **CloudWatch** and a **feature store**, and alert if the served
  threshold stops meeting the recall/FPR targets on live traffic.
- **Security**: the current `allow_origins=["*"]` CORS policy is fine for
  local development only -- production would restrict it to the actual
  frontend domain, add authentication (API keys / IAM), and put the API
  behind a WAF.
- **Retraining**: a scheduled retraining job (new data -> `train.py`
  equivalent -> new artifacts in S3 -> blue/green model swap) so the model
  adapts as fraud patterns evolve.

None of this was deployed for this project -- it's scoped to run entirely
locally, per the assignment.

---

## 8. Notes on Modularity

- `src/data_prep.py` and `src/model.py` are imported by **both**
  `scripts/train.py` and `app/main.py`, so feature engineering and
  preprocessing are guaranteed identical between training and serving
  (a common source of subtle bugs when the two are duplicated).
- All hyperparameters, thresholds, and file paths are named constants at
  the top of each script -- no magic numbers buried in logic.
- Every non-obvious modeling decision (why SMOTE ratio is tuned, why the
  Optuna objective is recall-at-fixed-FPR rather than PR-AUC, why the
  decision threshold isn't 0.5) is explained in a comment at the point
  of the decision.
