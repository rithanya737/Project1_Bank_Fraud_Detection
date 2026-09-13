"""
app/main.py
===========
FastAPI backend for the Bank Fraud Detection project.

Endpoints:
  GET  /health   -- basic health check (also reports whether the model
                     artifacts loaded successfully)
  POST /predict  -- takes a single transaction's features, returns the
                     fraud probability, a fraud flag (using the
                     cost-sensitive optimized threshold from training),
                     and the top SHAP-contributing features for that
                     specific prediction.

The trained model + preprocessor + threshold are loaded ONCE at process
startup (not per-request) via FastAPI's lifespan handler, which is the
recommended, efficient pattern for serving ML models.

Run (from the project root, with the venv active):
    uvicorn app.main:app --reload --port 8000
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Make `src` importable regardless of the working directory uvicorn is
# launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_prep import (
    KNOWN_COUNTRIES,
    KNOWN_MERCHANT_CATEGORIES,
    KNOWN_TRANSACTION_TYPES,
    engineer_features,
)
from src.model import get_shap_explainer, local_explanation

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Populated at startup by the lifespan handler
ml_state = {
    "preprocessor": None,
    "xgb_pipeline": None,
    "xgb_model": None,
    "explainer": None,
    "threshold": 0.5,
    "feature_names": [],
    "metadata": {},
    "loaded": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup: load model artifacts once ----
    try:
        ml_state["preprocessor"] = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
        ml_state["xgb_pipeline"] = joblib.load(os.path.join(MODELS_DIR, "xgb_pipeline.joblib"))
        ml_state["xgb_model"] = ml_state["xgb_pipeline"].named_steps["clf"]
        ml_state["explainer"] = get_shap_explainer(ml_state["xgb_model"])

        with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
            metadata = json.load(f)
        ml_state["metadata"] = metadata
        ml_state["threshold"] = metadata["threshold"]
        ml_state["feature_names"] = metadata["feature_names"]
        ml_state["loaded"] = True
        print(f"[startup] Model artifacts loaded. Threshold={ml_state['threshold']:.3f}")
    except FileNotFoundError as e:
        # Allow the API to start (so /health reports the problem clearly)
        # even if `scripts/train.py` hasn't been run yet.
        print(f"[startup] WARNING: could not load model artifacts: {e}")
        ml_state["loaded"] = False

    yield
    # ---- shutdown: nothing to clean up ----


app = FastAPI(
    title="Bank Fraud Detection API",
    description="Predicts whether a bank transaction is fraudulent using a trained XGBoost model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS so a locally-served static frontend (e.g. via
# `python -m http.server` on a different port) can call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Pydantic request / response models
# ----------------------------------------------------------------------
class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in dollars")
    timestamp: str = Field(..., description="ISO-8601 timestamp, e.g. 2025-07-15T02:30:00")
    transaction_type: str = Field(..., description=f"One of {KNOWN_TRANSACTION_TYPES}")
    merchant_category: str = Field(..., description=f"One of {KNOWN_MERCHANT_CATEGORIES}")
    country: str = Field(..., description=f"One of {KNOWN_COUNTRIES}")
    device_risk_score: float = Field(..., ge=0, le=1, description="Device fingerprint risk score, 0-1")
    ip_risk_score: float = Field(..., ge=0, le=1, description="IP reputation risk score, 0-1")

    @field_validator("transaction_type")
    @classmethod
    def validate_transaction_type(cls, v):
        if v not in KNOWN_TRANSACTION_TYPES:
            raise ValueError(f"transaction_type must be one of {KNOWN_TRANSACTION_TYPES}")
        return v

    @field_validator("merchant_category")
    @classmethod
    def validate_merchant_category(cls, v):
        if v not in KNOWN_MERCHANT_CATEGORIES:
            raise ValueError(f"merchant_category must be one of {KNOWN_MERCHANT_CATEGORIES}")
        return v

    @field_validator("country")
    @classmethod
    def validate_country(cls, v):
        if v not in KNOWN_COUNTRIES:
            raise ValueError(f"country must be one of {KNOWN_COUNTRIES}")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v):
        try:
            pd.to_datetime(v)
        except Exception:
            raise ValueError("timestamp must be a valid ISO-8601 datetime string")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 4500.00,
                "timestamp": "2025-07-15T02:30:00",
                "transaction_type": "WIRE_TRANSFER",
                "merchant_category": "cash_advance",
                "country": "NG",
                "device_risk_score": 0.82,
                "ip_risk_score": 0.77,
            }
        }


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    fraud_probability: float
    fraud_flag: bool
    threshold_used: float
    top_contributing_features: List[FeatureContribution]


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # allow the `model_loaded` field name

    status: str
    model_loaded: bool
    threshold: Optional[float] = None
    trained_at: Optional[str] = None


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if ml_state["loaded"] else "model_not_loaded",
        model_loaded=ml_state["loaded"],
        threshold=ml_state["threshold"] if ml_state["loaded"] else None,
        trained_at=ml_state["metadata"].get("trained_at") if ml_state["loaded"] else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    if not ml_state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Run `python scripts/train.py` first.",
        )

    # Build a single-row DataFrame with the raw columns, then apply the
    # EXACT same feature engineering + preprocessing used during training.
    raw_df = pd.DataFrame([transaction.model_dump()])
    engineered_df = engineer_features(raw_df)

    preprocessor = ml_state["preprocessor"]
    X = preprocessor.transform(engineered_df)

    xgb_pipeline = ml_state["xgb_pipeline"]
    fraud_probability = float(xgb_pipeline.predict_proba(X)[0, 1])
    threshold = ml_state["threshold"]
    fraud_flag = fraud_probability >= threshold

    # SHAP local explanation for this specific transaction
    explainer = ml_state["explainer"]
    feature_names = ml_state["feature_names"]
    contributions = local_explanation(explainer, X, feature_names, top_n=5)
    top_features = [
        FeatureContribution(
            feature=c["feature"],
            value=c["value"],
            shap_value=c["shap_value"],
            direction="increases_fraud_risk" if c["shap_value"] > 0 else "decreases_fraud_risk",
        )
        for c in contributions
    ]

    return PredictionResponse(
        fraud_probability=round(fraud_probability, 6),
        fraud_flag=fraud_flag,
        threshold_used=threshold,
        top_contributing_features=top_features,
    )


# ----------------------------------------------------------------------
# Serve the static frontend (optional convenience -- the frontend can
# also be served independently, see README.md).
# Mounted LAST so it doesn't shadow the /predict and /health routes above.
# ----------------------------------------------------------------------
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
