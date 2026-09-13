"""
data_prep.py
============
Reusable data-preparation code shared by the training script and the
FastAPI inference service. Keeping this logic in one place guarantees the
exact same feature engineering + encoding is applied at train time and at
prediction time (a common source of train/serve skew bugs if duplicated).

Pipeline:
  1. engineer_features(): derive `hour`, `day_of_week`, `is_night` from
     the raw `timestamp` column (fraud in this dataset is correlated with
     late-night activity -- see scripts/eda.py).
  2. build_preprocessor(): a scikit-learn ColumnTransformer that
     one-hot-encodes the categorical columns and passes numeric columns
     through untouched (tree models don't need scaling).
  3. load_and_split(): reads the CSV, engineers features, and performs a
     stratified train/test split (stratified because the fraud class is
     rare -- a plain random split risks skewing the fraud rate between
     train and test).
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Raw columns expected from the client / CSV (before feature engineering)
RAW_NUMERIC_FEATURES = ["amount", "device_risk_score", "ip_risk_score"]
RAW_CATEGORICAL_FEATURES = ["transaction_type", "merchant_category", "country"]

# Columns engineered from `timestamp`
ENGINEERED_NUMERIC_FEATURES = ["hour", "day_of_week", "is_night"]

# Final feature ordering used everywhere downstream
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COL = "is_fraud"

# Known categorical vocabulary (keeps one-hot columns stable even if a
# rare category is missing from a particular batch of new data)
KNOWN_TRANSACTION_TYPES = ["POS", "ONLINE", "ATM_WITHDRAWAL", "WIRE_TRANSFER", "BILL_PAYMENT"]
KNOWN_MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "fuel", "utilities", "online_retail",
    "electronics", "travel", "jewelry", "gambling", "cash_advance",
]
KNOWN_COUNTRIES = ["US", "GB", "IN", "DE", "FR", "CA", "AU", "BR", "NG", "RU", "CN", "UA"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour / day_of_week / is_night derived from `timestamp`.

    Accepts either a datetime64 column or ISO-format strings (as sent by
    the API from JSON).
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: one-hot encode categoricals, passthrough numerics.

    `handle_unknown="ignore"` means a category never seen during training
    (e.g. a new country code) won't crash inference -- it just produces an
    all-zero one-hot row for that feature, which is the safe default.
    """
    categories = [KNOWN_TRANSACTION_TYPES, KNOWN_MERCHANT_CATEGORIES, KNOWN_COUNTRIES]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> list:
    """Human-readable names for the columns produced by the preprocessor,
    in the exact order they appear in the transformed array. Needed for
    mapping SHAP values back to interpretable feature names.
    """
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return cat_names + NUMERIC_FEATURES


def load_and_split(
    csv_path: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the raw CSV, engineer features, and return a stratified
    train/test split of the RAW (pre-preprocessing) feature frame.
    """
    df = pd.read_csv(csv_path)
    df = engineer_features(df)

    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
