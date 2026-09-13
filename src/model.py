"""
model.py
========
Model training, tuning, evaluation, threshold optimization, and SHAP
explainability helpers, shared between scripts/train.py and the FastAPI
inference service (app/main.py loads the saved artifacts these functions
produce, and reuses `get_shap_explainer` / `local_explanation` at
request time).
"""

from typing import Dict, Tuple

import numpy as np
import optuna
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

RANDOM_STATE = 42
MAX_FPR_TARGET = 0.05


def recall_at_fixed_fpr(y_true, y_proba, max_fpr: float = MAX_FPR_TARGET) -> float:
    """Best recall achievable while keeping false-positive rate <= max_fpr.

    This is exactly the operating point the business cares about (see
    optimize_threshold below), so we use it directly as the Optuna
    tuning objective rather than a proxy metric like PR-AUC -- optimizing
    PR-AUC can favor models that are good "on average" across all
    thresholds while being mediocre at the specific low-FPR region we
    actually deploy at.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    feasible = tpr[fpr <= max_fpr]
    return float(feasible.max()) if len(feasible) else 0.0


# ----------------------------------------------------------------------
# Baseline model
# ----------------------------------------------------------------------
def train_baseline(X_train, y_train, model_type: str = "logistic_regression"):
    """Train a simple baseline model on SMOTE-resampled training data.

    SMOTE is applied INSIDE an imblearn Pipeline so that resampling only
    ever touches the training fold, never the held-out test data (which
    keeps the test-set evaluation honest / leakage-free).
    """
    if model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", clf),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ----------------------------------------------------------------------
# XGBoost + Optuna hyperparameter tuning
# ----------------------------------------------------------------------
def tune_xgboost_with_optuna(X_train, y_train, n_trials: int = 25, cv_folds: int = 3) -> dict:
    """Search XGBoost + SMOTE hyperparameters with Optuna.

    Each trial builds a SMOTE -> XGBoost pipeline and scores it with
    stratified k-fold cross-validation using `recall_at_fixed_fpr` --
    i.e. we directly optimize for "how much fraud do we catch while
    keeping false alarms under the target rate", the actual deployment
    objective, rather than a threshold-agnostic proxy like PR-AUC.

    `smote_sampling_strategy` (the ratio of synthetic-minority to
    majority samples SMOTE creates) is tuned ALONGSIDE the XGBoost
    hyperparameters: SMOTE's default fully balances the classes 1:1,
    which for a ~1.6%-fraud dataset means generating ~60x synthetic
    fraud examples. In this dataset's high-dimensional, mostly one-hot
    feature space that much synthetic oversampling produces noisy
    near-duplicate points that blur the true decision boundary. Letting
    Optuna search a more moderate ratio (e.g. minority = 10-50% of
    majority) consistently recovers a cleaner, better-generalizing
    boundary.
    """

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    y_train_arr = np.asarray(y_train)

    def objective(trial: optuna.Trial) -> float:
        smote_ratio = trial.suggest_float("smote_sampling_strategy", 0.1, 1.0)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "eval_metric": "aucpr",
            "n_jobs": -1,
        }

        fold_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train_arr):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train_arr[train_idx], y_train_arr[val_idx]

            pipeline = ImbPipeline([
                ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_STATE)),
                ("clf", XGBClassifier(**params)),
            ])
            pipeline.fit(X_fold_train, y_fold_train)
            proba = pipeline.predict_proba(X_fold_val)[:, 1]
            fold_scores.append(recall_at_fixed_fpr(y_fold_val, proba))

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params


def train_final_xgboost(X_train, y_train, best_params: dict) -> ImbPipeline:
    """Fit the final XGBoost model (with tuned hyperparameters, including
    the tuned SMOTE ratio) on the full training set.
    """
    params = dict(best_params)
    smote_ratio = params.pop("smote_sampling_strategy", 1.0)
    params.update({"random_state": RANDOM_STATE, "eval_metric": "aucpr", "n_jobs": -1})

    pipeline = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_STATE)),
        ("clf", XGBClassifier(**params)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate_at_threshold(y_true, y_proba, threshold: float) -> Dict[str, float]:
    """Compute fraud-detection metrics at a given decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = recall_score(y_true, y_pred, zero_division=0)  # fraud caught rate
    precision = precision_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # legit txns wrongly flagged
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    return {
        "threshold": threshold,
        "recall": recall,
        "precision": precision,
        "false_positive_rate": fpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


# ----------------------------------------------------------------------
# Cost-sensitive threshold optimization
# ----------------------------------------------------------------------
def optimize_threshold(
    y_true, y_proba, max_fpr: float = 0.05, min_recall: float = 0.80
) -> Tuple[float, Dict[str, float]]:
    """Pick a decision threshold that satisfies the business targets:
    fraud recall > `min_recall` AND false-positive rate < `max_fpr`.

    Strategy: scan a fine grid of thresholds, keep every threshold that
    satisfies BOTH constraints, and among those pick the one with the
    highest recall (ties broken by lowest FPR). If no threshold satisfies
    both constraints (e.g. on a very hard dataset), fall back to the
    threshold that satisfies the FPR cap and gets closest to the recall
    target -- flagging more fraud is preferred over silently failing.
    """
    thresholds = np.linspace(0.01, 0.99, 197)
    candidates = []
    for t in thresholds:
        m = evaluate_at_threshold(y_true, y_proba, t)
        candidates.append(m)

    feasible = [m for m in candidates if m["recall"] >= min_recall and m["false_positive_rate"] <= max_fpr]

    if feasible:
        # Among feasible thresholds, prefer the lowest FPR (most precise)
        # while still keeping recall above target.
        best = min(feasible, key=lambda m: (m["false_positive_rate"], -m["recall"]))
    else:
        # Relax: satisfy the FPR cap, get as close to the recall target as possible
        under_fpr = [m for m in candidates if m["false_positive_rate"] <= max_fpr]
        pool = under_fpr if under_fpr else candidates
        best = max(pool, key=lambda m: m["recall"])

    return best["threshold"], best


# ----------------------------------------------------------------------
# SHAP explainability
# ----------------------------------------------------------------------
def get_shap_explainer(xgb_model) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer for the trained XGBoost booster.

    TreeExplainer computes exact Shapley values efficiently for tree
    ensembles (no sampling / approximation needed like KernelExplainer).
    """
    return shap.TreeExplainer(xgb_model)


def global_feature_importance(explainer: shap.TreeExplainer, X_sample, feature_names) -> Dict[str, float]:
    """Mean absolute SHAP value per feature across a sample of rows --
    the standard "global importance" summary derived from local
    explanations.
    """
    shap_values = explainer.shap_values(X_sample)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs_shap.tolist()))
    return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))


def local_explanation(explainer: shap.TreeExplainer, x_row, feature_names, top_n: int = 5) -> list:
    """SHAP contribution for a single transaction, returned as a list of
    {feature, value, shap_value} dicts sorted by |shap_value| descending.
    Positive shap_value pushes the prediction TOWARD fraud; negative
    pushes it toward legitimate.
    """
    shap_values = explainer.shap_values(x_row)
    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    row_values = x_row[0] if hasattr(x_row, "ndim") and x_row.ndim == 2 else x_row

    contributions = [
        {
            "feature": feature_names[i],
            "value": float(row_values[i]),
            "shap_value": float(shap_values[i]),
        }
        for i in range(len(feature_names))
    ]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    return contributions[:top_n]
