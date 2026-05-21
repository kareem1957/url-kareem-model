"""
LightGBM phishing classifier.

This is our baseline model. Trained on the 65 hand-engineered features
from src/features/, it should hit ~96-98% accuracy on the held-out
test set. Deep learning models in later phases must beat this number
to justify their complexity.

The model is intentionally configured for INFERENCE SPEED in addition
to accuracy. Hyperparameters favor smaller, shallower trees (faster
prediction at the cost of a tiny accuracy drop) because the model will
serve real-time API requests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np

from src.features.feature_pipeline import FEATURE_NAMES, N_FEATURES
from src.utils.logger import get_logger


logger = get_logger(__name__)


# Hyperparameters chosen for production:
#   - num_leaves=63    : moderate complexity; 127 fits better but is slower
#   - max_depth=-1     : let leaves grow naturally; LightGBM is leaf-wise
#   - learning_rate=0.05: low LR + many estimators = better generalization
#   - n_estimators=500 : capped; early stopping decides the real number
#   - min_child_samples=20 : prevents trees memorizing tiny phishing groups
#   - reg_alpha/lambda : L1+L2 regularization, mild
#   - is_unbalance=True : handles the 30/70 class imbalance automatically
#   - n_jobs=-1        : use all CPU cores during training
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "is_unbalance": True,
    "feature_fraction": 0.9,    # bagging on features per tree
    "bagging_fraction": 0.9,    # bagging on rows per tree
    "bagging_freq": 5,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 42,
}


@dataclass
class TrainedLGBM:
    """Wrapper bundling the trained model with its schema and metadata."""

    model: lgb.LGBMClassifier
    feature_names: tuple[str, ...]
    n_features: int
    metrics: dict[str, float]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return phishing probability for each row in X."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}. "
                "Feature pipeline schema mismatch."
            )
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return integer predictions (0=legit, 1=phishing)."""
        return (self.predict_proba(X) >= threshold).astype(np.int8)


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 50,
) -> TrainedLGBM:
    """
    Train a LightGBM classifier with early stopping on the validation set.

    Early stopping is critical: we set n_estimators=500 as an upper bound,
    but training stops automatically when validation AUC plateaus. This
    saves training time AND prevents overfitting.
    """
    if X_train.shape[1] != N_FEATURES:
        raise ValueError(
            f"X_train has {X_train.shape[1]} features; "
            f"feature pipeline produces {N_FEATURES}. Schema mismatch."
        )

    params = {**DEFAULT_PARAMS, **(params or {})}

    logger.info(
        "lgbm_training_starting",
        train_rows=len(X_train),
        val_rows=len(X_val),
        n_features=X_train.shape[1],
        params={k: v for k, v in params.items() if k != "metric"},
    )

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=20),  # log every 20 rounds
        ],
    )

    # Collect metrics on the validation set
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(np.int8)
    metrics = _compute_metrics(y_val, val_pred, val_proba)
    logger.info("lgbm_training_complete", **metrics)

    return TrainedLGBM(
        model=model,
        feature_names=FEATURE_NAMES,
        n_features=N_FEATURES,
        metrics=metrics,
    )


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute the metrics most relevant for phishing detection.

    Accuracy alone is misleading on imbalanced data. For phishing:
      - Recall (TPR)    : fraction of actual phishing we catch
      - Precision       : fraction of "phishing" alerts that are real
      - FPR             : fraction of legit URLs we wrongly flag
      - ROC-AUC         : threshold-independent discrimination quality
    Operators tune the classification threshold based on whether the
    business prefers fewer missed phish (raise recall) or fewer false
    alarms (raise precision).
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        confusion_matrix,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def save_model(model: TrainedLGBM, path: Path) -> None:
    """Persist the trained model bundle. We store BOTH the model and the
    feature schema so inference code can verify alignment at load time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.model,
        "feature_names": model.feature_names,
        "n_features": model.n_features,
        "metrics": model.metrics,
    }
    joblib.dump(payload, path, compress=3)
    logger.info("lgbm_saved", path=str(path), size_mb=path.stat().st_size / 1e6)


def load_model(path: Path) -> TrainedLGBM:
    """Load a previously saved model. Verifies schema alignment."""
    payload = joblib.load(path)
    if payload["n_features"] != N_FEATURES:
        raise RuntimeError(
            f"Model was trained on {payload['n_features']} features but "
            f"current pipeline produces {N_FEATURES}. Refusing to load."
        )
    if tuple(payload["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError(
            "Model feature names differ from current pipeline schema. "
            "Retrain the model after schema changes."
        )
    logger.info("lgbm_loaded", path=str(path), metrics=payload["metrics"])
    return TrainedLGBM(
        model=payload["model"],
        feature_names=payload["feature_names"],
        n_features=payload["n_features"],
        metrics=payload["metrics"],
    )