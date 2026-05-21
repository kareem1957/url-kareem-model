"""
Train the LightGBM phishing classifier.

End-to-end pipeline:
  1. Load train/val/test splits from data/processed/
  2. Extract features for each split (uses src/features/feature_pipeline)
  3. Train LightGBM with early stopping on validation
  4. Evaluate on the held-out test set
  5. Persist the model and metrics to artifacts/

Run from project root:
    python scripts/train_lgbm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import configure_logging  # noqa: E402
from config.settings import settings  # noqa: E402
from src.features.feature_pipeline import (  # noqa: E402
    N_FEATURES,
    extract_features,
)
from src.models.lgbm_model import (  # noqa: E402
    _compute_metrics,
    save_model,
    train_lgbm,
)
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger(__name__)


def _build_feature_matrix(
    df: pd.DataFrame,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for every URL in a DataFrame. Skips rows where
    extraction fails."""
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    failures = 0

    for url, label in tqdm(
        zip(df["url"], df["label"]),
        total=len(df),
        desc=f"Extracting features ({name})",
        ascii=True,
    ):
        try:
            vec, _ = extract_features(url)
            X_list.append(vec)
            y_list.append(int(label))
        except Exception:
            failures += 1
            continue

    logger.info(
        f"{name}_features_built",
        rows=len(X_list),
        failures=failures,
        n_features=N_FEATURES,
    )
    return np.vstack(X_list), np.array(y_list, dtype=np.int8)


def main() -> int:
    configure_logging()
    start = time.time()

    print("=" * 60)
    print("LightGBM Phishing Classifier - Training")
    print("=" * 60)
    print()

    # 1. Load splits
    train_df = pd.read_parquet(settings.processed_data_dir / "train.parquet")
    val_df = pd.read_parquet(settings.processed_data_dir / "val.parquet")
    test_df = pd.read_parquet(settings.processed_data_dir / "test.parquet")
    print(f"Loaded splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print()

    # 2. Feature extraction (the slow step; ~3-5 min on ~330k URLs)
    X_train, y_train = _build_feature_matrix(train_df, "train")
    X_val, y_val = _build_feature_matrix(val_df, "val")
    X_test, y_test = _build_feature_matrix(test_df, "test")
    print()
    print(f"Feature matrices: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    print()

    # 3. Train
    trained = train_lgbm(X_train, y_train, X_val, y_val)
    print()
    print("Validation metrics:")
    print(json.dumps(trained.metrics, indent=2))

    # 4. Final evaluation on the held-out test set
    test_proba = trained.predict_proba(X_test)
    test_pred = (test_proba >= 0.5).astype(np.int8)
    test_metrics = _compute_metrics(y_test, test_pred, test_proba)
    print()
    print("=" * 60)
    print("TEST SET METRICS (final, untouched until now)")
    print("=" * 60)
    print(json.dumps(test_metrics, indent=2))

    # 5. Persist model and metrics
    model_path = settings.models_dir / "lgbm_v1.joblib"
    save_model(trained, model_path)

    metrics_path = settings.artifacts_dir / "metrics" / "lgbm_v1_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump({"val": trained.metrics, "test": test_metrics}, f, indent=2)

    elapsed = time.time() - start
    print()
    print(f"Model saved to:   {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Total time: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())