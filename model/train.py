"""
XGBoost Training with Walk-Forward Validation
===============================================
Trains Direction (classifier) and Magnitude (regressor) models.
Uses expanding-window walk-forward CV to avoid look-ahead bias.
"""
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error,
)

from data.store import Store
from data.features import build_features
from model.targets import generate_targets
from config import (
    XGB_CLASSIFIER_PARAMS, XGB_REGRESSOR_PARAMS,
    EARLY_STOPPING_ROUNDS, WF_TRAIN_MONTHS, WF_TEST_MONTHS,
    MODEL_DIR,
)

logger = logging.getLogger(__name__)


def train(store: Store, save: bool = True) -> dict:
    """
    Full training pipeline:
    1. Build features and targets
    2. Align them on timestamp
    3. Run walk-forward validation
    4. Train final models on all data
    5. Save models and return metrics

    Returns dict with metrics and model paths.
    """
    logger.info("=== Starting Training Pipeline ===")

    # 1. Build features and targets
    features = build_features(store)
    targets = generate_targets(store)

    if features.empty or targets.empty:
        logger.error("Cannot train: no features or targets")
        return {"error": "insufficient data"}

    # 2. Merge features and targets on timestamp
    merged = pd.merge(features, targets, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])

    logger.info(f"Merged dataset: {len(merged)} rows")

    # Separate feature columns from target columns
    target_cols = ["direction_7d", "magnitude_7d", "future_close"]
    feature_cols = [c for c in merged.columns if c not in target_cols + ["timestamp"]]

    # Deduplicate column names first
    if merged.columns.duplicated().any():
        cols = []
        seen = {}
        for c in merged.columns:
            if c in seen:
                seen[c] += 1
                cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                cols.append(c)
        merged.columns = cols
        feature_cols = [c for c in merged.columns if c not in target_cols + ["timestamp"]]

    # Force all feature columns to float64
    for col in feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")

    X = merged[feature_cols]
    y_dir = merged["direction_7d"]
    y_mag = merged["magnitude_7d"]
    timestamps = merged["timestamp"]

    # Handle NaN in features (XGBoost handles this natively, but log it)
    nan_pct = X.isna().mean()
    high_nan = nan_pct[nan_pct > 0.5]
    if not high_nan.empty:
        logger.warning(f"Features with >50% NaN (consider dropping): {dict(high_nan)}")

    # 3. Walk-forward validation
    logger.info("Running walk-forward validation...")
    wf_results = _walk_forward_cv(X, y_dir, y_mag, timestamps)

    # 4. Train final models on ALL data
    logger.info("Training final models on full dataset...")

    clf = XGBClassifier(**XGB_CLASSIFIER_PARAMS)
    clf.fit(X, y_dir)

    reg = XGBRegressor(**XGB_REGRESSOR_PARAMS)
    reg.fit(X, y_mag)

    # 5. Save models
    result = {
        "walk_forward": wf_results,
        "n_samples": len(merged),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "trained_at": datetime.utcnow().isoformat(),
    }

    if save:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        clf_path = MODEL_DIR / f"direction_clf_{ts}.pkl"
        reg_path = MODEL_DIR / f"magnitude_reg_{ts}.pkl"
        meta_path = MODEL_DIR / f"meta_{ts}.pkl"

        # Also save as "latest" for easy loading
        clf_latest = MODEL_DIR / "direction_clf_latest.pkl"
        reg_latest = MODEL_DIR / "magnitude_reg_latest.pkl"
        meta_latest = MODEL_DIR / "meta_latest.pkl"

        for path in [clf_path, clf_latest]:
            with open(path, "wb") as f:
                pickle.dump(clf, f)

        for path in [reg_path, reg_latest]:
            with open(path, "wb") as f:
                pickle.dump(reg, f)

        for path in [meta_path, meta_latest]:
            with open(path, "wb") as f:
                pickle.dump(result, f)

        result["model_paths"] = {
            "classifier": str(clf_path),
            "regressor": str(reg_path),
            "meta": str(meta_path),
        }
        logger.info(f"Models saved to {MODEL_DIR}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "clf_importance": clf.feature_importances_,
        "reg_importance": reg.feature_importances_,
    }).sort_values("clf_importance", ascending=False)

    result["feature_importance"] = importance.to_dict("records")

    _print_summary(result)
    return result


def _walk_forward_cv(
    X: pd.DataFrame,
    y_dir: pd.Series,
    y_mag: pd.Series,
    timestamps: pd.Series,
) -> dict:
    """
    Expanding-window walk-forward cross-validation.
    Train on months 1..N, test on month N+1. Expand window each fold.
    """
    min_date = timestamps.min()
    max_date = timestamps.max()

    # Start testing after WF_TRAIN_MONTHS
    test_start = min_date + relativedelta(months=WF_TRAIN_MONTHS)

    all_dir_preds = []
    all_dir_actuals = []
    all_dir_probs = []
    all_mag_preds = []
    all_mag_actuals = []
    fold = 0

    current_test_start = test_start

    while current_test_start < max_date:
        current_test_end = current_test_start + relativedelta(months=WF_TEST_MONTHS)

        train_mask = timestamps < current_test_start
        test_mask = (timestamps >= current_test_start) & (timestamps < current_test_end)

        X_train, X_test = X[train_mask], X[test_mask]
        y_dir_train, y_dir_test = y_dir[train_mask], y_dir[test_mask]
        y_mag_train, y_mag_test = y_mag[train_mask], y_mag[test_mask]

        if len(X_train) < 100 or len(X_test) < 10:
            current_test_start = current_test_end
            continue

        # Direction model
        clf = XGBClassifier(**XGB_CLASSIFIER_PARAMS)
        clf.fit(
            X_train, y_dir_train,
            eval_set=[(X_test, y_dir_test)],
            verbose=False,
        )
        dir_probs = clf.predict_proba(X_test)[:, 1]
        dir_preds = (dir_probs >= 0.5).astype(int)

        # Magnitude model
        reg = XGBRegressor(**XGB_REGRESSOR_PARAMS)
        reg.fit(
            X_train, y_mag_train,
            eval_set=[(X_test, y_mag_test)],
            verbose=False,
        )
        mag_preds = reg.predict(X_test)

        all_dir_preds.extend(dir_preds)
        all_dir_actuals.extend(y_dir_test)
        all_dir_probs.extend(dir_probs)
        all_mag_preds.extend(mag_preds)
        all_mag_actuals.extend(y_mag_test)

        fold += 1
        current_test_start = current_test_end

    if not all_dir_actuals:
        logger.warning("No walk-forward folds completed")
        return {"error": "insufficient data for walk-forward"}

    # Aggregate metrics
    dir_actual = np.array(all_dir_actuals)
    dir_pred = np.array(all_dir_preds)
    dir_prob = np.array(all_dir_probs)
    mag_actual = np.array(all_mag_actuals)
    mag_pred = np.array(all_mag_preds)

    metrics = {
        "n_folds": fold,
        "n_test_samples": len(dir_actual),
        "direction": {
            "accuracy": accuracy_score(dir_actual, dir_pred),
            "precision": precision_score(dir_actual, dir_pred, zero_division=0),
            "recall": recall_score(dir_actual, dir_pred, zero_division=0),
            "f1": f1_score(dir_actual, dir_pred, zero_division=0),
            "auc_roc": roc_auc_score(dir_actual, dir_prob) if len(np.unique(dir_actual)) > 1 else None,
            "baseline_accuracy": max(dir_actual.mean(), 1 - dir_actual.mean()),
        },
        "magnitude": {
            "mae": mean_absolute_error(mag_actual, mag_pred),
            "rmse": np.sqrt(mean_squared_error(mag_actual, mag_pred)),
            "directional_accuracy": np.mean(np.sign(mag_pred) == np.sign(mag_actual)),
        },
    }

    return metrics


def _print_summary(result: dict):
    """Print a readable training summary."""
    wf = result.get("walk_forward", {})

    print("\n" + "=" * 65)
    print("      BTC ORACLE - TRAINING SUMMARY")
    print("=" * 65)
    print(f"Samples: {result['n_samples']}  |  Features: {result['n_features']}")
    print(f"Walk-forward folds: {wf.get('n_folds', 'N/A')}")
    print("-" * 65)

    d = wf.get("direction", {})
    if d:
        print(f"\nDIRECTION MODEL (7-day):")
        print(f"  Accuracy:  {d.get('accuracy', 0):.1%}  (baseline: {d.get('baseline_accuracy', 0):.1%})")
        print(f"  Precision: {d.get('precision', 0):.1%}  |  Recall: {d.get('recall', 0):.1%}")
        print(f"  F1 Score:  {d.get('f1', 0):.1%}  |  AUC-ROC: {d.get('auc_roc', 'N/A')}")

    m = wf.get("magnitude", {})
    if m:
        print(f"\nMAGNITUDE MODEL (7-day):")
        print(f"  MAE:  {m.get('mae', 0):.2f}%")
        print(f"  RMSE: {m.get('rmse', 0):.2f}%")
        print(f"  Directional accuracy: {m.get('directional_accuracy', 0):.1%}")

    # Top features
    imp = result.get("feature_importance", [])
    if imp:
        print(f"\nTOP 10 FEATURES (classifier):")
        for i, row in enumerate(imp[:10]):
            bar = "█" * int(row["clf_importance"] * 50)
            print(f"  {i+1:2d}. {row['feature']:<30s} {bar} ({row['clf_importance']:.3f})")

    print("=" * 65)
