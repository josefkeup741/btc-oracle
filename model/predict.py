"""
Live Prediction
================
Loads the latest trained models and generates a prediction
from the most recent feature snapshot.
"""
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from data.store import Store
from data.features import build_features
from config import MODEL_DIR

logger = logging.getLogger(__name__)


def predict(store: Store) -> dict | None:
    """
    Generate a prediction using the latest stored data and trained models.

    Returns:
        dict with keys:
            - timestamp: when prediction was made
            - direction: "UP" or "DOWN"
            - direction_probability: float 0-1
            - magnitude_pct: expected % change
            - confidence: "HIGH", "MEDIUM", or "LOW"
            - features_used: dict of current feature values
    """
    # Load models
    clf_path = MODEL_DIR / "direction_clf_latest.pkl"
    reg_path = MODEL_DIR / "magnitude_reg_latest.pkl"
    meta_path = MODEL_DIR / "meta_latest.pkl"

    for path in [clf_path, reg_path, meta_path]:
        if not path.exists():
            logger.error(f"Model not found: {path}. Run training first.")
            return None

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(reg_path, "rb") as f:
        reg = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Build features from latest data
    features = build_features(store)
    if features.empty:
        logger.error("No features available for prediction")
        return None

    # Use the most recent row
    latest = features.iloc[[-1]]
    feature_cols = meta.get("feature_names", [])

    # build_features can return duplicate column names (IBIT/Deribit
    # collisions from the asof join); dedup to match the training layout.
    latest = latest.copy()
    if latest.columns.duplicated().any():
        cols, seen = [], {}
        for c in latest.columns:
            if c in seen:
                seen[c] += 1
                cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                cols.append(c)
        latest.columns = cols

    # Add any expected columns the live row is missing, as NaN.
    for c in feature_cols:
        if c not in latest.columns:
            latest[c] = np.nan

    X = latest[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")

    # Predict
    dir_prob = clf.predict_proba(X)[0, 1]  # Probability of "UP"
    magnitude = reg.predict(X)[0]

    direction = "UP" if dir_prob >= 0.5 else "DOWN"

    # Confidence based on probability distance from 0.5
    prob_strength = abs(dir_prob - 0.5) * 2  # Normalize to 0-1
    if prob_strength > 0.3:
        confidence = "HIGH"
    elif prob_strength > 0.15:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Agreement check: do direction and magnitude models agree?
    models_agree = (direction == "UP" and magnitude > 0) or (direction == "DOWN" and magnitude < 0)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_horizon": "7 days",
        "direction": direction,
        "direction_probability": round(dir_prob, 4),
        "magnitude_pct": round(magnitude, 2),
        "confidence": confidence,
        "models_agree": models_agree,
        "features_snapshot": {col: round(float(X[col].iloc[0]), 4) if pd.notna(X[col].iloc[0]) else None for col in feature_cols},
    }

    _print_prediction(result)
    return result


def _print_prediction(result: dict):
    """Print a readable prediction."""
    print("\n" + "=" * 55)
    print("      BTC ORACLE - 7 DAY PREDICTION")
    print("=" * 55)

    d = result["direction"]
    prob = result["direction_probability"]
    mag = result["magnitude_pct"]
    conf = result["confidence"]
    agree = result["models_agree"]

    arrow = "↑" if d == "UP" else "↓"
    emoji = "🟢" if d == "UP" else "🔴"

    print(f"\n  {emoji} Direction:  {d} {arrow}")
    print(f"  📊 Probability: {prob:.1%}")
    print(f"  📏 Expected:    {mag:+.1f}%")
    print(f"  🎯 Confidence:  {conf}")
    print(f"  🤝 Models Agree: {'Yes' if agree else 'NO — treat with caution'}")

    if not agree:
        print(f"\n  ⚠️  Direction model says {d} but magnitude model says {mag:+.1f}%.")
        print(f"     This disagreement suggests uncertainty. Consider waiting.")

    print("\n" + "=" * 55)
