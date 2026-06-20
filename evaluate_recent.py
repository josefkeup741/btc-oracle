#!/usr/bin/env python3
"""
BTC Oracle - Recent-Data-Only Evaluation
==========================================
The full-history retrain dilutes the alternative-data signal because options,
whale, and macro-return features are NaN for ~96% of the dataset (everything
before April 2026). This script isolates the period where ALL features are
actually present and asks the real question:

    On the recent period, does adding options + whale + prediction-market data
    beat a model trained on only the always-available features
    (price, technicals, macro levels, Fear & Greed)?

It also applies a whale-flow fix inline (converting cumulative exchange balances
to period-over-period deltas) so we can measure whether that helps before making
it permanent.

IMPORTANT CAVEAT printed at the end: with ~80 days of data and a 7-day prediction
horizon, the number of *independent* test samples is small (~10-15). Treat the
accuracy numbers as directional hints, not verdicts. Feature importance and the
paper-trading log are more trustworthy signals of whether this works.

Usage:
    python evaluate_recent.py
"""
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from data.store import Store
from data.features import build_features
from model.targets import generate_targets
from config import PREDICTION_HORIZON_PERIODS

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_recent")

HORIZON = PREDICTION_HORIZON_PERIODS  # 42 periods = 7 days of 4h candles
EMBARGO = HORIZON                      # gap between train and test to kill label leakage

# Feature name fragments that count as "alternative data" (only present recently)
ALT_FRAGMENTS = (
    "put_call_ratio", "consensus_bias", "consensus_spread", "near_term_skew",
    "far_term_skew", "skew_divergence", "options_divergence",
    "net_exchange_flow", "net_flow_delta", "large_tx_volume", "whale_flow",
    "btc_primary_prob", "polymarket_prob",
)


def try_import_models():
    """Return a dict of {name: (ClassifierClass, params)} for whatever is installed."""
    models = {}
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (XGBClassifier, dict(
            max_depth=4, learning_rate=0.05, n_estimators=300,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            eval_metric="logloss", random_state=42, verbosity=0,
        ))
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = (LGBMClassifier, dict(
            max_depth=4, learning_rate=0.05, n_estimators=300,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=5,
            random_state=42, verbose=-1,
        ))
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = (CatBoostClassifier, dict(
            depth=4, learning_rate=0.05, iterations=300,
            random_state=42, verbose=0, allow_writing_files=False,
        ))
    except Exception:
        pass
    if not models:
        # Fallback so the script always runs (e.g. in a bare environment)
        from sklearn.ensemble import GradientBoostingClassifier
        models["GradientBoosting(sklearn)"] = (GradientBoostingClassifier, dict(
            max_depth=4, learning_rate=0.05, n_estimators=300, random_state=42,
        ))
    return models


def prep_dataset(store: Store):
    """Build features + targets, merge, dedup columns, coerce numeric, apply whale delta."""
    features = build_features(store)
    targets = generate_targets(store)
    if features.empty or targets.empty:
        return None

    merged = pd.merge(features, targets, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], format="mixed", utc=True)

    # Dedup duplicate column names (IBIT/Deribit collisions from the asof join)
    if merged.columns.duplicated().any():
        cols, seen = [], {}
        for c in merged.columns:
            if c in seen:
                seen[c] += 1
                cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                cols.append(c)
        merged.columns = cols

    target_cols = ["direction_7d", "magnitude_7d", "future_close"]
    feat_cols = [c for c in merged.columns if c not in target_cols + ["timestamp"]]
    for c in feat_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float64")

    # --- WHALE FIX (inline) ---
    # The collector stores cumulative lifetime exchange balances, so the raw
    # net_exchange_flow is a near-constant. The period-over-period change is the
    # actual flow. Replace the cumulative column's role with its diff.
    if "net_exchange_flow" in merged.columns:
        merged["net_flow_delta"] = merged["net_exchange_flow"].diff()
        if "net_flow_delta" not in feat_cols:
            feat_cols.append("net_flow_delta")
        # drop the near-constant cumulative from the feature set
        feat_cols = [c for c in feat_cols if c != "net_exchange_flow"]

    return merged, feat_cols, target_cols


def recent_slice(merged: pd.DataFrame, feat_cols):
    """Keep only rows where alternative-data features are actually populated."""
    alt_cols = [c for c in feat_cols if any(f in c for f in ALT_FRAGMENTS)]
    if not alt_cols:
        return merged, alt_cols
    # First row where at least one alt feature is non-NaN
    present = merged[alt_cols].notna().any(axis=1)
    if not present.any():
        return merged.iloc[0:0], alt_cols
    first_idx = present.idxmax()
    return merged.loc[first_idx:].reset_index(drop=True), alt_cols


def drop_dead_features(df, feat_cols):
    """Drop features that are entirely NaN or constant on this slice (no signal)."""
    keep = []
    for c in feat_cols:
        col = df[c]
        if col.notna().sum() == 0:
            continue
        if col.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return keep


def chrono_split(df):
    """70/30 chronological split with an embargo gap to prevent label leakage."""
    n = len(df)
    train_end = int(n * 0.70)
    test_start = train_end + EMBARGO
    if test_start >= n - 5:
        # not enough room for embargo; shrink train
        train_end = max(EMBARGO + 30, int(n * 0.55))
        test_start = train_end + EMBARGO
    train = df.iloc[:train_end]
    test = df.iloc[test_start:]
    return train, test


def evaluate_feature_set(merged, feat_cols, label_name):
    """Train + test on a chronological split, return metrics dict."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    feat_cols = drop_dead_features(merged, feat_cols)
    if len(feat_cols) == 0:
        return None

    train, test = chrono_split(merged)
    if len(train) < 30 or len(test) < 7:
        return {"error": f"insufficient rows (train={len(train)}, test={len(test)})"}

    X_train = train[feat_cols]
    y_train = train["direction_7d"].astype(int)
    X_test = test[feat_cols]
    y_test = test["direction_7d"].astype(int)

    # Median-imputed copies for any model that can't handle NaN natively
    # (XGBoost/LightGBM/CatBoost handle NaN; sklearn fallback does not).
    medians = X_train.median()
    X_train_imp = X_train.fillna(medians)
    X_test_imp = X_test.fillna(medians)

    models = try_import_models()
    results = {}
    importances = {}
    for name, (Cls, params) in models.items():
        try:
            clf = Cls(**params)
            try:
                clf.fit(X_train, y_train)
                proba = clf.predict_proba(X_test)[:, 1]
            except ValueError:
                # Model can't handle NaN -> retry on imputed data
                clf = Cls(**params)
                clf.fit(X_train_imp, y_train)
                proba = clf.predict_proba(X_test_imp)[:, 1]
            pred = (proba >= 0.5).astype(int)
            results[name] = {
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
            }
            if hasattr(clf, "feature_importances_"):
                importances[name] = pd.Series(clf.feature_importances_, index=feat_cols)
        except Exception as e:
            results[name] = {"error": str(e)[:80]}

    baseline = max(y_test.mean(), 1 - y_test.mean())
    indep_samples = max(1, len(test) // HORIZON)
    return {
        "label": label_name,
        "n_features": len(feat_cols),
        "n_train": len(train),
        "n_test": len(test),
        "indep_samples": indep_samples,
        "baseline": baseline,
        "models": results,
        "importances": importances,
        "feat_cols": feat_cols,
    }


def main():
    store = Store()
    prep = prep_dataset(store)
    if prep is None:
        print("No data to evaluate."); return
    merged, feat_cols, target_cols = prep

    recent, alt_cols = recent_slice(merged, feat_cols)
    print("\n" + "=" * 66)
    print("   BTC ORACLE - RECENT-DATA-ONLY EVALUATION")
    print("=" * 66)
    if len(recent) == 0:
        print("No rows with alternative-data features present yet."); return

    print(f"Recent window: {recent['timestamp'].min().date()} -> {recent['timestamp'].max().date()}")
    print(f"Rows in window: {len(recent)}  |  Alt-data features: {len(alt_cols)}")

    # Feature set A: everything
    all_res = evaluate_feature_set(recent, feat_cols, "ALL features")
    # Feature set B: only always-available (exclude alternative data)
    core_cols = [c for c in feat_cols if not any(f in c for f in ALT_FRAGMENTS)]
    core_res = evaluate_feature_set(recent, core_cols, "CORE only (no options/whale/pm)")

    for res in (core_res, all_res):
        if not res or "error" in (res or {}):
            print(f"\n[{res.get('label','?') if res else '?'}] "
                  f"{res.get('error','no result') if res else 'no result'}")
            continue
        print("\n" + "-" * 66)
        print(f"[{res['label']}]  features={res['n_features']}  "
              f"train={res['n_train']}  test={res['n_test']}  "
              f"(~{res['indep_samples']} independent 7d samples)")
        print(f"  Majority-class baseline: {res['baseline']:.1%}")
        for mname, m in res["models"].items():
            if "error" in m:
                print(f"  {mname:<22s} ERROR: {m['error']}")
            else:
                delta = m["accuracy"] - res["baseline"]
                flag = "  <-- beats baseline" if delta > 0 else ""
                print(f"  {mname:<22s} acc={m['accuracy']:.1%}  "
                      f"f1={m['f1']:.1%}  (vs base {delta:+.1%}){flag}")

    # Feature importance from ALL-features model (first available model)
    if all_res and all_res.get("importances"):
        first_model = next(iter(all_res["importances"]))
        imp = all_res["importances"][first_model].sort_values(ascending=False)
        print("\n" + "-" * 66)
        print(f"TOP 15 FEATURES on recent window ({first_model}):")
        for i, (feat, val) in enumerate(imp.head(15).items()):
            tag = " [ALT]" if any(f in feat for f in ALT_FRAGMENTS) else ""
            bar = "#" * int(val / (imp.max() + 1e-9) * 30)
            print(f"  {i+1:2d}. {feat:<28s} {bar} {val:.3f}{tag}")

    print("\n" + "=" * 66)
    print("READ THIS: with ~80 days of data and a 7-day horizon, there are only")
    print("~10-15 INDEPENDENT test samples. Accuracy here is a directional hint,")
    print("not proof. The feature-importance ranking (which signals the model")
    print("leans on when they're present) and the paper-trading log going forward")
    print("are the trustworthy measures. Re-run this monthly as data grows.")
    print("=" * 66)


if __name__ == "__main__":
    main()
