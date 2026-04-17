"""
Feature Engineering Pipeline
=============================
Builds model-ready features from raw stored data.
Every feature is numeric. No strings, no categoricals.
"""
import pandas as pd
import numpy as np
import logging

from data.store import Store

logger = logging.getLogger(__name__)


def build_features(store: Store, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Pull all data from store, join on timestamp, and compute derived features.
    Returns a DataFrame where each row is a timestep and columns are the 25 features.
    """
    df = store.join_all(start, end)

    if df.empty:
        logger.warning("No data from store.join_all()")
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)

    # === DERIVED FEATURES (computed from raw stored values) ===

    # Fear & Greed derived
    if "fng_value" in df.columns:
        df["fng_7d_avg"] = df["fng_value"].rolling(42, min_periods=1).mean()  # 42 periods = 7 days of 4h
        df["fng_delta_24h"] = df["fng_value"] - df["fng_value"].shift(6)  # 6 periods = 24h
    else:
        df["fng_7d_avg"] = np.nan
        df["fng_delta_24h"] = np.nan

    # Consensus bias spread (short-term vs long-term sentiment divergence)
    if "consensus_bias_7d" in df.columns and "consensus_bias_90d" in df.columns:
        df["consensus_spread_7d_90d"] = df["consensus_bias_7d"] - df["consensus_bias_90d"]
    else:
        df["consensus_spread_7d_90d"] = np.nan

    # Options divergence (Deribit vs IBIT)
    # Use 30d bucket for divergence since IBIT 7d bucket is often empty
    deribit_30d = df.get("consensus_bias_30d")
    ibit_30d = df.get("consensus_bias_30d_ibit_options")
    if deribit_30d is not None and ibit_30d is not None:
        df["options_divergence"] = deribit_30d - ibit_30d
    else:
        # Fall back to aggregate bias
        deribit_bias = df.get("consensus_bias_pct")
        ibit_bias = df.get("consensus_bias_pct_ibit_options")
        if deribit_bias is not None and ibit_bias is not None:
            df["options_divergence"] = deribit_bias - ibit_bias
        else:
            df["options_divergence"] = np.nan

    # Deribit skew divergence
    if "near_term_skew" in df.columns and "far_term_skew" in df.columns:
        df["deribit_skew_divergence"] = df["near_term_skew"] - df["far_term_skew"]
    else:
        df["deribit_skew_divergence"] = np.nan

    # Whale flow trend (7-day slope of net exchange flow)
    if "net_exchange_flow" in df.columns:
        df["whale_flow_7d_trend"] = df["net_exchange_flow"].rolling(42, min_periods=6).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 6 else np.nan,
            raw=False,
        )
    else:
        df["whale_flow_7d_trend"] = np.nan

    # Polymarket probability delta (computed from stored history)
    if "btc_primary_prob" in df.columns:
        df["polymarket_prob_delta_computed"] = (
            df["btc_primary_prob"] - df["btc_primary_prob"].shift(6)
        )
    else:
        df["polymarket_prob_delta_computed"] = np.nan

    # Price-derived features
    if "close" in df.columns:
        df["returns_24h"] = df["close"].pct_change(6) * 100  # 6 periods = 24h
        df["returns_7d"] = df["close"].pct_change(42) * 100  # 42 periods = 7 days
        df["volatility_7d"] = df["close"].pct_change().rolling(42).std() * 100
    else:
        df["returns_24h"] = np.nan
        df["returns_7d"] = np.nan
        df["volatility_7d"] = np.nan

    # === SELECT FINAL FEATURE COLUMNS ===
    feature_cols = [
        # Fear & Greed (3)
        "fng_value",
        "fng_7d_avg",
        "fng_delta_24h",
        # Deribit Options (8)
        "put_call_ratio",       # from deribit_options table
        "consensus_bias_pct",   # all expirations
        "consensus_bias_7d",    # options expiring within 7 days
        "consensus_bias_30d",   # options expiring within 30 days
        "consensus_bias_90d",   # options expiring within 90 days
        "consensus_spread_7d_90d",  # short vs long term divergence
        "near_term_skew",
        "far_term_skew",
        "deribit_skew_divergence",
        # IBIT Options (5)
        # These get suffixed by join_all if column names collide
        "consensus_bias_pct_ibit_options" if "consensus_bias_pct_ibit_options" in df.columns else "consensus_bias_pct",
        "consensus_bias_7d_ibit_options" if "consensus_bias_7d_ibit_options" in df.columns else "consensus_bias_7d",
        "consensus_bias_30d_ibit_options" if "consensus_bias_30d_ibit_options" in df.columns else "consensus_bias_30d",
        "near_term_skew_ibit_options" if "near_term_skew_ibit_options" in df.columns else "near_term_skew",
        "options_divergence",
        # Whale Activity (3)
        "net_exchange_flow",
        "large_tx_volume_btc",
        "whale_flow_7d_trend",
        # Polymarket (2)
        "btc_primary_prob",
        "polymarket_prob_delta_computed",
        # Technicals (6)
        "ema_ratio",
        "rsi_14",
        "atr_14",
        "obv_slope",
        "bb_pct_b",
        "macd_hist",
        # Macro (6)
        "dxy",
        "dxy_7d_return",
        "us10y_yield",
        "sp500_7d_return",
        "qqq_7d_return",
        "oil_price",
        "oil_7d_return",
        "btc_gold_ratio",
        "m2_supply",
        "m2_yoy_growth",
        # Price-derived (3)
        "returns_24h",
        "returns_7d",
        "volatility_7d",
    ]

    # Only keep columns that actually exist
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.info(f"Missing feature columns (will be NaN): {missing}")
        for col in missing:
            df[col] = np.nan
        available = feature_cols

    features = df[["timestamp"] + available].copy()
    features.columns = ["timestamp"] + [_clean_col_name(c) for c in available]

    logger.info(f"Built {len(features)} rows with {len(available)} features")
    return features


def _clean_col_name(col: str) -> str:
    """Normalize column names for the model."""
    # Remove table suffixes from join_all
    suffixes = ["_ibit_options", "_deribit_options", "_whale_activity", "_polymarket"]
    for s in suffixes:
        col = col.replace(s, "")
    return col
