"""
Target/Label Generation
========================
Creates prediction targets from price data.
Direction (binary classification) and Magnitude (regression).
"""
import pandas as pd
import numpy as np
import logging

from data.store import Store
from config import PREDICTION_HORIZON_PERIODS

logger = logging.getLogger(__name__)


def generate_targets(store: Store, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Generate prediction targets from stored price data.

    Returns DataFrame with columns:
        - timestamp: the prediction point
        - direction_7d: 1 if price higher in 7 days, 0 if lower
        - magnitude_7d: percentage change over 7 days
        - future_close: the actual future close price (for evaluation only)
    """
    prices = store.query("btc_price", start, end)

    if prices.empty:
        logger.warning("No price data for target generation")
        return pd.DataFrame()

    prices = prices.sort_values("timestamp").reset_index(drop=True)

    # Future price = close price PREDICTION_HORIZON_PERIODS ahead
    prices["future_close"] = prices["close"].shift(-PREDICTION_HORIZON_PERIODS)

    # Direction: 1 = up, 0 = down
    prices["direction_7d"] = (prices["future_close"] > prices["close"]).astype(int)

    # Magnitude: percentage change
    prices["magnitude_7d"] = (
        (prices["future_close"] - prices["close"]) / prices["close"]
    ) * 100

    # Drop rows where we don't have future data (last 7 days)
    targets = prices[["timestamp", "direction_7d", "magnitude_7d", "future_close"]].dropna()

    logger.info(
        f"Generated {len(targets)} targets. "
        f"Up: {targets['direction_7d'].sum()} ({targets['direction_7d'].mean():.1%}), "
        f"Down: {(1 - targets['direction_7d']).sum()}"
    )

    return targets
