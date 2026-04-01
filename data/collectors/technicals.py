"""
Technical Indicators Collector
==============================
Computes RSI, EMA ratio, ATR, MACD histogram, Bollinger %B, and OBV slope
from stored BTC price data. Unlike other collectors, this reads from the
store rather than an external API.
"""
import pandas as pd
import numpy as np

from data.collectors.base import BaseCollector
from config import (
    EMA_FAST, EMA_SLOW, RSI_PERIOD, ATR_PERIOD,
    BB_PERIOD, BB_STD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    OBV_SLOPE_PERIOD,
)


class TechnicalsCollector(BaseCollector):
    """
    Unlike other collectors, this one requires price data already in the store.
    Call compute_from_df() directly with a price DataFrame, or compute_latest()
    with a Store instance.
    """

    table_name = "technicals"

    def __init__(self, store=None):
        super().__init__()
        self._store = store

    def collect(self) -> dict | None:
        """
        Compute technicals from the latest stored price data.
        Needs at least EMA_SLOW + 10 rows of price data.
        """
        if self._store is None:
            self.logger.error("TechnicalsCollector requires a Store instance")
            return None

        df = self._store.query("btc_price")
        if df.empty or len(df) < EMA_SLOW + 20:
            self.logger.warning(f"Not enough price data ({len(df)} rows) for technicals")
            return None

        df = df.sort_values("timestamp").reset_index(drop=True)
        result = self.compute_from_df(df)

        if result is None or result.empty:
            return None

        # Return the latest row
        latest = result.iloc[-1].to_dict()
        # Remove timestamp if present (added by runner)
        latest.pop("timestamp", None)
        return latest

    def compute_from_df(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Compute all technical indicators from a price DataFrame.
        Expects columns: open, high, low, close, volume (optional).
        Returns a DataFrame with indicator columns and timestamp.
        """
        if len(df) < EMA_SLOW + 20:
            return None

        result = pd.DataFrame()
        result["timestamp"] = df["timestamp"]

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0, index=df.index)

        # 1. EMA Ratio (trend)
        ema_fast = close.ewm(span=EMA_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=EMA_SLOW, adjust=False).mean()
        result["ema_ratio"] = ema_fast / ema_slow

        # 2. RSI (momentum)
        result["rsi_14"] = self._rsi(close, RSI_PERIOD)

        # 3. ATR (volatility)
        result["atr_14"] = self._atr(high, low, close, ATR_PERIOD)

        # 4. OBV Slope (volume trend)
        obv = self._obv(close, volume)
        result["obv_slope"] = obv.rolling(OBV_SLOPE_PERIOD).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == OBV_SLOPE_PERIOD else np.nan,
            raw=False,
        )

        # 5. Bollinger Band %B (mean reversion)
        bb_mid = close.rolling(BB_PERIOD).mean()
        bb_std = close.rolling(BB_PERIOD).std()
        bb_upper = bb_mid + BB_STD * bb_std
        bb_lower = bb_mid - BB_STD * bb_std
        bb_range = bb_upper - bb_lower
        result["bb_pct_b"] = np.where(bb_range != 0, (close - bb_lower) / bb_range, 0.5)

        # 6. MACD Histogram (trend strength)
        ema_macd_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_macd_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_macd_fast - ema_macd_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        result["macd_hist"] = macd_line - signal_line

        return result

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        return (volume * direction).cumsum()
