"""
IBIT ETF Options Collector
==========================
Refactored from btcSmartDumbIBIT.ipynb.
Fetches IBIT option chains via yfinance, computes OI-weighted breakevens,
time-bucketed consensus bias (7d/30d/90d), and near/far-term skew.

IBIT captures traditional finance flow (hedge funds, RIAs, retail brokerage)
vs Deribit which captures crypto-native flow. Divergences between the two
are a signal: institutional disagreement across participant bases.

Note: IBIT only trades during US market hours. Collections outside market
hours will return None (handled gracefully by the runner).
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

from data.collectors.base import BaseCollector
from config import IBIT_TICKER, IBIT_MIN_OI_THRESHOLD


class IBITCollector(BaseCollector):

    table_name = "ibit_options"

    # Time bucket boundaries (days to expiration)
    NEAR_TERM_DAYS = 30
    FAR_TERM_DAYS = 90

    def collect(self) -> dict | None:
        ticker = yf.Ticker(IBIT_TICKER)

        # Get current price
        try:
            hist = ticker.history(period="1d")
            if hist.empty:
                self.logger.warning("No IBIT price data (market may be closed)")
                return None
            current_price = hist["Close"].iloc[-1]
        except Exception as e:
            self.logger.error(f"Failed to get IBIT price: {e}")
            return None

        # Get all expiration dates
        try:
            expirations = ticker.options
        except Exception as e:
            self.logger.error(f"Failed to get option expirations: {e}")
            return None

        if not expirations:
            self.logger.warning("No option expirations available")
            return None

        # Parse all options into a unified list
        now = datetime.now(timezone.utc)
        all_options = []

        for date_str in expirations:
            try:
                exp_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                days_to_exp = (exp_date - now).days
                if days_to_exp < 0:
                    continue

                chain = ticker.option_chain(date_str)

                # Process calls
                calls = chain.calls
                if not calls.empty:
                    calls = calls[calls["openInterest"] > 0].copy()
                    for _, row in calls.iterrows():
                        all_options.append({
                            "expiration": exp_date,
                            "type": "C",
                            "strike": row["strike"],
                            "oi": row["openInterest"],
                            "breakeven": row["strike"] + row["lastPrice"],
                            "days_to_exp": days_to_exp,
                        })

                # Process puts
                puts = chain.puts
                if not puts.empty:
                    puts = puts[puts["openInterest"] > 0].copy()
                    for _, row in puts.iterrows():
                        all_options.append({
                            "expiration": exp_date,
                            "type": "P",
                            "strike": row["strike"],
                            "oi": row["openInterest"],
                            "breakeven": row["strike"] - row["lastPrice"],
                            "days_to_exp": days_to_exp,
                        })

            except Exception as e:
                self.logger.debug(f"Skipping expiration {date_str}: {e}")
                continue

        if not all_options:
            self.logger.warning("No valid IBIT options parsed")
            return None

        df = pd.DataFrame(all_options)
        calls = df[df["type"] == "C"]
        puts = df[df["type"] == "P"]

        total_call_oi = calls["oi"].sum()
        total_put_oi = puts["oi"].sum()
        total_oi = total_call_oi + total_put_oi

        if total_oi < IBIT_MIN_OI_THRESHOLD:
            self.logger.warning(f"Total OI {total_oi} below threshold {IBIT_MIN_OI_THRESHOLD}")
            return None

        if total_call_oi == 0:
            self.logger.warning("Zero call OI for IBIT")
            return None

        # === Aggregate metrics ===
        pcr = total_put_oi / total_call_oi
        w_call_be = (calls["breakeven"] * calls["oi"]).sum() / total_call_oi
        w_put_be = (puts["breakeven"] * puts["oi"]).sum() / total_put_oi if total_put_oi > 0 else 0

        consensus_all = (
            (calls["breakeven"] * calls["oi"]).sum()
            + (puts["breakeven"] * puts["oi"]).sum()
        ) / total_oi
        bias_all = ((consensus_all - current_price) / current_price) * 100

        # === Time-bucketed consensus bias ===
        bias_7d = self._consensus_for_window(df, current_price, max_days=7)
        bias_30d = self._consensus_for_window(df, current_price, max_days=30)
        bias_90d = self._consensus_for_window(df, current_price, max_days=90)

        # === Near/far-term skew ===
        near = df[df["days_to_exp"] <= self.NEAR_TERM_DAYS]
        near_call_oi = near[near["type"] == "C"]["oi"].sum()
        near_put_oi = near[near["type"] == "P"]["oi"].sum()
        near_skew = near_put_oi / near_call_oi if near_call_oi > 0 else None

        far = df[df["days_to_exp"] > self.FAR_TERM_DAYS]
        far_call_oi = far[far["type"] == "C"]["oi"].sum()
        far_put_oi = far[far["type"] == "P"]["oi"].sum()
        far_skew = far_put_oi / far_call_oi if far_call_oi > 0 else None

        return {
            "ibit_price": current_price,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "put_call_ratio": pcr,
            "weighted_call_breakeven": w_call_be,
            "weighted_put_breakeven": w_put_be,
            "consensus_price": consensus_all,
            "consensus_bias_pct": bias_all,
            "consensus_bias_7d": bias_7d,
            "consensus_bias_30d": bias_30d,
            "consensus_bias_90d": bias_90d,
            "near_term_skew": near_skew,
            "far_term_skew": far_skew,
        }

    def _consensus_for_window(self, df: pd.DataFrame, spot: float, max_days: int) -> float | None:
        """
        Compute OI-weighted consensus bias for options expiring within max_days.
        Returns percentage bias vs spot, or None if insufficient data.
        """
        window = df[df["days_to_exp"] <= max_days]
        if window.empty:
            return None

        total_oi = window["oi"].sum()
        if total_oi < 100:  # Minimum OI for meaningful signal
            return None

        weighted_sum = (window["breakeven"] * window["oi"]).sum()
        consensus = weighted_sum / total_oi
        return ((consensus - spot) / spot) * 100
