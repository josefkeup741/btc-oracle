"""
IBIT ETF Options Collector
==========================
Refactored from btcSmartDumbIBIT.ipynb.
Fetches IBIT option chains via yfinance, computes OI-weighted breakevens.
Only runs during market hours (yfinance returns stale data otherwise).
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

from data.collectors.base import BaseCollector
from config import IBIT_TICKER, IBIT_MIN_OI_THRESHOLD


class IBITCollector(BaseCollector):

    table_name = "ibit_options"

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

        all_call_oi = 0
        all_put_oi = 0
        call_weighted_sum = 0
        put_weighted_sum = 0

        for date in expirations:
            try:
                chain = ticker.option_chain(date)

                # Process calls
                calls = chain.calls
                if not calls.empty:
                    calls = calls[calls["openInterest"] > 0].copy()
                    if not calls.empty:
                        calls["breakeven"] = calls["strike"] + calls["lastPrice"]
                        c_oi = calls["openInterest"].sum()
                        all_call_oi += c_oi
                        call_weighted_sum += (calls["breakeven"] * calls["openInterest"]).sum()

                # Process puts
                puts = chain.puts
                if not puts.empty:
                    puts = puts[puts["openInterest"] > 0].copy()
                    if not puts.empty:
                        puts["breakeven"] = puts["strike"] - puts["lastPrice"]
                        p_oi = puts["openInterest"].sum()
                        all_put_oi += p_oi
                        put_weighted_sum += (puts["breakeven"] * puts["openInterest"]).sum()

            except Exception as e:
                self.logger.debug(f"Skipping expiration {date}: {e}")
                continue

        total_oi = all_call_oi + all_put_oi
        if total_oi < IBIT_MIN_OI_THRESHOLD:
            self.logger.warning(f"Total OI {total_oi} below threshold {IBIT_MIN_OI_THRESHOLD}")
            return None

        if all_call_oi == 0:
            self.logger.warning("Zero call OI for IBIT")
            return None

        pcr = all_put_oi / all_call_oi
        w_call_be = call_weighted_sum / all_call_oi
        w_put_be = put_weighted_sum / all_put_oi if all_put_oi > 0 else 0
        consensus = (call_weighted_sum + put_weighted_sum) / total_oi
        bias_pct = ((consensus - current_price) / current_price) * 100

        return {
            "ibit_price": current_price,
            "total_call_oi": all_call_oi,
            "total_put_oi": all_put_oi,
            "put_call_ratio": pcr,
            "weighted_call_breakeven": w_call_be,
            "weighted_put_breakeven": w_put_be,
            "consensus_price": consensus,
            "consensus_bias_pct": bias_pct,
        }
