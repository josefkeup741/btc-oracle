"""
Deribit Options Collector
=========================
Refactored from btcSmartDumbDeribit.ipynb.
Fetches all BTC options from Deribit public API, computes OI-weighted
breakevens, put/call ratios, near/far-term skew, and max pain.
"""
import requests
import pandas as pd
from datetime import datetime, timezone

from data.collectors.base import BaseCollector
from config import (
    DERIBIT_URL, DERIBIT_MIN_OI,
    DERIBIT_NEAR_TERM_DAYS, DERIBIT_FAR_TERM_DAYS,
)


class DeribitCollector(BaseCollector):

    table_name = "deribit_options"

    def collect(self) -> dict | None:
        params = {"currency": "BTC", "kind": "option"}
        resp = requests.get(DERIBIT_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "result" not in data or len(data["result"]) == 0:
            self.logger.warning("No results from Deribit API")
            return None

        btc_price = data["result"][0]["underlying_price"]

        # Parse all options into a DataFrame
        options = []
        now = datetime.now(timezone.utc)

        for item in data["result"]:
            name = item["instrument_name"]
            parts = name.split("-")
            if len(parts) != 4:
                continue

            try:
                strike = float(parts[2])
                exp_date = datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=timezone.utc)
            except (ValueError, IndexError):
                continue

            if exp_date < now:
                continue

            oi = item.get("open_interest", 0)
            if oi <= 0:
                continue

            mark_price_usd = item["mark_price"] * btc_price
            opt_type = parts[3]  # 'C' or 'P'

            if opt_type == "C":
                breakeven = strike + mark_price_usd
            else:
                breakeven = strike - mark_price_usd

            days_to_exp = (exp_date - now).days

            options.append({
                "expiration": exp_date,
                "type": opt_type,
                "strike": strike,
                "oi": oi,
                "breakeven": breakeven,
                "days_to_exp": days_to_exp,
            })

        if not options:
            self.logger.warning("No valid options parsed")
            return None

        df = pd.DataFrame(options)
        calls = df[df["type"] == "C"]
        puts = df[df["type"] == "P"]

        total_call_oi = calls["oi"].sum()
        total_put_oi = puts["oi"].sum()

        if total_call_oi == 0:
            self.logger.warning("Zero call OI")
            return None

        pcr = total_put_oi / total_call_oi

        # OI-weighted breakevens
        w_call_be = (calls["breakeven"] * calls["oi"]).sum() / total_call_oi
        w_put_be = (puts["breakeven"] * puts["oi"]).sum() / total_put_oi if total_put_oi > 0 else 0

        total_oi = total_call_oi + total_put_oi
        consensus = (
            (calls["breakeven"] * calls["oi"]).sum()
            + (puts["breakeven"] * puts["oi"]).sum()
        ) / total_oi

        consensus_bias = ((consensus - btc_price) / btc_price) * 100

        # Near-term skew (< 30 days)
        near = df[df["days_to_exp"] <= DERIBIT_NEAR_TERM_DAYS]
        near_calls_oi = near[near["type"] == "C"]["oi"].sum()
        near_puts_oi = near[near["type"] == "P"]["oi"].sum()
        near_skew = near_puts_oi / near_calls_oi if near_calls_oi > 0 else None

        # Far-term skew (> 90 days)
        far = df[df["days_to_exp"] > DERIBIT_FAR_TERM_DAYS]
        far_calls_oi = far[far["type"] == "C"]["oi"].sum()
        far_puts_oi = far[far["type"] == "P"]["oi"].sum()
        far_skew = far_puts_oi / far_calls_oi if far_calls_oi > 0 else None

        # Max pain for nearest monthly expiry (most OI)
        max_pain = self._calc_max_pain(df, btc_price)

        return {
            "btc_spot": btc_price,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "put_call_ratio": pcr,
            "weighted_call_breakeven": w_call_be,
            "weighted_put_breakeven": w_put_be,
            "consensus_price": consensus,
            "consensus_bias_pct": consensus_bias,
            "near_term_skew": near_skew,
            "far_term_skew": far_skew,
            "max_pain_30d": max_pain,
        }

    def _calc_max_pain(self, df: pd.DataFrame, spot: float) -> float | None:
        """
        Max pain: the strike price where total option holder losses are maximized
        (i.e., where most options expire worthless). Computed for the nearest
        high-OI expiration.
        """
        # Find the nearest expiration with meaningful OI
        exp_oi = df.groupby("expiration")["oi"].sum()
        exp_oi = exp_oi[exp_oi >= DERIBIT_MIN_OI].sort_index()

        if exp_oi.empty:
            return None

        target_exp = exp_oi.index[0]
        subset = df[df["expiration"] == target_exp]
        strikes = sorted(subset["strike"].unique())

        if len(strikes) < 3:
            return None

        min_pain = float("inf")
        max_pain_strike = strikes[0]

        for test_strike in strikes:
            total_pain = 0
            for _, row in subset.iterrows():
                if row["type"] == "C":
                    # Call holder loss if price settles at test_strike
                    intrinsic = max(0, test_strike - row["strike"])
                    # They paid the breakeven - strike, so loss = premium - intrinsic
                    # Simplified: pain to call holder = max(0, row['strike'] - test_strike) * oi
                    # Actually, max pain = sum of ITM value * OI for all options
                    pain = max(0, test_strike - row["strike"]) * row["oi"]
                else:
                    pain = max(0, row["strike"] - test_strike) * row["oi"]
                total_pain += pain

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return max_pain_strike
