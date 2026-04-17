"""
Macro Data Collector
====================
Collects macroeconomic indicators relevant to BTC price prediction:
- DXY (US Dollar Index)
- US 10-Year Treasury Yield
- S&P 500
- QQQ (NASDAQ 100)
- WTI Crude Oil
- BTC/Gold ratio
- US M2 Money Supply + M2 YoY growth rate (via FRED API)

All market data from yfinance (free, no key).
M2 data from FRED API (free key required).
"""
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone, timedelta

from data.collectors.base import BaseCollector

# FRED API key from environment
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


class MacroCollector(BaseCollector):

    table_name = "macro"

    # yfinance tickers
    TICKERS = {
        "dxy": "DX-Y.NYB",      # US Dollar Index
        "us10y": "^TNX",         # 10-Year Treasury Yield
        "sp500": "^GSPC",        # S&P 500
        "qqq": "QQQ",            # NASDAQ 100 ETF
        "oil": "CL=F",           # WTI Crude Oil
        "gold": "GC=F",          # Gold futures
        "btc": "BTC-USD",        # Bitcoin (for BTC/gold ratio)
    }

    def collect(self) -> dict | None:
        data = {}

        # === yfinance market data ===
        try:
            # Fetch all tickers at once for efficiency
            tickers_str = " ".join(self.TICKERS.values())
            download = yf.download(tickers_str, period="8d", progress=False, group_by="ticker")

            for name, ticker in self.TICKERS.items():
                try:
                    if len(self.TICKERS) > 1:
                        series = download[ticker]["Close"].dropna()
                    else:
                        series = download["Close"].dropna()

                    if len(series) >= 2:
                        current = float(series.iloc[-1])
                        prev_7d = float(series.iloc[0]) if len(series) >= 5 else float(series.iloc[0])

                        data[f"{name}_price"] = current
                        data[f"{name}_7d_return"] = ((current - prev_7d) / prev_7d) * 100
                    elif len(series) == 1:
                        data[f"{name}_price"] = float(series.iloc[-1])
                        data[f"{name}_7d_return"] = None
                except Exception as e:
                    self.logger.debug(f"Failed to get {name}: {e}")
                    data[f"{name}_price"] = None
                    data[f"{name}_7d_return"] = None

        except Exception as e:
            self.logger.warning(f"yfinance batch download failed: {e}")

        # === BTC/Gold ratio ===
        btc_price = data.get("btc_price")
        gold_price = data.get("gold_price")
        if btc_price and gold_price and gold_price > 0:
            data["btc_gold_ratio"] = btc_price / gold_price
        else:
            data["btc_gold_ratio"] = None

        # === FRED M2 Money Supply ===
        m2, m2_yoy = self._get_m2()
        data["m2_supply"] = m2
        data["m2_yoy_growth"] = m2_yoy

        # Clean up - remove the raw btc and gold prices since we have them
        # in the price collector already. Keep the ratios and returns.
        result = {
            "dxy": data.get("dxy_price"),
            "dxy_7d_return": data.get("dxy_7d_return"),
            "us10y_yield": data.get("us10y_price"),
            "sp500_7d_return": data.get("sp500_7d_return"),
            "qqq_7d_return": data.get("qqq_7d_return"),
            "oil_price": data.get("oil_price"),
            "oil_7d_return": data.get("oil_7d_return"),
            "btc_gold_ratio": data.get("btc_gold_ratio"),
            "m2_supply": data.get("m2_supply"),
            "m2_yoy_growth": data.get("m2_yoy_growth"),
        }

        # Check we got at least some data
        non_null = sum(1 for v in result.values() if v is not None)
        if non_null < 3:
            self.logger.warning(f"Only {non_null} macro fields populated")
            return None

        return result

    def _get_m2(self) -> tuple[float | None, float | None]:
        """
        Fetch US M2 money supply from FRED.
        Returns (latest_m2_billions, yoy_growth_pct).
        M2 is released weekly on Tuesdays.
        """
        if not FRED_API_KEY:
            self.logger.warning(
                "FRED_API_KEY not set. Skipping M2 data. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            return None, None

        try:
            # WM2NS = M2 money stock, not seasonally adjusted, weekly
            resp = requests.get(
                FRED_BASE_URL,
                params={
                    "series_id": "WM2NS",
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 60,  # ~1 year of weekly data
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            observations = data.get("observations", [])
            if not observations:
                self.logger.warning("No M2 observations from FRED")
                return None, None

            # Filter out missing values
            valid = [(o["date"], float(o["value"])) for o in observations if o["value"] != "."]

            if not valid:
                return None, None

            latest_m2 = valid[0][1]  # Most recent value (billions)

            # YoY growth: compare to ~52 weeks ago
            if len(valid) >= 52:
                year_ago_m2 = valid[51][1]
                yoy_growth = ((latest_m2 - year_ago_m2) / year_ago_m2) * 100
            else:
                yoy_growth = None

            self.logger.info(f"M2: ${latest_m2:.0f}B, YoY: {yoy_growth:.1f}%" if yoy_growth else f"M2: ${latest_m2:.0f}B")

            return latest_m2, yoy_growth

        except Exception as e:
            self.logger.warning(f"FRED API error: {e}")
            return None, None

    def collect_historical(self, days: int = 2000) -> list[dict]:
        """
        Backfill macro data from yfinance and FRED.
        Returns list of daily dicts with 'timestamp'.
        """
        rows = []

        try:
            # Fetch all market data
            tickers_str = " ".join(self.TICKERS.values())
            download = yf.download(tickers_str, period=f"{days}d", progress=False, group_by="ticker")

            if download.empty:
                self.logger.warning("No historical macro data from yfinance")
                return rows

            # Get dates from index
            dates = download.index

            for i, date in enumerate(dates):
                row = {"timestamp": date.strftime("%Y-%m-%d 00:00:00")}

                for name, ticker in self.TICKERS.items():
                    try:
                        val = float(download[ticker]["Close"].iloc[i])
                        if not np.isnan(val):
                            row[f"{name}_price"] = val
                    except (KeyError, IndexError):
                        pass

                # BTC/Gold ratio
                btc = row.get("btc_price")
                gold = row.get("gold_price")
                if btc and gold and gold > 0:
                    row["btc_gold_ratio"] = btc / gold

                # Compute 7d returns
                if i >= 5:
                    for name, ticker in self.TICKERS.items():
                        try:
                            current = float(download[ticker]["Close"].iloc[i])
                            prev = float(download[ticker]["Close"].iloc[i - 5])
                            if not np.isnan(current) and not np.isnan(prev) and prev > 0:
                                row[f"{name}_7d_return"] = ((current - prev) / prev) * 100
                        except (KeyError, IndexError):
                            pass

                # Map to final schema
                final_row = {
                    "timestamp": row["timestamp"],
                    "dxy": row.get("dxy_price"),
                    "dxy_7d_return": row.get("dxy_7d_return"),
                    "us10y_yield": row.get("us10y_price"),
                    "sp500_7d_return": row.get("sp500_7d_return"),
                    "qqq_7d_return": row.get("qqq_7d_return"),
                    "oil_price": row.get("oil_price"),
                    "oil_7d_return": row.get("oil_7d_return"),
                    "btc_gold_ratio": row.get("btc_gold_ratio"),
                    "m2_supply": None,  # Filled separately from FRED
                    "m2_yoy_growth": None,
                }

                rows.append(final_row)

        except Exception as e:
            self.logger.error(f"Historical macro backfill error: {e}")

        # Backfill M2 data from FRED
        if FRED_API_KEY:
            try:
                m2_data = self._backfill_m2()
                # Merge M2 into rows by nearest date
                if m2_data:
                    m2_df = pd.DataFrame(m2_data)
                    m2_df["date"] = pd.to_datetime(m2_df["date"])

                    for row in rows:
                        row_date = pd.to_datetime(row["timestamp"])
                        # Find nearest M2 reading on or before this date
                        mask = m2_df["date"] <= row_date
                        if mask.any():
                            nearest = m2_df[mask].iloc[-1]
                            row["m2_supply"] = nearest["m2"]
                            row["m2_yoy_growth"] = nearest.get("m2_yoy")

            except Exception as e:
                self.logger.warning(f"M2 backfill error: {e}")

        self.logger.info(f"Backfilled {len(rows)} historical macro rows")
        return rows

    def _backfill_m2(self) -> list[dict]:
        """Fetch full M2 history from FRED."""
        resp = requests.get(
            FRED_BASE_URL,
            params={
                "series_id": "WM2NS",
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "asc",
                "observation_start": "2019-01-01",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        observations = [o for o in data.get("observations", []) if o["value"] != "."]

        for i, obs in enumerate(observations):
            m2 = float(obs["value"])
            m2_yoy = None
            if i >= 52:
                year_ago = float(observations[i - 52]["value"])
                m2_yoy = ((m2 - year_ago) / year_ago) * 100

            results.append({
                "date": obs["date"],
                "m2": m2,
                "m2_yoy": m2_yoy,
            })

        return results
