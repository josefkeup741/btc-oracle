"""
BTC Price Collector
===================
Fetches current BTC OHLCV data. Uses CoinGecko free API.
Also supports historical backfill via market_chart endpoint.
"""
import requests
import pandas as pd
from datetime import datetime, timezone

from data.collectors.base import BaseCollector
from config import COINGECKO_PRICE_URL, COINGECKO_MARKET_CHART_URL


class PriceCollector(BaseCollector):

    table_name = "btc_price"

    def collect(self) -> dict | None:
        """Fetch current BTC price. For OHLCV we use the market_chart endpoint."""
        # Get recent 1-day data at hourly granularity, take the latest candle
        resp = requests.get(
            COINGECKO_MARKET_CHART_URL,
            params={
                "vs_currency": "usd",
                "days": "1",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        if not prices:
            self.logger.warning("No price data from CoinGecko")
            return None

        # CoinGecko returns [timestamp_ms, value] pairs
        # Get the last few entries to compute OHLC over the latest ~4h window
        # For a single snapshot, we use the latest price as close and approximate OHLC
        latest_price = prices[-1][1]

        # Approximate 4h OHLC from the last ~4 hours of data points
        # CoinGecko returns ~every 5 min for 1-day range
        four_hours_points = prices[-48:] if len(prices) >= 48 else prices  # ~48 * 5min = 4h
        price_values = [p[1] for p in four_hours_points]

        latest_volume = volumes[-1][1] if volumes else 0

        return {
            "open": price_values[0] if price_values else latest_price,
            "high": max(price_values) if price_values else latest_price,
            "low": min(price_values) if price_values else latest_price,
            "close": latest_price,
            "volume": latest_volume,
        }

    def collect_historical(self, days: int = 365) -> list[dict]:
        """
        Backfill daily OHLC data from CoinGecko.
        For days > 90, CoinGecko returns daily granularity.
        Returns list of dicts with 'timestamp'.
        """
        resp = requests.get(
            COINGECKO_MARKET_CHART_URL,
            params={
                "vs_currency": "usd",
                "days": str(days),
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        # Build volume lookup
        vol_map = {}
        for ts_ms, vol in volumes:
            vol_map[ts_ms] = vol

        rows = []
        for ts_ms, price in prices:
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            rows.append({
                "timestamp": ts.isoformat(),
                "open": price,  # CoinGecko daily only gives close, we approximate
                "high": price,
                "low": price,
                "close": price,
                "volume": vol_map.get(ts_ms, 0),
            })

        self.logger.info(f"Fetched {len(rows)} historical price entries")
        return rows
