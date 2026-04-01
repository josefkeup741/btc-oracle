"""
Fear & Greed Index Collector
=============================
Fetches the current Bitcoin Fear & Greed Index from alternative.me.
Also supports fetching historical data for backfill.
"""
import requests

from data.collectors.base import BaseCollector
from config import FEAR_GREED_URL


class FearGreedCollector(BaseCollector):

    table_name = "fear_greed"

    def collect(self) -> dict | None:
        resp = requests.get(
            FEAR_GREED_URL,
            params={"limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if "data" not in data or len(data["data"]) == 0:
            self.logger.warning("No FnG data returned")
            return None

        entry = data["data"][0]
        return {
            "fng_value": float(entry["value"]),
            "fng_classification": entry["value_classification"],
        }

    def collect_historical(self, days: int = 365) -> list[dict]:
        """
        Fetch historical FnG data for backfill.
        Returns list of dicts with 'timestamp' included.
        The API returns daily data going back to 2018.
        """
        resp = requests.get(
            FEAR_GREED_URL,
            params={"limit": days, "format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        rows = []
        for entry in data.get("data", []):
            # API returns unix timestamp
            from datetime import datetime, timezone
            ts = datetime.fromtimestamp(int(entry["timestamp"]), tz=timezone.utc)
            rows.append({
                "timestamp": ts.isoformat(),
                "fng_value": float(entry["value"]),
                "fng_classification": entry["value_classification"],
            })

        self.logger.info(f"Fetched {len(rows)} historical FnG entries")
        return rows
