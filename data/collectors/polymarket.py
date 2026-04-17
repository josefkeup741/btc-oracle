"""
Polymarket Collector (Fixed)
=============================
Fetches BTC-related prediction market data from Polymarket.
Updated to use the events endpoint and broader search terms.

Note: Polymarket BTC markets come and go. When no active market exists,
we store None values. The model handles this via NaN.
"""
import requests

from data.collectors.base import BaseCollector
from config import POLYMARKET_CLOB_URL


class PolymarketCollector(BaseCollector):

    table_name = "polymarket"

    # Broader search to catch more market formats
    BTC_KEYWORDS = ["bitcoin", "btc", "crypto", "cryptocurrency"]
    PRICE_KEYWORDS = [
        "price", "above", "below", "reach", "hit", "ath", "all-time",
        "100k", "150k", "200k", "50k", "75k", "end of",
        "by december", "by june", "by march", "this year", "this month",
    ]

    def collect(self) -> dict | None:
        markets = self._search_markets()

        if not markets:
            self.logger.info("No active BTC price markets found on Polymarket")
            return {
                "btc_primary_prob": None,
                "btc_primary_question": None,
                "market_volume_24h": 0,
                "prob_delta_24h": None,
            }

        # Sort by volume, take highest-volume market
        markets.sort(key=lambda x: x["volume"], reverse=True)
        primary = markets[0]
        total_volume = sum(m["volume"] for m in markets)

        self.logger.info(
            f"Found {len(markets)} BTC markets. "
            f"Primary: '{primary['question'][:60]}' prob={primary['probability']:.2f}"
        )

        return {
            "btc_primary_prob": primary["probability"],
            "btc_primary_question": primary["question"][:200],
            "market_volume_24h": total_volume,
            "prob_delta_24h": None,  # Computed from history in features.py
        }

    def _search_markets(self) -> list[dict]:
        """Search for BTC price markets using multiple approaches."""
        all_markets = []

        # Approach 1: /markets endpoint
        try:
            resp = requests.get(
                f"{POLYMARKET_CLOB_URL}/markets",
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    data = data.get("data", data.get("markets", []))
                if isinstance(data, list):
                    all_markets.extend(data)
        except Exception as e:
            self.logger.debug(f"Markets endpoint error: {e}")

        # Approach 2: Search with query parameter
        for query in ["bitcoin", "btc price", "crypto"]:
            try:
                resp = requests.get(
                    f"{POLYMARKET_CLOB_URL}/markets",
                    params={"tag": query},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        data = data.get("data", data.get("markets", []))
                    if isinstance(data, list):
                        all_markets.extend(data)
            except Exception:
                continue

        # Deduplicate by question/title
        seen = set()
        unique_markets = []
        for m in all_markets:
            key = m.get("question", m.get("title", ""))
            if key and key not in seen:
                seen.add(key)
                unique_markets.append(m)

        # Filter for BTC price markets
        btc_markets = []
        for market in unique_markets:
            question = (market.get("question", "") or market.get("title", "")).lower()
            description = (market.get("description", "")).lower()
            combined = question + " " + description

            is_btc = any(kw in combined for kw in self.BTC_KEYWORDS)
            is_price = any(kw in combined for kw in self.PRICE_KEYWORDS)
            is_active = market.get("active", True) and not market.get("closed", False)

            if is_btc and is_price and is_active:
                volume = 0
                try:
                    volume = float(market.get("volume", 0) or market.get("volumeNum", 0) or 0)
                except (ValueError, TypeError):
                    pass

                probability = self._extract_probability(market)
                if probability is not None:
                    btc_markets.append({
                        "question": market.get("question", market.get("title", "Unknown")),
                        "probability": probability,
                        "volume": volume,
                    })

        return btc_markets

    def _extract_probability(self, market: dict) -> float | None:
        """Extract the 'Yes' probability from various market formats."""
        # Method 1: tokens array
        tokens = market.get("tokens", [])
        for token in tokens:
            outcome = (token.get("outcome", "") or "").lower()
            if outcome == "yes":
                try:
                    return float(token.get("price", 0))
                except (ValueError, TypeError):
                    pass

        # Method 2: outcomePrices array
        prices = market.get("outcomePrices")
        if prices and isinstance(prices, list):
            try:
                return float(prices[0])
            except (ValueError, TypeError, IndexError):
                pass

        # Method 3: direct probability field
        prob = market.get("probability")
        if prob is not None:
            try:
                return float(prob)
            except (ValueError, TypeError):
                pass

        return None
