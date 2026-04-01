"""
Polymarket Collector
====================
Fetches BTC-related prediction market data from Polymarket's CLOB API.
Looks for active markets about Bitcoin price targets and extracts
implied probabilities.
"""
import requests

from data.collectors.base import BaseCollector
from config import POLYMARKET_CLOB_URL


class PolymarketCollector(BaseCollector):

    table_name = "polymarket"

    # Keywords to identify BTC-relevant markets
    BTC_KEYWORDS = ["bitcoin", "btc", "crypto"]
    # Keywords to identify price-related markets specifically
    PRICE_KEYWORDS = ["price", "above", "below", "reach", "hit", "ath", "all-time high"]

    def collect(self) -> dict | None:
        """
        Search for active BTC price prediction markets.
        Extract the highest-volume market's probability as the primary signal.
        """
        try:
            # Search for BTC-related markets
            # Polymarket CLOB API: GET /markets for listing
            resp = requests.get(
                f"{POLYMARKET_CLOB_URL}/markets",
                timeout=15,
            )
            resp.raise_for_status()
            markets = resp.json()

            # Handle both list and dict responses
            if isinstance(markets, dict):
                markets = markets.get("data", markets.get("markets", []))
            if not isinstance(markets, list):
                self.logger.warning(f"Unexpected Polymarket response format: {type(markets)}")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Polymarket API error: {e}")
            return None

        # Filter for BTC price markets
        btc_markets = []
        for market in markets:
            question = (market.get("question", "") or market.get("title", "")).lower()
            description = (market.get("description", "")).lower()
            combined = question + " " + description

            is_btc = any(kw in combined for kw in self.BTC_KEYWORDS)
            is_price = any(kw in combined for kw in self.PRICE_KEYWORDS)
            is_active = market.get("active", True) and not market.get("closed", False)

            if is_btc and is_price and is_active:
                volume = float(market.get("volume", 0) or market.get("volumeNum", 0) or 0)
                # Get the "Yes" probability
                tokens = market.get("tokens", [])
                yes_prob = None
                for token in tokens:
                    if token.get("outcome", "").lower() == "yes":
                        yes_prob = float(token.get("price", 0))
                        break

                # Some markets store probability directly
                if yes_prob is None:
                    yes_prob = float(market.get("outcomePrices", [0.5])[0]) if market.get("outcomePrices") else None

                if yes_prob is not None:
                    btc_markets.append({
                        "question": market.get("question", market.get("title", "Unknown")),
                        "probability": yes_prob,
                        "volume": volume,
                    })

        if not btc_markets:
            self.logger.info("No active BTC price markets found on Polymarket")
            # Return empty but valid data rather than None so we still get a timestamp
            return {
                "btc_primary_prob": None,
                "btc_primary_question": None,
                "market_volume_24h": 0,
                "prob_delta_24h": None,
            }

        # Sort by volume, take the highest-volume market as primary signal
        btc_markets.sort(key=lambda x: x["volume"], reverse=True)
        primary = btc_markets[0]

        total_volume = sum(m["volume"] for m in btc_markets)

        return {
            "btc_primary_prob": primary["probability"],
            "btc_primary_question": primary["question"][:200],  # Truncate for storage
            "market_volume_24h": total_volume,
            "prob_delta_24h": None,  # Computed in features.py from stored history
        }
