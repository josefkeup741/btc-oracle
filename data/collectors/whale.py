"""
Whale Activity Collector (Free Version)
=========================================
Tracks large BTC transactions and exchange flows using free APIs:
1. Blockchair API - recent large transactions (free: 30 req/min)
2. Blockchain.com API - known exchange wallet balance tracking

No API key required. Zero cost.

The key insight: we don't need Whale Alert's labeling service.
Exchange cold wallet addresses are publicly known. By tracking their
balance changes between collection periods, we get net exchange flow
(the most predictive whale signal) for free.
"""
import requests
from datetime import datetime, timezone

from data.collectors.base import BaseCollector


class WhaleCollector(BaseCollector):

    table_name = "whale_activity"

    # Well-known exchange cold/hot wallet addresses (publicly documented)
    # Source: OXT, Arkham, and blockchain explorers
    # We track cumulative received/sent; the *delta* between collections = flow
    EXCHANGE_ADDRESSES = {
        "binance": [
            "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
            "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h",
        ],
        "coinbase": [
            "3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb",
            "395xRileQEAMv1MzSNaLt3YDpSoiHg6RCi",
        ],
        "bitfinex": [
            "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
        ],
        "kraken": [
            "bc1qr4dl5wa7kl8yu792dceg9z5knl2gkn220lk7a9",
        ],
    }

    # Free API endpoints
    BLOCKCHAIR_TX_URL = "https://api.blockchair.com/bitcoin/transactions?s=output_total(desc)&limit=10"
    BLOCKCHAIN_BALANCE_URL = "https://blockchain.info/balance?active={addresses}"

    def collect(self) -> dict | None:
        large_tx_count, large_tx_volume = self._get_large_transactions()
        exchange_inflow, exchange_outflow, net_flow = self._get_exchange_flows()

        avg_size = large_tx_volume / large_tx_count if large_tx_count > 0 else 0

        return {
            "large_tx_count": large_tx_count,
            "large_tx_volume_btc": large_tx_volume,
            "exchange_inflow_btc": exchange_inflow,
            "exchange_outflow_btc": exchange_outflow,
            "net_exchange_flow": net_flow,
            "whale_tx_avg_size": avg_size,
        }

    def _get_large_transactions(self) -> tuple[int, float]:
        """
        Get recent large BTC transactions from Blockchair.
        Returns (count, total_volume_btc) for transactions >= 100 BTC.
        """
        try:
            resp = requests.get(self.BLOCKCHAIR_TX_URL, timeout=15)
            if resp.status_code != 200:
                self.logger.warning(f"Blockchair status {resp.status_code}")
                return 0, 0.0

            data = resp.json()
            txs = data.get("data", [])

            count = 0
            volume = 0.0
            for tx in txs:
                output_btc = tx.get("output_total", 0) / 1e8
                if output_btc >= 100:
                    count += 1
                    volume += output_btc

            return count, volume

        except Exception as e:
            self.logger.warning(f"Blockchair error: {e}")
            return 0, 0.0

    def _get_exchange_flows(self) -> tuple[float, float, float]:
        """
        Track known exchange wallet balances via Blockchain.com.

        Returns (total_received, total_sent, net_flow) in BTC.

        These are CUMULATIVE values. The model learns from the *change*
        between collection periods (computed in features.py via
        whale_flow_7d_trend). A rising net_flow means more BTC flowing
        into exchanges over time (bearish). A falling net_flow means
        accumulation (bullish).
        """
        try:
            all_addresses = []
            for addrs in self.EXCHANGE_ADDRESSES.values():
                all_addresses.extend(addrs)

            addr_str = "|".join(all_addresses)
            resp = requests.get(
                self.BLOCKCHAIN_BALANCE_URL.format(addresses=addr_str),
                timeout=15,
            )

            if resp.status_code != 200:
                self.logger.warning(f"Blockchain.com status {resp.status_code}")
                return 0.0, 0.0, 0.0

            balances = resp.json()

            total_received = 0.0
            total_sent = 0.0

            for addr, info in balances.items():
                total_received += info.get("total_received", 0) / 1e8
                total_sent += info.get("total_sent", 0) / 1e8

            net_flow = total_received - total_sent

            self.logger.info(
                f"Exchange balances: received={total_received:.2f} BTC, "
                f"sent={total_sent:.2f} BTC, net={net_flow:.2f} BTC"
            )

            return total_received, total_sent, net_flow

        except Exception as e:
            self.logger.warning(f"Blockchain.com error: {e}")
            return 0.0, 0.0, 0.0
